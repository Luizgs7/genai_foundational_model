"""F2-H10 — LLM open source como codificador de clientes (ajuste fino com LoRA).

Três modos, sobre o histórico em texto de f2h10_textos.py (uma linha por cesta):

  extrair      embedding de cada âncora = estado oculto da última camada no fim
               da linha da cesta-âncora (`--pooling ultimo`, o padrão do
               Qwen3-Embedding, que é causal e usa o último token) ou a média
               acumulada dos estados até ali (`--pooling media`, como no LLM4ES).
               Como o modelo é causal, o estado no fim de uma linha só depende das
               cestas até ela: uma passada por janela dá o embedding de todas as
               âncoras da janela, sem vazamento. Janela principal = últimas linhas
               que cabem em `--max_tokens`; âncoras anteriores ganham janelas extras
               (mesma regra da extração do backbone próprio).
  contrastivo  LoRA com objetivo contrastivo (CoLES): duas fatias contíguas
               aleatórias do histórico de TREINO do mesmo cliente devem gerar
               vetores próximos; os outros clientes do lote são os negativos
               (InfoNCE simétrica, temperatura `--tau`).
  lm           LoRA por próximo token sobre o histórico de treino (estilo LLM4ES).

Uso (Colab):
  python3 pipeline/common/f2h10_llm.py configs/hm_clientes.yaml contrastivo --modelo Qwen/Qwen3-Embedding-0.6B --saida f2h10_e3a
  python3 pipeline/common/f2h10_llm.py configs/hm_clientes.yaml extrair --modelo Qwen/Qwen3-Embedding-0.6B \\
          [--adaptador f2h10_e3a/adaptador] --saida f2h10_e3a
Gera runs/<nome>/<saida>/{embeddings_eventos.npy, embeddings_eventos_index.parquet, relatorio_<modo>.json}
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

sys.path.insert(0, "pipeline/common")
from schema import carregar_config, run_dir  # noqa: E402

INSTRUCAO = ("Instruct: Represente o histórico de compras deste cliente de moda para prever o seu comportamento futuro\n"
             "Query: ")


def carregar_textos(config, max_clientes=0):
    t = pl.read_parquet(os.path.join(run_dir(config, "f2h10"), "textos.parquet")).sort("entidade_id", "data", "evento_id")
    if max_clientes:
        t = t.join(t.select("entidade_id").unique().sort("entidade_id").head(max_clientes), on="entidade_id", how="semi")
    return t


def tokenizar(tok, t):
    """-> por cliente: lista de arrays de ids (uma por linha, terminando em \\n), evento_id e split das linhas."""
    ids = tok(t["linha"].to_list(), add_special_tokens=False)["input_ids"]
    nl = tok("\n", add_special_tokens=False)["input_ids"]
    ent = t["entidade_id"].to_numpy()
    ini = np.r_[0, np.flatnonzero(np.diff(ent)) + 1]
    fim = np.r_[ini[1:], len(ent)]
    ev = t["evento_id"].to_numpy()
    sp = t["split"].cast(pl.String).to_numpy()
    clientes = []
    for a, b in zip(ini, fim):
        clientes.append((int(ent[a]), [np.asarray(ids[k] + nl, dtype=np.int64) for k in range(a, b)], ev[a:b], sp[a:b]))
    return clientes


def montar_modelo(a, treino):
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.modelo)
    # um adaptador treinado no modo `lm` tem os nomes de camada do modelo com cabeça de linguagem:
    # a extração precisa carregar a mesma classe (--causal_lm), senão o LoRA não casa com nada
    cls = AutoModelForCausalLM if (a.modo == "lm" or a.causal_lm) else AutoModel
    m = cls.from_pretrained(a.modelo, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    if a.adaptador:
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, os.path.join(run_dir(a.config_obj, ""), a.adaptador))
        carregado = sum(float(p.detach().float().abs().sum()) for n, p in m.named_parameters() if "lora_B" in n)
        assert carregado > 0, "o adaptador LoRA não foi aplicado (pesos lora_B zerados): classe do modelo diferente da do treino?"
        m = m.merge_and_unload()
    if treino:
        from peft import LoraConfig, get_peft_model
        m.gradient_checkpointing_enable()
        m.enable_input_require_grads()
        m = get_peft_model(m, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
        m.print_trainable_parameters()
    return tok, m.cuda()


def estados(m, ids, att):
    out = m(input_ids=ids, attention_mask=att, output_hidden_states=True)
    return out.hidden_states[-1] if hasattr(out, "hidden_states") and out.hidden_states is not None else out.last_hidden_state


# ------------------------------------------------------------------------- extrair

def extrair(a, config):
    t0 = time.time()
    tok, m = montar_modelo(a, treino=False)
    m.eval()
    anc = pl.read_parquet(os.path.join(run_dir(config, "rotulos_downstream"), "ancoras.parquet"))
    t = carregar_textos(config, a.max_clientes)
    eh_anc = t.select(pl.col("evento_id").is_in(anc["evento_id"].implode())).to_series().to_numpy()
    pre = np.asarray(tok(INSTRUCAO, add_special_tokens=False)["input_ids"], dtype=np.int64) if a.instrucao else np.zeros(0, np.int64)
    clientes = tokenizar(tok, t)
    print(f"{len(clientes):,} clientes tokenizados; {int(eh_anc.sum()):,} âncoras ({time.time() - t0:.0f}s)", flush=True)

    # janelas: (cliente, linha inicial, linha final exclusiva); âncoras = linhas cujo fim cai na janela
    janelas, k_ini = [], 0
    max_corpo = a.max_tokens - len(pre)
    alvo_linhas = []
    for ci, (_, linhas, ev, _) in enumerate(clientes):
        n = len(linhas)
        lens = np.array([len(x) for x in linhas])
        anc_c = np.flatnonzero(eh_anc[k_ini:k_ini + n])
        k_ini += n
        cum = np.r_[0, np.cumsum(lens)]
        coberto = n
        fim = n
        while True:
            # linhas [ini, fim) que cabem; uma linha maior que a janela é cortada à esquerda
            ini = min(int(np.searchsorted(cum, cum[fim] - max_corpo, side="left")), fim - 1)
            sel = anc_c[(anc_c >= ini) & (anc_c < coberto)]
            if len(sel):
                janelas.append((ci, ini, fim))
                alvo_linhas.append(sel)
            coberto = ini
            restantes = anc_c[anc_c < coberto]
            if not len(restantes):
                break
            fim = int(restantes.max()) + 1
    tam = np.array([min(a.max_tokens, len(pre) + sum(len(clientes[c][1][k]) for k in range(i, f))) for c, i, f in janelas])
    n_anc = sum(len(x) for x in alvo_linhas)
    print(f"{len(janelas):,} janelas, {n_anc:,} âncoras, {tam.sum() / 1e6:.1f}M tokens", flush=True)

    out_dir = run_dir(config, a.saida)
    os.makedirs(out_dir, exist_ok=True)
    dim = m.config.hidden_size
    emb = np.lib.format.open_memmap(os.path.join(out_dir, "embeddings_eventos.npy"), mode="w+", dtype=np.float16, shape=(n_anc, dim))
    idx_ev, idx_ent = np.zeros(n_anc, np.int64), np.zeros(n_anc, np.int64)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    ordem = np.argsort(tam, kind="stable")
    linha_out = np.cumsum([0] + [len(x) for x in alvo_linhas])

    def rodar(lote):
        T = int(max(tam[j] for j in lote))
        ids = np.full((len(lote), T), pad, np.int64)
        att = np.zeros((len(lote), T), np.int64)
        fins = []
        for r, j in enumerate(lote):
            c, i, f = janelas[j]
            seq = np.concatenate([pre] + clientes[c][1][i:f])[-a.max_tokens:]
            corte = len(np.concatenate([pre] + clientes[c][1][i:f])) - len(seq)
            ids[r, :len(seq)] = seq; att[r, :len(seq)] = 1
            ends = np.cumsum([len(pre)] + [len(x) for x in clientes[c][1][i:f]])[1:] - 1 - corte
            fins.append(ends[alvo_linhas[j] - i])
        ids_t, att_t = torch.from_numpy(ids).cuda(), torch.from_numpy(att).cuda()
        with torch.no_grad():
            h = estados(m, ids_t, att_t).float()
            if a.pooling == "media":
                cs = torch.cumsum(h * att_t[..., None], dim=1) / torch.cumsum(att_t, dim=1)[..., None].clamp(min=1)
                h = cs
            for r, j in enumerate(lote):
                v = F.normalize(h[r, torch.from_numpy(np.maximum(fins[r], 0)).cuda()], dim=-1).cpu().numpy()
                o = slice(linha_out[j], linha_out[j + 1])
                emb[o] = v.astype(np.float16)
                c, i, _ = janelas[j]
                idx_ev[o] = clientes[c][2][alvo_linhas[j]]
                idx_ent[o] = clientes[c][0]

    atual, maior, feitos, t_ult = [], 0, 0, time.time()
    for j in ordem:
        novo = max(maior, tam[j])
        if atual and (len(atual) + 1) * novo > a.tokens_por_lote:
            rodar(atual); feitos += len(atual)
            atual, novo = [], tam[j]
            if time.time() - t_ult > 120:
                print(f"  {feitos:,}/{len(janelas):,} janelas ({time.time() - t0:.0f}s)", flush=True); t_ult = time.time()
        atual.append(j)
        maior = novo
    if atual:
        rodar(atual)
    emb.flush()
    pl.DataFrame({"evento_id": idx_ev, "entidade_id": idx_ent.astype(np.int32), "row_idx": np.arange(n_anc)}).write_parquet(
        os.path.join(out_dir, "embeddings_eventos_index.parquet"))
    rel = {"modo": "extrair", "modelo": a.modelo, "adaptador": a.adaptador, "pooling": a.pooling, "instrucao": a.instrucao,
           "max_tokens": a.max_tokens, "n_ancoras": int(n_anc), "n_janelas": len(janelas), "tokens": int(tam.sum()),
           "dim": int(dim), "segundos": round(time.time() - t0, 1),
           "tokens_por_s": round(float(tam.sum()) / (time.time() - t0), 1)}
    json.dump(rel, open(os.path.join(out_dir, "relatorio_extrair.json"), "w"), indent=1)
    print(json.dumps(rel))


# ------------------------------------------------------------------------- treinos

def historicos_treino(tok, config, max_clientes=0):
    t = carregar_textos(config, max_clientes).filter(pl.col("split") == "train")
    return [c for c in tokenizar(tok, t) if len(c[1]) >= 1]


def fatia(rng, linhas, max_corpo):
    """fatia contígua aleatória de cestas (CoLES), cortada pela esquerda até caber."""
    n = len(linhas)
    comp = rng.randint(max(1, n // 4), n + 1) if n > 1 else 1
    ini = rng.randint(0, n - comp + 1)
    sel = linhas[ini:ini + comp]
    seq = np.concatenate(sel)
    return seq[-max_corpo:]


def lote_padded(seqs, pad):
    T = max(len(s) for s in seqs)
    ids = np.full((len(seqs), T), pad, np.int64)
    att = np.zeros((len(seqs), T), np.int64)
    for r, s in enumerate(seqs):
        ids[r, :len(s)] = s; att[r, :len(s)] = 1
    return torch.from_numpy(ids).cuda(), torch.from_numpy(att).cuda()


def treinar(a, config):
    t0 = time.time()
    tok, m = montar_modelo(a, treino=True)
    pre = np.asarray(tok(INSTRUCAO, add_special_tokens=False)["input_ids"], dtype=np.int64) if (a.instrucao and a.modo == "contrastivo") else np.zeros(0, np.int64)
    hist = historicos_treino(tok, config, a.max_clientes)
    if a.modo == "contrastivo":
        hist = [h for h in hist if len(h[1]) >= 2]
    print(f"{len(hist):,} clientes com histórico de treino ({time.time() - t0:.0f}s)", flush=True)
    rng = np.random.RandomState(a.seed)
    passos = a.passos or math.ceil(len(hist) * a.epocas / a.lote)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 100) * max(0.0, 1 - s / passos))
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    out_dir = run_dir(config, a.saida)
    os.makedirs(out_dir, exist_ok=True)
    ordem = rng.permutation(len(hist))
    log, p0, perdas, ini = [], 0, [], 0
    # estado completo a cada N passos (o Colab cai): pesos treináveis, otimizador, agenda de lr e geradores
    est_path = os.path.join(out_dir, "estado_treino.pt")
    treinaveis = {n: p for n, p in m.named_parameters() if p.requires_grad}
    if a.retomar and os.path.exists(est_path):
        est = torch.load(est_path, map_location="cuda", weights_only=False)
        with torch.no_grad():
            for n, v in est["params"].items():
                treinaveis[n].copy_(v)
        opt.load_state_dict(est["opt"]); sched.load_state_dict(est["sched"])
        rng.set_state(est["rng"]); torch.set_rng_state(est["torch_rng"]); torch.cuda.set_rng_state_all(est["cuda_rng"])
        ordem, p0, log, perdas, ini = est["ordem"], est["p0"], est["log"], est["perdas"], est["passo"]
        t0 -= est["segundos"]
        print(f"retomando do passo {ini}", flush=True)
    m.train()
    for passo in range(ini, passos):
        if p0 + a.lote > len(ordem):
            ordem, p0 = rng.permutation(len(hist)), 0
        lote = [hist[i] for i in ordem[p0:p0 + a.lote]]
        p0 += a.lote
        if a.modo == "contrastivo":
            max_corpo = a.max_tokens - len(pre)
            vs = [np.concatenate([pre, fatia(rng, h[1], max_corpo)]) for h in lote] + \
                 [np.concatenate([pre, fatia(rng, h[1], max_corpo)]) for h in lote]
            ids, att = lote_padded(vs, pad)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = estados(m, ids, att)
            fim = att.sum(1) - 1
            z = F.normalize(h[torch.arange(len(vs), device=h.device), fim].float(), dim=-1)
            za, zb = z[:a.lote], z[a.lote:]
            sim = za @ zb.T / a.tau
            alvo = torch.arange(a.lote, device=sim.device)
            perda = 0.5 * (F.cross_entropy(sim, alvo) + F.cross_entropy(sim.T, alvo))
        else:
            vs = [np.concatenate(h[1])[-a.max_tokens:] for h in lote]
            ids, att = lote_padded(vs, pad)
            rot = ids.masked_fill(att == 0, -100)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                perda = m(input_ids=ids, attention_mask=att, labels=rot).loss
        opt.zero_grad(set_to_none=True)
        perda.backward()
        torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], 1.0)
        opt.step(); sched.step()
        perdas.append(float(perda.detach()))
        if (passo + 1) % 50 == 0 or passo == passos - 1:
            reg = {"passo": passo + 1, "perda": float(np.mean(perdas[-50:])), "s": round(time.time() - t0, 1)}
            log.append(reg)
            print(json.dumps(reg), flush=True)
        if (passo + 1) % 500 == 0 or passo == passos - 1:
            m.save_pretrained(os.path.join(out_dir, "adaptador"))
        if a.estado_a_cada and (passo + 1) % a.estado_a_cada == 0 and passo != passos - 1:
            est = {"passo": passo + 1, "params": {n: p.detach().clone() for n, p in treinaveis.items()},
                   "opt": opt.state_dict(), "sched": sched.state_dict(), "rng": rng.get_state(),
                   "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                   "ordem": ordem, "p0": p0, "log": log, "perdas": perdas[-50:], "segundos": time.time() - t0}
            torch.save(est, est_path + ".tmp")
            os.replace(est_path + ".tmp", est_path)
    rel = {"modo": a.modo, "modelo": a.modelo, "passos": passos, "lote": a.lote, "lr": a.lr, "tau": a.tau,
           "max_tokens": a.max_tokens, "n_clientes": len(hist), "segundos": round(time.time() - t0, 1), "log": log}
    json.dump(rel, open(os.path.join(out_dir, f"relatorio_{a.modo}.json"), "w"), indent=1)
    if os.path.exists(est_path):
        os.remove(est_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("modo", choices=["extrair", "contrastivo", "lm"])
    ap.add_argument("--modelo", required=True)
    ap.add_argument("--adaptador", default=None, help="subpasta (em runs/<nome>) com o adaptador LoRA")
    ap.add_argument("--saida", required=True)
    ap.add_argument("--pooling", default="ultimo", choices=["ultimo", "media"])
    ap.add_argument("--sem_instrucao", dest="instrucao", action="store_false")
    ap.add_argument("--causal_lm", action="store_true", help="na extração: carrega o modelo com cabeça de linguagem (adaptador treinado no modo lm)")
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--tokens_por_lote", type=int, default=65536)
    ap.add_argument("--lote", type=int, default=32)
    ap.add_argument("--epocas", type=float, default=1.0)
    ap.add_argument("--passos", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_clientes", type=int, default=0, help="só para teste rápido")
    ap.add_argument("--retomar", action="store_true", help="continua de <saida>/estado_treino.pt, se existir")
    ap.add_argument("--estado_a_cada", type=int, default=100, help="passos entre gravações do estado de treino (0 = não grava)")
    a = ap.parse_args()
    config = carregar_config(a.config)
    a.config_obj = config
    torch.manual_seed(a.seed)
    if a.modo == "extrair":
        extrair(a, config)
    else:
        treinar(a, config)


if __name__ == "__main__":
    main()
