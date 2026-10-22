"""F2-H2 — Ajuste fino conjunto do backbone com a fusão DCNv2 (experimento H&M, fase 2).

Até aqui o backbone ficou CONGELADO: o embedding de cada âncora era extraído
uma vez (tarefa13_extrair_embeddings.py) e só a cabeça DCNv2 aprendia as
tarefas (tarefa13_treinar_fusao.py). Este script treina os DOIS juntos, de
ponta a ponta — a receita "joint fusion" que o Nubank relata ter superado o
LightGBM (nuFormer, arXiv 2507.23267): o gradiente das tarefas de negócio
ajusta também o transformer, que passa a representar o que importa para elas.

Tudo o mais é idêntico à fusão da H4, para a comparação ser justa:
  - mesmas âncoras, rótulos, splits, entradas de cadastro e causais e mesma
    cabeça DCNv2 (classes e pré-processamento importados de tarefa13_treinar_fusao);
  - cada âncora vê exatamente a mesma janela de contexto da extração de
    embeddings (janela principal + janelas extras), e o vetor usado é o estado
    oculto no último token do evento da âncora;
  - seleção pelo AUC de churn na validação; teste só com --avaliar_teste;
  - saídas no mesmo formato (tarefa13_relatorio.json, predicoes_<split>.parquet).

Sem vazamento: a atenção é causal, então o estado na âncora não vê tokens
posteriores; e só âncoras de TREINO geram gradiente.

Treino: AdamW com duas taxas (backbone pequena, cabeça maior), aquecimento
linear e decaimento linear; bf16 autocast. Checkpoint a cada N passos e
--retomar (o Colab cai). Opcional --frac_treino/--semente_amostra (F2-H1).

Uso: python3 pipeline/common/tarefa15_ajuste_conjunto.py configs/hm_clientes.yaml \\
        --pretreino pretreino --saida f2_h2_conjunto_s0 --seed 0 [--epocas 1] [--avaliar_teste] [--retomar]
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, "pipeline/common")
from modelo_decoder import arquitetura_efetiva, construir_modelo  # noqa: E402
from schema import amostra_clientes, caminho_tabela, carregar_config, ler_tabela, run_dir, tarefas_ativas  # noqa: E402
from sequencias_io import carregar_sequencias  # noqa: E402
from tarefa13_treinar_fusao import (CROSS_RANK, DEEP_HIDDEN, N_CROSS, DCNv2Fusao, carregar_dados,  # noqa: E402
                                    inferir_tipo_tarefa)
from tarefa8_rotulos import accuracy, auc_roc, f1_macro  # noqa: E402

TOKENS_TREINO = 32768
TOKENS_AVAL = 65536


def atribuir_janelas(seqs, posicoes, n_positions):
    """Mesma regra da extração de embeddings: janela principal = últimos
    n_positions tokens de seq_full[:-1]; âncoras anteriores ganham janelas
    extras terminando nelas. -> (posicoes filtradas, janelas DataFrame[ent, ini, fim],
    janela_id por âncora, posição local da âncora na janela)."""
    idx_ent = posicoes["entidade_id"].map(pd.Series(np.arange(len(seqs.ids)), index=seqs.ids))
    posicoes = posicoes.assign(_ent=idx_ent.to_numpy(dtype=np.int64)).sort_values(["_ent", "pos_full"]).reset_index(drop=True)
    ent_a = posicoes["_ent"].to_numpy()
    pos_a = posicoes["pos_full"].to_numpy(dtype=np.int64)
    total_a = seqs.len_full[ent_a] - 1
    fim_janela = total_a.copy()
    ok = pos_a < total_a
    fora = ok & (pos_a < total_a - n_positions)
    inicio_ent = np.r_[0, np.flatnonzero(np.diff(ent_a)) + 1]
    fim_ent = np.r_[inicio_ent[1:], len(ent_a)]
    tem_fora = np.add.reduceat(fora.astype(np.int64), inicio_ent) > 0
    for a, b in zip(inicio_ent[tem_fora], fim_ent[tem_fora]):
        f = total_a[a]
        for k in range(b - 1, a - 1, -1):
            if pos_a[k] < f - n_positions:
                f = pos_a[k] + 1
            fim_janela[k] = f
    posicoes, ent_a, pos_a, fim_janela = posicoes[ok].reset_index(drop=True), ent_a[ok], pos_a[ok], fim_janela[ok]
    base = int(fim_janela.max()) + 1
    chaves, janela_id = np.unique(ent_a * base + fim_janela, return_inverse=True)
    janelas = pd.DataFrame({"ent": chaves // base, "fim": chaves % base})
    janelas["ini"] = np.maximum(0, janelas["fim"].to_numpy() - n_positions)
    local = pos_a - janelas["ini"].to_numpy()[janela_id]
    return posicoes, janelas, janela_id, local


def lotes_por_orcamento(ordem, tam, orcamento):
    atual, maior = [], 0
    for k in ordem:
        novo = max(maior, tam[k])
        if atual and (len(atual) + 1) * novo > orcamento:
            yield atual
            atual, novo = [], tam[k]
        atual.append(k)
        maior = novo
    if atual:
        yield atual


def main(a):
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    t0 = time.time()
    config = carregar_config(a.config)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    autocast = lambda: torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=dev == "cuda")  # noqa: E731

    # ---------------- dados das âncoras (idênticos à fusão da H4) ----------------
    _emb, df, cat_estaticos, num_estaticos = carregar_dados(config, a.emb_indice, False, False)
    for campo in cat_estaticos:
        valores = sorted(df.loc[df["split"] == "train", campo].dropna().unique(), key=str)
        mapa = {v: i for i, v in enumerate(valores)}
        df[f"{campo}_id"] = df[campo].map(mapa).fillna(len(mapa)).astype(int)
    cards = [int(df[f"{c}_id"].max()) for c in cat_estaticos]
    for campo in num_estaticos:
        media = df.loc[df["split"] == "train", campo].mean()
        desvio = df.loc[df["split"] == "train", campo].std() or 1.0
        df[f"{campo}_norm"] = ((df[campo] - media) / desvio).fillna(0.0)
    tarefas_info = []
    for tcfg in tarefas_ativas(config):
        nome = tcfg.get("nome", tcfg["tipo"])
        tipo, mapa_cls = inferir_tipo_tarefa(df[f"{nome}_rotulo"])
        tarefas_info.append((nome, tipo, len(mapa_cls) if mapa_cls else 1, mapa_cls))
        if mapa_cls:
            df[f"{nome}_rotulo_id"] = df[f"{nome}_rotulo"].map(mapa_cls)
    tarefas_modelo = [(n, t, c) for n, t, c, _ in tarefas_info]
    if a.frac_treino < 1.0:
        fora = (df["split"] == "train") & ~amostra_clientes(df["entidade_id"], a.frac_treino, a.semente_amostra)
        df = df[~fora].reset_index(drop=True)

    # janelas de contexto por âncora (mesma regra da extração)
    arch = arquitetura_efetiva(json.load(open(os.path.join(run_dir(config, a.pretreino), "pretreino_config.json")))["arquitetura"])
    vocab = json.load(open(f"{run_dir(config, 'vocabulario')}/vocab.json", encoding="utf-8"))
    pad_id = vocab["PAD"]
    seqs = carregar_sequencias(config)
    posicoes = ler_tabela(caminho_tabela(config, "serializacao", "posicoes_evento"))
    posicoes = posicoes[posicoes["evento_id"].isin(df["evento_id"])]
    posicoes, janelas, janela_id, local = atribuir_janelas(seqs, posicoes, arch["n_positions"])
    df = df.set_index("evento_id").loc[posicoes["evento_id"]].reset_index()   # mesma ordem das âncoras
    split = df["split"].to_numpy()
    ini, fim, ent_j = janelas["ini"].to_numpy(), janelas["fim"].to_numpy(), janelas["ent"].to_numpy()
    tam = fim - ini

    cat_ids = [torch.tensor(df[f"{c}_id"].to_numpy(), dtype=torch.long) for c in cat_estaticos]
    num_vals = torch.tensor(df[[f"{c}_norm" for c in num_estaticos]].to_numpy(dtype=np.float32)) if num_estaticos else None
    alvos = {}
    for nome, tipo, _n, mapa_cls in tarefas_info:
        col = f"{nome}_rotulo_id" if mapa_cls else f"{nome}_rotulo"
        alvos[nome] = torch.tensor(df[col].fillna(-1 if mapa_cls else float("nan")).to_numpy(),
                                   dtype=torch.long if mapa_cls else torch.float32)
    ordem_anc = np.argsort(janela_id, kind="stable")
    jan_ord = janela_id[ordem_anc]
    anc_ini = np.searchsorted(jan_ord, np.arange(len(janelas)), side="left")
    anc_fim = np.searchsorted(jan_ord, np.arange(len(janelas)), side="right")

    def ancoras_da_janela(k, s):
        idx = ordem_anc[anc_ini[k]:anc_fim[k]]
        return idx[split[idx] == s]

    jan_split = {s: np.unique(janela_id[split == s]) for s in ("train", "val", "test")}
    print(f"âncoras: treino={int((split == 'train').sum()):,} val={int((split == 'val').sum()):,} "
          f"teste={int((split == 'test').sum()):,}; janelas de treino={len(jan_split['train']):,} "
          f"({int(tam[jan_split['train']].sum()):,} tokens por época)", flush=True)

    # ---------------- modelo: backbone pré-treinado + cabeça DCNv2 ----------------
    backbone = construir_modelo(arch, vocab).to(dev)
    backbone.carregar_pesos(torch.load(os.path.join(run_dir(config, a.pretreino), "checkpoints", "melhor.pt"), map_location=dev))
    cabeca = DCNv2Fusao(arch["n_embd"], cards, len(num_estaticos), N_CROSS, CROSS_RANK, DEEP_HIDDEN, tarefas_modelo).to(dev)
    pos_w = {}
    for nome, tipo, _n in tarefas_modelo:
        if tipo == "binaria":
            y = alvos[nome][torch.from_numpy(split == "train")]
            y = y[~torch.isnan(y)]
            pos_w[nome] = torch.tensor(float((y == 0).sum()) / max(float((y == 1).sum()), 1.0), device=dev)

    def forward(lote, s, treino):
        seqs_l = [seqs.full(ent_j[k])[ini[k]:fim[k]] for k in lote]
        T = max(len(x) for x in seqs_l)
        ids = np.full((len(seqs_l), T), pad_id, dtype=np.int64)
        for r, x in enumerate(seqs_l):
            ids[r, :len(x)] = x
        anc, lin, col = [], [], []
        for r, k in enumerate(lote):
            ii = ancoras_da_janela(k, s)
            anc.append(ii); lin.append(np.full(len(ii), r)); col.append(local[ii])
        anc, lin, col = np.concatenate(anc), np.concatenate(lin), np.concatenate(col)
        ids_t = torch.from_numpy(ids).to(dev)
        with autocast():
            h = backbone.estados(ids_t, (ids_t != pad_id).long())
            e = h[torch.from_numpy(lin).to(dev), torch.from_numpy(col).to(dev)].float()
        a_t = torch.from_numpy(anc)
        out = cabeca(e, [c[a_t].to(dev) for c in cat_ids], num_vals[a_t].to(dev) if num_vals is not None else None)
        return out, a_t

    def perda(out, a_t):
        total = 0.0
        for nome, tipo, _n in tarefas_modelo:
            y = alvos[nome][a_t].to(dev)
            if tipo == "binaria":
                m = ~torch.isnan(y)
                if m.any():
                    total = total + F.binary_cross_entropy_with_logits(out[nome].squeeze(-1)[m].float(), y[m], pos_weight=pos_w[nome])
            else:
                m = y >= 0
                if m.any():
                    total = total + F.cross_entropy(out[nome][m].float(), y[m])
        return total

    @torch.no_grad()
    def prever(s, max_janelas=None):
        backbone.eval(); cabeca.eval()
        js = jan_split[s]
        if max_janelas and len(js) > max_janelas:
            js = np.sort(np.random.RandomState(123).choice(js, max_janelas, replace=False))
        js = js[np.argsort(tam[js], kind="stable")]
        sc = {n: [] for n, _t, _c in tarefas_modelo}
        idxs = []
        for lote in lotes_por_orcamento(js, tam, TOKENS_AVAL):
            out, a_t = forward(lote, s, False)
            idxs.append(a_t.numpy())
            for nome, tipo, _n in tarefas_modelo:
                o = out[nome].float()
                sc[nome].append((torch.sigmoid(o.squeeze(-1)) if tipo == "binaria" else torch.softmax(o, -1)).cpu().numpy())
        backbone.train(); cabeca.train()
        idx = np.concatenate(idxs)
        return idx, {n: np.concatenate(v) for n, v in sc.items()}

    def metricas(idx, sc):
        r = {}
        for nome, tipo, _n in tarefas_modelo:
            y = alvos[nome][torch.from_numpy(idx)].numpy()
            if tipo == "binaria":
                r[nome] = {"auc": auc_roc(pd.Series(y), pd.Series(sc[nome]))}
            else:
                pred = sc[nome].argmax(1).astype(float)
                yv = np.where(y < 0, np.nan, y); pv = np.where(y < 0, np.nan, pred)
                r[nome] = {"accuracy": accuracy(pd.Series(yv), pd.Series(pv)), "f1_macro": f1_macro(pd.Series(yv), pd.Series(pv))}
        return r

    # ---------------- otimização ----------------
    rng = np.random.RandomState(a.seed)
    planos = []
    for _ in range(a.epocas):
        js = rng.permutation(jan_split["train"])
        bloco = 2000
        js = np.concatenate([b[np.argsort(tam[b], kind="stable")] for b in np.array_split(js, max(1, len(js) // bloco))])
        lotes = list(lotes_por_orcamento(js, tam, TOKENS_TREINO))
        planos.append([lotes[k] for k in rng.permutation(len(lotes))])
    total = sum(len(p) for p in planos)
    opt = torch.optim.AdamW([{"params": backbone.parameters(), "lr": a.lr_backbone, "weight_decay": 0.01},
                             {"params": cabeca.parameters(), "lr": a.lr_cabeca, "weight_decay": 1e-4}])
    aquec = min(500, total // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda p: min(1.0, (p + 1) / aquec) * max(0.1, 1 - p / total))

    out_dir = run_dir(config, a.saida)
    os.makedirs(out_dir, exist_ok=True)
    ult = os.path.join(out_dir, "ultimo.pt")
    passo, melhor, hist = 0, -1.0, []
    if a.retomar and os.path.exists(ult):
        ck = torch.load(ult, map_location=dev, weights_only=False)
        backbone.load_state_dict(ck["backbone"]); cabeca.load_state_dict(ck["cabeca"])
        opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        passo, melhor, hist = ck["passo"], ck["melhor"], ck["hist"]
        print(f"retomando do passo {passo}/{total}", flush=True)

    def salvar_ultimo():
        torch.save({"backbone": backbone.state_dict(), "cabeca": cabeca.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "passo": passo, "melhor": melhor, "hist": hist}, ult + ".tmp")
        os.replace(ult + ".tmp", ult)

    def avaliar_parcial():
        nonlocal melhor
        idx, sc = prever("val", a.val_janelas)
        m = metricas(idx, sc)
        chave = m[tarefas_modelo[0][0]].get("auc", m[tarefas_modelo[0][0]].get("accuracy"))
        hist.append({"passo": passo, "val_amostra": m})
        msg = "  ".join(f"{n}={list(v.values())[0]:.4f}" for n, v in m.items())
        print(f"[passo {passo}/{total}] val (amostra): {msg}  ({time.time() - t0:.0f}s)", flush=True)
        if chave > melhor:
            melhor = chave
            torch.save({"backbone": backbone.state_dict(), "cabeca": cabeca.state_dict()}, os.path.join(out_dir, "melhor.pt"))

    backbone.train(); cabeca.train()
    perdas, p_global = [], 0
    for ep, plano in enumerate(planos):
        for lote in plano:
            p_global += 1
            if p_global <= passo:
                continue
            out, a_t = forward(lote, "train", True)
            loss = perda(out, a_t)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(backbone.parameters()) + list(cabeca.parameters()), 1.0)
            opt.step(); sched.step()
            passo = p_global
            perdas.append(float(loss))
            if passo % 200 == 0:
                print(f"passo {passo}/{total} ép {ep + 1} loss={np.mean(perdas[-200:]):.4f} ({time.time() - t0:.0f}s)", flush=True)
            if passo % a.avaliar_a_cada == 0:
                avaliar_parcial()
            if passo % a.checkpoint_a_cada == 0:
                salvar_ultimo()
    avaliar_parcial()
    salvar_ultimo()

    # ---------------- avaliação final com o melhor checkpoint ----------------
    ck = torch.load(os.path.join(out_dir, "melhor.pt"), map_location=dev)
    backbone.load_state_dict(ck["backbone"]); cabeca.load_state_dict(ck["cabeca"])

    def gravar(s):
        idx, sc = prever(s, a.val_janelas if a.smoke else None)
        sub = df.iloc[idx]
        cols = {"evento_id": sub["evento_id"].to_numpy(), "entidade_id": sub["entidade_id"].to_numpy()}
        for nome, tipo, _n, mapa_cls in tarefas_info:
            if tipo == "binaria":
                cols[f"{nome}_score"] = sc[nome]
            else:
                for c, i in mapa_cls.items():
                    cols[f"{nome}_p_{c}"] = sc[nome][:, i]
        pd.DataFrame(cols).to_parquet(os.path.join(out_dir, f"predicoes_{s}.parquet"), index=False)
        return metricas(idx, sc)

    val_m = gravar("val")
    print("Validação (melhor checkpoint):", json.dumps(val_m), flush=True)
    test_m = gravar("test") if a.avaliar_teste else None
    if test_m:
        print("TESTE (avaliação única):", json.dumps(test_m), flush=True)
    rel = {"metodo": "ajuste fino conjunto backbone + DCNv2 (F2-H2)", "pretreino": a.pretreino, "seed": a.seed,
           "epocas": a.epocas, "lr_backbone": a.lr_backbone, "lr_cabeca": a.lr_cabeca, "passos": total,
           "frac_treino": a.frac_treino, "semente_amostra": a.semente_amostra,
           "n_train": int((split == "train").sum()), "n_val": int((split == "val").sum()), "n_test": int((split == "test").sum()),
           "metricas_validacao": val_m, "metricas_teste": test_m, "historico_val_amostra": hist,
           "tempo_total_segundos": time.time() - t0}
    with open(os.path.join(out_dir, "tarefa13_relatorio.json"), "w", encoding="utf-8") as f:
        json.dump(rel, f, ensure_ascii=False, indent=2, default=str)
    print(f"Gravado {out_dir} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--pretreino", default="pretreino")
    ap.add_argument("--emb_indice", default="fusao", help="subpasta cujo índice de embeddings define as âncoras (mesmas da H4)")
    ap.add_argument("--saida", default="f2_h2_conjunto_s0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epocas", type=int, default=1)
    ap.add_argument("--lr_backbone", type=float, default=5e-5)
    ap.add_argument("--lr_cabeca", type=float, default=3e-3)
    ap.add_argument("--avaliar_a_cada", type=int, default=2000)
    ap.add_argument("--val_janelas", type=int, default=60000, help="janelas de validação na avaliação parcial")
    ap.add_argument("--checkpoint_a_cada", type=int, default=500)
    ap.add_argument("--frac_treino", type=float, default=1.0)
    ap.add_argument("--semente_amostra", type=int, default=0)
    ap.add_argument("--avaliar_teste", action="store_true")
    ap.add_argument("--retomar", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="teste rápido: avaliação final só na amostra de validação")
    main(ap.parse_args())
