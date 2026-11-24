"""F3-H9 — ajuste fino conjunto do backbone do motor v2 com a cabeça DCNv2 (desenho em PLANO_FASE3.md).

O backbone pré-treinado (v2_pretreino) e a cabeça DCNv2 da fusão (tarefa13_treinar_fusao.py) são
treinados juntos sobre as âncoras de treino, com todas as tarefas do YAML ao mesmo tempo. Taxas
separadas (backbone pequena, cabeça maior), aquecimento e decaimento lineares, como na F2-H2 (v1).

Cada âncora usa exatamente o contexto da extração dos vetores (v2_extrair.janelas_sobrepostas: janela
de até `janela` posições, causal), então o estado de uma âncora só depende do que veio antes dela.
Entradas da cabeça (cadastro e estatísticas causais), codificação, pos_weight e formato das
predições são os da fusão; a avaliação (avaliacao.py) lê as predições sem mudança.

Uso: python3 pipeline/common/v2_ajuste_conjunto.py <config.yaml> --emb v2_fusao --saida f3h9_ft_s0 \\
        [--v2 v2] [--pretreino v2_pretreino] [--seed 0] [--epocas 3] [--avaliar_teste]
`--emb` só fornece a lista de âncoras (o índice da extração); os vetores congelados não são usados.
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
from schema import carregar_config, run_dir, tarefas_ativas  # noqa: E402
from tarefa13_treinar_fusao import (CROSS_RANK, DEEP_HIDDEN, N_CROSS, DCNv2Fusao, carregar_dados,  # noqa: E402
                                    inferir_tipo_tarefa)
from tarefa8_rotulos import accuracy, auc_roc  # noqa: E402
from v2_extrair import carregar_modelo, janelas_sobrepostas  # noqa: E402
from v2_modelo import DadosV2, montar_lote  # noqa: E402
from v2_pretreino import lotes_por_orcamento  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--emb", default="v2_fusao", help="pasta com embeddings_eventos_index.parquet (lista de âncoras)")
    ap.add_argument("--v2", default="v2", help="pasta da serialização do v2 em runs/<nome>/")
    ap.add_argument("--pretreino", default="v2_pretreino")
    ap.add_argument("--saida", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epocas", type=int, default=3)
    ap.add_argument("--lr_backbone", type=float, default=5e-5)
    ap.add_argument("--lr_cabeca", type=float, default=3e-3)
    ap.add_argument("--aquecimento", type=float, default=0.05, help="fração dos passos com aquecimento linear")
    ap.add_argument("--orcamento", type=int, default=60000, help="posições por lote no treino")
    ap.add_argument("--orcamento_aval", type=int, default=120000)
    ap.add_argument("--max_janelas_treino", type=int, default=0, help="só para testes rápidos")
    ap.add_argument("--avaliar_teste", action="store_true")
    ap.add_argument("--dispositivo", default="auto", choices=["auto", "cpu", "cuda"])
    a = ap.parse_args()
    t0 = time.time()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    rng = np.random.RandomState(a.seed)
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if a.dispositivo == "auto" else a.dispositivo
    amp = dev == "cuda"
    config = carregar_config(a.config)
    out = run_dir(config, a.saida)
    os.makedirs(out, exist_ok=True)

    # ---------------- dados da cabeça: os mesmos da fusão
    _emb, df, cat_estaticos, num_estaticos = carregar_dados(config, a.emb)
    del _emb
    cat_ids = []
    for campo in cat_estaticos:
        mapa = {v: i for i, v in enumerate(sorted(df.loc[df["split"] == "train", campo].dropna().unique(), key=str))}
        cat_ids.append((df[campo].map(mapa).fillna(len(mapa)).astype(int).to_numpy(), len(mapa)))
    num = np.zeros((len(df), len(num_estaticos)), np.float32)
    for j, campo in enumerate(num_estaticos):
        media = df.loc[df["split"] == "train", campo].mean()
        desvio = df.loc[df["split"] == "train", campo].std() or 1.0
        num[:, j] = ((df[campo] - media) / desvio).fillna(0.0).to_numpy(np.float32)
    tarefas_info, alvos = [], {}
    for t in tarefas_ativas(config):
        nome = t.get("nome", t["tipo"])
        tipo, mapa_cl = inferir_tipo_tarefa(df[f"{nome}_rotulo"])
        tarefas_info.append((nome, tipo, len(mapa_cl) if mapa_cl else 1, mapa_cl))
        if mapa_cl:
            alvos[nome] = torch.tensor(df[f"{nome}_rotulo"].map(mapa_cl).fillna(-1).to_numpy(), dtype=torch.long)
        else:
            alvos[nome] = torch.tensor(df[f"{nome}_rotulo"].to_numpy(dtype=np.float64), dtype=torch.float32)
    tarefas_modelo = [(n, t, c) for n, t, c, _ in tarefas_info]
    split = df["split"].astype(str).to_numpy()
    eh_treino = split == "train"
    pos_weights = {}
    for nome, tipo, _ in tarefas_modelo:
        if tipo == "binaria":
            y = alvos[nome][torch.from_numpy(eh_treino)]
            y = y[~torch.isnan(y)]
            pos_weights[nome] = torch.tensor(float((y == 0).sum()) / max(float((y == 1).sum()), 1.0), device=dev)
    cat_t = [torch.from_numpy(np.ascontiguousarray(c)) for c, _ in cat_ids]
    num_t = torch.from_numpy(num) if num_estaticos else None

    # ---------------- âncoras -> posições do v2 -> janela da extração
    d = DadosV2(run_dir(config, a.v2))
    modelo, ck = carregar_modelo(config, d, a.pretreino, dev)
    p_alvo = np.flatnonzero(np.asarray(d.tipo) == d.meta["tipo_alvo"])
    ids = np.asarray(d.id_ancora)[p_alvo]
    ordem = np.argsort(ids, kind="stable")
    ev = df["evento_id"].to_numpy()
    k = np.searchsorted(ids[ordem], ev)
    k = np.minimum(k, len(ordem) - 1)
    achou = ids[ordem][k] == ev
    if not achou.all():
        print(f"aviso: {int((~achou).sum()):,} âncoras sem posição no v2; ficam fora", flush=True)
    pos = np.where(achou, p_alvo[ordem[k]], -1)
    linhas_por_pos = np.argsort(pos, kind="stable")
    pos_ord = pos[linhas_por_pos]
    janelas = {}                                   # (ini, fim) -> (offsets na janela, linhas do df)
    for ini, fim, de in janelas_sobrepostas(d):
        lo, hi = np.searchsorted(pos_ord, de), np.searchsorted(pos_ord, fim)
        if hi > lo:
            r = linhas_por_pos[lo:hi]
            janelas[(ini, fim)] = (pos[r] - ini, r)
    print(f"{len(df):,} âncoras ({int(achou.sum()):,} no v2) em {len(janelas):,} janelas; "
          f"treino {int(eh_treino.sum()):,}, val {int((split == 'val').sum()):,}, teste {int((split == 'test').sum()):,}", flush=True)

    def janelas_de(sp):
        return [j for j, (_, r) in janelas.items() if (split[r] == sp).any()]

    jan_treino = janelas_de("train")
    if a.max_janelas_treino:
        jan_treino = [jan_treino[i] for i in rng.choice(len(jan_treino), min(a.max_janelas_treino, len(jan_treino)), replace=False)]

    # ---------------- modelo conjunto
    cabeca = DCNv2Fusao(ck["arquitetura"]["n_embd"], [c for _, c in cat_ids], len(num_estaticos),
                        N_CROSS, CROSS_RANK, DEEP_HIDDEN, tarefas_modelo).to(dev)
    otim = torch.optim.AdamW([{"params": modelo.parameters(), "lr": a.lr_backbone},
                              {"params": cabeca.parameters(), "lr": a.lr_cabeca}], weight_decay=1e-4)
    lotes_epoca = len(lotes_por_orcamento(d, jan_treino, a.orcamento))
    total = max(1, lotes_epoca * a.epocas)
    aquec = max(1, int(a.aquecimento * total))
    sched = torch.optim.lr_scheduler.LambdaLR(otim, lambda s: min(1.0, (s + 1) / aquec) * max(0.0, (total - s) / max(1, total - aquec)))

    def estados_ancoras(jl, lote, sp=None):
        """-> (estados das âncoras do lote, linhas do df) na ordem do lote."""
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            h = modelo.estados(lote)
        rr, tt, linhas = [], [], []
        for r, j in enumerate(jl):
            off, li = janelas[j]
            if sp is not None:
                m = split[li] == sp
                off, li = off[m], li[m]
            rr.append(np.full(len(li), r)); tt.append(off); linhas.append(li)
        rr, tt, linhas = np.concatenate(rr), np.concatenate(tt), np.concatenate(linhas)
        return h[torch.from_numpy(rr).to(dev), torch.from_numpy(tt).to(dev)].float(), linhas

    def cabeca_em(hs, linhas):
        li = torch.from_numpy(linhas)
        return cabeca(hs, [c[li].to(dev) for c in cat_t], num_t[li].to(dev) if num_t is not None else None)

    def perda(saidas, linhas):
        li = torch.from_numpy(linhas)
        tot = 0.0
        for nome, tipo, _ in tarefas_modelo:
            y = alvos[nome][li].to(dev)
            if tipo == "binaria":
                m = ~torch.isnan(y)
                if m.any():
                    tot = tot + F.binary_cross_entropy_with_logits(saidas[nome].squeeze(-1)[m], y[m], pos_weight=pos_weights[nome])
            else:
                m = y >= 0
                if m.any():
                    tot = tot + F.cross_entropy(saidas[nome][m], y[m])
        return tot

    @torch.no_grad()
    def prever(sp):
        modelo.eval(); cabeca.eval()
        res = {nome: [] for nome, _, _ in tarefas_modelo}
        todas = []
        for jl in lotes_por_orcamento(d, janelas_de(sp), a.orcamento_aval):
            lote = montar_lote(d, jl, d.meta["m_fim"] + 1, com_alvos=False, device=dev)
            hs, linhas = estados_ancoras(jl, lote, sp)
            s = cabeca_em(hs, linhas)
            for nome, tipo, _ in tarefas_modelo:
                res[nome].append((torch.sigmoid(s[nome].squeeze(-1)) if tipo == "binaria" else torch.softmax(s[nome].float(), -1)).float().cpu().numpy())
            todas.append(linhas)
        modelo.train(); cabeca.train()
        linhas = np.concatenate(todas)
        return linhas, {n: np.concatenate(v) for n, v in res.items()}

    def metricas(linhas, sc):
        out_m = {}
        for nome, tipo, _, mapa_cl in tarefas_info:
            y = alvos[nome][torch.from_numpy(linhas)].numpy()
            if tipo == "binaria":
                out_m[nome] = {"auc": auc_roc(pd.Series(y), pd.Series(sc[nome]))}
            else:
                ok = y >= 0
                out_m[nome] = {"accuracy": accuracy(pd.Series(np.where(ok, y, np.nan)), pd.Series(np.where(ok, sc[nome].argmax(1), np.nan)))}
        return out_m

    def gravar(linhas, sc, sp):
        sub = df.iloc[linhas]
        cols = {"evento_id": sub["evento_id"].to_numpy(), "entidade_id": sub["entidade_id"].to_numpy()}
        for nome, tipo, _, mapa_cl in tarefas_info:
            if tipo == "binaria":
                cols[f"{nome}_score"] = sc[nome]
            else:
                inv = {i: c for c, i in mapa_cl.items()}
                cols[f"{nome}_pred"] = [inv[i] for i in sc[nome].argmax(axis=1)]
                for c, i in mapa_cl.items():
                    cols[f"{nome}_p_{c}"] = sc[nome][:, i]
        pd.DataFrame(cols).to_parquet(os.path.join(out, f"predicoes_{sp}.parquet"), index=False)

    # ---------------- treino
    chave = tarefas_modelo[0][0]
    melhor, melhor_ep, hist = -1.0, 0, []
    modelo.train(); cabeca.train()
    print(f"backbone {sum(p.numel() for p in modelo.parameters()):,} parâmetros; cabeça {sum(p.numel() for p in cabeca.parameters()):,}; "
          f"{len(jan_treino):,} janelas de treino, {lotes_epoca} lotes por época, dispositivo {dev}", flush=True)
    for ep in range(1, a.epocas + 1):
        te = time.time(); perdas = []
        for n, jl in enumerate(lotes_por_orcamento(d, jan_treino, a.orcamento, rng)):
            lote = montar_lote(d, jl, d.meta["m_fim"] + 1, com_alvos=False, device=dev)
            hs, linhas = estados_ancoras(jl, lote, "train")
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                s = cabeca_em(hs, linhas)
            loss = perda({k_: v.float() for k_, v in s.items()}, linhas)
            otim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(modelo.parameters()) + list(cabeca.parameters()), 1.0)
            otim.step(); sched.step()
            perdas.append(float(loss.detach()))
            if n % 200 == 0:
                print(f"  época {ep} lote {n}/{lotes_epoca} perda {np.mean(perdas[-200:]):.4f} ({time.time() - te:.0f}s)", flush=True)
        lv, sv = prever("val")
        mv = metricas(lv, sv)
        score = mv[chave].get("auc", mv[chave].get("accuracy"))
        hist.append({"epoca": ep, "perda_treino": float(np.mean(perdas)), "val": mv, "segundos": round(time.time() - te, 1)})
        print(f"época {ep}: perda {np.mean(perdas):.4f}; val " + "; ".join(f"{k_} {list(v.values())[0]:.4f}" for k_, v in mv.items())
              + f" ({time.time() - te:.0f}s)", flush=True)
        if score > melhor:
            melhor, melhor_ep = score, ep
            torch.save({"backbone": modelo.state_dict(), "cabeca": cabeca.state_dict()}, os.path.join(out, "melhor.pt"))
        json.dump({"historico": hist, "melhor_epoca": melhor_ep}, open(os.path.join(out, "v2_ajuste_parcial.json"), "w"), indent=1)

    # ---------------- melhor época: predições de validação e (uma vez) de teste
    est = torch.load(os.path.join(out, "melhor.pt"), map_location=dev, weights_only=False)
    modelo.load_state_dict(est["backbone"]); cabeca.load_state_dict(est["cabeca"])
    lv, sv = prever("val"); gravar(lv, sv, "val")
    rel = {"melhor_epoca": melhor_ep, "historico": hist, "metricas_validacao": metricas(lv, sv),
           "argumentos": vars(a), "arquitetura": ck["arquitetura"]}
    if a.avaliar_teste:
        lt, st = prever("test"); gravar(lt, st, "test")
        rel["metricas_teste"] = metricas(lt, st)
        print("TESTE: " + "; ".join(f"{k_} {list(v.values())[0]:.4f}" for k_, v in rel["metricas_teste"].items()), flush=True)
    rel["segundos"] = round(time.time() - t0, 1)
    json.dump(rel, open(os.path.join(out, "v2_ajuste_relatorio.json"), "w"), indent=1, default=str)
    os.remove(os.path.join(out, "melhor.pt")) if os.environ.get("APAGAR_CHECKPOINT") else None
    print(f"melhor época {melhor_ep}; {rel['segundos']:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
