"""F3-H4 — recomendação direta do motor v2 para uma semana-alvo do protocolo da F2-H3.

Para cada cliente com compra na semana-alvo (runs/<nome>/f2h3/<semana>/alvos.parquet), o estado depois
da última posição antes da semana pontua o catálogo inteiro (log-probabilidade de cada artigo ser
comprado no próximo pedido). Grava, por cliente:
  - os `--topn` artigos de maior pontuação do modelo;
  - todos os artigos que o cliente já comprou antes da semana, com a pontuação do modelo, o número de
    compras e os dias desde a última (para o braço que soma o histórico à pontuação; f3h4_avaliar.py).

O backbone deve ter sido treinado com `v2_pretreino.py --corte <semana>` (eventos até a véspera).

Uso: python3 pipeline/common/f3h4_recomendar.py <config.yaml> --pretreino f3h4_pre_test_s0 --semana 2020-09-16
        [--topn 100] [--saida f3h4_rec_test_s0]
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import polars as pl
import torch

sys.path.insert(0, "pipeline/common")
from schema import carregar_config, run_dir  # noqa: E402
from v2_extrair import carregar_modelo  # noqa: E402
from v2_modelo import DadosV2, montar_lote  # noqa: E402
from v2_pretreino import lotes_por_orcamento  # noqa: E402


def historico(d, ents, corte):
    """(ent, item, n_compras, t_ultima) das linhas da fonte-alvo antes do corte, para as entidades dadas."""
    partes_p = []
    for e in ents:
        a, b = int(d.ent_ini[e]), int(d.ent_ini[e + 1])
        b = a + int(np.searchsorted(d.t[a:b], corte, side="left"))
        if b > a:
            partes_p.append(np.arange(a, b))
    p = np.concatenate(partes_p)
    p = p[np.asarray(d.tipo[p]) == 0]
    n = np.asarray(d.n_lin[p]).astype(np.int64)
    li = np.asarray(d.lin_ini[p])
    sel = np.repeat(li - np.r_[0, np.cumsum(n)[:-1]], n) + np.arange(int(n.sum()))
    return (pl.DataFrame({"ent": np.repeat(np.asarray(d.ent[p]), n), "item_idx": np.asarray(d.lin_item[sel]).astype(np.int32),
                          "t": np.repeat(np.asarray(d.t[p]), n)})
            .group_by(["ent", "item_idx"]).agg(pl.len().alias("n_compras"), pl.col("t").max().alias("t_ult")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--pretreino", required=True)
    ap.add_argument("--semana", required=True, help="início da semana-alvo (AAAA-MM-DD), como em runs/<nome>/f2h3/")
    ap.add_argument("--dados", default="v2")
    ap.add_argument("--topn", type=int, default=100)
    ap.add_argument("--orcamento", type=int, default=120000)
    ap.add_argument("--saida", default=None)
    a = ap.parse_args()
    t0 = time.time()
    config = carregar_config(a.config)
    d = DadosV2(run_dir(config, a.dados))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    modelo, ck = carregar_modelo(config, d, a.pretreino, device)
    corte = int(np.datetime64(a.semana, "m").astype(np.int64))
    if ck["corte_min"] > corte:
        sys.exit(f"o backbone viu eventos até {np.datetime64(ck['corte_min'], 'm')}, depois do início da semana-alvo")
    alvos = pl.read_parquet(os.path.join(run_dir(config, "f2h3"), a.semana, "alvos.parquet"))
    ent_id = pl.read_parquet(os.path.join(d.pasta, "entidades.parquet"))["entidade"].to_numpy()
    idx_de = {int(c): i for i, c in enumerate(ent_id)}
    clientes = alvos["cliente"].unique().to_numpy()
    ents = sorted(idx_de[int(c)] for c in clientes if int(c) in idx_de)
    item_de = np.r_[-1, pl.read_parquet(os.path.join(d.pasta, "itens.parquet")).sort("idx")["item"].to_numpy()]

    hist = historico(d, ents, corte)
    print(f"{len(clientes):,} clientes na semana, {len(ents):,} com histórico no v2; {hist.height:,} pares de histórico "
          f"({time.time() - t0:.0f}s)", flush=True)
    hist_por = {e: (g["item_idx"].to_numpy(), g["n_compras"].to_numpy(), g["t_ult"].to_numpy())
                for (e,), g in hist.partition_by("ent", as_dict=True).items()}

    jan = {}
    for ini, fim in d.janelas(corte, so_entidades=ents):
        e = int(d.ent[ini])
        if e not in jan or fim > jan[e][1]:
            jan[e] = (ini, fim)
    linhas = []
    with torch.no_grad():
        for jl in lotes_por_orcamento(d, list(jan.values()), a.orcamento):
            lote = montar_lote(d, jl, corte, com_alvos=False, device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                h = modelo.estados(lote)
            ult = lote["valido"].sum(1) - 1
            lp = modelo.pontuar_itens(h[torch.arange(len(jl), device=h.device), ult].float())
            v, i = lp.topk(a.topn, dim=-1)
            v, i = v.cpu().numpy(), i.cpu().numpy()
            for r, (ini, _) in enumerate(jl):
                e = int(d.ent[ini])
                hi, hn, ht = hist_por.get(e, (np.zeros(0, np.int32), np.zeros(0, np.int64), np.zeros(0, np.int64)))
                lph = lp[r, torch.from_numpy(hi.astype(np.int64)).to(lp.device)].cpu().numpy() if len(hi) else np.zeros(0, np.float32)
                itens = np.r_[i[r], hi].astype(np.int64)
                lps = np.r_[v[r], lph].astype(np.float32)
                rank = np.r_[np.arange(1, a.topn + 1), np.zeros(len(hi), np.int64)]
                n = np.r_[np.zeros(a.topn, np.int64), hn]
                dias = np.r_[np.full(a.topn, np.nan), (corte - ht) / 1440.0]
                linhas.append(pl.DataFrame({"cliente": np.full(len(itens), ent_id[e], np.int32), "artigo": item_de[itens].astype(np.int32),
                                            "logp": lps, "rank_modelo": rank.astype(np.int32), "n_compras": n.astype(np.int32),
                                            "dias_desde_ult": dias.astype(np.float32)}))
    rec = (pl.concat(linhas).group_by(["cliente", "artigo"])
           .agg(pl.col("logp").first(), pl.col("rank_modelo").max(), pl.col("n_compras").max(), pl.col("dias_desde_ult").min()))
    out = run_dir(config, a.saida or f"f3h4_rec_{a.semana}")
    os.makedirs(out, exist_ok=True)
    rec.write_parquet(os.path.join(out, f"rec_{a.semana}.parquet"))
    rel = {"semana": a.semana, "pretreino": a.pretreino, "corte_do_backbone": str(np.datetime64(ck["corte_min"], "m")),
           "epoca_do_backbone": ck.get("epoca"), "topn": a.topn, "clientes_semana": int(len(clientes)), "clientes_com_estado": len(jan),
           "pares": rec.height, "itens_vistos_no_corte": int(modelo.visto.sum().item()), "segundos": round(time.time() - t0, 1)}
    json.dump(rel, open(os.path.join(out, f"relatorio_{a.semana}.json"), "w"), indent=1)
    print(json.dumps(rel), flush=True)


if __name__ == "__main__":
    main()
