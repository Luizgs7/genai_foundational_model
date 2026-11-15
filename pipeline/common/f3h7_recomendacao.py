"""F3-H7/F3-H8 — recomendação na base sintética: MAP@12 do próximo pedido.

Alvo: os itens do primeiro pedido de cada cliente em [data, data + horizonte).
Referências (protocolo da F2-H3):
  POP  os 12 itens mais vendidos (linhas) nos 30 dias antes da data
  R1   recompra: os itens do próprio cliente antes da data, por número de compras e,
       no empate, pela compra mais recente; completa com POP
Modelos: as recomendações gravadas por `v2_extrair.py --recomendar <data>` (uma pasta por modelo).

Também mede, com o gabarito, quanto da compra conjunta plantada o modelo recupera:
para cada regra (A -> B), quando A está no último pedido antes da data, B aparece
entre as 12 recomendações? (`--regras`, só quando há gabarito.)

Uso: python3 pipeline/common/f3h7_recomendacao.py <config.yaml> --data 2025-01-06
        [--modelo nome=pasta ...] [--horizonte 21] [--saida f3h7_recomendacao]
"""

import argparse
import json
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, "pipeline/common")
from schema import carregar_config, run_dir  # noqa: E402

K = 12


def ap_k(recs, alvo):
    acertos, soma = 0, 0.0
    for i, r in enumerate(recs[:K]):
        if r in alvo:
            acertos += 1
            soma += acertos / (i + 1)
    return soma / min(len(alvo), K) if alvo else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--data", required=True)
    ap.add_argument("--modelo", nargs="*", default=[])
    ap.add_argument("--horizonte", type=int, default=21)
    ap.add_argument("--saida", default="f3h7_recomendacao")
    a = ap.parse_args()
    config = carregar_config(a.config)
    v2 = config["v2"]
    fonte = next(f for f in v2["fontes"] if f.get("alvo"))
    item, grupo = fonte["item"], fonte["agrupar_por"][0]
    ev = pl.read_parquet(config["fonte_canonica"], columns=["entidade_id", "data_evento", grupo, item])
    D = pl.lit(np.datetime64(a.data, "ms")).cast(pl.Datetime("ms"))
    antes = ev.filter(pl.col("data_evento") < D)
    depois = ev.filter((pl.col("data_evento") >= D) & (pl.col("data_evento") < D + pl.duration(days=a.horizonte)))
    # alvo: o primeiro pedido de cada cliente no horizonte
    prim = depois.group_by("entidade_id").agg(pl.col("data_evento").min().alias("t0"))
    alvo = (depois.join(prim, on="entidade_id").filter(pl.col("data_evento") == pl.col("t0"))
            .group_by("entidade_id").agg(pl.col(item).unique().alias("alvo")))
    alvo = {r["entidade_id"]: set(r["alvo"]) for r in alvo.iter_rows(named=True)}
    pop = (antes.filter(pl.col("data_evento") >= D - pl.duration(days=30)).group_by(item).len()
           .sort(["len", item], descending=[True, False])[item].head(K).to_list())
    hist = (antes.filter(pl.col("entidade_id").is_in(list(alvo))).group_by(["entidade_id", item])
            .agg(pl.len().alias("n"), pl.col("data_evento").max().alias("ult"))
            .sort(["entidade_id", "n", "ult"], descending=[False, True, True])
            .group_by("entidade_id", maintain_order=True).agg(pl.col(item).head(K)))
    r1 = {r["entidade_id"]: r[item] for r in hist.iter_rows(named=True)}

    def completar(lista):
        out = list(lista)
        for p in pop:
            if len(out) >= K:
                break
            if p not in out:
                out.append(p)
        return out

    sistemas = {"POP": {e: pop for e in alvo}, "R1": {e: completar(r1.get(e, [])) for e in alvo}}
    for m in a.modelo:
        nome, pasta = m.split("=", 1)
        rec = pl.read_parquet(os.path.join(run_dir(config, pasta), f"recomendacoes_{a.data}.parquet")).sort(["entidade_id", "posicao"])
        rec = rec.group_by("entidade_id", maintain_order=True).agg(pl.col("item"))
        rec = {r["entidade_id"]: r["item"] for r in rec.iter_rows(named=True)}
        sistemas[nome] = {e: completar(rec.get(e, [])) for e in alvo}

    ents = sorted(alvo)
    rng = np.random.RandomState(0)
    res = {"data": a.data, "horizonte_dias": a.horizonte, "n_clientes": len(ents), "map12": {}, "ic95_vs_R1": {}}
    aps = {s: np.array([ap_k(sistemas[s][e], alvo[e]) for e in ents]) for s in sistemas}
    for s, v in aps.items():
        res["map12"][s] = float(v.mean())
        if s != "R1":
            d = v - aps["R1"]
            bs = [d[rng.randint(0, len(d), len(d))].mean() for _ in range(300)]
            res["ic95_vs_R1"][s] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        print(f"{s:20s} MAP@12 = {v.mean():.4f}", flush=True)

    # compra conjunta plantada (só na base sintética)
    gab = os.path.join(os.path.dirname(v2["catalogo"]["tabela"]), "gabarito", "regras_compra_conjunta.parquet")
    if os.path.exists(gab):
        regras = pl.read_parquet(gab)
        ult_t = antes.group_by("entidade_id").agg(pl.col("data_evento").max().alias("tu"))
        ult = (antes.join(ult_t, on="entidade_id").filter(pl.col("data_evento") == pl.col("tu"))
               .select(["entidade_id", pl.col(item).alias("sku_a")]).unique())
        casos = ult.join(regras.select(["sku_a", "sku_b"]), on="sku_a")
        casos = casos.filter(pl.col("entidade_id").is_in(ents))
        res["compra_conjunta"] = {"n_casos": casos.height}
        for s in sistemas:
            rec_s = sistemas[s]
            hits = [r["sku_b"] in rec_s[r["entidade_id"]] for r in casos.iter_rows(named=True)]
            res["compra_conjunta"][s] = float(np.mean(hits)) if hits else None
        print("compra conjunta recuperada:", {s: round(v, 4) for s, v in res["compra_conjunta"].items() if s != "n_casos" and v is not None})
    out = run_dir(config, "f3h7")
    os.makedirs(out, exist_ok=True)
    json.dump(res, open(os.path.join(out, a.saida + ".json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
