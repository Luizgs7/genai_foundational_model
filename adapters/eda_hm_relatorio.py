"""Séries e distribuições para o relatório de análise exploratória (EDA) da
H&M, complementando runs/hm_clientes/eda/eda.json com o que os gráficos
precisam: volume semanal e participação online, histogramas (itens por
cliente, cestas por cliente, itens por cesta, intervalo entre cestas,
idade), preço por grupo de produto, concentração do catálogo, cadastro,
rótulos por split, cobertura de contexto — e a comparação com as bases
sintética e Olist.

Uso: scripts/rodar_limitado.sh -m 4G .venv/bin/python adapters/eda_hm_relatorio.py
Gera runs/hm_clientes/eda/eda_relatorio.json
"""

import json

import numpy as np
import pandas as pd
import polars as pl

D = "dados_hm"
OUT = "runs/hm_clientes/eda/eda_relatorio.json"


def hist(valores, bordas):
    cont, _ = np.histogram(valores, bins=bordas)
    return {"bordas": [float(b) for b in bordas], "contagens": cont.tolist()}


def main():
    r = {}
    tx = pl.read_parquet(f"{D}/transactions_compacto.parquet")
    art = pl.read_parquet(f"{D}/articles.parquet").with_columns(pl.col("article_id").cast(pl.Int32))
    cus = pl.read_parquet(f"{D}/customers_compacto.parquet")

    sem = (tx.group_by(pl.col("t_dat").dt.truncate("1w").alias("semana"))
           .agg(pl.len().alias("itens"), (pl.col("sales_channel_id") == 2).mean().alias("online"),
                pl.col("cliente").n_unique().alias("clientes"))
           .sort("semana"))
    r["semanal"] = {"semana": [str(d) for d in sem["semana"]], "itens": sem["itens"].to_list(),
                    "share_online": [round(x, 4) for x in sem["online"]], "clientes": sem["clientes"].to_list()}

    itens_cli = tx.group_by("cliente").len()["len"].to_numpy()
    r["hist_itens_cliente"] = hist(itens_cli, [1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 2000])

    cestas = tx.group_by(["cliente", "t_dat"]).agg(pl.len().alias("itens"))
    r["hist_itens_cesta"] = hist(cestas["itens"].to_numpy(), [1, 2, 3, 4, 5, 7, 10, 15, 25, 600])
    ces_cli = cestas.group_by("cliente").len()["len"].to_numpy()
    r["hist_cestas_cliente"] = hist(ces_cli, [1, 2, 3, 5, 10, 20, 50, 500])
    gaps = (cestas.sort(["cliente", "t_dat"])
            .with_columns((pl.col("t_dat") - pl.col("t_dat").shift(1).over("cliente")).dt.total_days().alias("g"))
            .drop_nulls("g")["g"].to_numpy())
    r["hist_gap_dias"] = hist(gaps, [1, 7, 14, 30, 45, 60, 90, 120, 180, 365, 732])
    r["gap_frac_maior_que"] = {str(n): float((gaps > n).mean()) for n in [30, 60, 90, 120, 180]}
    del cestas

    tx_g = tx.select(["article_id", "price"]).join(art.select(["article_id", "product_group_name", "index_group_name"]),
                                                   on="article_id", how="left")
    pg = (tx_g.group_by("product_group_name")
          .agg(pl.len().alias("n"), pl.col("price").quantile(0.25).alias("p25"), pl.col("price").median().alias("p50"),
               pl.col("price").quantile(0.75).alias("p75"))
          .sort("n", descending=True))
    r["preco_por_grupo"] = [{"grupo": g, "n": int(n), "p25": float(a), "p50": float(b), "p75": float(c)}
                            for g, n, a, b, c in pg.iter_rows()]
    ig = tx_g.group_by("index_group_name").len().sort("len", descending=True)
    r["share_index_group"] = {g: int(n) for g, n in ig.iter_rows()}
    del tx_g

    vend = tx.group_by("article_id").len()["len"].sort(descending=True).to_numpy()
    cum = np.cumsum(vend) / vend.sum()
    pontos = [10, 100, 500, 1000, 2000, 5000, 10000, 20000, 50000, len(vend)]
    r["concentracao_artigos"] = {"top_n": pontos, "share": [float(cum[min(p, len(cum)) - 1]) for p in pontos],
                                 "artigos_vendidos": int(len(vend)),
                                 "artigos_1_venda": int((vend == 1).sum())}

    idade = cus["age"].drop_nulls().to_numpy()
    r["hist_idade"] = hist(idade, [16, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 100])
    r["cadastro"] = {c: {str(a): int(b) for a, b in cus[c].value_counts(sort=True).iter_rows()}
                     for c in ["club_member_status", "fashion_news_frequency"]}
    r["cadastro"]["FN_nulo"] = float(cus["FN"].is_null().mean())
    r["cadastro"]["idade_nula"] = float(cus["age"].is_null().mean())
    cep = cus.group_by("postal_code").len()["len"].sort(descending=True).to_numpy()
    r["cadastro"]["cep_distintos"] = int(len(cep))
    r["cadastro"]["cep_maior"] = int(cep[0])
    r["cadastro"]["cep_mediana_clientes"] = float(np.median(cep))
    r["cadastro"]["clientes_sem_compra"] = int(cus.height - tx["cliente"].n_unique())
    del tx

    # rótulos por split (âncora = cesta)
    rot = pd.read_parquet("runs/hm_clientes/rotulos_downstream/rotulos.parquet",
                          columns=["split", "churn_rotulo", "proxima_online_rotulo", "next_category_rotulo", "ltv_90d_rotulo"])
    r["rotulos"] = {
        "cestas_por_split": rot["split"].astype(str).value_counts().to_dict(),
        "churn_taxa": rot.groupby(rot["split"].astype(str))["churn_rotulo"].mean().round(4).to_dict(),
        "online_taxa": rot.groupby(rot["split"].astype(str))["proxima_online_rotulo"].mean().round(4).to_dict(),
        "next_category": rot["next_category_rotulo"].astype(str).value_counts(normalize=True).round(4).to_dict(),
        "ltv_buckets": rot["ltv_90d_rotulo"].value_counts(normalize=True).sort_index().round(4).to_dict(),
    }
    with open("runs/hm_clientes/rotulos_downstream/relatorio.json") as f:
        r["rotulos"]["relatorio"] = json.load(f)
    del rot

    ent = pd.read_parquet("runs/hm_clientes/serializacao/entidades.parquet", columns=["len_full"])
    r["hist_tokens_cliente"] = hist(ent["len_full"].to_numpy(), [0, 64, 128, 256, 512, 1024, 2048, 4096, 25000])

    # comparação entre as 3 bases
    comp = {}
    for nome in ["synthetic_agro", "olist_sellers", "hm_clientes"]:
        e = pd.read_parquet(f"runs/{nome}/serializacao/entidades.parquet", columns=["len_full", "n_eventos", "len_blocos_train"])
        with open(f"runs/{nome}/vocabulario/vocab.json") as f:
            v = json.load(f)
        s = pd.read_parquet(f"runs/{nome}/splits/splits.parquet", columns=["data_evento"])
        comp[nome] = {"eventos": int(e["n_eventos"].sum()), "entidades": int(len(e)),
                      "eventos_por_entidade_mediana": float(e["n_eventos"].median()),
                      "eventos_por_entidade_media": float(e["n_eventos"].mean()),
                      "periodo": [str(s["data_evento"].min().date()), str(s["data_evento"].max().date())],
                      "vocab": len(v), "tokens": int(e["len_full"].sum()),
                      "tokens_treino": int((e["len_blocos_train"][e["len_blocos_train"] > 0] + 1).sum())}
    r["comparacao_bases"] = comp

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False, indent=1, default=str)
    print(f"Gravado {OUT}")


if __name__ == "__main__":
    main()
