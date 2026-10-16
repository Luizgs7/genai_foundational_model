"""EDA do dataset H&M (H1 do experimento) — números que orientam as escolhas
do adapter/config (`configs/hm_clientes.yaml`): granularidade, janelas de
split, horizonte de churn, campos tokenizados e n_positions.

Uso: scripts/rodar_limitado.sh .venv/bin/python adapters/eda_hm.py
(lê os Parquets compactos de adapters/compactar_hm.py)
Gera runs/hm_clientes/eda/eda.json (e imprime um resumo).
"""

import json
import os

import numpy as np
import polars as pl

DADOS = "dados_hm"
OUT = "runs/hm_clientes/eda"


def pct(serie, qs=(50, 75, 90, 95, 99)):
    arr = serie.to_numpy()
    return {f"p{q}": float(np.percentile(arr, q)) for q in qs} | {"media": float(arr.mean()), "max": float(arr.max())}


def main():
    os.makedirs(OUT, exist_ok=True)
    r = {}

    # arquivos compactos (adapters/compactar_hm.py): cliente int32, data date32
    tx = pl.read_parquet(f"{DADOS}/transactions_compacto.parquet").rename({"cliente": "customer_id"})
    art = pl.read_parquet(f"{DADOS}/articles.parquet").with_columns(pl.col("article_id").cast(pl.Int32))
    cus = pl.read_parquet(f"{DADOS}/customers_compacto.parquet")

    r["transacoes"] = tx.height
    r["clientes_com_compra"] = tx["customer_id"].n_unique()
    r["clientes_cadastro"] = cus.height
    r["artigos_vendidos"] = tx["article_id"].n_unique()
    r["artigos_cadastro"] = art.height
    r["periodo"] = [str(tx["t_dat"].min()), str(tx["t_dat"].max())]
    r["dias"] = (tx["t_dat"].max() - tx["t_dat"].min()).days + 1

    mensal = tx.group_by(pl.col("t_dat").dt.truncate("1mo").alias("mes")).len().sort("mes")
    r["volume_mensal"] = {str(m): int(n) for m, n in mensal.iter_rows()}
    r["canal"] = {str(k): int(v) for k, v in tx.group_by("sales_channel_id").len().iter_rows()}
    r["preco"] = pct(tx["price"])

    itens_cliente = tx.group_by("customer_id").len()["len"]
    r["itens_por_cliente"] = pct(itens_cliente)
    r["frac_clientes_1_item"] = float((itens_cliente == 1).mean())

    cestas = tx.group_by(["customer_id", "t_dat"]).agg(pl.len().alias("itens"), pl.col("price").sum().alias("valor"),
                                                       pl.col("sales_channel_id").max().alias("canal"))
    r["cestas"] = cestas.height
    r["itens_por_cesta"] = pct(cestas["itens"])
    cestas_cliente = cestas.group_by("customer_id").len()["len"]
    r["cestas_por_cliente"] = pct(cestas_cliente)
    r["frac_clientes_1_cesta"] = float((cestas_cliente == 1).mean())

    gaps = (cestas.sort(["customer_id", "t_dat"])
            .with_columns((pl.col("t_dat") - pl.col("t_dat").shift(1).over("customer_id")).dt.total_days().alias("gap"))
            .drop_nulls("gap")["gap"])
    r["gap_entre_cestas_dias"] = pct(gaps)
    fim = tx["t_dat"].max()
    r["frac_gap_maior_que"] = {str(n): float((gaps > n).mean()) for n in [30, 45, 60, 90, 120, 180]}

    # contagem por artigo primeiro (~105 mil linhas) e agregação ponderada —
    # o join direto das 31,8M linhas com as colunas de texto estoura a RAM
    vendas_art = tx.group_by("article_id").len().join(art, on="article_id", how="left")
    campos = ["product_type_name", "product_group_name", "graphical_appearance_name", "colour_group_name",
              "perceived_colour_master_name", "department_name", "index_name", "index_group_name", "section_name",
              "garment_group_name"]
    r["cardinalidade_campos_artigo"] = {}
    for c in campos:
        vc = vendas_art.group_by(c).agg(pl.col("len").sum()).sort("len", descending=True)
        top = vc.head(5)
        r["cardinalidade_campos_artigo"][c] = {
            "distintos_vendidos": int(vc.height),
            "top5": {str(a): round(int(b) / tx.height, 4) for a, b in top.iter_rows()},
        }
    vc_art = tx["article_id"].value_counts(sort=True)["count"].to_numpy()
    cum = np.cumsum(vc_art) / vc_art.sum()
    r["cobertura_top_n_artigos"] = {str(n): float(cum[min(n, len(cum)) - 1]) for n in [500, 1000, 2000, 5000, 10000]}

    r["clientes_faltantes"] = {c: float(cus[c].null_count() / cus.height) for c in cus.columns}
    r["clientes_valores"] = {c: {str(a): int(b) for a, b in cus[c].value_counts(sort=True).head(8).iter_rows()}
                             for c in ["FN", "Active", "club_member_status", "fashion_news_frequency"]}
    r["idade"] = pct(cus["age"].drop_nulls())
    r["postal_code_distintos"] = int(cus["postal_code"].n_unique())

    # comprimento de sequência em tokens, p/ 2 desenhos de bloco (k tokens por item)
    for k in [9, 11, 13]:
        r[f"tokens_por_cliente_k{k}"] = pct(itens_cliente * k + 2)

    with open(f"{OUT}/eda.json", "w", encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False, indent=2)
    print(json.dumps(r, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
