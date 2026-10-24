"""F2-H10 — histórico de compras em texto, para os braços com LLM.

Uma linha por cesta (cliente + dia), com tempo relativo por extenso e sem o
mês do calendário (lição da seção 7.1 do artigo: o token de mês distorce a
loss). Exemplo:

  +12 dias, loja: Bonnie sweater (Sweater, Black, Ladieswear, preço 7/10); Skinny jeans (Trousers, Blue, Divided, preço 4/10)

`preço k/10` é a faixa de preço do item dentro do seu grupo de produto (a mesma
discretização do tokenizador). Cestas com mais de MAX_ITENS itens listam os
primeiros e terminam com "e mais N itens".

Só os clientes da amostra da F2-H1 (fração `--frac`, semente 0): os braços com
LLM rodam nela para caber no Colab Pro. Cada linha carrega o `evento_id` da
âncora da cesta (último item), o mesmo que define as âncoras downstream.

Uso: python3 pipeline/common/f2h10_textos.py configs/hm_clientes.yaml [--frac 0.10]
Gera runs/<nome>/f2h10/textos.parquet (entidade_id, evento_id, data, split, linha)
"""

import argparse
import os
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, "pipeline/common")
from schema import amostra_clientes, carregar_config, run_dir  # noqa: E402

MAX_ITENS = 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--frac", type=float, default=0.10)
    ap.add_argument("--semente_amostra", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    config = carregar_config(a.config)
    rot = pl.read_parquet(os.path.join(run_dir(config, "rotulos_downstream"), "rotulos.parquet"),
                          columns=["evento_id", "entidade_id", "split", "usar_downstream"]).filter(pl.col("usar_downstream"))
    anc = pl.read_parquet(os.path.join(run_dir(config, "rotulos_downstream"), "ancoras.parquet"))
    rot = rot.join(anc, on="evento_id", how="semi")
    ents = rot["entidade_id"].to_numpy()
    na = amostra_clientes(ents, a.frac, a.semente_amostra)
    amostra = pl.DataFrame({"entidade_id": np.unique(ents[np.asarray(na)])})
    print(f"amostra: {amostra.height:,} clientes; âncoras de treino na amostra: "
          f"{rot.filter(pl.col('split') == 'train').join(amostra, on='entidade_id', how='semi').height:,}", flush=True)

    art = pl.read_parquet("dados_hm/articles.parquet", columns=["article_id", "prod_name", "product_type_name",
                                                                 "colour_group_name", "index_name"])
    ev = (pl.scan_parquet(config["fonte_canonica"]).select("evento_id", "entidade_id", "data_evento", "artigo", "canal")
          .join(amostra.lazy(), on="entidade_id", how="semi")
          .join(pl.scan_parquet(os.path.join(run_dir(config, "discretizacao"), "discretizado.parquet"))
                .select("evento_id", "preco_bucket", "split"), on="evento_id")
          .join(art.lazy().rename({"article_id": "artigo"}), on="artigo", how="left")
          .with_columns(pl.col("data_evento").dt.date().alias("data"))
          .sort("entidade_id", "data", "evento_id").collect())
    item = pl.format("{} ({}, {}, {}, preço {}/10)", pl.col("prod_name").fill_null("?"), pl.col("product_type_name").fill_null("?"),
                     pl.col("colour_group_name").fill_null("?"), pl.col("index_name").fill_null("?"), pl.col("preco_bucket") + 1)
    cestas = (ev.with_columns(item.alias("_item"))
              .group_by("entidade_id", "data", maintain_order=True)
              .agg(pl.col("evento_id").last(), pl.col("split").last(), pl.col("_item"),
                   (pl.col("canal").cast(pl.String) == "online").any().alias("_on"),
                   (pl.col("canal").cast(pl.String) != "online").any().alias("_loja")))
    gap = (pl.col("data") - pl.col("data").shift(1).over("entidade_id")).dt.total_days()
    canal = pl.when(pl.col("_on") & pl.col("_loja")).then(pl.lit("loja e online")).when(pl.col("_on")).then(pl.lit("online")).otherwise(pl.lit("loja"))
    n = pl.col("_item").list.len()
    itens = pl.col("_item").list.head(MAX_ITENS).list.join("; ")
    resto = pl.when(n > MAX_ITENS).then(pl.format("; e mais {} itens", n - MAX_ITENS)).otherwise(pl.lit(""))
    tempo = pl.when(gap.is_null()).then(pl.lit("primeira compra")).otherwise(pl.format("+{} dias", gap))
    cestas = cestas.with_columns(pl.format("{}, {}: {}{}", tempo, canal, itens, resto).alias("linha"))
    cestas = cestas.select("entidade_id", "evento_id", "data", "split", "linha")
    out = run_dir(config, "f2h10")
    os.makedirs(out, exist_ok=True)
    cestas.write_parquet(os.path.join(out, "textos.parquet"))
    cob = rot.join(amostra, on="entidade_id", how="semi").join(cestas.select("evento_id"), on="evento_id", how="anti").height
    print(f"{cestas.height:,} cestas, {ev.height:,} itens; âncoras sem linha: {cob}; "
          f"caracteres por cliente (mediana): {cestas.group_by('entidade_id').agg(pl.col('linha').str.len_chars().sum())['linha'].median():.0f} "
          f"({time.time() - t0:.0f}s)")
    print(cestas.head(3)["linha"].to_list())


if __name__ == "__main__":
    main()
