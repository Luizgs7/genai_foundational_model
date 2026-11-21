"""Adapter da base Synerise (RecSys Challenge 2025, arquivo bruto) -> schema canônico e fontes do motor v2.

Desenho registrado em PLANO_FASE3.md (F3-H7, camada 2). Entram os clientes com pelo menos uma compra
e todos os eventos deles. A data de corte da validação (fim - 84 dias) delimita o "treino" usado para
escolher o catálogo do modelo e as categorias, para nada do futuro entrar nessas escolhas.

  - catálogo do modelo: os TOP_SKU SKUs mais frequentes em compra e carrinho antes do corte
    (`sku_modelo`; os demais ficam nulos e a linha entra só com os atributos);
  - categoria: as TOP_CAT categorias mais compradas antes do corte, as demais em "outras";
  - faixa de preço: o arquivo traz 100 faixas (0 a 99); `faixa10` agrupa em 10; `valor` = faixa + 1
    (só a ordem do preço é conhecida; é o valor usado no LTV).

Saídas:
  runs/synerise/eventos_canonicos.parquet   compras: evento_id, entidade_id, data_evento, pedido_id, sku,
                                            sku_modelo, categoria, faixa_preco, faixa10, valor
  dados_synerise/preparado/{carrinho,busca,visita,produtos}.parquet

Uso: python3 adapters/adapter_synerise.py
"""

import os
import time

import polars as pl

BRUTO = "dados_synerise/extraido"
PREP = "dados_synerise/preparado"
RUN = "runs/synerise"
TOP_SKU = 100_000
TOP_CAT = 50
DIAS_VAL_TESTE = 84


def data(col):
    """Os arquivos trazem a data como texto, com ou sem os segundos."""
    return pl.coalesce(pl.col(col).str.strptime(pl.Datetime("ms"), "%Y-%m-%d %H:%M:%S", strict=False),
                       pl.col(col).str.strptime(pl.Datetime("ms"), "%Y-%m-%d %H:%M", strict=False))


def main():
    t0 = time.time()
    os.makedirs(PREP, exist_ok=True)
    os.makedirs(RUN, exist_ok=True)
    prop = pl.read_parquet(f"{BRUTO}/product_properties.parquet", columns=["sku", "category", "price"])
    buy = pl.read_parquet(f"{BRUTO}/product_buy.parquet").with_columns(data("timestamp").alias("t")).drop("timestamp")
    assert buy["t"].null_count() == 0
    fim = buy["t"].max()
    corte = fim - pl.duration(days=DIAS_VAL_TESTE)
    corte = pl.select(corte).item()
    clientes = buy.select("client_id").unique().sort("client_id").with_row_index("entidade_id").with_columns(pl.col("entidade_id").cast(pl.Int32))
    print(f"{clientes.height:,} clientes com compra; fim {fim}, corte do treino {corte} ({time.time() - t0:.0f}s)", flush=True)

    carr = pl.concat([
        pl.scan_parquet(f"{BRUTO}/add_to_cart.parquet").with_columns(pl.lit("adicao").alias("tipo")),
        pl.scan_parquet(f"{BRUTO}/remove_from_cart.parquet").with_columns(pl.lit("remocao").alias("tipo"))]
    ).join(clientes.lazy(), on="client_id", how="inner").with_columns(data("timestamp").alias("t")).drop("timestamp").collect()

    # escolhas feitas só com o período de treino
    freq = (pl.concat([buy.filter(pl.col("t") < corte).select("sku"), carr.filter(pl.col("t") < corte).select("sku")])
            .group_by("sku").len().sort(["len", "sku"], descending=[True, False]).head(TOP_SKU).select("sku"))
    cats = (buy.filter(pl.col("t") < corte).join(prop, on="sku", how="left").group_by("category").len()
            .sort(["len", "category"], descending=[True, False]).head(TOP_CAT)["category"].to_list())
    print(f"catálogo do modelo: {freq.height:,} SKUs; {len(cats)} categorias ({time.time() - t0:.0f}s)", flush=True)

    def atributos(df):
        return (df.join(prop, on="sku", how="left")
                .join(freq.with_columns(pl.col("sku").alias("sku_modelo")), on="sku", how="left")
                .with_columns(pl.when(pl.col("category").is_in(cats)).then(pl.col("category").cast(pl.Utf8))
                              .otherwise(pl.lit("outras")).alias("categoria"),
                              pl.col("price").cast(pl.Int32).alias("faixa_preco"),
                              (pl.col("price") // 10).cast(pl.Int32).cast(pl.Utf8).alias("faixa10"))
                .drop(["category", "price"]))

    ev = (atributos(buy.join(clientes, on="client_id", how="inner"))
          .sort(["entidade_id", "t", "sku"])
          .with_columns((((pl.col("entidade_id") != pl.col("entidade_id").shift(1)) | (pl.col("t") != pl.col("t").shift(1)))
                         .fill_null(True).cast(pl.Int64).cum_sum()).alias("pedido_id"),
                        (pl.col("faixa_preco") + 1).cast(pl.Float32).alias("valor"))
          .with_row_index("evento_id").with_columns(pl.col("evento_id").cast(pl.Int64))
          .rename({"t": "data_evento"}))
    cols = ["evento_id", "entidade_id", "data_evento", "pedido_id", "sku", "sku_modelo", "categoria", "faixa_preco", "faixa10", "valor"]
    ev.select(cols).write_parquet(f"{RUN}/eventos_canonicos.parquet")
    print(f"compras: {ev.height:,} linhas, {ev['pedido_id'].n_unique():,} cestas ({time.time() - t0:.0f}s)", flush=True)

    atributos(carr).select("entidade_id", pl.col("t").alias("datahora"), "sku_modelo", "tipo", "categoria", "faixa10") \
        .sort(["entidade_id", "datahora"]).write_parquet(f"{PREP}/carrinho.parquet")
    (pl.scan_parquet(f"{BRUTO}/search_query.parquet").select("client_id", "timestamp")
     .join(clientes.lazy(), on="client_id", how="inner").select("entidade_id", data("timestamp").alias("datahora"))
     .sort(["entidade_id", "datahora"]).collect().write_parquet(f"{PREP}/busca.parquet"))
    print(f"carrinho e busca gravados ({time.time() - t0:.0f}s)", flush=True)
    (pl.scan_parquet(f"{BRUTO}/page_visit.parquet").select("client_id", "timestamp")
     .join(clientes.lazy(), on="client_id", how="inner").select("entidade_id", data("timestamp").alias("datahora"))
     .sink_parquet(f"{PREP}/visita.parquet"))
    print(f"visita gravada ({time.time() - t0:.0f}s)", flush=True)
    (freq.rename({"sku": "sku_modelo"}).join(atributos(freq).select("sku_modelo", "categoria", "faixa10"), on="sku_modelo", how="left")
     .write_parquet(f"{PREP}/produtos.parquet"))
    for f in ("carrinho", "busca", "visita"):
        n = pl.scan_parquet(f"{PREP}/{f}.parquet").select(pl.len(), pl.col("datahora").null_count().alias("nulos")).collect().row(0)
        print(f"{f}: {n[0]:,} linhas, {n[1]:,} datas nulas", flush=True)


if __name__ == "__main__":
    main()
