"""Adapter Olist (Kaggle, olistbr/brazilian-ecommerce) para o motor v2 — F3-H7, camada 3 (teste funcional).

Entidade = VENDEDOR. As tabelas do Olist entram como os quatro formatos do motor v2 (MOTOR_V2.md):
  venda       itens   (alvo) linhas de pedido do vendedor: produto, categoria, preço, frete; pedido = order_id
  pagamento   evento  formas de pagamento dos pedidos do vendedor (tipo e parcelas), na data da compra
  avaliacao   evento  avaliações dos clientes, na data em que foram criadas (nota de 1 a 5)
  entrega     estado  na data da entrega ao cliente: dias de atraso contra a data prometida (vale até a próxima)
  cadastro    atributo fixo: UF do vendedor

O schema canônico (v1: rótulos, features e GBDT) é a tabela de vendas, com a nota e o atraso do pedido como
campos de rótulo (as receitas `risco_review_negativo` e `risco_atraso_entrega` do config antigo).

Saídas: runs/olist_v2/eventos_canonicos.parquet e dados_olist/preparado/{pagamento,avaliacao,entrega}.parquet
Uso: python3 adapters/adapter_olist_v2.py
"""

import os

import polars as pl

D = "dados_olist"
PREP = f"{D}/preparado"
RUN = "runs/olist_v2"


def main():
    os.makedirs(PREP, exist_ok=True)
    os.makedirs(RUN, exist_ok=True)
    dt = lambda c: pl.col(c).str.strptime(pl.Datetime("ms"), "%Y-%m-%d %H:%M:%S", strict=False)  # noqa: E731
    orders = (pl.read_csv(f"{D}/olist_orders_dataset.csv")
              .filter(~pl.col("order_status").is_in(["canceled", "unavailable"]))
              .with_columns(dt("order_purchase_timestamp").alias("compra"), dt("order_delivered_customer_date").alias("entregue"),
                            dt("order_estimated_delivery_date").alias("prometida")))
    items = pl.read_csv(f"{D}/olist_order_items_dataset.csv")
    sellers = pl.read_csv(f"{D}/olist_sellers_dataset.csv").select("seller_id", "seller_state")
    vend = items.select("seller_id").unique().sort("seller_id").with_row_index("entidade_id").with_columns(pl.col("entidade_id").cast(pl.Int32))
    ped = orders.select("order_id").unique().sort("order_id").with_row_index("pedido_id").with_columns(pl.col("pedido_id").cast(pl.Int64))
    prod = (pl.read_csv(f"{D}/olist_products_dataset.csv").select("product_id", "product_category_name")
            .join(pl.read_csv(f"{D}/product_category_name_translation.csv"), on="product_category_name", how="left")
            .select("product_id", pl.col("product_category_name_english").fill_null("sem_categoria").alias("categoria")))
    reviews = (pl.read_csv(f"{D}/olist_order_reviews_dataset.csv")
               .with_columns(dt("review_creation_date").alias("datahora"))
               .unique("order_id", keep="first", maintain_order=True))
    orders = orders.with_columns(
        pl.when(pl.col("entregue").is_null()).then(None)
        .otherwise((pl.col("entregue") > pl.col("prometida")).cast(pl.Float32)).alias("atraso"),
        ((pl.col("entregue") - pl.col("prometida")).dt.total_hours() / 24).cast(pl.Float32).alias("dias_atraso"))

    base = (items.join(orders.select("order_id", "compra", "atraso"), on="order_id", how="inner")
            .join(vend, on="seller_id").join(ped, on="order_id").join(prod, on="product_id", how="left")
            .join(reviews.select("order_id", "review_score"), on="order_id", how="left")
            .join(sellers, on="seller_id", how="left"))
    ev = (base.sort(["entidade_id", "compra", "pedido_id", "order_item_id"])
          .with_row_index("evento_id").with_columns(pl.col("evento_id").cast(pl.Int64))
          .select("evento_id", "entidade_id", pl.col("compra").alias("data_evento"), "pedido_id",
                  pl.col("product_id").alias("produto"), "categoria", pl.col("price").cast(pl.Float32).alias("preco"),
                  pl.col("freight_value").cast(pl.Float32).alias("frete"), pl.col("price").cast(pl.Float32).alias("valor"),
                  pl.col("review_score").cast(pl.Float32), "atraso", "seller_state"))
    ev.write_parquet(f"{RUN}/eventos_canonicos.parquet")

    # as outras tabelas, por vendedor (um pedido com dois vendedores conta para os dois)
    pv = base.select("order_id", "entidade_id").unique()
    pag = (pl.read_csv(f"{D}/olist_order_payments_dataset.csv").join(pv, on="order_id")
           .join(orders.select("order_id", "compra"), on="order_id")
           .select("entidade_id", pl.col("compra").alias("datahora"), pl.col("payment_type").alias("tipo"),
                   pl.col("payment_installments").clip(0, 12).cast(pl.Utf8).alias("parcelas")))
    aval = (reviews.join(pv, on="order_id").select("entidade_id", "datahora", pl.col("review_score").cast(pl.Utf8).alias("nota"))
            .drop_nulls("datahora"))
    ent = (orders.join(pv, on="order_id").drop_nulls("entregue")
           .select("entidade_id", pl.col("entregue").alias("data"), "dias_atraso",
                   (pl.col("dias_atraso") > 0).cast(pl.Utf8).alias("atrasou")))
    for nome, df, col in (("pagamento", pag, "datahora"), ("avaliacao", aval, "datahora"), ("entrega", ent, "data")):
        df.sort(["entidade_id", col]).write_parquet(f"{PREP}/{nome}.parquet")
        print(f"{nome}: {df.height:,} linhas, {df['entidade_id'].n_unique():,} vendedores")
    print(f"vendas: {ev.height:,} linhas, {ev['entidade_id'].n_unique():,} vendedores, {ev['pedido_id'].n_unique():,} pedidos; "
          f"{ev['data_evento'].min()} a {ev['data_evento'].max()}")


if __name__ == "__main__":
    main()
