"""Adapter Olist Brazilian E-commerce (Kaggle, olistbr/brazilian-ecommerce)
pro schema canônico do motor genérico — entidade = VENDEDOR (seller_id).

Granularidade: uma linha = um item de pedido (order_item) vendido por um
vendedor — mesma semântica "uma linha = um produto" da base sintética
(ver PLANO_PRODUTO.md, trade-off #5).

Campos de rótulo (não tokenizados, só usados pela Tarefa 8 genérica):
  - review_score: nota (1-5) do PRÓXIMO pedido do vendedor -> receita
    `risco_review_negativo`. Atributo do pedido inteiro (join por order_id),
    não do item — linhas-irmãs do mesmo pedido herdam a mesma nota.
  - atraso: 1 se `order_delivered_customer_date > order_estimated_delivery_date`,
    0 se entregue no prazo, NaN se ainda não entregue -> receita
    `risco_atraso_entrega`. Também atributo do pedido inteiro.

order_status filtrado pra excluir 'canceled'/'unavailable' (venda que não
se concretizou).

Saída: `runs/olist_sellers/eventos_canonicos.csv`.
"""

import os

import pandas as pd

DADOS_DIR = "dados_olist"
OUT_CSV = "runs/olist_sellers/eventos_canonicos.csv"

STATUS_EXCLUIDOS = {"canceled", "unavailable"}


def main():
    items = pd.read_csv(os.path.join(DADOS_DIR, "olist_order_items_dataset.csv"))

    orders = pd.read_csv(
        os.path.join(DADOS_DIR, "olist_orders_dataset.csv"),
        parse_dates=["order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"],
    )
    orders = orders[~orders["order_status"].isin(STATUS_EXCLUIDOS)]
    orders["atraso"] = (
        (orders["order_delivered_customer_date"] > orders["order_estimated_delivery_date"]).astype(float)
    )
    orders.loc[orders["order_delivered_customer_date"].isna(), "atraso"] = float("nan")

    products = pd.read_csv(os.path.join(DADOS_DIR, "olist_products_dataset.csv"), usecols=["product_id", "product_category_name"])
    traducao = pd.read_csv(os.path.join(DADOS_DIR, "product_category_name_translation.csv"))
    products = products.merge(traducao, on="product_category_name", how="left")
    products["categoria"] = products["product_category_name_english"].fillna("sem_categoria")

    pagamentos = pd.read_csv(os.path.join(DADOS_DIR, "olist_order_payments_dataset.csv"))
    pagamento_dominante = (
        pagamentos.sort_values("payment_value", ascending=False)
        .drop_duplicates(subset="order_id", keep="first")[["order_id", "payment_type"]]
    )

    reviews = pd.read_csv(os.path.join(DADOS_DIR, "olist_order_reviews_dataset.csv"), usecols=["order_id", "review_score"])
    reviews = reviews.drop_duplicates(subset="order_id", keep="first")

    sellers = pd.read_csv(os.path.join(DADOS_DIR, "olist_sellers_dataset.csv"), usecols=["seller_id", "seller_state"])

    df = items.merge(orders, on="order_id", how="inner", validate="many_to_one")
    df = df.merge(products, on="product_id", how="left", validate="many_to_one")
    df = df.merge(pagamento_dominante, on="order_id", how="left", validate="many_to_one")
    df = df.merge(reviews, on="order_id", how="left", validate="many_to_one")
    df = df.merge(sellers, on="seller_id", how="left", validate="many_to_one")

    df["payment_type"] = df["payment_type"].fillna("nao_definido")

    saida = pd.DataFrame({
        "evento_id": range(len(df)),
        "entidade_id": df["seller_id"],
        "data_evento": df["order_purchase_timestamp"],
        "categoria": df["categoria"],
        "produto": df["product_id"],
        "pagamento": df["payment_type"],
        "preco": df["price"],
        "frete": df["freight_value"],
        "review_score": df["review_score"],
        "atraso": df["atraso"],
        "seller_state": df["seller_state"],
    })
    saida = saida.dropna(subset=["entidade_id", "data_evento", "preco"])

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    saida.to_csv(OUT_CSV, index=False)
    print(f"Gravado {OUT_CSV} ({len(saida)} eventos, {saida['entidade_id'].nunique()} vendedores)")
    print(f"\nEventos por vendedor:\n{saida.groupby('entidade_id').size().describe(percentiles=[0.5, 0.75, 0.9, 0.99])}")
    print(f"\nreview_score presente: {saida['review_score'].notna().mean():.1%}")
    print(f"atraso determinável: {saida['atraso'].notna().mean():.1%}  (taxa de atraso entre determináveis: "
          f"{saida['atraso'].dropna().mean():.1%})")


if __name__ == "__main__":
    main()
