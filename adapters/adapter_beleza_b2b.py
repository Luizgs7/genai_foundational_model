"""Adapter da base sintética de beleza B2B -> schema canônico do motor genérico.

Entidade = CLIENTE (revendedor ou loja); evento = LINHA de pedido (um SKU), a
mesma semântica "uma linha = um produto" das outras bases. Mantém `pedido_id` e
a data com hora: o motor v2 agrupa as linhas por pedido; o v1 ancora por
cliente e dia (`ancora: cesta`).

Saída: runs/beleza_b2b_<tamanho>/eventos_canonicos.parquet com
  evento_id (ordem por cliente, data e pedido), entidade_id, data_evento, pedido_id,
  campos do item: sku, categoria, tipo_produto, familia, marca, faixa_preco;
  campos da linha: quantidade, preco (unitário), valor (quantidade x preço), desconto (0/1);
  campos do pedido: canal, forma_pagamento, online (0/1: aplicativo ou site);
  cadastro: tipo_cliente, uf, canal_aquisicao, idade.

Uso: python3 adapters/adapter_beleza_b2b.py mini|pequeno|grande
"""

import os
import sys

import polars as pl


def main():
    tamanho = sys.argv[1]
    d = f"dados_beleza_b2b/{tamanho}"
    out = f"runs/beleza_b2b_{tamanho}"
    os.makedirs(out, exist_ok=True)
    it = pl.read_parquet(f"{d}/itens_pedido.parquet")
    ped = pl.read_parquet(f"{d}/pedidos.parquet", columns=["pedido_id", "canal", "forma_pagamento"])
    prod = pl.read_parquet(f"{d}/produtos.parquet", columns=["sku", "categoria", "tipo_produto", "familia", "marca", "faixa_preco"])
    cli = pl.read_parquet(f"{d}/clientes.parquet", columns=["cliente_id", "tipo_cliente", "uf", "canal_aquisicao", "idade"])
    ev = (it.join(ped, on="pedido_id", how="left").join(prod, on="sku", how="left").join(cli, on="cliente_id", how="left")
          .sort(["cliente_id", "data", "pedido_id", "sku"])
          .with_columns([pl.col("cliente_id").cast(pl.Int32).alias("entidade_id"), pl.col("data").alias("data_evento"),
                         pl.col("preco_unitario").cast(pl.Float32).alias("preco"),
                         (pl.col("quantidade") * pl.col("preco_unitario")).cast(pl.Float32).alias("valor"),
                         pl.col("em_promocao").cast(pl.Float32).alias("desconto"),
                         pl.col("canal").is_in(["aplicativo", "site"]).cast(pl.Float32).alias("online")])
          .with_row_index("evento_id").with_columns(pl.col("evento_id").cast(pl.Int64)))
    cols = ["evento_id", "entidade_id", "data_evento", "pedido_id", "sku", "categoria", "tipo_produto", "familia", "marca", "faixa_preco",
            "quantidade", "preco", "valor", "desconto", "canal", "forma_pagamento", "online", "tipo_cliente", "uf", "canal_aquisicao", "idade"]
    ev.select(cols).write_parquet(f"{out}/eventos_canonicos.parquet")
    print(f"{ev.height:,} linhas, {ev['entidade_id'].n_unique():,} clientes, {ev['pedido_id'].n_unique():,} pedidos -> {out}/eventos_canonicos.parquet")


if __name__ == "__main__":
    main()
