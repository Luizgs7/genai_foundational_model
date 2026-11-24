"""MODELO DE ADAPTER para uma base nova (copie como adapters/adapter_<empresa>.py e preencha).

O adapter é a única parte de código específica da empresa. Ele só lê as tabelas brutas,
filtra, renomeia e grava os arquivos que o YAML (configs/modelo_empresa.yaml) declara.
Nada de modelagem aqui: variáveis, rótulos e o modelo saem do motor.

Saídas obrigatórias:
  runs/<nome>/eventos_canonicos.parquet   a fonte-alvo (linhas de pedido), com:
      evento_id    Int64, único, em ordem de (entidade_id, data_evento, pedido_id)
      entidade_id  Int32 (cliente); converta ids de texto em inteiros ANTES de qualquer join grande
      data_evento  Datetime
      pedido_id    Int64 (agrupa as linhas de um pedido)
      + os campos categóricos, numéricos, de valor e estáticos declarados no YAML
Saídas opcionais (uma por fonte extra do bloco v2 do YAML):
  dados_<empresa>/preparado/<fonte>.parquet   entidade_id + a coluna de data declarada + os campos

Regras (ver a skill motor-v2-nova-base e o agente auditor-vazamento):
  - tudo o que depende de frequência (catálogo top-N, categorias top-K) usa só dados ANTERIORES
    ao corte da validação, para não vazar o futuro;
  - campos que só existem depois do pedido (nota, devolução, atraso) entram apenas como campo
    de rótulo (receita campo_futuro), nunca como entrada da mesma posição;
  - rode pesado com scripts/rodar_limitado.sh -m 7G numa máquina de pouca memória.

Uso: python3 adapters/adapter_<empresa>.py
"""

import os

import polars as pl

BRUTO = "dados_minha_empresa/bruto"          # onde a empresa deixa as extrações
PREP = "dados_minha_empresa/preparado"
RUN = "runs/minha_empresa"


def main():
    os.makedirs(PREP, exist_ok=True)
    os.makedirs(RUN, exist_ok=True)

    # 1. clientes: id de texto -> inteiro, uma vez
    clientes = (pl.scan_parquet(f"{BRUTO}/clientes.parquet")
                .select("id_cliente", "tipo_cliente", "uf").collect()
                .sort("id_cliente").with_row_index("entidade_id")
                .with_columns(pl.col("entidade_id").cast(pl.Int32)))

    # 2. fonte-alvo: linhas de pedido
    linhas = (pl.scan_parquet(f"{BRUTO}/itens_pedido.parquet")
              .join(clientes.lazy().select("id_cliente", "entidade_id", "tipo_cliente", "uf"), on="id_cliente")
              .with_columns(pl.col("data_pedido").cast(pl.Datetime("ms")).alias("data_evento"))
              .collect())
    pedidos = linhas.select("id_pedido").unique().sort("id_pedido").with_row_index("pedido_id")
    ev = (linhas.join(pedidos.with_columns(pl.col("pedido_id").cast(pl.Int64)), on="id_pedido")
          .sort(["entidade_id", "data_evento", "pedido_id"])
          .with_row_index("evento_id").with_columns(pl.col("evento_id").cast(pl.Int64))
          .select("evento_id", "entidade_id", "data_evento", "pedido_id", "sku", "categoria",
                  "quantidade", "valor", "desconto", "canal", "tipo_cliente", "uf"))
    ev.write_parquet(f"{RUN}/eventos_canonicos.parquet")

    # 3. fontes extras (uma por bloco do YAML)
    nav = (pl.scan_parquet(f"{BRUTO}/navegacao.parquet")
           .join(clientes.lazy().select("id_cliente", "entidade_id"), on="id_cliente")
           .select("entidade_id", pl.col("momento").cast(pl.Datetime("ms")).alias("datahora"), "tipo_pagina")
           .sort(["entidade_id", "datahora"]).collect())
    nav.write_parquet(f"{PREP}/navegacao.parquet")

    print(f"pedidos: {ev.height:,} linhas, {ev['entidade_id'].n_unique():,} clientes; "
          f"navegação: {nav.height:,} eventos")


if __name__ == "__main__":
    main()
