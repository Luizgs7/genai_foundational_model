"""Adapter de corte temporal: copia uma tabela canônica pronta só com os eventos ANTES de uma data.

Serve para avaliar o motor num corte anterior da mesma base (F3-H10: a H&M terminando em
25/03/2020). Os ids (evento_id, pedido_id, entidade_id) e a ordem ficam os mesmos da tabela de
origem; as janelas de validação e teste saem do YAML, contadas a partir do último dia que sobra.
As colunas categóricas são gravadas com as categorias da origem em ordem alfabética (como na tabela
canônica da H&M): o polars gravaria a ordem de aparição, e isso muda o desempate da moda nos rótulos.

Uso: python3 adapters/cortar_canonica.py <origem.parquet> <destino.parquet> <AAAA-MM-DD (exclusiva)>
"""
import os
import sys
from datetime import datetime

import polars as pl


def main():
    origem, destino, fim = sys.argv[1], sys.argv[2], datetime.strptime(sys.argv[3], "%Y-%m-%d")
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    lf = pl.scan_parquet(origem)
    cats = [c for c, t in lf.collect_schema().items() if t == pl.Categorical]
    niveis = {c: sorted(lf.select(pl.col(c).cast(pl.Utf8).drop_nulls().unique()).collect()[c].to_list()) for c in cats}
    (lf.filter(pl.col("data_evento") < fim)
       .with_columns([pl.col(c).cast(pl.Utf8).cast(pl.Enum(niveis[c])) for c in cats])
       .sink_parquet(destino))
    r = pl.scan_parquet(destino).select(pl.len().alias("n"), pl.col("data_evento").min().alias("ini"),
                                        pl.col("data_evento").max().alias("fim"),
                                        pl.col("entidade_id").n_unique().alias("clientes")).collect()
    print(f"corte antes de {sys.argv[3]}: {r['n'][0]:,} eventos, {r['clientes'][0]:,} clientes, "
          f"{r['ini'][0]} a {r['fim'][0]}")


if __name__ == "__main__":
    main()
