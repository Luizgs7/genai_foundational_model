"""Adapter da base sintética agro pro schema canônico do motor genérico.

Quase 1:1 — só renomeia colunas pros nomes canônicos e cria um `evento_id`
estável (mesma convenção usada em `pipeline/rotulos_downstream/tarefa8_rotulos.py`
na branch `luiz_g`: o índice de linha do CSV bruto).

Saída: `runs/synthetic_agro/eventos_canonicos.csv`.
"""

import pandas as pd

SOURCE_CSV = "base_sintetica_embeddings_100k_v2.csv"
OUT_CSV = "runs/synthetic_agro/eventos_canonicos.csv"

RENOMEAR = {
    "cpf": "entidade_id",
    "data_compra": "data_evento",
    "categoria_produto": "categoria",
    "marca": "marca",
    "fabricante": "fabricante",
    "cod_produto": "produto",
    "quantidade": "quantidade",
    "valor_total": "valor",
    "desconto": "desconto",
    "forma_pagamento": "pagamento",
    "canal_venda": "canal",
    "uf": "uf",
    "sexo": "sexo",
    "data_nascimento": "data_nascimento",
}


def main():
    df = pd.read_csv(SOURCE_CSV, encoding="utf-8-sig")
    df["evento_id"] = df.index
    df = df.rename(columns=RENOMEAR)
    colunas = ["evento_id"] + list(RENOMEAR.values())
    df = df[colunas]
    df.to_csv(OUT_CSV, index=False)
    print(f"Gravado {OUT_CSV} ({len(df)} eventos, {df['entidade_id'].nunique()} entidades)")


if __name__ == "__main__":
    main()
