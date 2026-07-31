"""Tarefa 4 — Discretização (bucketing por quantil, por categoria_produto).

Ver .claude/skills/customer-sequence-serialization/SKILL.md para o racional
(por que quantil, por que separado por categoria, por que calibrar só no treino).

Gera:
  - pipeline/artifacts/buckets.json: limites de quantil por categoria, para
    valor_total e desconto (10 buckets cada), calculados só no split de treino.
  - pipeline/artifacts/discretizado.csv: transacao_id, categoria_produto,
    valor_total, valor_total_bucket, desconto, desconto_bucket, split.
"""

import json

import numpy as np
import pandas as pd

SOURCE_CSV = "base_sintetica_embeddings_100k_v2.csv"
SPLITS_CSV = "pipeline/artifacts/splits.csv"
BUCKETS_JSON = "pipeline/artifacts/buckets.json"
OUTPUT_CSV = "pipeline/artifacts/discretizado.csv"

N_BUCKETS = 10
CAMPOS = ["valor_total", "desconto"]


def calcular_edges(train_df, campo):
    edges_por_categoria = {}
    for categoria, grupo in train_df.groupby("categoria_produto"):
        percentis = np.linspace(0, 100, N_BUCKETS + 1)[1:-1]  # 9 edges internos -> 10 buckets
        edges = np.quantile(grupo[campo].values, percentis / 100)
        edges_por_categoria[categoria] = edges.tolist()
    return edges_por_categoria


def aplicar_bucket(valor, edges):
    return int(np.searchsorted(edges, valor, side="right"))


def main():
    df = pd.read_csv(SOURCE_CSV, encoding="utf-8-sig", usecols=["categoria_produto", "valor_total", "desconto"])
    df["transacao_id"] = df.index

    splits = pd.read_csv(SPLITS_CSV, usecols=["transacao_id", "split"])
    df = df.merge(splits, on="transacao_id", validate="one_to_one")

    train_df = df[df["split"] == "train"]

    todos_edges = {campo: calcular_edges(train_df, campo) for campo in CAMPOS}
    with open(BUCKETS_JSON, "w", encoding="utf-8") as f:
        json.dump(todos_edges, f, ensure_ascii=False, indent=2)

    for campo in CAMPOS:
        df[f"{campo}_bucket"] = df.apply(
            lambda row, c=campo: aplicar_bucket(row[c], todos_edges[c][row["categoria_produto"]]),
            axis=1,
        )

    colunas_saida = ["transacao_id", "categoria_produto", "split"] + CAMPOS + [f"{c}_bucket" for c in CAMPOS]
    df[colunas_saida].to_csv(OUTPUT_CSV, index=False)

    print(f"Gravado {BUCKETS_JSON} (limites de quantil por categoria, calibrados no treino).")
    print(f"Gravado {OUTPUT_CSV} com {len(df)} linhas.\n")

    print("--- Ocupação dos buckets no TREINO ---")
    for campo in CAMPOS:
        print(f"\n{campo}:")
        ocupacao = train_df.assign(
            bucket=train_df.apply(lambda row, c=campo: aplicar_bucket(row[c], todos_edges[c][row["categoria_produto"]]), axis=1)
        ).groupby(["categoria_produto", "bucket"]).size().unstack(fill_value=0)
        print(ocupacao)

    print("\n--- Checagem: buckets aplicados em val/test (fora do range de treino ainda mapeiam 0-9) ---")
    for split_name in ["val", "test"]:
        sub = df[df["split"] == split_name]
        for campo in CAMPOS:
            b = sub[f"{campo}_bucket"]
            print(f"{split_name}/{campo}: min_bucket={b.min()} max_bucket={b.max()} n_buckets_distintos={b.nunique()}")


if __name__ == "__main__":
    main()
