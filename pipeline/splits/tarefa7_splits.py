"""Tarefa 7 — Splits de treino/validação/teste sem leakage.

Ver .claude/skills/temporal-customer-splits/SKILL.md para o racional.

Gera pipeline/splits/splits.csv com uma linha por transação de
base_sintetica_embeddings_100k_v2.csv, contendo:
  - transacao_id: índice da linha no CSV de origem (para join posterior)
  - cpf, data_compra
  - split: "train" | "val" | "test"
  - elegivel_downstream: True se o cliente (cpf) tem >=2 transações no total
"""

from datetime import timedelta

import pandas as pd

SOURCE_CSV = "base_sintetica_embeddings_100k_v2.csv"
OUTPUT_CSV = "pipeline/splits/splits.csv"

# TEST_WINDOW_DAYS=180 (não 90): a Tarefa 8 define churn com um horizonte de
# N=121 dias (censura: só sabemos que um cliente "não voltou" se passaram
# >=N dias até o fim do dataset sem nova compra). Com uma janela de teste
# menor que N, NENHUM evento de teste consegue satisfazer essa condição —
# o split fica estruturalmente incapaz de avaliar churn (taxa 0% garantida,
# não é o modelo acertando tudo). 180 dias dá margem confortável acima de N
# (verificado por simulação: ~63% das linhas de teste viram determináveis,
# taxa de churn resultante ~14%, não-degenerada). Custo aceito: treino cai
# de 82,4% para 74,4% do total de linhas.
TEST_WINDOW_DAYS = 180
VAL_WINDOW_DAYS = 90


def main():
    df = pd.read_csv(SOURCE_CSV, encoding="utf-8-sig", usecols=["cpf", "data_compra"])
    df["data_compra"] = pd.to_datetime(df["data_compra"])
    df["transacao_id"] = df.index

    date_end = df["data_compra"].max()
    test_cutoff = date_end - timedelta(days=TEST_WINDOW_DAYS)
    val_cutoff = test_cutoff - timedelta(days=VAL_WINDOW_DAYS)

    def rotular(data):
        if data < val_cutoff:
            return "train"
        elif data < test_cutoff:
            return "val"
        else:
            return "test"

    df["split"] = df["data_compra"].map(rotular)

    # Checagem de não-overlap (assert, não checagem visual).
    max_train = df.loc[df["split"] == "train", "data_compra"].max()
    min_val = df.loc[df["split"] == "val", "data_compra"].min()
    max_val = df.loc[df["split"] == "val", "data_compra"].max()
    min_test = df.loc[df["split"] == "test", "data_compra"].min()
    assert max_train < min_val, f"Overlap treino/val: {max_train} >= {min_val}"
    assert max_val < min_test, f"Overlap val/teste: {max_val} >= {min_test}"

    contagem_por_cliente = df.groupby("cpf").size()
    elegiveis = contagem_por_cliente[contagem_por_cliente >= 2].index
    df["elegivel_downstream"] = df["cpf"].isin(elegiveis)

    saida = df[["transacao_id", "cpf", "data_compra", "split", "elegivel_downstream"]]
    saida.to_csv(OUTPUT_CSV, index=False)

    print(f"val_cutoff={val_cutoff.date()}  test_cutoff={test_cutoff.date()}")
    print(f"max(train)={max_train}  min(val)={min_val}  max(val)={max_val}  min(test)={min_test}")
    print("\nContagem de linhas por split:")
    print(df["split"].value_counts())
    print("\nContagem de clientes (cpf) distintos por split:")
    print(df.groupby("split")["cpf"].nunique())
    print(f"\nClientes elegíveis para downstream (>=2 transações): {len(elegiveis)} de {contagem_por_cliente.shape[0]}")
    print(f"Gravado {OUTPUT_CSV} com {len(saida)} linhas.")


if __name__ == "__main__":
    main()
