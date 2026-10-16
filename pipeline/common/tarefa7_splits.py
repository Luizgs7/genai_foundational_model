"""Motor genérico — Splits de treino/validação/teste sem leakage.

Versão schema-agnóstica de `pipeline/splits/tarefa7_splits.py` (branch
`luiz_g`) — mesma lógica (corte por data, sem sorteio, checado por assert de
não-overlap), parametrizada via config em vez de hardcoded pra base
sintética. Ver .claude/skills/temporal-customer-splits/SKILL.md.

Uso: python3 pipeline/common/tarefa7_splits.py <config.yaml>

Gera runs/<nome>/splits/splits.parquet com: evento_id, entidade_id, data_evento,
split, elegivel_downstream (entidade com >=2 eventos no total).
"""

import sys
from datetime import timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, "pipeline/common")
from schema import carregar_config, gravar_tabela, ler_fonte  # noqa: E402


def main(config_path):
    config = carregar_config(config_path)
    df = ler_fonte(config, ["evento_id", "entidade_id", "data_evento"])
    df["data_evento"] = pd.to_datetime(df["data_evento"])

    val_window = config["splits"]["val_window_days"]
    test_window = config["splits"]["test_window_days"]

    date_end = df["data_evento"].max()
    test_cutoff = date_end - timedelta(days=test_window)
    val_cutoff = test_cutoff - timedelta(days=val_window)

    # vetorizado (antes: .map com função Python por linha) — mesma regra
    df["split"] = np.where(df["data_evento"] < val_cutoff, "train",
                           np.where(df["data_evento"] < test_cutoff, "val", "test"))

    max_train = df.loc[df["split"] == "train", "data_evento"].max()
    min_val = df.loc[df["split"] == "val", "data_evento"].min()
    max_val = df.loc[df["split"] == "val", "data_evento"].max()
    min_test = df.loc[df["split"] == "test", "data_evento"].min()
    assert max_train < min_val, f"Overlap treino/val: {max_train} >= {min_val}"
    assert max_val < min_test, f"Overlap val/teste: {max_val} >= {min_test}"

    contagem_por_entidade = df.groupby("entidade_id").size()
    elegiveis = contagem_por_entidade[contagem_por_entidade >= 2].index
    df["elegivel_downstream"] = df["entidade_id"].isin(elegiveis)

    out_csv = gravar_tabela(df, config, "splits", "splits")

    print(f"val_cutoff={val_cutoff.date()}  test_cutoff={test_cutoff.date()}")
    print(f"max(train)={max_train}  min(val)={min_val}  max(val)={max_val}  min(test)={min_test}")
    print("\nContagem de linhas por split:")
    print(df["split"].value_counts())
    print(f"\nEntidades ({config['entidade_label']}) elegíveis para downstream (>=2 eventos): "
          f"{len(elegiveis)} de {contagem_por_entidade.shape[0]}")
    print(f"Gravado {out_csv} com {len(df)} linhas.")


if __name__ == "__main__":
    main(sys.argv[1])
