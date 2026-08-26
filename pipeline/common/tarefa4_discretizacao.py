"""Motor genérico — Discretização (bucketing por quantil).

Versão schema-agnóstica de `pipeline/discretizacao/tarefa4_discretizacao.py`
(branch `luiz_g`) — mesma lógica (quantil calibrado só no treino, opcionalmente
por grupo categórico), agora pra uma lista arbitrária de campos numéricos
declarada no config (`campos_numericos`, cada um com `agrupar_por` opcional
e `n_buckets` próprio).

Uso: python3 pipeline/common/tarefa4_discretizacao.py <config.yaml>

Gera runs/<nome>/discretizacao/{buckets.json, discretizado.csv}.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "pipeline/common")
from schema import carregar_config, run_dir  # noqa: E402

SEM_GRUPO = "__GLOBAL__"


def calcular_edges(train_df, campo, agrupar_por, n_buckets):
    percentis = np.linspace(0, 100, n_buckets + 1)[1:-1]
    edges_por_grupo = {}
    if agrupar_por:
        for grupo, sub in train_df.groupby(agrupar_por):
            edges_por_grupo[grupo] = np.quantile(sub[campo].values, percentis / 100).tolist()
    else:
        edges_por_grupo[SEM_GRUPO] = np.quantile(train_df[campo].values, percentis / 100).tolist()
    return edges_por_grupo


def aplicar_bucket(valor, edges):
    return int(np.searchsorted(edges, valor, side="right"))


def main(config_path):
    config = carregar_config(config_path)
    campos_num = config["campos_numericos"]
    if not campos_num:
        print("Nenhum campo numérico configurado — nada a discretizar.")
        return

    colunas_necessarias = {"evento_id", "entidade_id"} | {c["nome"] for c in campos_num}
    colunas_necessarias |= {c["agrupar_por"] for c in campos_num if c.get("agrupar_por")}
    df = pd.read_csv(config["fonte_canonica"], usecols=list(colunas_necessarias))

    splits_csv = os.path.join(run_dir(config, "splits"), "splits.csv")
    splits = pd.read_csv(splits_csv, usecols=["evento_id", "split"])
    df = df.merge(splits, on="evento_id", validate="one_to_one")
    train_df = df[df["split"] == "train"]

    todos_edges = {}
    for campo_cfg in campos_num:
        campo = campo_cfg["nome"]
        agrupar_por = campo_cfg.get("agrupar_por")
        n_buckets = campo_cfg.get("n_buckets", 10)
        todos_edges[campo] = {
            "agrupar_por": agrupar_por,
            "n_buckets": n_buckets,
            "edges": calcular_edges(train_df, campo, agrupar_por, n_buckets),
        }

    out_dir = run_dir(config, "discretizacao")
    os.makedirs(out_dir, exist_ok=True)
    buckets_json = os.path.join(out_dir, "buckets.json")
    with open(buckets_json, "w", encoding="utf-8") as f:
        json.dump(todos_edges, f, ensure_ascii=False, indent=2)

    for campo_cfg in campos_num:
        campo = campo_cfg["nome"]
        agrupar_por = campo_cfg.get("agrupar_por")
        edges_por_grupo = todos_edges[campo]["edges"]

        def bucket_da_linha(row, campo=campo, agrupar_por=agrupar_por, edges_por_grupo=edges_por_grupo):
            chave = row[agrupar_por] if agrupar_por else SEM_GRUPO
            edges = edges_por_grupo.get(chave, edges_por_grupo.get(SEM_GRUPO))
            if edges is None:
                return -1  # grupo nunca visto no treino (raro, sem baseline pra bucketizar)
            return aplicar_bucket(row[campo], edges)

        df[f"{campo}_bucket"] = df.apply(bucket_da_linha, axis=1)

    colunas_saida = ["evento_id", "entidade_id", "split"] + [c["nome"] for c in campos_num] + \
        [f"{c['nome']}_bucket" for c in campos_num]
    out_csv = os.path.join(out_dir, "discretizado.csv")
    df[colunas_saida].to_csv(out_csv, index=False)

    print(f"Gravado {buckets_json}")
    print(f"Gravado {out_csv} com {len(df)} linhas.")
    for campo_cfg in campos_num:
        campo = campo_cfg["nome"]
        b = df[f"{campo}_bucket"]
        print(f"{campo}: min_bucket={b.min()} max_bucket={b.max()} n_distintos={b.nunique()} "
              f"n_grupo_desconhecido={(b == -1).sum()}")


if __name__ == "__main__":
    main(sys.argv[1])
