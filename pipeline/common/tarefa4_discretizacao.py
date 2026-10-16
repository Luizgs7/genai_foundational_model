"""Motor genérico — Discretização (bucketing por quantil).

Versão schema-agnóstica de `pipeline/discretizacao/tarefa4_discretizacao.py`
(branch `luiz_g`) — mesma lógica (quantil calibrado só no treino, opcionalmente
por grupo categórico), agora pra uma lista arbitrária de campos numéricos
declarada no config (`campos_numericos`, cada um com `agrupar_por` opcional
e `n_buckets` próprio).

Uso: python3 pipeline/common/tarefa4_discretizacao.py <config.yaml>

Gera runs/<nome>/discretizacao/{buckets.json, discretizado.parquet}.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, "pipeline/common")
from schema import caminho_tabela, carregar_config, gravar_tabela, ler_fonte, ler_tabela, run_dir  # noqa: E402

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


def aplicar_buckets(df, campo, agrupar_por, edges_por_grupo):
    """Vetorizado (antes: df.apply linha a linha). Mesma regra:
    searchsorted(side=right) nos edges do grupo da linha; grupo nunca visto no
    treino cai nos edges globais se existirem, senão vira -1."""
    saida = np.full(len(df), -1, dtype=np.int64)
    valores = df[campo].to_numpy(dtype=float)
    if agrupar_por:
        chaves = df[agrupar_por]
        for grupo, idx in chaves.groupby(chaves, sort=False).indices.items():
            edges = edges_por_grupo.get(grupo, edges_por_grupo.get(SEM_GRUPO))
            if edges is not None:
                saida[idx] = np.searchsorted(edges, valores[idx], side="right")
    else:
        edges = edges_por_grupo.get(SEM_GRUPO)
        if edges is not None:
            saida[:] = np.searchsorted(edges, valores, side="right")
    return saida


def main(config_path):
    config = carregar_config(config_path)
    campos_num = config["campos_numericos"]
    if not campos_num:
        print("Nenhum campo numérico configurado — nada a discretizar.")
        return

    colunas_necessarias = {"evento_id", "entidade_id"} | {c["nome"] for c in campos_num}
    colunas_necessarias |= {c["agrupar_por"] for c in campos_num if c.get("agrupar_por")}
    df = ler_fonte(config, sorted(colunas_necessarias))

    splits = ler_tabela(caminho_tabela(config, "splits", "splits"), ["evento_id", "split"])
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
        df[f"{campo}_bucket"] = aplicar_buckets(df, campo, campo_cfg.get("agrupar_por"), todos_edges[campo]["edges"])

    colunas_saida = ["evento_id", "entidade_id", "split"] + [c["nome"] for c in campos_num] + \
        [f"{c['nome']}_bucket" for c in campos_num]
    out_csv = gravar_tabela(df[colunas_saida], config, "discretizacao", "discretizado")

    print(f"Gravado {buckets_json}")
    print(f"Gravado {out_csv} com {len(df)} linhas.")
    for campo_cfg in campos_num:
        campo = campo_cfg["nome"]
        b = df[f"{campo}_bucket"]
        print(f"{campo}: min_bucket={b.min()} max_bucket={b.max()} n_distintos={b.nunique()} "
              f"n_grupo_desconhecido={(b == -1).sum()}")


if __name__ == "__main__":
    main(sys.argv[1])
