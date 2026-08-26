"""Motor genérico — Serialização das sequências por entidade.

Versão schema-agnóstica de `pipeline/serializacao/tarefa3_serializacao.py`
(branch `luiz_g`). Ordem fixa dentro de cada bloco de evento: EVT, os
campos categóricos do config (na ordem declarada), os campos numéricos do
config (bucket, na ordem declarada), [recência, se não for o 1º evento da
entidade e usa_recencia], [mês, se usa_mes].

Correção estrutural desta generalização (ver PLANO_PRODUTO.md): como o
número de campos por bloco agora é configurável por empresa, a Tarefa 13
não pode mais reconstruir a posição de âncora de um evento por fórmula
aritmética fixa. Este script grava explicitamente, por evento, o índice do
seu último token dentro de `seq_full` — `runs/<nome>/serializacao/posicoes_evento.csv`.

Uso: python3 pipeline/common/tarefa3_serializacao.py <config.yaml>

Gera runs/<nome>/serializacao/{sequencias.json, posicoes_evento.csv}.
"""

import json
import os
import sys

import pandas as pd

sys.path.insert(0, "pipeline/common")
from schema import carregar_config, run_dir  # noqa: E402

RECENCIA_EDGES = [7, 15, 30, 60, 90, 180, 365]
RECENCIA_LABELS = ["0-7", "8-15", "16-30", "31-60", "61-90", "91-180", "181-365", ">365"]


def recencia_bucket(gap_dias):
    for edge, label in zip(RECENCIA_EDGES, RECENCIA_LABELS):
        if gap_dias <= edge:
            return label
    return RECENCIA_LABELS[-1]


def carregar_top_n_mapa(config):
    path = os.path.join(run_dir(config, "vocabulario"), "top_n_mapa.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def montar_evento(config, vocab, top_n_mapa, row, gap_dias):
    tokens = ["EVT"]

    for campo_cfg in config["campos_categoricos"]:
        campo = campo_cfg["nome"]
        valor = row[campo]
        if campo_cfg["estrategia"] == "top_n_outros":
            info = top_n_mapa[campo]
            tok = info["mapa"].get(str(valor), info["outros_token"])
        else:
            tok = f"{campo.upper()}_{valor}"
        tokens.append(tok)

    for campo_cfg in config["campos_numericos"]:
        campo = campo_cfg["nome"]
        bucket = row[f"{campo}_bucket"]
        agrupar_por = campo_cfg.get("agrupar_por")
        if agrupar_por:
            grupo = row[agrupar_por]
            tokens.append(f"{campo.upper()}_BUCKET_{grupo}_{bucket}")
        else:
            tokens.append(f"{campo.upper()}_BUCKET_{bucket}")

    if gap_dias is not None and config.get("usa_recencia", True):
        tokens.append(f"RECENCIA_{recencia_bucket(gap_dias)}")
    if config.get("usa_mes", True):
        tokens.append(f"MES_{row['mes']}")

    return [vocab.get(t, vocab["UNK"]) for t in tokens]


def montar_sequencias(config, df, vocab, top_n_mapa):
    bos, eos = vocab["BOS"], vocab["EOS"]
    sequencias = {}
    posicoes = []  # linhas de posicoes_evento.csv

    for entidade_id, grupo in df.groupby("entidade_id", sort=False):
        grupo = grupo.sort_values(["data_evento", "evento_id"]).reset_index(drop=True)
        blocos, splits_evento, evento_ids = [], [], []
        data_anterior = None
        for _, row in grupo.iterrows():
            gap = None if data_anterior is None else (row["data_evento"] - data_anterior).days
            blocos.append(montar_evento(config, vocab, top_n_mapa, row, gap))
            splits_evento.append(row["split"])
            evento_ids.append(row["evento_id"])
            data_anterior = row["data_evento"]

        # pos_full[i] = índice (0-indexado) do último token do i-ésimo bloco
        # dentro de `[bos] + blocos...` (sem contar o EOS final) — âncora
        # explícita, não reconstruída por fórmula (ver docstring do módulo).
        cursor = 1  # depois do bos
        pos_full = []
        for bloco in blocos:
            cursor += len(bloco)
            pos_full.append(cursor - 1)
        for evento_id, split_evento, pos in zip(evento_ids, splits_evento, pos_full):
            posicoes.append({"evento_id": evento_id, "entidade_id": entidade_id, "pos_full": pos})

        seq_full = [bos] + [tid for bloco in blocos for tid in bloco] + [eos]
        blocos_train = [b for b, s in zip(blocos, splits_evento) if s == "train"]
        seq_train = [bos] + [tid for bloco in blocos_train for tid in bloco] + [eos] if blocos_train else None
        blocos_train_val = [b for b, s in zip(blocos, splits_evento) if s in ("train", "val")]
        seq_train_val = (
            [bos] + [tid for bloco in blocos_train_val for tid in bloco] + [eos] if blocos_train_val else None
        )

        sequencias[entidade_id] = {
            "n_eventos": len(grupo),
            "seq_full": seq_full,
            "seq_train": seq_train,
            "seq_train_val": seq_train_val,
        }

    return sequencias, posicoes


def main(config_path):
    config = carregar_config(config_path)

    colunas = ["evento_id", "entidade_id", "data_evento"] + \
        [c["nome"] for c in config["campos_categoricos"]] + \
        [c.get("agrupar_por") for c in config["campos_numericos"] if c.get("agrupar_por")]
    colunas = [c for c in dict.fromkeys(colunas) if c]  # dedup preservando ordem

    fonte = pd.read_csv(config["fonte_canonica"], usecols=colunas)
    fonte["data_evento"] = pd.to_datetime(fonte["data_evento"])
    fonte["mes"] = fonte["data_evento"].dt.month

    splits_csv = os.path.join(run_dir(config, "splits"), "splits.csv")
    splits = pd.read_csv(splits_csv, usecols=["evento_id", "split"])
    df = fonte.merge(splits, on="evento_id", validate="one_to_one")

    if config["campos_numericos"]:
        disc_csv = os.path.join(run_dir(config, "discretizacao"), "discretizado.csv")
        cols_bucket = ["evento_id"] + [f"{c['nome']}_bucket" for c in config["campos_numericos"]]
        disc = pd.read_csv(disc_csv, usecols=cols_bucket)
        df = df.merge(disc, on="evento_id", validate="one_to_one")

    vocab_json = os.path.join(run_dir(config, "vocabulario"), "vocab.json")
    with open(vocab_json, encoding="utf-8") as f:
        vocab = json.load(f)
    top_n_mapa = carregar_top_n_mapa(config)

    sequencias, posicoes = montar_sequencias(config, df, vocab, top_n_mapa)

    out_dir = run_dir(config, "serializacao")
    os.makedirs(out_dir, exist_ok=True)
    seq_json = os.path.join(out_dir, "sequencias.json")
    with open(seq_json, "w", encoding="utf-8") as f:
        json.dump(sequencias, f)

    pos_csv = os.path.join(out_dir, "posicoes_evento.csv")
    pd.DataFrame(posicoes).to_csv(pos_csv, index=False)

    print(f"Gravado {seq_json} com {len(sequencias)} entidades ({config['entidade_label']}s).")
    print(f"Gravado {pos_csv} com {len(posicoes)} posições de âncora.")

    comprimentos = pd.Series({k: len(s["seq_full"]) for k, s in sequencias.items()})
    print("\n--- Distribuição do comprimento de sequência (em tokens, seq_full) ---")
    print(comprimentos.describe(percentiles=[0.5, 0.9, 0.99]))

    n_sem_train = sum(1 for s in sequencias.values() if s["seq_train"] is None)
    print(f"\nEntidades sem nenhum evento de treino (seq_train=null): {n_sem_train}")


if __name__ == "__main__":
    main(sys.argv[1])
