"""Tarefa 3 — Serialização das sequências por cliente.

Ver .claude/skills/customer-sequence-serialization/SKILL.md para o racional
(ordem fixa de campos, delta-recência em vez de data absoluta, ausência de
dado = ausência de token).

Monta, por `cpf`, a sequência ordenada de tokens (evento a evento) usando o
vocabulário/tokenizer da Tarefa 5 e os buckets já calculados na Tarefa 4
(via `pipeline/artifacts/discretizado.csv`).

Ordem fixa dos campos dentro de cada evento (ver ARQUITETURA.md, Estágio 1/2):
    EVT, categoria, marca, fabricante, produto, valor_bucket, desconto_bucket,
    quantidade, pagamento, canal, [recência], mês

`recência` é omitida no primeiro evento de cada cliente (não existe evento
anterior para calcular o delta) — ausência de dado vira ausência de token,
não um bucket sentinela artificial.

Gera pipeline/artifacts/sequencias.json:
    {cpf: {"n_eventos": int, "seq_full": [ids...], "seq_train": [ids...] | null}}

`seq_full` cobre todo o histórico do cliente; `seq_train` é o prefixo
truncado no corte de treino (Tarefa 7) — nunca inclui eventos de val/teste,
usado como corpus de pré-treino (Tarefa 6). Como o corte é por data, os
eventos de treino de um cliente são sempre um prefixo cronológico do
histórico completo (sem intercalação com val/teste).
"""

import json
import sys

import pandas as pd

sys.path.insert(0, "pipeline")
from tarefa5_vocabulario import (  # noqa: E402
    TokenizerFechado,
    token_canal,
    token_categoria,
    token_desconto_bucket,
    token_fabricante,
    token_marca,
    token_mes,
    token_pagamento,
    token_produto,
    token_quantidade,
    token_valor_bucket,
)

SOURCE_CSV = "base_sintetica_embeddings_100k_v2.csv"
DISCRETIZADO_CSV = "pipeline/artifacts/discretizado.csv"
VOCAB_JSON = "pipeline/artifacts/vocab.json"
OUTPUT_JSON = "pipeline/artifacts/sequencias.json"

RECENCIA_EDGES = [7, 15, 30, 60, 90, 180, 365]
RECENCIA_LABELS = ["0-7", "8-15", "16-30", "31-60", "61-90", "91-180", "181-365", ">365"]


def token_recencia(label):
    return f"RECENCIA_{label}"


def recencia_bucket(gap_dias):
    for edge, label in zip(RECENCIA_EDGES, RECENCIA_LABELS):
        if gap_dias <= edge:
            return label
    return RECENCIA_LABELS[-1]


def montar_evento(tokenizer, row, gap_dias):
    tokens = [
        "EVT",
        token_categoria(row["categoria_produto"]),
        token_marca(row["marca"]),
        token_fabricante(row["fabricante"]),
        token_produto(row["cod_produto"]),
        token_valor_bucket(row["categoria_produto"], row["valor_total_bucket"]),
        token_desconto_bucket(row["categoria_produto"], row["desconto_bucket"]),
        token_quantidade(row["quantidade"]),
        token_pagamento(row["forma_pagamento"]),
        token_canal(row["canal_venda"]),
    ]
    if gap_dias is not None:
        tokens.append(token_recencia(recencia_bucket(gap_dias)))
    tokens.append(token_mes(row["mes"]))
    return [tokenizer.encode(t) for t in tokens]


def montar_sequencias(df, tokenizer):
    bos, eos = tokenizer.encode("BOS"), tokenizer.encode("EOS")
    sequencias = {}
    for cpf, grupo in df.groupby("cpf", sort=False):
        grupo = grupo.sort_values(["data_compra", "transacao_id"]).reset_index(drop=True)
        blocos, splits_evento = [], []
        data_anterior = None
        for _, row in grupo.iterrows():
            gap = None if data_anterior is None else (row["data_compra"] - data_anterior).days
            blocos.append(montar_evento(tokenizer, row, gap))
            splits_evento.append(row["split"])
            data_anterior = row["data_compra"]

        seq_full = [bos] + [tid for bloco in blocos for tid in bloco] + [eos]
        blocos_train = [b for b, s in zip(blocos, splits_evento) if s == "train"]
        seq_train = [bos] + [tid for bloco in blocos_train for tid in bloco] + [eos] if blocos_train else None

        sequencias[cpf] = {"n_eventos": len(grupo), "seq_full": seq_full, "seq_train": seq_train}
    return sequencias


def main():
    fonte = pd.read_csv(
        SOURCE_CSV,
        encoding="utf-8-sig",
        usecols=["cpf", "data_compra", "categoria_produto", "marca", "fabricante", "cod_produto",
                  "quantidade", "forma_pagamento", "canal_venda"],
    )
    fonte["transacao_id"] = fonte.index
    fonte["data_compra"] = pd.to_datetime(fonte["data_compra"])
    fonte["mes"] = fonte["data_compra"].dt.month

    disc = pd.read_csv(DISCRETIZADO_CSV, usecols=["transacao_id", "split", "valor_total_bucket", "desconto_bucket"])
    df = fonte.merge(disc, on="transacao_id", validate="one_to_one")

    tokenizer = TokenizerFechado.load(VOCAB_JSON)

    sequencias = montar_sequencias(df, tokenizer)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(sequencias, f)
    print(f"Gravado {OUTPUT_JSON} com {len(sequencias)} clientes.\n")

    comprimentos = pd.Series({cpf: len(s["seq_full"]) for cpf, s in sequencias.items()})
    n_eventos = pd.Series({cpf: s["n_eventos"] for cpf, s in sequencias.items()})
    print("--- Distribuição do comprimento de sequência (em tokens, seq_full) ---")
    print(comprimentos.describe(percentiles=[0.5, 0.9, 0.99]))
    print(f"\n--- Distribuição do nº de eventos por cliente ---")
    print(n_eventos.describe(percentiles=[0.5, 0.9, 0.99]))

    n_sem_train = sum(1 for s in sequencias.values() if s["seq_train"] is None)
    print(f"\nClientes sem nenhum evento de treino (seq_train=null): {n_sem_train}")

    print("\n--- Validação: clientes de 1 evento geram [BOS][EVT]...[EOS] ---")
    unicos = [cpf for cpf, s in sequencias.items() if s["n_eventos"] == 1]
    cpf_1ev = unicos[0]
    seq = sequencias[cpf_1ev]["seq_full"]
    assert tokenizer.decode(seq[0]) == "BOS"
    assert tokenizer.decode(seq[1]) == "EVT"
    assert tokenizer.decode(seq[-1]) == "EOS"
    print(f"OK: cliente {cpf_1ev} (1 evento) -> {[tokenizer.decode(i) for i in seq]}")

    print(f"\n--- Spot-check manual: 5 clientes decodificados ---")
    amostra = list(sequencias.keys())[:5]
    for cpf in amostra:
        s = sequencias[cpf]
        decodificado = [tokenizer.decode(i) for i in s["seq_full"]]
        print(f"\ncpf={cpf} | n_eventos={s['n_eventos']} | len(seq_full)={len(s['seq_full'])}")
        print(decodificado)
        original = df[df["cpf"] == cpf].sort_values(["data_compra", "transacao_id"])
        print(original[["data_compra", "categoria_produto", "marca", "cod_produto", "valor_total_bucket",
                          "desconto_bucket", "quantidade", "forma_pagamento", "canal_venda", "split"]].to_string(index=False))


if __name__ == "__main__":
    main()
