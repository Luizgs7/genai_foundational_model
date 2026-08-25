"""Tarefa 5 — Vocabulário e tokenizer fechado (sem BPE).

Ver .claude/skills/customer-sequence-serialization/SKILL.md para o racional.

Gera pipeline/vocabulario/vocab.json (token -> id) e valida que 100% dos
valores categóricos/bucket/mês observados no dataset mapeiam para um token
válido (sem UNK).
"""

import json

import pandas as pd

SOURCE_CSV = "base_sintetica_embeddings_100k_v2.csv"
BUCKETS_JSON = "pipeline/discretizacao/buckets.json"
VOCAB_JSON = "pipeline/vocabulario/vocab.json"

RECENCIA_BUCKETS = ["0-7", "8-15", "16-30", "31-60", "61-90", "91-180", "181-365", ">365"]
ESTRUTURAIS = ["BOS", "EOS", "EVT", "PAD", "UNK"]


class TokenizerFechado:
    def __init__(self, vocab):
        self.token_to_id = vocab
        self.id_to_token = {v: k for k, v in vocab.items()}

    def encode(self, token):
        if token not in self.token_to_id:
            return self.token_to_id["UNK"]
        return self.token_to_id[token]

    def decode(self, token_id):
        return self.id_to_token[token_id]

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.token_to_id, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            vocab = json.load(f)
        return cls(vocab)


def token_categoria(v):
    return f"CATEGORIA_{v}"


def token_marca(v):
    return f"MARCA_{v}"


def token_fabricante(v):
    return f"FABRICANTE_{v}"


def token_pagamento(v):
    return f"PAGAMENTO_{v}"


def token_canal(v):
    return f"CANAL_{v}"


def token_produto(v):
    return f"PRODUTO_{v}"


def token_quantidade(v):
    return f"QTD_{v}"


def token_valor_bucket(categoria, bucket):
    return f"VALOR_BUCKET_{categoria}_{bucket}"


def token_desconto_bucket(categoria, bucket):
    return f"DESCONTO_BUCKET_{categoria}_{bucket}"


def token_recencia(label):
    return f"RECENCIA_{label}"


def token_mes(m):
    return f"MES_{m}"


def construir_vocabulario(df, buckets):
    tokens = list(ESTRUTURAIS)

    for v in sorted(df["categoria_produto"].unique()):
        tokens.append(token_categoria(v))
    for v in sorted(df["marca"].unique()):
        tokens.append(token_marca(v))
    for v in sorted(df["fabricante"].unique()):
        tokens.append(token_fabricante(v))
    for v in sorted(df["forma_pagamento"].unique()):
        tokens.append(token_pagamento(v))
    for v in sorted(df["canal_venda"].unique()):
        tokens.append(token_canal(v))
    for v in sorted(df["cod_produto"].unique()):
        tokens.append(token_produto(v))
    for v in range(1, 11):
        tokens.append(token_quantidade(v))

    categorias = sorted(buckets["valor_total"].keys())
    for categoria in categorias:
        for bucket in range(10):
            tokens.append(token_valor_bucket(categoria, bucket))
    for categoria in categorias:
        for bucket in range(10):
            tokens.append(token_desconto_bucket(categoria, bucket))

    for label in RECENCIA_BUCKETS:
        tokens.append(token_recencia(label))
    for m in range(1, 13):
        tokens.append(token_mes(m))

    assert len(tokens) == len(set(tokens)), "Tokens duplicados no vocabulário"
    return {tok: i for i, tok in enumerate(tokens)}


def validar_cobertura(df, tokenizer, buckets):
    df = df.copy()
    df["mes"] = pd.to_datetime(df["data_compra"]).dt.month

    checagens = {
        "categoria_produto": df["categoria_produto"].map(token_categoria),
        "marca": df["marca"].map(token_marca),
        "fabricante": df["fabricante"].map(token_fabricante),
        "forma_pagamento": df["forma_pagamento"].map(token_pagamento),
        "canal_venda": df["canal_venda"].map(token_canal),
        "cod_produto": df["cod_produto"].map(token_produto),
        "quantidade": df["quantidade"].map(token_quantidade),
        "mes": df["mes"].map(token_mes),
    }

    total_unk = 0
    for campo, serie_tokens in checagens.items():
        ids = serie_tokens.map(tokenizer.encode)
        n_unk = (ids == tokenizer.encode("UNK")).sum()
        total_unk += n_unk
        print(f"  {campo}: {len(serie_tokens)} valores checados, {n_unk} caíram em UNK")

    for campo, buckets_por_categoria in buckets.items():
        token_fn = token_valor_bucket if campo == "valor_total" else token_desconto_bucket
        for categoria, edges in buckets_por_categoria.items():
            for bucket in range(10):
                tok = token_fn(categoria, bucket)
                if tok not in tokenizer.token_to_id:
                    total_unk += 1
                    print(f"  FALTA token de bucket: {tok}")

    for label in RECENCIA_BUCKETS:
        assert token_recencia(label) in tokenizer.token_to_id

    return total_unk


def main():
    df = pd.read_csv(
        SOURCE_CSV,
        encoding="utf-8-sig",
        usecols=["categoria_produto", "marca", "fabricante", "forma_pagamento",
                  "canal_venda", "cod_produto", "quantidade", "data_compra"],
    )
    with open(BUCKETS_JSON, encoding="utf-8") as f:
        buckets = json.load(f)

    vocab = construir_vocabulario(df, buckets)
    tokenizer = TokenizerFechado(vocab)
    tokenizer.save(VOCAB_JSON)

    print(f"Vocabulário gerado com {len(vocab)} tokens -> {VOCAB_JSON}\n")

    # Teste de encode/decode
    exemplo = token_categoria("Máquinas")
    id_exemplo = tokenizer.encode(exemplo)
    assert tokenizer.decode(id_exemplo) == exemplo
    print(f"Teste encode/decode OK: '{exemplo}' <-> id {id_exemplo}\n")

    print("--- Validação de cobertura (0 esperado em cada campo) ---")
    total_unk = validar_cobertura(df, tokenizer, buckets)
    print(f"\nTotal de valores mapeados para UNK indevidamente: {total_unk}")
    assert total_unk == 0, "Há valores conhecidos caindo em UNK — vocabulário incompleto"
    print("OK: 100% dos valores categóricos do dataset mapeiam para um token válido.")


if __name__ == "__main__":
    main()
