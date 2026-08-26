"""Motor genérico — Vocabulário e tokenizer fechado (sem BPE).

Versão schema-agnóstica de `pipeline/vocabulario/tarefa5_vocabulario.py`
(branch `luiz_g`). Constrói um token por valor distinto de cada campo
categórico do config, mais um token por bucket numérico (grupo × posição),
mais recência/mês se ativos.

Duas estratégias por campo categórico (config `campos_categoricos[].estrategia`):
  - `fechado`: um token por valor distinto observado em TODO o dataset (não
    só treino — é enumeração de um domínio fechado, não uma estatística
    calibrável, mesmo raciocínio já usado na base sintética).
  - `top_n_outros`: token só pros N valores mais frequentes **no treino**
    (aqui sim calibrado só no treino, porque é uma escolha estatística de
    frequência, não enumeração de domínio) + um token `OUTROS` pro resto.
    Necessário pra campos de alta cardinalidade (ex. produto na Olist,
    ~33 mil valores — inviável como vocabulário fechado).

Uso: python3 pipeline/common/tarefa5_vocabulario.py <config.yaml>

Gera runs/<nome>/vocabulario/vocab.json e runs/<nome>/vocabulario/top_n_mapa.json
(mapa valor->token pros campos top_n_outros, usado na serialização).
"""

import json
import os
import sys

import pandas as pd

sys.path.insert(0, "pipeline/common")
from schema import carregar_config, run_dir  # noqa: E402

RECENCIA_LABELS = ["0-7", "8-15", "16-30", "31-60", "61-90", "91-180", "181-365", ">365"]
ESTRUTURAIS = ["BOS", "EOS", "EVT", "PAD", "UNK"]


def token_categorico(campo, valor):
    return f"{campo.upper()}_{valor}"


def token_bucket(campo, grupo, bucket):
    if grupo is None:
        return f"{campo.upper()}_BUCKET_{bucket}"
    return f"{campo.upper()}_BUCKET_{grupo}_{bucket}"


def token_recencia(label):
    return f"RECENCIA_{label}"


def token_mes(m):
    return f"MES_{m}"


def construir_vocabulario(config, df_eventos, splits, buckets):
    tokens = list(ESTRUTURAIS)
    top_n_mapa = {}  # {campo: {valor_original: token}}

    train_ids = set(splits.loc[splits["split"] == "train", "evento_id"])
    df_train = df_eventos[df_eventos["evento_id"].isin(train_ids)]

    for campo_cfg in config["campos_categoricos"]:
        campo = campo_cfg["nome"]
        if campo_cfg["estrategia"] == "fechado":
            for v in sorted(df_eventos[campo].dropna().unique(), key=str):
                tokens.append(token_categorico(campo, v))
        elif campo_cfg["estrategia"] == "top_n_outros":
            top_n = campo_cfg["top_n"]
            contagem = df_train[campo].value_counts()
            top_valores = contagem.index[:top_n].tolist()
            mapa = {}
            for v in sorted(top_valores, key=str):
                tok = token_categorico(campo, v)
                tokens.append(tok)
                mapa[v] = tok
            outros_tok = f"{campo.upper()}_OUTROS"
            tokens.append(outros_tok)
            top_n_mapa[campo] = {"mapa": mapa, "outros_token": outros_tok}
        else:
            raise ValueError(f"Estratégia desconhecida: {campo_cfg['estrategia']}")

    for campo_cfg in config["campos_numericos"]:
        campo = campo_cfg["nome"]
        info = buckets[campo]
        n_buckets = info["n_buckets"]
        for grupo in sorted(info["edges"].keys(), key=str):
            grupo_tok = None if grupo == "__GLOBAL__" else grupo
            for bucket in range(n_buckets):
                tokens.append(token_bucket(campo, grupo_tok, bucket))

    if config.get("usa_recencia", True):
        for label in RECENCIA_LABELS:
            tokens.append(token_recencia(label))
    if config.get("usa_mes", True):
        for m in range(1, 13):
            tokens.append(token_mes(m))

    assert len(tokens) == len(set(tokens)), "Tokens duplicados no vocabulário"
    vocab = {tok: i for i, tok in enumerate(tokens)}
    return vocab, top_n_mapa


def main(config_path):
    config = carregar_config(config_path)

    colunas = ["evento_id"] + [c["nome"] for c in config["campos_categoricos"]]
    df_eventos = pd.read_csv(config["fonte_canonica"], usecols=colunas)

    splits_csv = os.path.join(run_dir(config, "splits"), "splits.csv")
    splits = pd.read_csv(splits_csv, usecols=["evento_id", "split"])

    buckets_json = os.path.join(run_dir(config, "discretizacao"), "buckets.json")
    buckets = {}
    if os.path.exists(buckets_json):
        with open(buckets_json, encoding="utf-8") as f:
            buckets = json.load(f)

    vocab, top_n_mapa = construir_vocabulario(config, df_eventos, splits, buckets)

    out_dir = run_dir(config, "vocabulario")
    os.makedirs(out_dir, exist_ok=True)
    vocab_json = os.path.join(out_dir, "vocab.json")
    with open(vocab_json, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    top_n_json = os.path.join(out_dir, "top_n_mapa.json")
    with open(top_n_json, "w", encoding="utf-8") as f:
        json.dump({k: {"mapa": {str(kk): vv for kk, vv in v["mapa"].items()}, "outros_token": v["outros_token"]}
                   for k, v in top_n_mapa.items()}, f, ensure_ascii=False, indent=2)

    print(f"Vocabulário gerado com {len(vocab)} tokens -> {vocab_json}")
    if top_n_mapa:
        for campo, info in top_n_mapa.items():
            print(f"  top_n_outros/{campo}: {len(info['mapa'])} valores mapeados + 1 token OUTROS")

    print("\n--- Cobertura (campos fechado — 0 esperado fora do vocabulário) ---")
    for campo_cfg in config["campos_categoricos"]:
        campo = campo_cfg["nome"]
        if campo_cfg["estrategia"] != "fechado":
            continue
        tokens_gerados = df_eventos[campo].dropna().map(lambda v, c=campo: token_categorico(c, v))
        faltando = (~tokens_gerados.isin(vocab)).sum()
        print(f"  {campo}: {len(tokens_gerados)} valores, {faltando} fora do vocabulário")
        assert faltando == 0, f"Campo fechado {campo!r} com valores fora do vocabulário"

    print("\n--- Cobertura (campos top_n_outros — taxa em OUTROS é esperada, não erro) ---")
    for campo, info in top_n_mapa.items():
        mapa = info["mapa"]
        serie = df_eventos[campo].dropna()
        taxa_outros = (~serie.isin(mapa.keys())).mean()
        print(f"  {campo}: {taxa_outros:.1%} dos valores caem em OUTROS")

    print("\nOK.")


if __name__ == "__main__":
    main(sys.argv[1])
