"""Motor genérico — Serialização das sequências por entidade.

Versão schema-agnóstica de `pipeline/serializacao/tarefa3_serializacao.py`
(branch `luiz_g`). Ordem fixa dentro de cada bloco de evento: EVT, os
campos categóricos do config (na ordem declarada), os campos numéricos do
config (bucket, na ordem declarada), [recência, se não for o 1º evento da
entidade e usa_recencia], [mês, se usa_mes].

Correção estrutural da generalização (ver GENERALIZACAO.md): como o número
de campos por bloco é configurável por empresa, a Tarefa 13 não reconstrói a
posição de âncora de um evento por fórmula fixa — este script grava, por
evento, o índice do seu último token dentro de `seq_full`
(`posicoes_evento.parquet`).

Escala (experimento H&M, 31,8M eventos): a versão anterior montava cada
bloco com `iterrows` e gravava todas as sequências num `sequencias.json` —
inviável nessa escala (horas de CPU e dezenas de GB de RAM). Esta versão é
vetorizada (uma matriz evento × posição-no-bloco de ids de token) e grava
um único array de tokens + uma tabela de offsets por entidade. A semântica
é idêntica — ver `pipeline/common/teste_regressao.py`, que compara contra
os artefatos legados token a token. Leitura: `sequencias_io.py`.

Uso: python3 pipeline/common/tarefa3_serializacao.py <config.yaml>

Gera runs/<nome>/serializacao/{tokens.npy, entidades.parquet, posicoes_evento.parquet}.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "pipeline/common")
from schema import (caminho_tabela, carregar_config, gravar_tabela, juntar_por_evento, ler_fonte,  # noqa: E402
                    ler_tabela, mem, reordenar, run_dir)

RECENCIA_EDGES = [7, 15, 30, 60, 90, 180, 365]
RECENCIA_LABELS = ["0-7", "8-15", "16-30", "31-60", "61-90", "91-180", "181-365", ">365"]
SPLIT_ORDEM = {"train": 0, "val": 1, "test": 2}
TAMANHO_PEDACO = 4_000_000  # eventos por pedaço da matriz de tokens


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


def ids_por_valor(serie, token_de_valor, vocab, valor_nulo_id):
    """Mapeia cada linha de `serie` pro id do token `token_de_valor(valor)`,
    calculando o token uma vez por valor distinto (vetorizado). Valor nulo ->
    `valor_nulo_id` (mesmo destino que o legado dava a f"..._nan")."""
    codigos, uniques = pd.factorize(serie)
    ids_uniques = np.array([vocab.get(token_de_valor(v), vocab["UNK"]) for v in uniques], dtype=np.int32)
    return np.where(codigos >= 0, ids_uniques[np.maximum(codigos, 0)], valor_nulo_id).astype(np.int32)


def matriz_tokens(config, df, vocab, top_n_mapa):
    """Matriz (n_eventos, n_posicoes_no_bloco) de ids de token; -1 marca
    posição ausente (recência do 1º evento da entidade)."""
    colunas = [np.full(len(df), vocab["EVT"], dtype=np.int32)]

    for campo_cfg in config["campos_categoricos"]:
        campo = campo_cfg["nome"]
        if campo_cfg["estrategia"] == "top_n_outros":
            info = top_n_mapa[campo]
            outros = vocab.get(info["outros_token"], vocab["UNK"])
            mapa = info["mapa"]
            colunas.append(ids_por_valor(
                df[campo], lambda v, m=mapa, o=info["outros_token"]: m.get(str(v), o), vocab, outros))
        else:
            unk_nulo = vocab.get(f"{campo.upper()}_nan", vocab["UNK"])
            colunas.append(ids_por_valor(df[campo], lambda v, c=campo: f"{c.upper()}_{v}", vocab, unk_nulo))

    for campo_cfg in config["campos_numericos"]:
        campo = campo_cfg["nome"]
        agrupar_por = campo_cfg.get("agrupar_por")
        bucket = df[f"{campo}_bucket"]
        if agrupar_por:
            # combina (grupo, bucket) num código inteiro em vez de tuplas Python
            cod_g, uniq_g = pd.factorize(df[agrupar_por])
            cod_b, uniq_b = pd.factorize(bucket)
            chave = np.where((cod_g >= 0) & (cod_b >= 0), cod_g.astype(np.int64) * len(uniq_b) + cod_b, -1)
            ids_chave = {}
            for k in np.unique(chave[chave >= 0]):
                g, b = uniq_g[k // len(uniq_b)], uniq_b[k % len(uniq_b)]
                ids_chave[k] = vocab.get(f"{campo.upper()}_BUCKET_{g}_{b}", vocab["UNK"])
            lookup = np.full(len(uniq_g) * max(len(uniq_b), 1), vocab["UNK"], dtype=np.int32)
            for k, v in ids_chave.items():
                lookup[k] = v
            colunas.append(np.where(chave >= 0, lookup[np.maximum(chave, 0)], vocab["UNK"]).astype(np.int32))
        else:
            colunas.append(ids_por_valor(bucket, lambda b, c=campo: f"{c.upper()}_BUCKET_{b}", vocab, vocab["UNK"]))

    if config.get("usa_recencia", True):
        gap = df["_gap_dias"].to_numpy()
        primeiro = np.isnan(gap)
        idx_label = np.searchsorted(RECENCIA_EDGES, np.where(primeiro, 0, gap), side="left")
        ids_label = np.array([vocab.get(f"RECENCIA_{l}", vocab["UNK"]) for l in RECENCIA_LABELS], dtype=np.int32)
        colunas.append(np.where(primeiro, -1, ids_label[idx_label]).astype(np.int32))
    if config.get("usa_mes", True):
        colunas.append(ids_por_valor(df["mes"], lambda m: f"MES_{m}", vocab, vocab["UNK"]))

    return np.stack(colunas, axis=1)


def montar_sequencias(config, df, vocab, top_n_mapa):
    bos, eos = vocab["BOS"], vocab["EOS"]

    # ordem das entidades = 1ª aparição (mesma do legado groupby(sort=False));
    # dentro da entidade, ordem temporal com desempate por evento_id
    df["_ent"], uniques_ent = pd.factorize(df["entidade_id"])
    df = reordenar(df, ["_ent", "data_evento", "evento_id"])
    mem("reordenado")

    anterior = df.groupby("_ent", sort=False)["data_evento"].shift(1)
    df["_gap_dias"] = (df["data_evento"] - anterior).dt.days.astype(np.float32)
    del anterior
    df.drop(columns=["data_evento"], inplace=True)  # não é mais usada (mês já extraído)
    mem("gaps")

    # tamanho de cada bloco sem materializar a matriz inteira:
    # EVT + categóricos + numéricos + [recência, exceto no 1º evento] + [mês]
    n_fixo = 1 + len(config["campos_categoricos"]) + len(config["campos_numericos"]) + int(config.get("usa_mes", True))
    tem_recencia = config.get("usa_recencia", True)
    L = np.full(len(df), n_fixo + int(tem_recencia), dtype=np.int64)
    if tem_recencia:
        L[np.isnan(df["_gap_dias"].to_numpy())] -= 1

    ent = df["_ent"].to_numpy()
    n_ent = len(uniques_ent)
    blocos_por_ent = np.bincount(ent, weights=L, minlength=n_ent).astype(np.int64)
    len_full = blocos_por_ent + 2
    offset = np.concatenate([[0], np.cumsum(len_full)[:-1]])

    total = int(len_full.sum())
    vmax = max(vocab.values())
    dtype = np.int16 if vmax < np.iinfo(np.int16).max else np.int32
    tokens = np.empty(total, dtype=dtype)
    tokens[offset] = bos
    tokens[offset + len_full - 1] = eos
    mem("tokens alocados")

    # matriz de tokens em pedaços de eventos (memória limitada); o token k
    # (contando só tokens de bloco, em ordem global) vai pra posição
    # k + 2*ent + 1 do array final (cada entidade anterior somou BOS+EOS).
    cum_L = np.cumsum(L)
    for a in range(0, len(df), TAMANHO_PEDACO):
        b = min(a + TAMANHO_PEDACO, len(df))
        M = matriz_tokens(config, df.iloc[a:b], vocab, top_n_mapa)
        presente = M >= 0
        assert (presente.sum(axis=1) == L[a:b]).all()
        k0 = int(cum_L[a - 1]) if a > 0 else 0
        k = k0 + np.arange(int(L[a:b].sum()), dtype=np.int64)
        tokens[k + 2 * np.repeat(ent[a:b], L[a:b]) + 1] = M[presente]
        del M, presente
        mem("pedaço")

    # pos_full = índice do último token do bloco dentro de seq_full
    inicio_ent = np.concatenate([[0], np.cumsum(blocos_por_ent)[:-1]])
    pos_full = cum_L - inicio_ent[ent]

    # blocos de treino / treino+val precisam ser prefixo (splits por data)
    split = df["split"].astype("category")
    split_rank = np.array([SPLIT_ORDEM[c] for c in split.cat.categories], dtype=np.int8)[split.cat.codes.to_numpy()]
    rank_max_ate_aqui = pd.Series(split_rank).groupby(ent).cummax().to_numpy()
    assert (split_rank == rank_max_ate_aqui).all(), "split fora de ordem temporal dentro de uma entidade"
    len_train = np.bincount(ent, weights=np.where(split_rank == 0, L, 0), minlength=n_ent).astype(np.int64)
    len_train_val = np.bincount(ent, weights=np.where(split_rank <= 1, L, 0), minlength=n_ent).astype(np.int64)

    entidades = pd.DataFrame({
        "entidade_id": uniques_ent,
        "offset": offset,
        "len_full": len_full,
        "n_eventos": np.bincount(ent, minlength=n_ent),
        "len_blocos_train": len_train,
        "len_blocos_train_val": len_train_val,
    })
    posicoes = pd.DataFrame({"evento_id": df["evento_id"].to_numpy(),
                             "entidade_id": df["entidade_id"].to_numpy(), "pos_full": pos_full})
    return tokens, entidades, posicoes


def main(config_path):
    config = carregar_config(config_path)

    colunas = ["evento_id", "entidade_id", "data_evento"] + \
        [c["nome"] for c in config["campos_categoricos"]] + \
        [c.get("agrupar_por") for c in config["campos_numericos"] if c.get("agrupar_por")]
    colunas = [c for c in dict.fromkeys(colunas) if c]  # dedup preservando ordem

    fonte = ler_fonte(config, colunas)
    fonte["data_evento"] = pd.to_datetime(fonte["data_evento"])
    fonte["mes"] = fonte["data_evento"].dt.month

    splits = ler_tabela(caminho_tabela(config, "splits", "splits"), ["evento_id", "split"])
    df = juntar_por_evento(fonte, splits, ["split"])
    del splits

    if config["campos_numericos"]:
        cols_bucket = ["evento_id"] + [f"{c['nome']}_bucket" for c in config["campos_numericos"]]
        disc = ler_tabela(caminho_tabela(config, "discretizacao", "discretizado"), cols_bucket)
        df = juntar_por_evento(df, disc, cols_bucket[1:])
        del disc

    with open(os.path.join(run_dir(config, "vocabulario"), "vocab.json"), encoding="utf-8") as f:
        vocab = json.load(f)
    top_n_mapa = carregar_top_n_mapa(config)
    mem("entrada pronta")

    tokens, entidades, posicoes = montar_sequencias(config, df, vocab, top_n_mapa)
    mem("montado")

    out_dir = run_dir(config, "serializacao")
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "tokens.npy"), tokens)
    gravar_tabela(entidades, config, "serializacao", "entidades")
    gravar_tabela(posicoes, config, "serializacao", "posicoes_evento")
    legado = os.path.join(out_dir, "sequencias.json")
    if os.path.exists(legado):
        os.remove(legado)  # formato antigo — substituído por tokens.npy + entidades.parquet

    print(f"Gravado {out_dir}/tokens.npy ({len(tokens):,} tokens, {tokens.dtype}) e "
          f"entidades.parquet ({len(entidades):,} {config['entidade_label']}s)")
    print(f"Gravado {out_dir}/posicoes_evento.parquet com {len(posicoes):,} posições de âncora.")

    print("\n--- Distribuição do comprimento de sequência (em tokens, seq_full) ---")
    print(entidades["len_full"].describe(percentiles=[0.5, 0.9, 0.99]))

    n_sem_train = int((entidades["len_blocos_train"] == 0).sum())
    print(f"\nEntidades sem nenhum evento de treino (seq_train=null): {n_sem_train}")


if __name__ == "__main__":
    main(sys.argv[1])
