"""Adapter H&M Personalized Fashion Recommendations -> schema canônico do
motor genérico. Entidade = CLIENTE; evento = ITEM comprado (uma linha de
transactions = um artigo; mesma semântica "uma linha = um produto" da base
sintética e da Olist). A âncora downstream é a CESTA (cliente + dia), ver
`ancora: cesta` em configs/hm_clientes.yaml.

Lê os Parquets compactos de `adapters/compactar_hm.py` e junta os atributos
do artigo por índice (sem join de texto em 31,8M linhas — memória). Colunas
de texto ficam como `category` do pandas.

Saída: runs/hm_clientes/eventos_canonicos.parquet com
  evento_id (ordem original do arquivo), entidade_id (int32), data_evento,
  campos tokenizados: indice, grupo_produto, tipo_produto, cor, linha_vestuario,
           canal, artigo, preco;
  campos de rótulo: grupo_produto_alvo (grupo com classes <1% fundidas em
           "Outros"), online (0/1);
  campos estáticos do cliente (só fusão/GBDT): clube, newsletter, cep_top
           (top-1000 CEPs + OUTROS), idade, fn, active.

Uso: scripts/rodar_limitado.sh .venv/bin/python adapters/adapter_hm_clientes.py
"""

import os
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

DADOS = "dados_hm"
OUT = "runs/hm_clientes/eventos_canonicos.parquet"
CAMPOS_ARTIGO = {
    "index_name": "indice",
    "product_group_name": "grupo_produto",
    "product_type_name": "tipo_produto",
    "colour_group_name": "cor",
    "garment_group_name": "linha_vestuario",
}
LIMIAR_CLASSE_RARA = 0.01
TOP_CEPS = 1000


def categoria_por_indice(valores, posicoes):
    """Categorical a partir de um vetor pequeno (1 valor por artigo/cliente)
    indexado por `posicoes` — evita materializar strings por linha."""
    cat = pd.Categorical(valores)
    return pd.Categorical.from_codes(cat.codes[posicoes], categories=cat.categories)


def main():
    t0 = time.time()
    tx = pq.read_table(f"{DADOS}/transactions_compacto.parquet").to_pandas()
    n = len(tx)
    print(f"transações: {n:,} ({time.time()-t0:.0f}s)")

    art = pd.read_parquet(f"{DADOS}/articles.parquet", columns=["article_id"] + list(CAMPOS_ARTIGO))
    art = art.sort_values("article_id").reset_index(drop=True)
    pos_art = np.searchsorted(art["article_id"].to_numpy(), tx["article_id"].to_numpy())
    assert (art["article_id"].to_numpy()[pos_art] == tx["article_id"].to_numpy()).all(), "artigo fora do cadastro"

    cus = pd.read_parquet(f"{DADOS}/customers_compacto.parquet").sort_values("cliente").reset_index(drop=True)
    assert (cus["cliente"].to_numpy() == np.arange(len(cus))).all()
    pos_cli = tx["cliente"].to_numpy()

    saida = pd.DataFrame({
        "evento_id": np.arange(n, dtype=np.int64),
        "entidade_id": tx["cliente"].to_numpy(),
        "data_evento": pd.to_datetime(tx["t_dat"]),
    })
    for orig, novo in CAMPOS_ARTIGO.items():
        saida[novo] = categoria_por_indice(art[orig].to_numpy(), pos_art)
    saida["canal"] = pd.Categorical.from_codes((tx["sales_channel_id"].to_numpy() - 1).astype(np.int8),
                                               categories=["loja", "online"])
    saida["artigo"] = tx["article_id"].to_numpy()
    saida["preco"] = tx["price"].to_numpy()

    # rótulos
    share = saida["grupo_produto"].value_counts(normalize=True)
    raras = set(share[share < LIMIAR_CLASSE_RARA].index)
    alvo = saida["grupo_produto"].astype(str).where(~saida["grupo_produto"].isin(raras), "Outros")
    saida["grupo_produto_alvo"] = pd.Categorical(alvo)
    saida["online"] = (tx["sales_channel_id"].to_numpy() == 2).astype(np.float32)
    print(f"grupo_produto_alvo: {saida['grupo_produto_alvo'].nunique()} classes "
          f"({len(raras)} grupos raros fundidos em 'Outros')")

    # estáticos do cliente (por índice)
    cep_cont = cus["postal_code"].value_counts()
    top = set(cep_cont.index[:TOP_CEPS])
    cep_top = np.where(cus["postal_code"].isin(top), cus["postal_code"].astype(str), "OUTROS")
    saida["clube"] = categoria_por_indice(cus["club_member_status"].fillna("NA").to_numpy(), pos_cli)
    saida["newsletter"] = categoria_por_indice(cus["fashion_news_frequency"].fillna("NA").replace("None", "NONE").to_numpy(), pos_cli)
    saida["cep_top"] = categoria_por_indice(cep_top, pos_cli)
    saida["idade"] = cus["age"].to_numpy()[pos_cli]
    saida["fn"] = cus["FN"].fillna(0).to_numpy()[pos_cli]          # nulo = não assinou (convenção do dataset)
    saida["active"] = cus["Active"].fillna(0).to_numpy()[pos_cli]
    del tx

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    saida.to_parquet(OUT, index=False)
    print(f"Gravado {OUT}: {len(saida):,} eventos, {saida['entidade_id'].nunique():,} clientes "
          f"({time.time()-t0:.0f}s)")
    print(saida.dtypes.to_string())


if __name__ == "__main__":
    main()
