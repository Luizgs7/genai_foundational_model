"""F3-H1 — campo derivado `desconto` (0/1 por item) para a tarefa proxima_com_desconto.

A H&M não traz um campo de promoção, mas o preço do mesmo artigo varia no tempo.
Definição (registrada em PLANO_FASE3.md antes de rodar):

  - preço de referência do artigo = o maior valor da MEDIANA DIÁRIA de preço
    observado até a véspera. É causal: só usa vendas passadas (de qualquer cliente);
  - um item tem desconto quando preço < (1 − d) × referência;
  - itens sem referência (primeiro dia de venda do artigo) não têm desconto.

Calibração, só com eventos do TREINO e antes de treinar qualquer modelo: para
d em {5%, 10%, 20%} e agregação por cesta em {algum item, maioria dos itens},
mede-se a taxa de cestas com desconto. Regra de escolha, fixada antes de olhar
os números:
  1) d = 10% com "algum item", se a taxa ficar entre 20% e 80%;
  2) senão, a combinação com taxa mais próxima de 50%.

Gera, em runs/<nome>/f3h1/:
  - desconto.parquet (evento_id, desconto para o d escolhido);
  - relatorio.json (taxas por combinação e por split, escolha);
e grava uma cópia da fonte canônica com a coluna `desconto`
(runs/<nome>/eventos_canonicos_f3h1.parquet), que o config da F3-H1 usa.

Uso: python3 pipeline/common/f3h1_desconto.py configs/hm_clientes.yaml
"""

import json
import os
import sys
import time

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, "pipeline/common")
from schema import caminho_tabela, carregar_config, run_dir  # noqa: E402

DS = [0.05, 0.10, 0.20]


def main():
    t0 = time.time()
    config = carregar_config(sys.argv[1])
    ft = config["features_tradicionais"]
    item, valor = ft["campo_item"], ft["campo_valor"]
    fonte = config["fonte_canonica"]
    ev = pl.read_parquet(fonte, columns=["evento_id", "entidade_id", "data_evento", item, valor])
    ev = ev.with_columns(pl.col("data_evento").dt.date().alias("dia")).drop("data_evento")
    # mediana diária por artigo e maior mediana até a véspera
    diario = (ev.group_by([item, "dia"]).agg(pl.col(valor).median().alias("mediana"))
              .sort([item, "dia"])
              .with_columns(pl.col("mediana").cum_max().shift(1).over(item).alias("referencia")))
    ev = ev.join(diario.select([item, "dia", "referencia"]), on=[item, "dia"], how="left")
    razao = (pl.col(valor) / pl.col("referencia"))
    ev = ev.with_columns([((razao < 1 - d).fill_null(False)).cast(pl.Int8).alias(f"d{int(d * 100)}") for d in DS])
    print(f"referência calculada: {diario.height:,} pares artigo-dia; itens sem referência: "
          f"{ev['referencia'].null_count() / ev.height:.2%} ({time.time() - t0:.0f}s)", flush=True)

    splits = pl.read_parquet(caminho_tabela(config, "splits", "splits"), columns=["evento_id", "split"])
    ev = ev.join(splits.with_columns(pl.col("split").cast(pl.Utf8)), on="evento_id", how="left")
    cestas = ev.group_by(["entidade_id", "dia"]).agg(
        [pl.col("split").first()] + [pl.col(f"d{int(d * 100)}").max().alias(f"algum_{int(d * 100)}") for d in DS]
        + [(pl.col(f"d{int(d * 100)}").mean() > 0.5).alias(f"maioria_{int(d * 100)}") for d in DS])
    taxas = {}
    for sp in ("train", "val", "test"):
        c = cestas.filter(pl.col("split") == sp)
        taxas[sp] = {f"{ag}_{int(d * 100)}": float(c[f"{ag}_{int(d * 100)}"].cast(pl.Float64).mean())
                     for ag in ("algum", "maioria") for d in DS}
    tr = taxas["train"]
    if 0.20 <= tr["algum_10"] <= 0.80:
        escolha, regra = "algum_10", "regra 1: d = 10% com algum item, taxa do treino dentro de 20–80%"
    else:
        escolha = min(tr, key=lambda k: abs(tr[k] - 0.5))
        regra = "regra 2: combinação com taxa do treino mais próxima de 50%"
    ag, d = escolha.split("_")
    print("taxa de cestas com desconto no treino:", {k: round(v, 3) for k, v in tr.items()})
    print(f"escolha: {escolha} ({regra})", flush=True)

    out = run_dir(config, "f3h1")
    os.makedirs(out, exist_ok=True)
    col = ev.select(["evento_id", pl.col(f"d{d}").cast(pl.Float32).alias("desconto")]).sort("evento_id")
    col.write_parquet(os.path.join(out, "desconto.parquet"))
    rel = {"definicao": "preço < (1 − d) × maior mediana diária do artigo até a véspera",
           "taxa_itens_com_desconto": {f"d{int(x * 100)}": float(ev[f"d{int(x * 100)}"].mean()) for x in DS},
           "itens_sem_referencia": float(ev["referencia"].null_count() / ev.height),
           "taxa_cestas_com_desconto": taxas, "escolha": escolha, "regra": regra,
           "agregacao_cesta": "max" if ag == "algum" else "media (limiar 0,5)", "d": int(d) / 100}
    json.dump(rel, open(os.path.join(out, "relatorio.json"), "w"), indent=1, ensure_ascii=False)
    del ev, cestas, diario

    # cópia da fonte canônica com a coluna nova (em blocos, para caber na memória)
    destino = os.path.join(run_dir(config, ""), "eventos_canonicos_f3h1.parquet")
    pf = pq.ParquetFile(fonte)
    ids = col["evento_id"].to_numpy()          # ordenado por evento_id
    valores = col["desconto"].to_numpy()
    w = None
    for g in range(pf.num_row_groups):
        t = pf.read_row_group(g)
        e = t["evento_id"].to_numpy()
        pos = np.searchsorted(ids, e)
        assert (ids[pos] == e).all()
        t = t.append_column("desconto", pa.array(valores[pos], type=pa.float32()))
        if w is None:
            w = pq.ParquetWriter(destino, t.schema, compression="zstd")
        w.write_table(t)
    w.close()
    print(f"gravado {destino} ({os.path.getsize(destino) / 1e6:.0f} MB) e {out}/relatorio.json ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
