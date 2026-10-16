"""Compacta os Parquets brutos da H&M (saída de baixar_hm.py) em streaming.

Os brutos guardam `customer_id` como texto hexadecimal de 64 caracteres e a
data como texto — carregar as 31,8M linhas assim estoura a RAM de uma
máquina de 15 GB (derrubou o VS Code na 1ª tentativa de EDA). Aqui cada
lote de 2M linhas é convertido e gravado direto, sem materializar o todo:

  transactions_compacto.parquet: t_dat (date32), cliente (int32),
      article_id (int32), price (float32), sales_channel_id (int8)
  customers_compacto.parquet:    cliente (int32), FN, Active (float32),
      club_member_status, fashion_news_frequency (string), age (float32),
      postal_code (int32 — código denso do CEP hasheado)
  mapa_clientes.parquet:         cliente (int32) <-> customer_id (texto original)

Uso: .venv/bin/python adapters/compactar_hm.py   (via scripts/rodar_limitado.sh)
"""

import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

DADOS = "dados_hm"
LOTE = 2_000_000


def main():
    t0 = time.time()
    cus = pq.read_table(f"{DADOS}/customers.parquet")
    ids = cus["customer_id"].combine_chunks()
    assert pc.count_distinct(ids).as_py() == len(ids), "customer_id duplicado no cadastro"
    codigo = pa.array(np.arange(len(ids), dtype=np.int32))
    pq.write_table(pa.table({"cliente": codigo, "customer_id": ids}), f"{DADOS}/mapa_clientes.parquet")

    cep = cus["postal_code"].combine_chunks().dictionary_encode()
    pq.write_table(pa.table({
        "cliente": codigo,
        "FN": pc.cast(cus["FN"], pa.float32()),
        "Active": pc.cast(cus["Active"], pa.float32()),
        "club_member_status": cus["club_member_status"],
        "fashion_news_frequency": cus["fashion_news_frequency"],
        "age": pc.cast(cus["age"], pa.float32()),
        "postal_code": pc.cast(cep.indices, pa.int32()),
    }), f"{DADOS}/customers_compacto.parquet")
    print(f"customers: {len(ids):,} clientes ({time.time()-t0:.0f}s)")
    del cus

    pf = pq.ParquetFile(f"{DADOS}/transactions.parquet")
    schema = pa.schema([("t_dat", pa.date32()), ("cliente", pa.int32()), ("article_id", pa.int32()),
                        ("price", pa.float32()), ("sales_channel_id", pa.int8())])
    n = 0
    with pq.ParquetWriter(f"{DADOS}/transactions_compacto.parquet", schema, compression="zstd") as w:
        for lote in pf.iter_batches(batch_size=LOTE):
            cli = pc.index_in(lote.column("customer_id"), value_set=ids)
            assert cli.null_count == 0, "transação com cliente fora do cadastro"
            data = pc.cast(pc.strptime(lote.column("t_dat"), format="%Y-%m-%d", unit="s"), pa.date32())
            w.write_table(pa.table({
                "t_dat": data,
                "cliente": pc.cast(cli, pa.int32()),
                "article_id": pc.cast(lote.column("article_id"), pa.int32()),
                "price": pc.cast(lote.column("price"), pa.float32()),
                "sales_channel_id": pc.cast(lote.column("sales_channel_id"), pa.int8()),
            }, schema=schema))
            n += lote.num_rows
            print(f"  transações: {n:,}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"Concluído em {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
