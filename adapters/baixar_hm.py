"""Download do dataset H&M Personalized Fashion Recommendations.

Fonte: mirror no HuggingFace `einrafh/hnm-fashion-recommendations-data`
(mesmos dados da competição do Kaggle de 2022: articles ~106 mil,
customers ~1,37M, transactions ~31,8M). Licença não informada no mirror —
dados de competição, uso restrito a experimento/pesquisa.

Grava cada subset como Parquet em `dados_hm/` (git-ignorado), que é o que o
adapter (`adapters/adapter_hm_clientes.py`) lê.

Uso: .venv/bin/python adapters/baixar_hm.py
"""

import os
import time

from datasets import load_dataset

REPO = "einrafh/hnm-fashion-recommendations-data"
SUBSETS = ["articles", "customers", "transactions"]
OUT_DIR = "dados_hm"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for subset in SUBSETS:
        destino = os.path.join(OUT_DIR, f"{subset}.parquet")
        if os.path.exists(destino):
            print(f"{destino} já existe, pulando")
            continue
        t0 = time.time()
        ds = load_dataset(REPO, subset, split="train")
        ds.to_parquet(destino)
        print(f"{subset}: {ds.num_rows:,} linhas, colunas={ds.column_names} -> {destino} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
