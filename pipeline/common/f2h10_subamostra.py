"""F2-H10 — recorta um diretório de embeddings para os clientes da amostra.

Os braços com LLM só têm embeddings dos clientes da amostra (f2h10/textos.parquet).
Para que E0 e E2 treinem e escolham a época exatamente nas mesmas âncoras, os
embeddings deles (extraídos para todas as âncoras) são recortados aqui.

Uso: python3 pipeline/common/f2h10_subamostra.py configs/hm_clientes.yaml --de final_s0/fusao --para f2h10_e0
"""

import argparse
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, "pipeline/common")
from schema import carregar_config, run_dir  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--de", required=True)
    ap.add_argument("--para", required=True)
    a = ap.parse_args()
    config = carregar_config(a.config)
    amostra = pl.read_parquet(os.path.join(run_dir(config, "f2h10"), "textos.parquet"), columns=["entidade_id"]).unique()
    src = run_dir(config, a.de)
    idx = pl.read_parquet(os.path.join(src, "embeddings_eventos_index.parquet"))
    sel = idx.join(amostra, on="entidade_id", how="semi").sort("row_idx")
    emb = np.load(os.path.join(src, "embeddings_eventos.npy"), mmap_mode="r")
    dst = run_dir(config, a.para)
    os.makedirs(dst, exist_ok=True)
    np.save(os.path.join(dst, "embeddings_eventos.npy"), np.asarray(emb[sel["row_idx"].to_numpy()]))
    sel.with_columns(pl.Series("row_idx", np.arange(sel.height))).write_parquet(os.path.join(dst, "embeddings_eventos_index.parquet"))
    print(f"{sel.height:,} de {idx.height:,} âncoras -> {dst}")


if __name__ == "__main__":
    main()
