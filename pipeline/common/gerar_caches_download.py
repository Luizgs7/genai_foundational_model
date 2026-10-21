"""Gera, no Colab, arquivos compactos para rodar localmente o que falta da H10
(o túnel baixa ~140 KB/s; os embeddings brutos têm 4 GB):
  caches/pca32_<nome>.npz          projeção PCA-32 (float32) exatamente como o
                                   tarefa14_gbdt.py a calcula com --avaliar_teste
  caches/<fusao>/predicoes_test.parquet   predições de teste da fusão em float32
Uso: python3 pipeline/common/gerar_caches_download.py configs/hm_clientes.yaml
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "pipeline/common")
from schema import carregar_config, run_dir  # noqa: E402
from tarefa14_gbdt import carregar  # noqa: E402


def main(config_path):
    config = carregar_config(config_path)
    R = run_dir(config, "")
    out = os.path.join(R, "caches")
    os.makedirs(out, exist_ok=True)
    pca = lambda emb, nome: carregar(config, argparse.Namespace(  # noqa: E731
        features="hibrido", emb=emb, pca_k=32, sem_cadastro=False, pca_cache=os.path.join(out, f"pca32_{nome}.npz")),
        ["train", "val", "test"])
    def marcar(caminho):   # o puxador local só baixa arquivos com marca de pronto
        open(caminho + ".pronto", "w").close()

    pca("fusao", "h3")
    marcar(os.path.join(out, "pca32_h3.npz"))
    for d in sorted(glob.glob(os.path.join(R, "h10_fusao_d_completo_s*"))):
        df = pd.read_parquet(os.path.join(d, "predicoes_test.parquet"))
        for c in df.columns:
            if df[c].dtype == np.float64:
                df[c] = df[c].astype(np.float32)
        os.makedirs(os.path.join(out, os.path.basename(d)), exist_ok=True)
        df.to_parquet(os.path.join(out, os.path.basename(d), "predicoes_test.parquet"), index=False)
        marcar(os.path.join(out, os.path.basename(d), "predicoes_test.parquet"))
        print("compactado", d)
    pca("final_s0/embeddings", "final_s0")
    marcar(os.path.join(out, "pca32_final_s0.npz"))
    print("caches prontos")


if __name__ == "__main__":
    main(sys.argv[1])
