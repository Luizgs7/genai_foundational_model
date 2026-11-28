"""Grava o PCA-32 do vetor do v2 (caches/pca32_v2.npz) sem rodar a HPO do GBDT híbrido.

Usa a mesma função de carga do tarefa14_gbdt.py (mesmas âncoras, mesma ordem, PCA ajustada em
até 500 mil âncoras de treino com semente 0), então o arquivo é idêntico ao que o híbrido gravaria.
A clusterização (f3h2_clusterizacao.py) e o híbrido leem este cache.

Uso: python3 pipeline/common/v2_pca_cache.py configs/<base>.yaml [--emb emb_v2] [--pca_k 32]
"""
import argparse
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, "pipeline/common")
from schema import carregar_config, run_dir  # noqa: E402
from tarefa14_gbdt import carregar  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--emb", default="emb_v2")
    ap.add_argument("--pca_k", type=int, default=32)
    ap.add_argument("--saida", default=None, help="padrão: runs/<nome>/caches/pca<k>_v2.npz")
    a = ap.parse_args()
    config = carregar_config(a.config)
    cache = a.saida or os.path.join(run_dir(config, "caches"), f"pca{a.pca_k}_v2.npz")
    if os.path.exists(cache):
        print(f"já existe: {cache}")
        return
    args = SimpleNamespace(features="emb_pca", emb=a.emb, pca_k=a.pca_k, pca_cache=cache,
                           sem_cadastro=False, sem_prefixo=None)
    carregar(config, args, ["train", "val", "test"])
    assert os.path.exists(cache), cache


if __name__ == "__main__":
    main()
