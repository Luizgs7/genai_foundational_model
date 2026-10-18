"""Backbone final do experimento H&M — fecha a H8 e repete H4/H6/H7 com ele.

A arquitetura vencedora das ablações (escolhida na validação) é treinada com
o orçamento COMPLETO da H3 (3 épocas, paciência 1) em várias sementes:
  1. pré-treino            -> runs/<nome>/final_s<S>/pretreino/
  2. extração (fp16)       -> runs/<nome>/final_s<S>/embeddings/
  3. fusão completa (d)    -> runs/<nome>/final_s<S>/fusao/          (H4)
  4. só na semente 0: GBDT só-embedding (PCA-k) e híbrido, com o mesmo
     orçamento de HPO da H6/H7 -> runs/<nome>/gbdt/final_{emb_pca,hibrido}_lgbm/
A variante (e) da fusão (sem embedding) não depende do backbone e não é refeita.

Etapas concluídas são puladas (retomável após queda do Colab).

Uso: python3 pipeline/common/experimento_backbone_final.py <config_base.yaml> \\
        --arq posicao=rope n_embd=256 n_layer=6 n_head=4 n_kv_heads=4 [--seeds 0 1 2] [--pca_k 32]
"""

import argparse
import copy
import json
import os
import subprocess
import sys

import yaml

sys.path.insert(0, "pipeline/common")
from experimento_ablacao import REFERENCIA  # noqa: E402
from schema import run_dir  # noqa: E402


def valor(v):
    for conv in (int, float):
        try:
            return conv(v)
        except ValueError:
            pass
    return {"true": True, "false": False}.get(v.lower(), v)


def rodar(cmd):
    print(f"\n$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--arq", nargs="+", required=True, help="chave=valor sobre a referência das ablações")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--pca_k", type=int, default=32)
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()

    base = yaml.safe_load(open(args.config, encoding="utf-8"))
    arq = {**REFERENCIA, **{k: valor(v) for k, v in (a.split("=", 1) for a in args.arq)}}
    py = sys.executable
    for s in args.seeds:
        cfg = copy.deepcopy(base)
        cfg["arquitetura"] = arq
        cfg["pretreino"] = {**base.get("pretreino", {}), "seed": s}
        sub = f"final_s{s}"
        os.makedirs(run_dir(cfg, sub), exist_ok=True)
        path = os.path.join(run_dir(cfg, sub), "config.yaml")
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
        print(f"\n######## backbone final semente {s}: {arq}", flush=True)

        if not os.path.exists(os.path.join(run_dir(cfg, sub), "pretreino", "pretreino_relatorio.json")):
            rodar([py, "-u", "pipeline/common/tarefa_pretreino.py", path, "--retomar", "--saida", f"{sub}/pretreino"])
        if not os.path.exists(os.path.join(run_dir(cfg, sub), "embeddings", "embeddings_eventos_index.parquet")):
            rodar([py, "-u", "pipeline/common/tarefa13_extrair_embeddings.py", path, "--pretreino", f"{sub}/pretreino",
                   "--saida", f"{sub}/embeddings", "--dtype", "float16"])
        if not os.path.exists(os.path.join(run_dir(cfg, sub), "fusao", "tarefa13_relatorio.json")):
            rodar([py, "-u", "pipeline/common/tarefa13_treinar_fusao.py", path, "--emb", f"{sub}/embeddings",
                   "--saida", f"{sub}/fusao", "--seed", str(s), "--dispositivo", "auto"])
        if s != args.seeds[0]:
            continue
        for feats, saida in (("emb_pca", "final_emb_pca_lgbm"), ("hibrido", "final_hibrido_lgbm")):
            rel = os.path.join(run_dir(cfg, "gbdt"), saida, "relatorio.json")
            feito = json.load(open(rel))["tarefas"] if os.path.exists(rel) else {}
            g = [py, "-u", "pipeline/common/tarefa14_gbdt.py", path, "--emb", f"{sub}/embeddings", "--features", feats,
                 "--modelo", "lgbm", "--pca_k", str(args.pca_k), "--saida", saida, "--amostra_hpo", "300000",
                 "--threads", str(args.threads)]
            if not {"churn", "proxima_online"} <= set(feito):
                rodar(g + ["--tarefas", "churn,proxima_online", "--n_trials", "20"])
            if not {"next_category", "ltv_90d"} <= set(feito):
                rodar(g + ["--tarefas", "next_category,ltv_90d", "--n_trials", "12"])
    print("backbone final concluído", flush=True)


if __name__ == "__main__":
    main()
