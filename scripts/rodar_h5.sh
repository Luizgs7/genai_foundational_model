#!/usr/bin/env bash
# H5 — baseline tradicional (features de engenharia) com LightGBM e XGBoost.
# Binárias com 40 tentativas de HPO; multiclasse (mais caras: 1 árvore por
# classe por iteração) com 25. Cada chamada acrescenta a tarefa ao mesmo
# relatorio.json do conjunto features+modelo.
set -euo pipefail
C=configs/hm_clientes.yaml
PY=".venv/bin/python pipeline/common/tarefa14_gbdt.py $C --features trad"
for M in ${MODELOS:-lgbm xgb}; do
  scripts/rodar_limitado.sh -m 7G $PY --modelo $M --tarefas churn,proxima_online --n_trials 40
  scripts/rodar_limitado.sh -m 7G $PY --modelo $M --tarefas next_category,ltv_90d --n_trials 25
done
