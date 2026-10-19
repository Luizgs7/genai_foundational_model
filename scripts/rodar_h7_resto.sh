#!/usr/bin/env bash
# H6/H7 — restante após o corte de 2026-10-19: o GBDT sobre o embedding bruto
# (384 dims) é ~5x mais caro que o PCA-32 e já perdeu dele em churn/online na
# H6, então o multiclasse do bruto e o híbrido bruto foram cortados.
#   CPU: híbrido LightGBM (H7) e híbrido sem cadastro (H9)
#   GPU: híbrido XGBoost (robustez ao algoritmo) e XGBoost LTV da H5
set -uo pipefail
C=configs/hm_clientes.yaml
R=runs/hm_clientes/gbdt
P="python3 -u pipeline/common/tarefa14_gbdt.py $C --emb fusao --pca_k 32"
tem4() { [ -f $R/$1/relatorio.json ] && python3 -c "import json,sys;sys.exit(0 if len(json.load(open('$R/$1/relatorio.json'))['tarefas'])==4 else 1)"; }
roda() {  # $1=features $2=modelo $3=saida $4=threads [extras]
  local f=$1 m=$2 s=$3 t=$4; shift 4
  tem4 $s && return
  $P --features $f --modelo $m --saida $s --threads $t --amostra_hpo 300000 --tarefas churn,proxima_online --n_trials 20 "$@"
  $P --features $f --modelo $m --saida $s --threads $t --amostra_hpo 300000 --tarefas next_category,ltv_90d --n_trials 12 "$@"
}
(
  export XGB_DEVICE=cuda
  roda hibrido xgb hibrido_xgb 4
  python3 -c "import json,sys;sys.exit(0 if 'ltv_90d' in json.load(open('$R/trad_xgb/relatorio.json'))['tarefas'] else 1)" \
    || python3 -u pipeline/common/tarefa14_gbdt.py $C --features trad --modelo xgb --tarefas ltv_90d --n_trials 25 --threads 4
  echo "[$(date)] GPU (xgb) concluído"
) >> /content/h7_xgb.log 2>&1 &
roda hibrido lgbm hibrido_lgbm 8
roda hibrido lgbm hibrido_lgbm_sem_cadastro 8 --sem_cadastro
echo "[$(date)] CPU (lgbm) concluído"
wait
