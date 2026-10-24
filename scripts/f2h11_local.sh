#!/usr/bin/env bash
# F2-H11, braço LightGBM (CPU, local): mesmas N linhas de contexto do TabPFN/TabICL.
cd "$(dirname "$0")/.."
while pgrep -f "tarefa14_gbdt.py configs/hm_clientes.yaml --features trad" > /dev/null; do sleep 60; done
for H in "" "--hibrido"; do for N in 10000 50000 300000; do
  scripts/rodar_limitado.sh -m 6G python3 -u pipeline/common/f2h11_tabular.py configs/hm_clientes.yaml --modelo lgbm --n $N $H
done; done
echo FIM
