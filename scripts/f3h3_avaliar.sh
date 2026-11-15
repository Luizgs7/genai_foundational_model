#!/usr/bin/env bash
# F3-H3: fusão com o embedding do v2 contra a fusão do v1 (H4, backbone H3) e contra o GBDT
# tradicional, no teste, nas 4 tarefas antigas (rótulo de churn corrigido). Bootstrap por cliente.
cd "$(dirname "$0")/.."
C=configs/hm_clientes.yaml
S=runs/hm_clientes/f3h3/bootstrap_test.json
mkdir -p runs/hm_clientes/f3h3; rm -f $S
for t in churn next_category ltv_90d proxima_online; do
  python3 -u pipeline/common/avaliacao.py $C --tarefa $t --split test --B 300 --saida $S \
    --a "caches/h10_fusao_d_completo_s*" --b "v2_fusao_s*"
  [ -f runs/hm_clientes/gbdt/h10_trad_lgbm/predicoes_test_${t}_seed0.parquet ] && \
    python3 -u pipeline/common/avaliacao.py $C --tarefa $t --split test --B 300 --saida $S \
      --a gbdt/h10_trad_lgbm --b "v2_fusao_s*" --b "caches/h10_fusao_d_completo_s*"
done 2>&1 | grep -E "auc|accuracy"
