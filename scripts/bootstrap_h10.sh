#!/usr/bin/env bash
# H10 — deltas no TESTE com IC 95% por bootstrap por cliente (avaliacao.py).
# Cada linha responde a uma pergunta do experimento:
#   H7  híbrido vs tradicional (GBDT)            -> o embedding agrega ao GBDT?
#   H6  só-embedding vs tradicional (GBDT)
#   H4  fusão completa vs GBDT tradicional        -> o modelo fundacional ganha da régua?
#   H9  fusão completa vs sem embedding / sem cadastro
#   H8  backbone final vs referência (fusão)
set -euo pipefail
C=configs/hm_clientes.yaml
A="python3 pipeline/common/avaliacao.py $C --split test --B ${B:-300} --saida runs/hm_clientes/h10_bootstrap.json"
rm -f runs/hm_clientes/h10_bootstrap.json
for t in churn proxima_online next_category ltv_90d; do
  $A --tarefa $t --a gbdt/h10_trad_lgbm --b gbdt/h10_hibrido_lgbm --b gbdt/h10_emb_pca_lgbm --b gbdt/h10_final_hibrido_lgbm \
     --b "h10_fusao_d_completo_s*" --b "h10_final_fusao_s*"
  $A --tarefa $t --a "h10_fusao_e_cad_caus_s*" --b "h10_fusao_d_completo_s*"
  $A --tarefa $t --a "h10_fusao_c_emb_caus_s*" --b "h10_fusao_d_completo_s*"
  $A --tarefa $t --a "h10_fusao_d_completo_s*" --b "h10_final_fusao_s*"
  $A --tarefa $t --a gbdt/h10_hibrido_lgbm_sem_cadastro --b gbdt/h10_hibrido_lgbm
  $A --tarefa $t --a gbdt/h10_trad_xgb --b gbdt/h10_hibrido_xgb
done
echo "bootstrap H10 concluído"
