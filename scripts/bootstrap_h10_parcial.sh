#!/usr/bin/env bash
# Bootstrap no teste só para os modelos já concluídos (fusões e XGBoost) —
# seguro contra a perda do runtime antes do fim dos LightGBM da H10.
set -uo pipefail
C=configs/hm_clientes.yaml
A="python3 pipeline/common/avaliacao.py $C --split test --B ${B:-300} --saida runs/hm_clientes/h10_bootstrap_parcial.json"
rm -f runs/hm_clientes/h10_bootstrap_parcial.json
for t in churn proxima_online next_category ltv_90d; do
  $A --tarefa $t --a gbdt/h10_trad_xgb --b gbdt/h10_hibrido_xgb --b "h10_fusao_d_completo_s*" --b "h10_final_fusao_s*" --b "h10_fusao_e_cad_caus_s*"
  $A --tarefa $t --a "h10_fusao_e_cad_caus_s*" --b "h10_fusao_d_completo_s*"
  $A --tarefa $t --a "h10_fusao_c_emb_caus_s*" --b "h10_fusao_d_completo_s*"
  $A --tarefa $t --a "h10_fusao_d_completo_s*" --b "h10_final_fusao_s*"
done
echo "bootstrap parcial H10 concluído"
