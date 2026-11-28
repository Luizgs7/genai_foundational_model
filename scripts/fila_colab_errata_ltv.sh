#!/usr/bin/env bash
# Errata de 27/11: o baseline de LTV vazava o futuro (EXPERIMENTOS_HM.md). Refaz, no corte atual da H&M,
# os braços que usavam a feature e onde ela pesa (churn e LTV; nas outras tarefas ela tem < 1% do ganho):
#   T  LightGBM com as 97 features corrigidas, HPO de 40 tentativas, 3 sementes (como a H10)
#   H  T + PCA-32 do v2 (como a F3-H6)
#   E  a fusão conjunta do v2 (como a F3-H3/F3-H6); a cabeça só de churn da F3-H5 não usava a feature
# e o bootstrap por cliente no teste. Espera os rótulos e features corrigidos (passo 0 de fila_colab_f3h10.sh).
cd /content/genai_model
source scripts/colab_persistencia.sh
export LD_LIBRARY_PATH=/usr/lib64-nvidia:${LD_LIBRARY_PATH:-}
C=configs/hm_clientes.yaml
R=runs/hm_clientes
PY="python3 -u"
t() { echo "== $* $(date)"; }
until [ -f $R/rotulos_downstream/.baseline_ltv_corrigido ] && [ -f $R/features_tradicionais/features_info.json ] \
      && [ -f $R/caches/pca32_v2.npz ]; do sleep 60; done
t "rótulos e features corrigidos prontos"

G="$PY pipeline/common/tarefa14_gbdt.py $C --modelo lgbm --n_trials 40 --seeds 0,1,2 --threads 6 --avaliar_teste --tarefas churn,ltv_90d"
feito() { [ -f $R/gbdt/$1/predicoes_test_ltv_90d_seed2.parquet ]; }
feito errata_trad_lgbm || { t "T"; $G --features trad --saida errata_trad_lgbm || exit 1; }
for S in 0 1 2; do
  [ -f $R/errata_v2_fusao_s$S/predicoes_test.parquet ] || { t "E fusão conjunta s$S"; $PY pipeline/common/tarefa13_treinar_fusao.py $C \
      --emb v2_fusao --seed $S --saida errata_v2_fusao_s$S --dispositivo auto --avaliar_teste || exit 1; }
done
persistir
feito errata_hibrido_v2_lgbm || { t "H"; $G --features hibrido --emb v2_fusao --pca_k 32 --pca_cache $R/caches/pca32_v2.npz \
    --saida errata_hibrido_v2_lgbm || exit 1; }
persistir

# bootstrap por cliente, teste: contra o T corrigido; e o T antigo contra o corrigido (o tamanho do vazamento)
O=$R/errata_ltv/bootstrap_test.json; mkdir -p $R/errata_ltv; rm -f $O
A="$PY pipeline/common/avaliacao.py $C --split test --B 300 --saida $O"
$A --tarefa churn --a gbdt/errata_trad_lgbm --b gbdt/errata_hibrido_v2_lgbm --b "errata_v2_fusao_s*" --b "f3h5_churn_s*" \
   --b gbdt/h10_trad_lgbm_media || exit 1
$A --tarefa ltv_90d --a gbdt/errata_trad_lgbm --b gbdt/errata_hibrido_v2_lgbm --b "errata_v2_fusao_s*" \
   --b gbdt/h10_trad_lgbm_media || exit 1
persistir
echo "FIM $(date)"
