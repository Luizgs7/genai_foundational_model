#!/usr/bin/env bash
# F3-H7, controle das ablações: a régua T sem as features de cada fonte, na base sintética pequena.
# Pergunta: a fonte que o gabarito liga a cada problema pesa pouco também para T? (então o critério da
# F3-H7 pedia mais efeito do que o gerador planta). Hiperparâmetros congelados do T completo (sem HPO).
cd "$(dirname "$0")/.."
source scripts/colab_persistencia.sh
restaurar
C=configs/beleza_b2b_pequeno.yaml; R=runs/beleza_b2b_pequeno
t() { echo "== $* $(date)"; }
[ -f $R/features_tradicionais/features.parquet ] || { t "preparo da base pequena"; scripts/preparar_beleza.sh pequeno; }
for fonte in navegacao credito incentivo desconto; do
  S=f3h7_trad_sem_$fonte
  [ -f $R/gbdt/$S/predicoes_test_sem_pedido_7d_seed2.parquet ] || { t "T sem $fonte"; python3 -u pipeline/common/tarefa14_gbdt.py $C \
      --features trad --modelo lgbm --params_de f3h7_trad_lgbm --seeds 0,1,2 --threads 6 --avaliar_teste --sem_prefixo ${fonte}_ --saida $S; }
done
persistir
mkdir -p $R/f3h7; O=$R/f3h7/controle_T.json; rm -f $O
for tt in churn proxima_com_desconto ltv_90d proxima_online sem_pedido_7d; do
  python3 -u pipeline/common/avaliacao.py $C --tarefa $tt --split test --B 300 --saida $O --a gbdt/f3h7_trad_lgbm \
      --b gbdt/f3h7_trad_sem_navegacao --b gbdt/f3h7_trad_sem_credito --b gbdt/f3h7_trad_sem_incentivo --b gbdt/f3h7_trad_sem_desconto
done 2>&1 | grep -E "auc|accuracy|Error"
persistir
echo "FIM $(date)"
