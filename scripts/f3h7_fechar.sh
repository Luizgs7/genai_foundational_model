#!/usr/bin/env bash
# F3-H7, camada 1 (base sintética pequena): fecha a bateria depois das fusões da fila f3b.
#   H: GBDT híbrido (features tradicionais com as fontes extras + embedding v2 em PCA-32)
#   clusterização: ARI contra os segmentos plantados (rfm, trad, E, H)
#   recomendação: direta do modelo, com todas as fontes e sem navegação, contra R1 e POP
#   avaliação: T, E e H nas 5 tarefas; cada ablação contra o v2 com todas as fontes
cd "$(dirname "$0")/.."
export LD_LIBRARY_PATH=/usr/lib64-nvidia:${LD_LIBRARY_PATH:-}
C=configs/beleza_b2b_pequeno.yaml; R=runs/beleza_b2b_pequeno
t() { echo "== $* $(date)"; }
while [ ! -f $R/fus_v2_sem_desconto_s2/predicoes_test.parquet ]; do sleep 60; done
[ -f $R/gbdt/f3h7_hibrido_lgbm/relatorio.json ] || { t "H hibrido"; python3 -u pipeline/common/tarefa14_gbdt.py $C --features hibrido --emb emb_v2 \
    --modelo lgbm --n_trials 40 --seeds 0,1,2 --threads 8 --avaliar_teste --pca_k 32 --saida f3h7_hibrido_lgbm > /content/f3h7_hibrido.log 2>&1; }
t "segmentos"; python3 -u pipeline/common/f3h7_segmentos.py $C --espacos rfm trad E:emb_v2 H:emb_v2 E:emb_v2_sem_navegacao --saida segmentos
D=2025-01-06
for V in v2 v2_sem_navegacao; do
  [ -f $R/rec_$V/recomendacoes_$D.parquet ] || { t "recomendacao $V"; python3 -u pipeline/common/v2_extrair.py $C --dados $V --pretreino pre_$V \
      --recomendar $D --k 12 --saida rec_$V; }
done
t "recomendacao: medida"; python3 -u pipeline/common/f3h7_recomendacao.py $C --data $D --modelo v2=rec_v2 v2_sem_navegacao=rec_v2_sem_navegacao --saida recomendacao
S=$R/f3h7/bootstrap_test.json; rm -f $S
A="python3 -u pipeline/common/avaliacao.py $C --split test --B 300 --saida $S"
for tt in churn proxima_com_desconto ltv_90d proxima_online sem_pedido_7d; do
  $A --tarefa $tt --a gbdt/f3h7_trad_lgbm --b "fus_v2_s*" --b gbdt/f3h7_hibrido_lgbm
  $A --tarefa $tt --a "fus_v2_s*" --b "fus_v2_sem_navegacao_s*" --b "fus_v2_sem_credito_s*" --b "fus_v2_sem_incentivo_s*" --b "fus_v2_sem_desconto_s*"
done 2>&1 | grep -E "auc|accuracy"
echo "FIM $(date)"
