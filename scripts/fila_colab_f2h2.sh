#!/usr/bin/env bash
# F2-H2 (ajuste fino conjunto), rodada 2: 3 sementes com predições de teste e
# bootstrap NA HORA, após cada semente, contra a fusão congelada (H4) e o GBDT
# tradicional, com as predições destes enviadas da máquina local.
cd /content/genai_model
export LD_LIBRARY_PATH=/usr/lib64-nvidia:${LD_LIBRARY_PATH:-}
C=configs/hm_clientes.yaml
R=runs/hm_clientes
[ -f $R/fusao/embeddings_eventos_index.parquet ] || python3 -u pipeline/common/tarefa13_extrair_embeddings.py $C --pretreino pretreino --saida fusao --dtype float16
echo "[$(date)] extração ok"
for S in 0 1 2; do
  ini=$(date +%s)
  [ -f $R/f2_h2_conjunto_s$S/predicoes_test.parquet ] || python3 -u pipeline/common/tarefa15_ajuste_conjunto.py $C \
    --saida f2_h2_conjunto_s$S --seed $S --avaliar_teste --retomar
  echo "TEMPO f2_h2_conjunto_s$S $(( $(date +%s) - ini )) s"
  n=$(ls $R/f2_h2_conjunto_s*/predicoes_test.parquet 2>/dev/null | wc -l)
  out=$R/f2_h2_bootstrap_${n}sementes.json; rm -f $out
  for t in churn proxima_online next_category ltv_90d; do
    python3 pipeline/common/avaliacao.py $C --split test --B 300 --saida $out --tarefa $t \
      --a "caches/h10_fusao_d_completo_s*" --b "f2_h2_conjunto_s*"
    python3 pipeline/common/avaliacao.py $C --split test --B 300 --saida $out --tarefa $t \
      --a gbdt/h10_trad_lgbm --b "f2_h2_conjunto_s*"
  done
  echo "[$(date)] bootstrap com $n sementes ok"
done
echo "[$(date)] F2-H2 concluída"
