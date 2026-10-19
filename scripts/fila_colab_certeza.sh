#!/usr/bin/env bash
# Fila de "certeza" (2026-10-19), pedida depois da perda do runtime no meio do
# bootstrap da H10. Três trilhas em paralelo, cada etapa retomável:
#   GPU : extrações (H3 e backbone final) -> fusões da H10 com predições
#         -> sementes extras de ref/pos_rope (H8) -> diagnóstico de cópia pareado
#   CPU1: GBDT LightGBM da H10 (hiperparâmetros congelados) com predições de teste
#   CPU2: variantes cortadas por custo: embedding bruto multiclasse (H6) e híbrido bruto (H7)
#   BOOT: cada comparação do bootstrap roda assim que as duas predições existem
#         (resultado incremental em h10_bootstrap_v2.json, sincronizado)
cd /content/genai_model
export LD_LIBRARY_PATH=/usr/lib64-nvidia:${LD_LIBRARY_PATH:-}
C=configs/hm_clientes.yaml
R=runs/hm_clientes
P=python3

(
  [ -f $R/fusao/embeddings_eventos_index.parquet ] || $P -u pipeline/common/tarefa13_extrair_embeddings.py $C --pretreino pretreino --saida fusao --dtype float16
  for S in 0 1 2; do
    [ -f $R/final_s$S/embeddings/embeddings_eventos_index.parquet ] || \
      $P -u pipeline/common/tarefa13_extrair_embeddings.py $R/final_s$S/config.yaml --pretreino final_s$S/pretreino --saida final_s$S/embeddings --dtype float16
  done
  echo "[$(date)] extrações ok"
  GPU=1 bash scripts/rodar_h10.sh
  $P -u pipeline/common/experimento_ablacao.py $C ref pos_rope ref@s1 pos_rope@s1 ref@s2 pos_rope@s2
  $P -u pipeline/common/diagnostico_copia.py $C --n_entidades 5000 --saida diagnostico_copia_pareado.json --modelos \
    nope_s0=abl_ref/pretreino nope_s1=abl_ref_s1/pretreino nope_s2=abl_ref_s2/pretreino \
    rope_s0=abl_pos_rope/pretreino rope_s1=abl_pos_rope_s1/pretreino rope_s2=abl_pos_rope_s2/pretreino
  echo "[$(date)] GPU concluída"
) >> /content/certeza_gpu.log 2>&1 &

(
  until [ -f $R/fusao/embeddings_eventos_index.parquet ] && [ -f $R/final_s0/embeddings/embeddings_eventos_index.parquet ]; do sleep 120; done
  CPU=1 THREADS=6 bash scripts/rodar_h10.sh
  echo "[$(date)] CPU1 concluída"
) >> /content/certeza_cpu1.log 2>&1 &

(
  until [ -f $R/fusao/embeddings_eventos_index.parquet ]; do sleep 120; done
  G="$P -u pipeline/common/tarefa14_gbdt.py $C --emb fusao --pca_k 32 --modelo lgbm --amostra_hpo 300000 --threads 6"
  tem() { python3 -c "import json,sys;sys.exit(0 if '$2' in json.load(open('$R/gbdt/$1/relatorio.json'))['tarefas'] else 1)" 2>/dev/null; }
  tem emb_bruto_lgbm next_category && tem emb_bruto_lgbm ltv_90d || $G --features emb --saida emb_bruto_lgbm --tarefas next_category,ltv_90d --n_trials 12
  tem hibrido_bruto_lgbm proxima_online || $G --features hibrido_bruto --saida hibrido_bruto_lgbm --tarefas churn,proxima_online --n_trials 20
  tem hibrido_bruto_lgbm ltv_90d || $G --features hibrido_bruto --saida hibrido_bruto_lgbm --tarefas next_category,ltv_90d --n_trials 12
  echo "[$(date)] CPU2 concluída"
) >> /content/certeza_cpu2.log 2>&1 &

(
  pronto() {  # $1 = pasta (relativa a runs/hm_clientes), $2 = tarefa
    case "$1" in
      gbdt/*) [ -f "$R/$1/predicoes_test_$2_seed2.parquet" ] ;;
      *) [ "$(ls $R/${1}/predicoes_test.parquet 2>/dev/null | wc -l)" -ge 3 ] ;;
    esac
  }
  comp() {  # $1 tarefa, $2 A, $3 B
    until pronto "$2" "$1" && pronto "$3" "$1"; do sleep 300; done
    $P pipeline/common/avaliacao.py $C --split test --B 300 --saida $R/h10_bootstrap_v2.json --tarefa $1 --a "$2" --b "$3"
  }
  for t in churn proxima_online next_category ltv_90d; do
    comp $t "h10_fusao_e_cad_caus_s*" "h10_fusao_d_completo_s*"
    comp $t "h10_fusao_c_emb_caus_s*" "h10_fusao_d_completo_s*"
    comp $t "h10_fusao_d_completo_s*" "h10_final_fusao_s*"
  done
  for t in churn proxima_online next_category ltv_90d; do
    comp $t gbdt/h10_trad_lgbm "h10_fusao_d_completo_s*"
    comp $t gbdt/h10_trad_lgbm "h10_final_fusao_s*"
    comp $t gbdt/h10_trad_lgbm gbdt/h10_hibrido_lgbm
    comp $t gbdt/h10_trad_lgbm gbdt/h10_emb_pca_lgbm
    comp $t gbdt/h10_trad_lgbm gbdt/h10_final_hibrido_lgbm
    comp $t gbdt/h10_hibrido_lgbm_sem_cadastro gbdt/h10_hibrido_lgbm
  done
  echo "[$(date)] BOOT concluído"
) >> /content/certeza_boot.log 2>&1 &
wait
echo "[$(date)] fila de certeza concluída"
