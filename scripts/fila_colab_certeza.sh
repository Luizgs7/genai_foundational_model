#!/usr/bin/env bash
# Fila de "certeza" (v2, 2026-10-20). Sessões do Colab têm caído a cada ~5 h,
# então a ordem prioriza o que dá certeza às conclusões centrais e cada etapa
# entrega um resultado pequeno (json sincronizado) assim que termina:
#   GPU: extração H3 + backbone final -> fusões da H10 com predições (d e final)
#        -> sementes extras de ref/pos_rope (H8) -> diagnóstico de mês pareado
#   CPU: para cada GBDT da H10 (hiperparâmetros congelados): treina com predições
#        de teste e roda NA HORA as comparações de bootstrap dele (h10_bootstrap_v2.json);
#        por fim as variantes cortadas (embedding bruto multiclasse, híbrido bruto)
# Tudo pula o que já está pronto (relatório completo / comparação já no json).
cd /content/genai_model
export LD_LIBRARY_PATH=/usr/lib64-nvidia:${LD_LIBRARY_PATH:-}
C=configs/hm_clientes.yaml
R=runs/hm_clientes
P=python3
pronto_emb() { [ -f $R/$1/embeddings_eventos_index.parquet ]; }

(
  pronto_emb fusao || $P -u pipeline/common/tarefa13_extrair_embeddings.py $C --pretreino pretreino --saida fusao --dtype float16
  for S in 0 1 2; do
    pronto_emb final_s$S/embeddings || $P -u pipeline/common/tarefa13_extrair_embeddings.py $R/final_s$S/config.yaml \
      --pretreino final_s$S/pretreino --saida final_s$S/embeddings --dtype float16
  done
  echo "[$(date)] extrações ok"
  F="$P -u pipeline/common/tarefa13_treinar_fusao.py"
  for S in 0 1 2; do
    [ -f $R/h10_fusao_d_completo_s$S/predicoes_test.parquet ] || $F $C --avaliar_teste --dispositivo auto --emb fusao --seed $S --saida h10_fusao_d_completo_s$S
    [ -f $R/h10_final_fusao_s$S/predicoes_test.parquet ] || $F $R/final_s$S/config.yaml --avaliar_teste --dispositivo auto \
      --emb final_s$S/embeddings --seed $S --saida h10_final_fusao_s$S
  done
  echo "[$(date)] fusões H10 (predições) ok"
  $P -u pipeline/common/experimento_ablacao.py $C pos_rope@s1 ref@s2 pos_rope@s2
  $P -u pipeline/common/diagnostico_copia.py $C --n_entidades 5000 --saida diagnostico_copia_pareado.json --modelos \
    nope_s0=abl_ref/pretreino nope_s1=abl_ref_s1/pretreino nope_s2=abl_ref_s2/pretreino \
    rope_s0=abl_pos_rope/pretreino rope_s1=abl_pos_rope_s1/pretreino rope_s2=abl_pos_rope_s2/pretreino
  echo "[$(date)] GPU concluída"
) >> /content/certeza_gpu.log 2>&1 &

(
  B="$P pipeline/common/avaliacao.py $C --split test --B 300 --saida $R/h10_bootstrap_v2.json"
  tem_comp() { python3 -c "import json,sys,os;f='$R/h10_bootstrap_v2.json';r=json.load(open(f)) if os.path.exists(f) else [];sys.exit(0 if any(x['tarefa']=='$1' and x['a'].endswith('$2') and x['b'].endswith('$3') for x in r) else 1)"; }
  fus_ok() { [ "$(ls $R/$1/predicoes_test.parquet 2>/dev/null | wc -l)" -ge 3 ]; }
  comp() {  # $1 tarefa $2 A $3 B — roda se ainda não estiver no json
    tem_comp "$1" "$2" "$3" || $B --tarefa $1 --a "$2" --b "$3"
  }
  completo() { python3 -c "import json,sys;r=json.load(open('$R/gbdt/$1/relatorio.json'));sys.exit(0 if len(r['tarefas'])==4 and all('test' in x['por_semente'][0] for x in r['tarefas'].values()) else 1)" 2>/dev/null; }
  G="$P -u pipeline/common/tarefa14_gbdt.py $C --avaliar_teste --seeds 0,1,2 --threads 12 --pca_k 32 --modelo lgbm"
  gb() {  # $1 origem dos hiperparâmetros, $2 features, $3 emb, [extras]
    local o=$1 f=$2 e=$3; shift 3
    completo h10_$o && [ -f $R/gbdt/h10_$o/predicoes_test_ltv_90d_seed2.parquet ] || $G --features $f --emb $e --params_de $o --saida h10_$o "$@"
  }
  until pronto_emb fusao && pronto_emb final_s0/embeddings; do sleep 120; done
  T="churn proxima_online next_category ltv_90d"
  gb trad_lgbm trad fusao
  until fus_ok "h10_fusao_d_completo_s*" && fus_ok "h10_final_fusao_s*"; do sleep 120; done
  for t in $T; do comp $t gbdt/h10_trad_lgbm "h10_fusao_d_completo_s*"; comp $t gbdt/h10_trad_lgbm "h10_final_fusao_s*"; done
  echo "[$(date)] bootstrap trad x fusões ok"
  gb hibrido_lgbm hibrido fusao
  for t in $T; do comp $t gbdt/h10_trad_lgbm gbdt/h10_hibrido_lgbm; done
  echo "[$(date)] bootstrap híbrido ok"
  gb emb_pca_lgbm emb_pca fusao
  for t in $T; do comp $t gbdt/h10_trad_lgbm gbdt/h10_emb_pca_lgbm; done
  echo "[$(date)] bootstrap só-embedding ok"
  gb hibrido_lgbm_sem_cadastro hibrido fusao --sem_cadastro
  for t in $T; do comp $t gbdt/h10_hibrido_lgbm_sem_cadastro gbdt/h10_hibrido_lgbm; done
  echo "[$(date)] bootstrap cadastro no GBDT ok"
  gb final_hibrido_lgbm hibrido final_s0/embeddings
  for t in $T; do comp $t gbdt/h10_trad_lgbm gbdt/h10_final_hibrido_lgbm; done
  echo "[$(date)] bootstrap híbrido final ok"
  # variantes cortadas por custo (menor prioridade)
  V="$P -u pipeline/common/tarefa14_gbdt.py $C --emb fusao --pca_k 32 --modelo lgbm --amostra_hpo 300000 --threads 12"
  tem() { python3 -c "import json,sys;sys.exit(0 if '$2' in json.load(open('$R/gbdt/$1/relatorio.json'))['tarefas'] else 1)" 2>/dev/null; }
  tem emb_bruto_lgbm next_category && tem emb_bruto_lgbm ltv_90d || $V --features emb --saida emb_bruto_lgbm --tarefas next_category,ltv_90d --n_trials 12
  tem hibrido_bruto_lgbm proxima_online || $V --features hibrido_bruto --saida hibrido_bruto_lgbm --tarefas churn,proxima_online --n_trials 20
  tem hibrido_bruto_lgbm ltv_90d || $V --features hibrido_bruto --saida hibrido_bruto_lgbm --tarefas next_category,ltv_90d --n_trials 12
  echo "[$(date)] CPU concluída"
) >> /content/certeza_cpu.log 2>&1 &
wait
echo "[$(date)] fila de certeza concluída"
