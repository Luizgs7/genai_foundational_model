#!/usr/bin/env bash
# H10 — avaliação ÚNICA no teste, com tudo congelado na validação:
#   GBDT: hiperparâmetros reaproveitados das execuções da H5/H6/H7 (--params_de),
#         3 sementes, sem nova HPO;
#   fusão: mesma configuração e sementes da H4/H9, agora com --avaliar_teste,
#          para o backbone de referência (H3) e para o backbone final (H8).
# Depois, bootstrap por cliente no teste para os deltas que respondem às perguntas.
# Uso (Colab): GPU=1 CPU=1 bash scripts/rodar_h10.sh   (as duas metades podem rodar em paralelo)
set -euo pipefail
C=configs/hm_clientes.yaml
export LD_LIBRARY_PATH=/usr/lib64-nvidia:${LD_LIBRARY_PATH:-}
R=runs/hm_clientes
T=${THREADS:-6}
G="python3 -u pipeline/common/tarefa14_gbdt.py $C --avaliar_teste --seeds 0,1,2 --threads $T"
F="python3 -u pipeline/common/tarefa13_treinar_fusao.py $C --avaliar_teste --dispositivo auto"

gb() { :; }
if [ "${CPU:-0}" = 1 ] || [ "${XGB:-0}" = 1 ]; then
  gb() {  # $1=origem dos hiperparâmetros $2=features $3=--emb  [extras]   (modelo: $M, padrão lgbm)
    local o=$1 f=$2 e=$3; shift 3
    [ -f $R/gbdt/h10_$o/relatorio.json ] && python3 -c "import json,sys;r=json.load(open('$R/gbdt/h10_$o/relatorio.json'));sys.exit(0 if len(r['tarefas'])==4 else 1)" && return
    until python3 -c "import json,sys;sys.exit(0 if len(json.load(open('$R/gbdt/$o/relatorio.json'))['tarefas'])==4 else 1)" 2>/dev/null; do
      sleep 300   # origem dos hiperparâmetros ainda rodando
    done
    $G --features $f --modelo ${M:-lgbm} --emb $e --pca_k 32 --params_de $o --saida h10_$o "$@"
  }
fi
if [ "${XGB:-0}" = 1 ]; then   # XGBoost na GPU (XGB_DEVICE=cuda)
  M=xgb gb trad_xgb trad fusao
  M=xgb gb hibrido_xgb hibrido fusao
  echo "H10 XGBoost concluída"
fi
if [ "${CPU:-0}" = 1 ]; then
  gb trad_lgbm trad fusao                       # H5
  gb hibrido_lgbm hibrido fusao                 # H7 (backbone H3)
  gb emb_pca_lgbm emb_pca fusao                 # H6 (backbone H3)
  gb final_hibrido_lgbm hibrido final_s0/embeddings   # H7 com o backbone final
  gb hibrido_lgbm_sem_cadastro hibrido fusao --sem_cadastro   # H9 no GBDT
  echo "H10 GBDT concluída"
fi

if [ "${GPU:-0}" = 1 ]; then
  for S in 0 1 2; do   # backbone de referência (H3): não depende do backbone final
    [ -f $R/h10_fusao_d_completo_s$S/predicoes_test.parquet ] || $F --emb fusao --seed $S --saida h10_fusao_d_completo_s$S
    [ -f $R/h10_fusao_e_cad_caus_s$S/predicoes_test.parquet ] || $F --emb fusao --seed $S --saida h10_fusao_e_cad_caus_s$S --sem_embedding
    [ -f $R/h10_fusao_c_emb_caus_s$S/predicoes_test.parquet ] || $F --emb fusao --seed $S --saida h10_fusao_c_emb_caus_s$S --sem_cadastro
  done
  for S in 0 1 2; do   # backbone final: espera a semente terminar (pré-treino + extração + fusão de validação)
    until [ -f $R/final_s$S/fusao/tarefa13_relatorio.json ]; do sleep 300; done
    [ -f $R/h10_final_fusao_s$S/predicoes_test.parquet ] || \
      python3 -u pipeline/common/tarefa13_treinar_fusao.py $R/final_s$S/config.yaml --avaliar_teste --dispositivo auto \
        --emb final_s$S/embeddings --seed $S --saida h10_final_fusao_s$S
  done
  echo "H10 fusão concluída"
fi
