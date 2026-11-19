#!/usr/bin/env bash
# F3-H4 (PLANO_FASE3.md): recomendação direta do motor v2 na H&M, protocolo da F2-H3.
#   backbone atualizado até a véspera da semana-alvo: validação (05/03/2020, semente 0) e teste
#   (16/09/2020, sementes 0-2); 8 épocas como na F3-H3; recomendações do catálogo inteiro + histórico;
#   MAP@12 contra o R1 corrigido e o R2 da F2-H3.
# Depois: E3b (F2-H10) e TabPFN de 300 mil âncoras (F2-H11), que ficaram pendentes.
cd /content/genai_model
source scripts/colab_persistencia.sh
restaurar
export LD_LIBRARY_PATH=/usr/lib64-nvidia:${LD_LIBRARY_PATH:-}
C=configs/hm_clientes.yaml; R=runs/hm_clientes
t() { echo "== $* $(date)"; }
completo() { [ -f $R/$1/v2_relatorio.json ] && python3 -c "import json,sys; sys.exit(0 if len(json.load(open('$R/$1/v2_relatorio.json'))['historico']) >= 8 else 1)"; }
for job in val:2020-03-05:0 test:2020-09-16:0 test:2020-09-16:1 test:2020-09-16:2; do
  IFS=: read fase sem s <<< "$job"
  P=f3h4_pre_${fase}_s$s
  if ! completo $P; then
    t "F3-H4 pre-treino $P (corte $sem)"
    ( while sleep 600; do persistir; done ) & PERS=$!
    python3 -u pipeline/common/v2_pretreino.py $C --saida $P --corte $sem --epocas 8 --seed $s --retomar
    kill $PERS 2>/dev/null; persistir
  fi
  [ -f $R/f3h4_rec_${fase}_s$s/rec_$sem.parquet ] || { t "F3-H4 recomendacao $fase s$s"; python3 -u pipeline/common/f3h4_recomendar.py $C \
      --pretreino $P --semana $sem --saida f3h4_rec_${fase}_s$s; persistir; }
done
t "F3-H4 avaliacao"; python3 -u pipeline/common/f3h4_avaliar.py $C --val f3h4_rec_val_s0 --test f3h4_rec_test_s0 f3h4_rec_test_s1 f3h4_rec_test_s2
persistir
SEM_ESPERA=1 scripts/fila_colab_e3b.sh
