#!/usr/bin/env bash
# E3b da F2-H10 sozinho: espera a fila f3b terminar (a GPU livre) e roda o bloco do E3b com micro-lotes.
# O primeiro treino (lote 32 inteiro) estourou a memória da L4: só os logits do vocabulário pedem 11 GB.
cd /content/genai_model
source scripts/colab_persistencia.sh
export LD_LIBRARY_PATH=/usr/lib64-nvidia:${LD_LIBRARY_PATH:-}
C=configs/hm_clientes.yaml
R=runs/hm_clientes
F="python3 -u pipeline/common/tarefa13_treinar_fusao.py"
A="python3 -u pipeline/common/f2h10_avaliar.py $C"
t() { echo "== $* $(date)"; }
until grep -q "^FIM" /content/f3b.log; do sleep 60; done
# ---- 6. E3b da F2-H10 ----
QB=Qwen/Qwen3.5-0.8B-Base
if [ ! -f $R/f2h10_e3b/relatorio_lm.json ]; then
  t "E3b ensaio"
  if python3 -u pipeline/common/f2h10_llm.py $C lm --modelo $QB --saida f2h10_e3b_ensaio --lote 4 --micro 2 --max_tokens 128 --passos 2 --max_clientes 200 --estado_a_cada 0; then
    rm -rf $R/f2h10_e3b_ensaio
    t "E3b treino"
    ( while sleep 300; do persistir; done ) & PERS=$!
    python3 -u pipeline/common/f2h10_llm.py $C lm --modelo $QB --saida f2h10_e3b --lote 32 --micro 8 --max_tokens 384 --passos 1000 --retomar
    kill $PERS 2>/dev/null
  else
    echo "E3b: o ensaio falhou; treino não iniciado"
  fi
fi
if [ -f $R/f2h10_e3b/relatorio_lm.json ]; then
  [ -f $R/f2h10_e3b/embeddings_eventos_index.parquet ] || { t "E3b extracao"; python3 -u pipeline/common/f2h10_llm.py $C extrair --modelo $QB --causal_lm \
      --adaptador f2h10_e3b/adaptador --pooling media --sem_instrucao --saida f2h10_e3b; }
  persistir
  for S in 0 1 2; do
    [ -f $R/f2h10_fusao_e3b_s$S/predicoes_test.parquet ] || { t "fusao e3b s$S"; $F $C --emb f2h10_e3b --seed $S --min_passos 4000 \
        --saida f2h10_fusao_e3b_s$S --dispositivo auto --avaliar_teste; }
  done
  persistir
  [ -f $R/f2h10/resultado_test_e3b.json ] || { t "avaliacao E3b"; $A --saida f2h10/resultado_test_e3b.json \
      --braco T=gbdt/f2h10_trad --braco "E0=f2h10_fusao_e0_s*" --braco "E3a0=f2h10_fusao_e3a0_s*" --braco "E3a=f2h10_fusao_e3a_s*" \
      --braco "E3b=f2h10_fusao_e3b_s*" --ref E0 --ref T --ref E3a; }
  persistir
fi

echo "FIM $(date)"
