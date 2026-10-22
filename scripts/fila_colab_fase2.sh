#!/usr/bin/env bash
# Fase 2 no Colab (GPU), depois de fila_colab_fase2_base.sh. Cada etapa pula o
# que já tem relatório; os tempos de parede vão para o log (F2-H6 mede custo).
cd /content/genai_model
export LD_LIBRARY_PATH=/usr/lib64-nvidia:${LD_LIBRARY_PATH:-}
C=configs/hm_clientes.yaml
R=runs/hm_clientes
F="python3 -u pipeline/common/tarefa13_treinar_fusao.py"
feito() { [ -f $R/$1/tarefa13_relatorio.json ]; }
until [ -f $R/fusao/embeddings_eventos_index.parquet ]; do sleep 60; done

# 1) F2-H2: teste rápido do ajuste fino conjunto (1% dos clientes, avaliação em amostra)
feito f2_h2_smoke || python3 -u pipeline/common/tarefa15_ajuste_conjunto.py $C --saida f2_h2_smoke --frac_treino 0.01 \
  --avaliar_a_cada 100 --val_janelas 3000 --checkpoint_a_cada 100 --smoke
echo "[$(date)] F2-H2 smoke ok"

# 2) F2-H6: fusão de tarefa única para a pergunta nova (3 sementes, cronometrada)
for S in 0 1 2; do
  ini=$(date +%s)
  feito f2h6_fusao_s$S || $F configs/hm_clientes_f2h6_fusao.yaml --emb fusao --seed $S --saida f2h6_fusao_s$S --dispositivo auto --avaliar_teste
  echo "TEMPO f2h6_fusao_s$S $(( $(date +%s) - ini )) s"
done
echo "[$(date)] F2-H6 fusão ok"

# 3) F2-H1: fusão com fração dos clientes rotulados (com e sem embedding)
for FR in 0.01 0.05 0.10 0.25; do
  for S in 0 1 2; do
    feito f2h1_fusao_d_f${FR}_s$S || $F $C --emb fusao --seed $S --frac_treino $FR --semente_amostra $S --min_passos 4000 \
      --saida f2h1_fusao_d_f${FR}_s$S --dispositivo auto --avaliar_teste
    feito f2h1_fusao_e_f${FR}_s$S || $F $C --emb fusao --seed $S --frac_treino $FR --semente_amostra $S --min_passos 4000 \
      --sem_embedding --saida f2h1_fusao_e_f${FR}_s$S --dispositivo auto --avaliar_teste
  done
done
echo "[$(date)] F2-H1 fusões ok"

# 4) F2-H2: ajuste fino conjunto completo, 3 sementes (longo; retomável)
for S in 0 1 2; do
  ini=$(date +%s)
  feito f2_h2_conjunto_s$S || python3 -u pipeline/common/tarefa15_ajuste_conjunto.py $C --saida f2_h2_conjunto_s$S --seed $S \
    --avaliar_teste --retomar
  echo "TEMPO f2_h2_conjunto_s$S $(( $(date +%s) - ini )) s"
done
echo "[$(date)] fase 2 GPU concluída"
