#!/usr/bin/env bash
# F2-H10 no Colab (L4): embeddings dos braços E0, E2, E3a0 e E3a na amostra de
# 10% dos clientes, fusões DCNv2 (3 sementes) e ajuste fino conjunto E1.
# Idempotente: cada etapa pula se a saída já existe (o Colab cai).
# Uso: setsid nohup scripts/fila_colab_f2h10.sh > /content/f2h10.log 2>&1 &
cd /content/genai_model
C=configs/hm_clientes.yaml
R=runs/hm_clientes
Q=Qwen/Qwen3-Embedding-0.6B
F="python3 -u pipeline/common/tarefa13_treinar_fusao.py"
t() { echo "== $* $(date)"; }

# E0: backbone H3 (embeddings da fase 2), recortado para a amostra
[ -f $R/f2h10_e0/embeddings_eventos_index.parquet ] || { t E0; python3 -u pipeline/common/f2h10_subamostra.py $C --de fusao --para f2h10_e0; }

# E3a0: Qwen3-Embedding sem ajuste
[ -f $R/f2h10_e3a0/embeddings_eventos_index.parquet ] || { t E3a0; python3 -u pipeline/common/f2h10_llm.py $C extrair --modelo $Q --saida f2h10_e3a0; }

# E2: CoLES nos nossos tokens
[ -f $R/f2h10_e2/pretreino/relatorio_coles.json ] || { t E2 treino; python3 -u pipeline/common/f2h10_coles.py $C --arch_de final_s0/config.yaml --saida f2h10_e2; }
[ -f $R/f2h10_e2_todas/embeddings_eventos_index.parquet ] || { t E2 extracao; python3 -u pipeline/common/tarefa13_extrair_embeddings.py $C \
    --pretreino f2h10_e2/pretreino --saida f2h10_e2_todas --dtype float16 --arch_de f2h10_e2/config.yaml; }
[ -f $R/f2h10_e2/embeddings_eventos_index.parquet ] || python3 -u pipeline/common/f2h10_subamostra.py $C --de f2h10_e2_todas --para f2h10_e2

# E3a: Qwen3-Embedding + LoRA contrastivo
[ -f $R/f2h10_e3a/relatorio_contrastivo.json ] || { t E3a treino; python3 -u pipeline/common/f2h10_llm.py $C contrastivo --modelo $Q \
    --saida f2h10_e3a --lote 32 --max_tokens 384 --passos 1000; }
[ -f $R/f2h10_e3a/embeddings_eventos_index.parquet ] || { t E3a extracao; python3 -u pipeline/common/f2h10_llm.py $C extrair --modelo $Q \
    --adaptador f2h10_e3a/adaptador --saida f2h10_e3a; }

# fusões (variante d), 3 sementes; todas as âncoras dos diretórios já são da amostra
for B in e0 e2 e3a0 e3a; do for S in 0 1 2; do
  [ -f $R/f2h10_fusao_${B}_s$S/predicoes_test.parquet ] || { t fusao $B s$S; $F $C --emb f2h10_$B --seed $S --min_passos 4000 \
      --saida f2h10_fusao_${B}_s$S --dispositivo auto --avaliar_teste; }
done; done

# E1: ajuste fino conjunto (receita F2-H2) na amostra
for S in 0 1 2; do
  [ -f $R/f2h10_e1_s$S/predicoes_test.parquet ] || { t E1 s$S; python3 -u pipeline/common/tarefa15_ajuste_conjunto.py $C \
      --saida f2h10_e1_s$S --seed $S --frac_treino 0.10 --semente_amostra 0 --avaliar_teste --retomar; }
done
echo "FIM $(date)"
