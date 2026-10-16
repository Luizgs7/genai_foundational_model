#!/usr/bin/env bash
# Roda no Colab (GPU) depois do pré-treino de referência (H3):
#   H4 — extração dos embeddings de todas as âncoras + fusão DCNv2 completa
#        (embedding + cadastro + causais) com 3 sementes;
#   H9 — grade de ablação de entradas da fusão, 3 sementes cada:
#        a) só embedding   b) embedding + cadastro   c) embedding + causais
#        e) cadastro + causais (sem embedding)   [d = completo = H4]
# Só validação: o teste fica para a H10 (avaliação única).
set -euo pipefail
C=configs/hm_clientes.yaml
export LD_LIBRARY_PATH=/usr/lib64-nvidia:${LD_LIBRARY_PATH:-}
F="python3 -u pipeline/common/tarefa13_treinar_fusao.py $C --emb fusao --dispositivo auto"

if [ ! -f runs/hm_clientes/fusao/embeddings_eventos_index.parquet ]; then
  python3 -u pipeline/common/tarefa13_extrair_embeddings.py $C --pretreino pretreino --saida fusao --dtype float16
fi
for S in 0 1 2; do
  [ -f runs/hm_clientes/fusao_d_completo_s$S/tarefa13_relatorio.json ] || $F --seed $S --saida fusao_d_completo_s$S
  [ -f runs/hm_clientes/fusao_a_emb_s$S/tarefa13_relatorio.json ] || $F --seed $S --saida fusao_a_emb_s$S --sem_cadastro --sem_causais
  [ -f runs/hm_clientes/fusao_b_emb_cad_s$S/tarefa13_relatorio.json ] || $F --seed $S --saida fusao_b_emb_cad_s$S --sem_causais
  [ -f runs/hm_clientes/fusao_c_emb_caus_s$S/tarefa13_relatorio.json ] || $F --seed $S --saida fusao_c_emb_caus_s$S --sem_cadastro
  [ -f runs/hm_clientes/fusao_e_cad_caus_s$S/tarefa13_relatorio.json ] || $F --seed $S --saida fusao_e_cad_caus_s$S --sem_embedding
done
echo "H4/H9 fusão concluídas"
