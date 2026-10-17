#!/usr/bin/env bash
# Fila do Colab: H3 (se o checkpoint de referência não existir) -> H4/H9 (GPU)
# -> H6/H7 (CPU, em paralelo) + H8 (GPU, ablações).
# Cada etapa pula o que já foi concluído — pode ser relançada após queda.
cd /content/genai_model
export LD_LIBRARY_PATH=/usr/lib64-nvidia:${LD_LIBRARY_PATH:-}
C=configs/hm_clientes.yaml
if ! python3 -c "import torch; torch.load('runs/hm_clientes/pretreino/checkpoints/melhor.pt', map_location='cpu')" 2>/dev/null; then
  echo "[$(date)] H3: checkpoint de referência ausente/inválido — pré-treino"
  python3 -u pipeline/common/tarefa_pretreino.py $C --retomar >> /content/pretreino_h3.log 2>&1 \
    && echo "[$(date)] H3 ok" || { echo "[$(date)] H3 FALHOU"; exit 1; }
fi
echo "[$(date)] iniciando H4/H9"
bash scripts/rodar_h4_h9_fusao.sh >> /content/h4_h9.log 2>&1 && echo "[$(date)] H4/H9 ok" || echo "[$(date)] H4/H9 FALHOU"
(THREADS=8 bash scripts/rodar_h6_h7.sh >> /content/h6_h7.log 2>&1 && echo "[$(date)] H6/H7 ok" || echo "[$(date)] H6/H7 FALHOU") &
python3 -u pipeline/common/experimento_ablacao.py $C \
  ref moderna tam_128x4 tam_256x6 tam_512x8 ctx_512 ctx_2048 pos_rope pos_aprendida ffn_swiglu norm_rms drop_0 attn_gqa2 attn_mqa \
  >> /content/h8.log 2>&1 && echo "[$(date)] H8 ok" || echo "[$(date)] H8 FALHOU"
wait
echo "[$(date)] fila concluída"
