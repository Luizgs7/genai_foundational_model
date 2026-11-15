#!/usr/bin/env bash
# Preparação de um runtime novo do Colab para a fila da fase 3 (scripts/fila_colab_f3.sh).
# Os pacotes de entrada ficam também no Drive: num runtime novo, se não vierem pelo túnel, saem de lá.
set -e
E=/content/drive/MyDrive/genai_model_artefatos/entrada
mkdir -p /content/genai_model $E && cd /content/genai_model
for p in bundle.tar codigo2.tar; do
  if [ -f /content/$p ]; then cp -u /content/$p $E/ 2>/dev/null || echo "sem espaço no Drive para $p"
  elif [ -f $E/$p ]; then cp $E/$p /content/; fi
done
tar xf /content/bundle.tar
tar xf /content/codigo2.tar
[ -d /content/v2_hm/v2 ] && { rm -rf runs/hm_clientes/v2; mv /content/v2_hm/v2 runs/hm_clientes/v2; }
[ -d runs/hm_clientes/v2 ] && cp -ru runs/hm_clientes/v2 $E/ 2>/dev/null || true
[ -d runs/hm_clientes/v2 ] || { [ -d $E/v2 ] && cp -r $E/v2 runs/hm_clientes/v2; } || true
chmod +x scripts/*.sh
pip uninstall -y torchao >/dev/null 2>&1 || true
pip install -q "tabpfn==9.2.0" tabicl optuna 2>&1 | tail -1
pip list 2>/dev/null | grep -i -E "^(tabpfn|tabicl|torchao|peft|optuna|polars) " || true
md5sum runs/hm_clientes/serializacao/entidades.parquet runs/hm_clientes/pretreino/checkpoints/melhor.pt runs/hm_clientes/v2/meta.json
ls runs/hm_clientes runs/hm_clientes_f3h1
