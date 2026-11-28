#!/usr/bin/env bash
# F3-H9, complemento: a comparação contra T nas 4 tarefas da H&M, que a fila original não gravou
# (a pasta de T não estava no Colab). Não reavalia nada novo: as mesmas previsões, o mesmo bootstrap.
cd /content/genai_model
source scripts/colab_persistencia.sh
R=runs/hm_clientes; C=configs/hm_clientes.yaml
E=/content/drive/MyDrive/genai_model_artefatos/entrada
for S_ in 0 1 2; do ln -sfn ../hm_clientes_f3h1/f3h9_ft_s$S_ $R/f3h9_ft_s$S_; done
[ -d $R/gbdt/h10_trad_lgbm_media ] || tar xf $E/h10_trad_lgbm_media.tar -C $R/gbdt || exit 1
S=runs/hm_clientes_f3h1/f3h9/bootstrap_contra_t.json; rm -f $S
for tf in churn proxima_online next_category ltv_90d; do
  echo "== $tf $(date)"
  python3 -u pipeline/common/avaliacao.py $C --tarefa $tf --split test --B 300 --saida $S \
      --a gbdt/h10_trad_lgbm_media --b "f3h9_ft_s*" --b "v2_fusao_s*" || exit 1
done
persistir
echo "FIM $(date)"
