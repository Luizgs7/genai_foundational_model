#!/usr/bin/env bash
# F3-H7, camada 2 (Synerise): bateria com e sem a navegação. Desenho em PLANO_FASE3.md.
# Entrada (enviada pronta): runs/synerise/{eventos_canonicos,splits,discretizacao,rotulos_downstream} e
# dados_synerise/preparado/. Idempotente: cada etapa pula o que já existe (o Drive devolve o que foi persistido).
cd /content/genai_model
source scripts/colab_persistencia.sh
export LD_LIBRARY_PATH=/usr/lib64-nvidia:${LD_LIBRARY_PATH:-}
C=configs/synerise.yaml
R=runs/synerise
PY="python3 -u"
F="$PY pipeline/common/tarefa13_treinar_fusao.py"
G="$PY pipeline/common/tarefa14_gbdt.py $C --modelo lgbm --seeds 0,1,2 --threads 12 --avaliar_teste"
NAV="carrinho_ busca_ visita_"
t() { echo "== $* $(date)"; }
( while sleep 600; do persistir; done ) & PERS=$!

# 1. features tradicionais (com os agregados genéricos das fontes extras) e as duas serializações do v2
[ -f $R/features_tradicionais/features.parquet ] || { t "features"; $PY pipeline/common/features_tradicionais.py $C; }
[ -f $R/v2/meta.json ] || { t "serializacao v2"; $PY pipeline/common/v2_serializacao.py $C --dados v2; }
[ -f $R/v2_so_compra/meta.json ] || { t "serializacao so compra"; $PY pipeline/common/v2_serializacao.py $C --dados v2_so_compra \
    --sem_fonte carrinho busca visita; }
persistir

# 2. E: pré-treino, extração e fusão conjunta (3 sementes), com e sem a navegação
for V in v2 v2_so_compra; do
  if [ ! -f $R/pre_$V/v2_relatorio.json ] || ! python3 -c "import json,sys; sys.exit(0 if len(json.load(open('$R/pre_$V/v2_relatorio.json'))['historico']) >= 8 else 1)"; then
    t "pre-treino $V"; $PY pipeline/common/v2_pretreino.py $C --dados $V --saida pre_$V --epocas 8 --retomar; persistir
  fi
  [ -f $R/emb_$V/embeddings_eventos_index.parquet ] || { t "extracao $V"; $PY pipeline/common/v2_extrair.py $C --dados $V --pretreino pre_$V --saida emb_$V; }
  for S in 0 1 2; do
    [ -f $R/fus_${V}_s$S/predicoes_test.parquet ] || { t "fusao $V s$S"; $F $C --emb emb_$V --seed $S --saida fus_${V}_s$S \
        --dispositivo auto --avaliar_teste; }
  done
  persistir
done

# 3. T: LightGBM com HPO; T sem navegação com os mesmos hiperparâmetros (controle da camada 1)
[ -f $R/gbdt/syn_trad_lgbm/relatorio.json ] || { t "T"; $G --features trad --n_trials 40 --saida syn_trad_lgbm; }
[ -f $R/gbdt/syn_trad_so_compra_lgbm/relatorio.json ] || { t "T so compra"; $G --features trad --params_de syn_trad_lgbm \
    --sem_prefixo $NAV --saida syn_trad_so_compra_lgbm; }
persistir

# 4. H: T + PCA-32 do vetor, com e sem a navegação
[ -f $R/gbdt/syn_hibrido_lgbm/relatorio.json ] || { t "H"; $G --features hibrido --emb emb_v2 --pca_k 32 --n_trials 40 \
    --pca_cache $R/caches/pca32_v2.npz --saida syn_hibrido_lgbm; }
[ -f $R/gbdt/syn_hibrido_so_compra_lgbm/relatorio.json ] || { t "H so compra"; $G --features hibrido --emb emb_v2_so_compra --pca_k 32 \
    --params_de syn_hibrido_lgbm --sem_prefixo $NAV --pca_cache $R/caches/pca32_v2_so_compra.npz --saida syn_hibrido_so_compra_lgbm; }
persistir

# 5. clusterização (protocolo da F3-H2): E e H com todas as fontes; E só com a compra como espaço extra
for f in val test; do
  [ -f $R/syn_cluster/resultado_$f.json ] || { t "cluster $f"; $PY pipeline/common/f3h2_clusterizacao.py $C --fase $f \
      --emb_cache caches/pca32_v2.npz --emb_extra SO_COMPRA=caches/pca32_v2_so_compra.npz --saida syn_cluster; }
done

# 6. bootstrap por cliente
S=$R/syn_bootstrap_test.json; rm -f $S
A="$PY pipeline/common/avaliacao.py $C --split test --B 300 --saida $S"
for tf in churn next_category ltv_30d; do
  $A --tarefa $tf --a gbdt/syn_trad_lgbm --b "fus_v2_s*" --b gbdt/syn_hibrido_lgbm          # E e H contra T
  $A --tarefa $tf --a "fus_v2_so_compra_s*" --b "fus_v2_s*"                                 # o que a navegação dá ao E
  $A --tarefa $tf --a gbdt/syn_trad_so_compra_lgbm --b gbdt/syn_trad_lgbm                   # ... e ao T (controle)
  $A --tarefa $tf --a gbdt/syn_hibrido_so_compra_lgbm --b gbdt/syn_hibrido_lgbm             # ... e ao H
done
kill $PERS 2>/dev/null
persistir
echo "FIM $(date)"
