#!/usr/bin/env bash
# F3-H10 — segmentos estáveis com o vetor do v2, num corte ainda não visto.
# Desenho, regra de escolha e critério em PLANO_FASE3.md (registrados antes de rodar). Idempotente.
#   A. corte atual, só validação: K-means com 3 (reprodução da F3-H6), 10 e 30 reinícios; a regra
#      registrada escolhe o número de reinícios (ou fecha a hipótese sem teste);
#   B. corte novo (dados até 25/03/2020): o pipeline inteiro pelo motor, só com YAML e o adapter de corte;
#   C. clusterização do corte novo com o número escolhido: validação (forma) e teste, uma vez.
cd /content/genai_model
source scripts/colab_persistencia.sh
export LD_LIBRARY_PATH=/usr/lib64-nvidia:${LD_LIBRARY_PATH:-}
C=configs/hm_clientes.yaml
C2=configs/hm_clientes_corte2.yaml
R=runs/hm_clientes
R2=runs/hm_clientes_corte2
PY="python3 -u"
t() { echo "== $* $(date)"; }
( while sleep 600; do persistir; done ) & PERS=$!
trap "kill $PERS 2>/dev/null" EXIT

# 0. corte atual: rótulos e features refeitos com o baseline de LTV corrigido (errata de 27/11); os
#    antigos ficam em rotulos_downstream_ate_27nov. Os dois preparos (atual e novo) não rodam juntos (memória).
if [ ! -f $R/rotulos_downstream/.baseline_ltv_corrigido ]; then
  t "corte atual: rótulos e features com o baseline corrigido"
  [ -d $R/rotulos_downstream ] && { rm -rf $R/rotulos_downstream_ate_27nov; mv $R/rotulos_downstream $R/rotulos_downstream_ate_27nov; }
  rm -rf $R/features_tradicionais
  $PY motor.py $C preparar || exit 1
  touch $R/rotulos_downstream/.baseline_ltv_corrigido
fi

# B (em segundo plano): o corte novo pelo motor. O preparo é CPU; o pré-treino espera a GPU livre.
(
  for e in preparar serializar pretreinar extrair; do
    t "corte2 $e"; $PY motor.py $C2 $e --dispositivo cuda --epocas 8 --threads 6 || { echo "FALHOU corte2 $e"; exit 1; }
  done
  [ -f $R2/caches/pca32_v2.npz ] || $PY pipeline/common/v2_pca_cache.py $C2 --emb emb_v2 || exit 1
  t "corte2 pronto"
) > /content/f3h10_corte2.log 2>&1 &
CORTE2=$!

# A. corte atual: o PCA-32 do v2 (o mesmo cache da F3-H6) e a validação com 3, 10 e 30 reinícios
[ -f $R/v2_fusao/embeddings_eventos_index.parquet ] || { t "extração v2 (corte atual)"; \
    $PY pipeline/common/v2_extrair.py $C --pretreino v2_pretreino --saida v2_fusao || exit 1; }
[ -f $R/caches/pca32_v2.npz ] || { t "PCA-32 v2 (corte atual)"; $PY pipeline/common/v2_pca_cache.py $C --emb v2_fusao || exit 1; }
for n in 3 10 30; do
  [ -f $R/f3h10_val_n$n/resultado_val.json ] || { t "val n_init=$n"; $PY pipeline/common/f3h2_clusterizacao.py $C --fase val \
      --emb_cache caches/pca32_v2.npz --saida f3h10_val_n$n --n_init $n --ks 8 || exit 1; }
done
persistir

# regra registrada: o menor de {10, 30} em que o ARI do E fica no máximo 0,05 abaixo do ARI do melhor T
N=$($PY - <<'EOF'
import json
esc = "nenhum"
for n in (10, 30):
    d = json.load(open(f"runs/hm_clientes/f3h10_val_n{n}/resultado_val.json"))
    p, mt = d["principal"], d["melhor_T"]
    print(f"n_init={n}: ARI E {p['E']['ari']:.3f} (sementes {[round(x, 3) for x in p['E']['ari_por_semente']]}); "
          f"melhor T {mt} {p[mt]['ari']:.3f}", file=__import__("sys").stderr)
    if p["E"]["ari"] >= p[mt]["ari"] - 0.05:
        esc = n
        break
print(esc)
EOF
)
t "escolha do número de reinícios: $N"
echo "$N" > $R/f3h10_escolha_n_init.txt
if [ "$N" = "nenhum" ]; then
  echo "nenhum candidato passa na validação: a hipótese fecha sem teste (regra registrada)"
  kill $CORTE2 2>/dev/null; persistir; kill $PERS; echo "FIM $(date)"; exit 0
fi

# C. corte novo: espera o pipeline e clusteriza com o número escolhido (forma na validação, teste uma vez)
wait $CORTE2 || { echo "corte2 falhou; ver /content/f3h10_corte2.log"; tail -30 /content/f3h10_corte2.log; exit 1; }
for f in val test; do
  [ -f $R2/f3h10_cluster/resultado_$f.json ] || { t "corte2 cluster $f (n_init=$N)"; $PY pipeline/common/f3h2_clusterizacao.py $C2 \
      --fase $f --emb_cache caches/pca32_v2.npz --saida f3h10_cluster --n_init $N || exit 1; }
done
persistir
kill $PERS
echo "FIM $(date)"
