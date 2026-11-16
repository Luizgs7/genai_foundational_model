# Persistência dos artefatos do Colab no Google Drive, para um runtime novo não
# recomeçar do zero. Só age se o Drive estiver montado no notebook:
#   from google.colab import drive; drive.mount('/content/drive')
# Sem o Drive, as funções não fazem nada e a fila roda igual.
#
# Vão para o Drive só os diretórios da amostra da F2-H10/F2-H11 (≈3 GB no total).
# Os embeddings de todas as âncoras (fusao, f2h10_e2_todas: 4 GB e 2,7 GB) ficam
# fora: refazê-los leva 23 e 6 min de L4.
#
# Uso (a partir de /content/genai_model): source scripts/colab_persistencia.sh
RUNS=runs/hm_clientes
DRIVE=/content/drive/MyDrive/genai_model_artefatos/hm_clientes

persistir() {
  [ -d /content/drive/MyDrive ] || return 0
  mkdir -p $DRIVE
  local d rel
  for d in $RUNS/f2h10_e0 $RUNS/f2h10_e2 $RUNS/f2h10_e3a0 $RUNS/f2h10_e3a $RUNS/f2h10_e3b $RUNS/f2h10_fusao_* $RUNS/f2h10_e1_s* \
           $RUNS/gbdt/f2h11_tabicl_* $RUNS/gbdt/f2h11_tabpfn_* $RUNS/f2h10/resultado*.json $RUNS/f2h11/resultado*.json \
           $RUNS/v2_pretreino* $RUNS/v2_fusao_s* $RUNS/v2_rec* $RUNS/v2b_pretreino* $RUNS/v2b_fusao_s* $RUNS/f3h3; do
    [ -e "$d" ] || continue
    rel=$(dirname "${d#$RUNS/}")
    mkdir -p "$DRIVE/$rel"
    cp -ru "$d" "$DRIVE/$rel/" 2>/dev/null
  done
  # F3-H1: fusões de tarefa única (run próprio, com ligações para o run da H&M)
  for d in runs/hm_clientes_f3h1/f3h1_fusao_s*; do
    [ -e "$d" ] || continue
    mkdir -p $DRIVE/../hm_clientes_f3h1 && cp -ru "$d" $DRIVE/../hm_clientes_f3h1/ 2>/dev/null
  done
  # F3-H7/F3-H8: pré-treinos e fusões da base sintética (os dados se regeneram com scripts/preparar_beleza.sh)
  for d in runs/beleza_b2b_*/pre_* runs/beleza_b2b_*/fus_* runs/beleza_b2b_*/gbdt runs/beleza_b2b_*/f3h7 runs/beleza_b2b_*/f3h8 runs/beleza_b2b_*/rec_*; do
    [ -e "$d" ] || continue
    rel=$(basename $(dirname "$d"))
    mkdir -p $DRIVE/../$rel && cp -ru "$d" $DRIVE/../$rel/ 2>/dev/null
  done
}

restaurar() {
  [ -d $DRIVE ] || return 0
  echo "== restaurando do Drive $(date)"
  cp -ru $DRIVE/. $RUNS/ 2>/dev/null
  [ -d $DRIVE/../hm_clientes_f3h1 ] && { mkdir -p runs/hm_clientes_f3h1; cp -ru $DRIVE/../hm_clientes_f3h1/. runs/hm_clientes_f3h1/ 2>/dev/null; }
  for b in $DRIVE/../beleza_b2b_*; do
    [ -d "$b" ] || continue
    mkdir -p runs/$(basename $b) && cp -ru $b/. runs/$(basename $b)/ 2>/dev/null
  done
  return 0
}
