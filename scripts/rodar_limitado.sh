#!/usr/bin/env bash
# Roda um comando pesado num cgroup com teto de memória (systemd-run --user,
# sem sudo). Se o processo passar do teto, o kernel mata SÓ ele — o resto da
# máquina (VS Code, terminal) não é afetado. Sem isso, um job que estoura a
# RAM leva a máquina inteira pro swap e derruba o editor (aconteceu na EDA
# da H&M com 31,8M linhas).
#
# Uso: scripts/rodar_limitado.sh [-m 7G] <comando> [args...]
#   -m  teto de RAM (padrão 7G; a máquina tem 15 GB, ~5 GB ficam pro sistema/editor)
# Também roda com prioridade baixa de CPU e I/O.
set -euo pipefail
LIMITE=7G
if [[ "${1:-}" == "-m" ]]; then LIMITE="$2"; shift 2; fi
# pyarrow por padrão usa um pool próprio que não devolve memória ao sistema
# depois de ler um Parquet grande; o alocador do sistema devolve.
export ARROW_DEFAULT_MEMORY_POOL=system
exec systemd-run --user --scope --quiet -p MemoryMax="$LIMITE" -p MemorySwapMax=0 \
  --setenv=ARROW_DEFAULT_MEMORY_POOL=system \
  nice -n 10 ionice -c3 "$@"
