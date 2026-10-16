"""Motor genérico — Dimensionamento da arquitetura (Tarefa 6 generalizada).

Ferramenta de decisão, sem treino:
  1. Cobertura de `n_positions` candidatos — por ENTIDADE (sequência inteira
     cabe na janela) e por EVENTO (o evento ainda tem embedding, isto é, cai
     dentro dos últimos n_positions tokens da sua entidade). A lição da Olist
     é que olhar só por entidade engana: poucas entidades de cauda longa
     concentram muitos eventos.
  2. Parâmetros reais da arquitetura do config (instanciada via
     `modelo_decoder.py`, CPU) e a razão tokens de treino / parâmetro.

Uso: python3 pipeline/common/tarefa6_arquitetura.py <config.yaml> [n_positions candidatos...]
"""

import json
import sys

import numpy as np

sys.path.insert(0, "pipeline/common")
from modelo_decoder import arquitetura_efetiva, construir_modelo  # noqa: E402
from schema import caminho_tabela, carregar_config, ler_tabela, run_dir  # noqa: E402
from sequencias_io import carregar_sequencias  # noqa: E402


def cobertura(seqs, posicoes, candidatos):
    total = seqs.len_full - 1
    idx = {e: i for i, e in enumerate(seqs.ids)} if seqs.ids.dtype == object else None
    if idx is not None:
        ent_evento = posicoes["entidade_id"].map(idx).to_numpy()
    else:
        ordem = np.argsort(seqs.ids)
        ent_evento = ordem[np.searchsorted(seqs.ids[ordem], posicoes["entidade_id"].to_numpy())]
    pos = posicoes["pos_full"].to_numpy()
    print(f"{'n_positions':>12} {'% entidades inteiras':>22} {'% eventos com embedding':>25}")
    for n in candidatos:
        ent_ok = (total <= n).mean()
        ev_ok = (pos >= np.maximum(0, total[ent_evento] - n)).mean()
        print(f"{n:>12} {ent_ok:>21.1%} {ev_ok:>24.1%}")


def main(config_path, candidatos):
    config = carregar_config(config_path)
    arch = arquitetura_efetiva(config.get("arquitetura", {}))
    with open(f"{run_dir(config, 'vocabulario')}/vocab.json", encoding="utf-8") as f:
        vocab = json.load(f)
    seqs = carregar_sequencias(config)
    posicoes = ler_tabela(caminho_tabela(config, "serializacao", "posicoes_evento"), ["entidade_id", "pos_full"])

    comp = seqs.len_full
    print(f"vocab_size={len(vocab)}  entidades={len(seqs):,}  tokens={int(comp.sum()):,}")
    print("Comprimento de seq_full (tokens): " + "  ".join(
        f"p{q}={np.percentile(comp, q):.0f}" for q in [50, 75, 90, 95, 99]) + f"  máx={comp.max()}")
    tokens_treino = int(np.sum(seqs.len_train[seqs.len_train > 0] + 1))
    print(f"Tokens de treino (alvos de NTP): {tokens_treino:,}\n")

    cobertura(seqs, posicoes, candidatos or [256, 512, 1024, 2048, 4096])

    modelo = construir_modelo({**arch, "attn_implementation": "sdpa"}, vocab)
    total, treinaveis = modelo.n_parametros()
    print(f"\nArquitetura do config: {arch}")
    print(f"Parâmetros: {total/1e6:.2f}M ({treinaveis/1e6:.2f}M treináveis) | "
          f"tokens de treino por parâmetro treinável: {tokens_treino/treinaveis:.1f}")


if __name__ == "__main__":
    main(sys.argv[1], [int(x) for x in sys.argv[2:]])
