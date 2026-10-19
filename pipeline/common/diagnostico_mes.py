"""Diagnóstico do token de MÊS (experimento H&M, H8).

O `diagnostico_copia.py` mostrou que, fora o token MES_*, modelos NoPE e RoPE
têm a mesma loss de pré-treino; toda a diferença (~0,73 nats) vem do mês.
Este script mostra o mecanismo: para cada modelo e split fora do tempo de
treino (val, teste), a loss nos tokens de mês, quanto da massa de
probabilidade fica em tokens de mês e qual mês o modelo prevê (argmax) contra
o mês real.

Uso: python3 pipeline/common/diagnostico_mes.py configs/hm_clientes.yaml \\
        --modelos nope_h3=pretreino rope_final_s0=final_s0/pretreino [--n_entidades 300]
Gera runs/<nome>/relatorios/diagnostico_mes.json
"""

import argparse
import collections
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "pipeline/common")
from modelo_decoder import arquitetura_efetiva, construir_modelo  # noqa: E402
from schema import carregar_config, run_dir  # noqa: E402
from sequencias_io import carregar_sequencias  # noqa: E402
from tarefa_pretreino import exemplos_de_avaliacao, montar_lote_avaliacao  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--modelos", nargs="+", required=True)
    ap.add_argument("--n_entidades", type=int, default=300)
    args = ap.parse_args()
    config = carregar_config(args.config)
    vocab = json.load(open(f"{run_dir(config, 'vocabulario')}/vocab.json", encoding="utf-8"))
    id2 = {v: k for k, v in vocab.items()}
    pad = vocab["PAD"]
    meses = [vocab[f"MES_{k}"] for k in range(1, 13)]
    seqs = carregar_sequencias(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = {}
    for split in ("val", "test"):
        ex, _ = exemplos_de_avaliacao(seqs, split)
        sel = np.random.RandomState(1).choice(len(ex), min(args.n_entidades, len(ex)), replace=False)
        exs = [ex[k] for k in sorted(sel)]
        for spec in args.modelos:
            nome, pasta = spec.split("=", 1)
            d = run_dir(config, pasta)
            arch = arquitetura_efetiva(json.load(open(os.path.join(d, "pretreino_config.json")))["arquitetura"])
            m = construir_modelo(arch, vocab).to(device)
            m.carregar_pesos(torch.load(os.path.join(d, "checkpoints", "melhor.pt"), map_location=device))
            m.eval()
            ids, alvo = montar_lote_avaliacao(seqs, exs, split, arch["n_positions"], pad)
            prev, real, massa, nll = collections.Counter(), collections.Counter(), [], []
            with torch.no_grad():
                for b in range(0, len(ids), 16):
                    x = torch.from_numpy(ids[b:b + 16]).to(device)
                    lp = torch.log_softmax(m(x, (x != pad).long()).float(), -1).cpu()
                    for r in range(x.shape[0]):
                        for p in np.flatnonzero(alvo[b + r]):
                            t = int(ids[b + r, p])
                            if p == 0 or not id2[t].startswith("MES_"):
                                continue
                            q = lp[r, p - 1]
                            prev[id2[int(q.argmax())]] += 1
                            real[id2[t]] += 1
                            massa.append(float(q[meses].exp().sum()))
                            nll.append(-float(q[t]))
            out.setdefault(split, {})[nome] = {
                "n_tokens_mes": len(nll), "nll_mes": float(np.mean(nll)), "massa_em_tokens_mes": float(np.mean(massa)),
                "acerto_argmax": sum(min(prev[k], real[k]) for k in real) / max(1, len(nll)),
                "meses_reais": dict(real.most_common()), "meses_previstos_argmax": dict(prev.most_common())}
            print(f"{split} {nome}: nll_mes={np.mean(nll):.3f}  real={real.most_common(4)}  previsto={prev.most_common(4)}")
    os.makedirs(run_dir(config, "relatorios"), exist_ok=True)
    caminho = os.path.join(run_dir(config, "relatorios"), "diagnostico_mes.json")
    json.dump(out, open(caminho, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"Gravado {caminho}")


if __name__ == "__main__":
    main()
