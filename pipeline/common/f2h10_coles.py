"""F2-H10, braço E2 — CoLES sobre os nossos tokens (controle sem LLM).

Mesmo decoder do backbone final (RoPE + SwiGLU + GQA, 256x6), iniciado do zero,
mas treinado com o objetivo contrastivo do CoLES em vez de próximo token: duas
fatias contíguas aleatórias de blocos de evento do histórico de TREINO do mesmo
cliente devem gerar vetores próximos; os outros clientes do lote são os
negativos (InfoNCE simétrica). O vetor de uma fatia é o estado no último token
do seu último bloco — a mesma posição que a extração usa para a âncora.

Treina só nos clientes da amostra da F2-H10 (os mesmos dos braços com LLM),
para que E2 x E3a isole o conhecimento prévio do LLM, e E2 x E0 o objetivo.

Uso: python3 pipeline/common/f2h10_coles.py configs/hm_clientes.yaml --arch_de final_s0/config.yaml --saida f2h10_e2
Gera runs/<nome>/<saida>/{config.yaml, pretreino/checkpoints/melhor.pt, pretreino/relatorio_coles.json}
"""

import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

sys.path.insert(0, "pipeline/common")
from modelo_decoder import arquitetura_efetiva, construir_modelo  # noqa: E402
from schema import carregar_config, run_dir  # noqa: E402
from sequencias_io import carregar_sequencias  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--arch_de", required=True)
    ap.add_argument("--saida", default="f2h10_e2")
    ap.add_argument("--passos", type=int, default=6000)
    ap.add_argument("--lote", type=int, default=256)
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    torch.manual_seed(a.seed)
    rng = np.random.RandomState(a.seed)
    config = carregar_config(a.config)
    base = run_dir(config, "")
    arch = arquitetura_efetiva(carregar_config(os.path.join(base, a.arch_de))["arquitetura"])
    vocab = json.load(open(os.path.join(run_dir(config, "vocabulario"), "vocab.json"), encoding="utf-8"))
    pad = vocab["PAD"]
    seqs = carregar_sequencias(config)

    amostra = pl.read_parquet(os.path.join(run_dir(config, "f2h10"), "textos.parquet"), columns=["entidade_id"]).unique()
    pos = (pl.read_parquet(os.path.join(run_dir(config, "serializacao"), "posicoes_evento.parquet"))
           .join(amostra, on="entidade_id", how="semi")
           .join(pl.read_parquet(os.path.join(run_dir(config, "splits"), "splits.parquet"), columns=["evento_id", "split"]),
                 on="evento_id").filter(pl.col("split") == "train").sort("entidade_id", "pos_full"))
    mapa = {int(e): i for i, e in enumerate(seqs.ids)}
    ent = pos["entidade_id"].to_numpy()
    pf = pos["pos_full"].to_numpy()
    ini = np.r_[0, np.flatnonzero(np.diff(ent)) + 1]
    fim = np.r_[ini[1:], len(ent)]
    clientes = [(mapa[int(ent[s])], pf[s:f]) for s, f in zip(ini, fim) if f - s >= 2]
    print(f"{len(clientes):,} clientes com >= 2 eventos de treino ({time.time() - t0:.0f}s)", flush=True)

    model = construir_modelo(arch, vocab).cuda()
    nucleo = model.nucleo
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.1, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 300) * max(0.05, 1 - s / a.passos))

    def visao(c):
        e, fins = clientes[c]
        n = len(fins)
        comp = rng.randint(max(1, n // 4), n + 1)
        i = rng.randint(0, n - comp + 1)
        f = fins[i + comp - 1] + 1                  # exclusivo: termina no último token do bloco
        s = 1 if i == 0 else fins[i - 1] + 1        # começa no bloco i (BOS fora)
        s = max(s, f - a.max_tokens)
        return seqs.full(e)[s:f]

    out = run_dir(config, a.saida)
    os.makedirs(os.path.join(out, "pretreino", "checkpoints"), exist_ok=True)
    shutil.copy(os.path.join(base, a.arch_de), os.path.join(out, "config.yaml"))
    log, perdas = [], []
    model.train()
    for passo in range(a.passos):
        lote = rng.choice(len(clientes), a.lote, replace=False)
        vs = [visao(c) for c in lote] + [visao(c) for c in lote]
        T = max(len(v) for v in vs)
        ids = np.full((len(vs), T), pad, np.int64)
        for r, v in enumerate(vs):
            ids[r, :len(v)] = v
        fim_v = torch.tensor([len(v) - 1 for v in vs], device="cuda")
        ids_t = torch.from_numpy(ids).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = nucleo.estados(ids_t)
        z = F.normalize(h[torch.arange(len(vs), device="cuda"), fim_v].float(), dim=-1)
        sim = z[:a.lote] @ z[a.lote:].T / a.tau
        alvo = torch.arange(a.lote, device="cuda")
        perda = 0.5 * (F.cross_entropy(sim, alvo) + F.cross_entropy(sim.T, alvo))
        opt.zero_grad(set_to_none=True)
        perda.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        perdas.append(float(perda))
        if (passo + 1) % 100 == 0:
            acc = float((sim.argmax(1) == alvo).float().mean())
            reg = {"passo": passo + 1, "perda": float(np.mean(perdas[-100:])), "acerto_lote": acc, "s": round(time.time() - t0, 1)}
            log.append(reg)
            print(json.dumps(reg), flush=True)
        if (passo + 1) % 1000 == 0 or passo == a.passos - 1:
            torch.save(model.pesos(), os.path.join(out, "pretreino", "checkpoints", "melhor.pt"))
    rel = {"passos": a.passos, "lote": a.lote, "max_tokens": a.max_tokens, "lr": a.lr, "tau": a.tau,
           "n_clientes": len(clientes), "segundos": round(time.time() - t0, 1), "log": log}
    json.dump(rel, open(os.path.join(out, "pretreino", "relatorio_coles.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
