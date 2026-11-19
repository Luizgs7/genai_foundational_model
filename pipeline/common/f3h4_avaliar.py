"""F3-H4 — MAP@12 da recomendação direta do motor v2 no protocolo da F2-H3.

Braços (mesmos clientes, alvos e populares da F2-H3; listas com menos de 12 completadas pelos populares):
  R1  recompra (mais recente primeiro) + populares, exatamente como em f2h3_ranquear.py
  M   só o modelo: os 12 artigos de maior log-probabilidade no catálogo inteiro
  MH  modelo + histórico do próprio cliente:
        pontuação = logp + beta * log(1 + nº de compras do artigo) + gama / (1 + dias desde a última / 7)
      (só os artigos já comprados recebem o termo). beta e gama saem de uma grade na semana de
      validação e ficam congelados no teste.

Critério (PLANO_FASE3.md, F3-H4): acima do R1 (0,0229 no teste); empate com pelo menos 95% do R2
(0,0324); forma forte acima do R2. O R2 vem de runs/<nome>/f2h3/resultado_test.json (só a média).

Uso: python3 pipeline/common/f3h4_avaliar.py <config.yaml> --val f3h4_rec_val_s0 --test f3h4_rec_test_s0 f3h4_rec_test_s1 ...
"""

import argparse
import json
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, "pipeline/common")
from schema import carregar_config, run_dir  # noqa: E402

SEMANA = {"val": "2020-03-05", "test": "2020-09-16"}
K = 12
BETAS = [0, 0.5, 1, 2, 3, 5, 8, 12]
GAMAS = [0, 1, 2, 4, 8]
B_BOOT = 300


def map12(listas, alvos, clientes):
    ap = np.zeros(len(clientes))
    for i, c in enumerate(clientes):
        reais, acertos, s = alvos[c], 0, 0.0
        for k, art in enumerate(listas.get(c, [])[:K]):
            if art in reais:
                acertos += 1
                s += acertos / (k + 1)
        ap[i] = s / min(len(reais), K)
    return ap


def completar(listas, populares, clientes):
    out = {}
    for c in clientes:
        l = list(listas.get(c, []))
        if len(l) < K:
            vistos = set(l)
            l += [p for p in populares[:2 * K] if p not in vistos][:K - len(l)]
        out[c] = l
    return out


def top12(df, score):
    t = (df.with_columns(score.alias("s")).select("cliente", "artigo", "s").sort(["cliente", "s"], descending=[False, True])
         .group_by("cliente", maintain_order=True).head(K).group_by("cliente", maintain_order=True).agg("artigo"))
    return dict(zip(t["cliente"].to_list(), t["artigo"].to_list()))


def pontuacao(beta, gama):
    hist = pl.col("n_compras") > 0
    return pl.col("logp") + pl.when(hist).then(beta * pl.col("n_compras").cast(pl.Float64).log1p()
                                               + gama / (1 + pl.col("dias_desde_ult") / 7)).otherwise(0.0)


def semana(config, fase):
    d = os.path.join(run_dir(config, "f2h3"), SEMANA[fase])
    alvos = pl.read_parquet(os.path.join(d, "alvos.parquet"))
    alvos = {c: set(a) for c, a in alvos.group_by("cliente").agg("artigo").iter_rows()}
    clientes = sorted(alvos)
    pop = pl.read_parquet(os.path.join(d, "populares.parquet")).sort("pop_rank")["artigo"].to_list()
    cand = pl.read_parquet(os.path.join(d, "candidatos.parquet"), columns=["cliente", "artigo", "src_rep", "par_dias_desde_ult"])
    rep = cand.filter(pl.col("src_rep") == 1)
    r1 = top12(rep, 1e6 - pl.col("par_dias_desde_ult").fill_null(1e5))
    return alvos, clientes, pop, map12(completar(r1, pop, clientes), alvos, clientes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--val", nargs="+", required=True, help="pastas de f3h4_recomendar.py na semana de validação")
    ap.add_argument("--test", nargs="+", required=True, help="pastas na semana de teste (uma por semente)")
    ap.add_argument("--saida", default="resultado")
    a = ap.parse_args()
    config = carregar_config(a.config)
    res = {"semanas": SEMANA, "grade": {"beta": BETAS, "gama": GAMAS}}

    # ---- validação: grade de beta e gama
    alvos, clientes, pop, ap_r1 = semana(config, "val")
    recs = [pl.read_parquet(os.path.join(run_dir(config, p), f"rec_{SEMANA['val']}.parquet")) for p in a.val]
    grade = {}
    for b in BETAS:
        for g in GAMAS:
            aps = [map12(completar(top12(r, pontuacao(b, g)), pop, clientes), alvos, clientes) for r in recs]
            grade[f"{b}|{g}"] = float(np.mean(aps, axis=0).mean())
            print(f"val beta={b} gama={g} MAP@12={grade[f'{b}|{g}']:.5f}", flush=True)
    melhor = max(grade, key=grade.get)
    beta, gama = (float(x) for x in melhor.split("|"))
    res["val"] = {"n_clientes": len(clientes), "R1": float(ap_r1.mean()), "M": grade["0|0"], "MH": grade[melhor],
                  "beta": beta, "gama": gama, "grade": grade}
    print(f"val: R1 {ap_r1.mean():.5f}  M {grade['0|0']:.5f}  MH {grade[melhor]:.5f} (beta={beta}, gama={gama})", flush=True)

    # ---- teste: avaliado uma vez, com beta e gama congelados
    alvos, clientes, pop, ap_r1 = semana(config, "test")
    por = {"M": [], "MH": []}
    for p in a.test:
        r = pl.read_parquet(os.path.join(run_dir(config, p), f"rec_{SEMANA['test']}.parquet"))
        por["M"].append(map12(completar(top12(r, pontuacao(0, 0)), pop, clientes), alvos, clientes))
        por["MH"].append(map12(completar(top12(r, pontuacao(beta, gama)), pop, clientes), alvos, clientes))
    ap_b = {"R1": ap_r1, "M": np.mean(por["M"], axis=0), "MH": np.mean(por["MH"], axis=0)}
    r2 = json.load(open(os.path.join(run_dir(config, "f2h3"), "resultado_test.json")))["map12"]["R2"]
    W = np.random.RandomState(0).poisson(1.0, (B_BOOT, len(clientes))).astype(np.float64)
    res["test"] = {"n_clientes": len(clientes), "map12": {b: float(v.mean()) for b, v in ap_b.items()} | {"R2_f2h3": r2},
                   "map12_por_semente": {b: [float(x.mean()) for x in por[b]] for b in por}, "vs_R1": {}, "frac_R2": {}}
    for b in ("M", "MH"):
        dlt = ap_b[b] - ap_r1
        lo, hi = np.percentile((W @ dlt) / W.sum(axis=1), [2.5, 97.5])
        res["test"]["vs_R1"][b] = {"delta": float(dlt.mean()), "ic95": [float(lo), float(hi)]}
        res["test"]["frac_R2"][b] = float(ap_b[b].mean() / r2)
    print(json.dumps(res["test"], indent=1), flush=True)
    out = run_dir(config, "f3h4")
    os.makedirs(out, exist_ok=True)
    json.dump(res, open(os.path.join(out, a.saida + ".json"), "w"), indent=1)


if __name__ == "__main__":
    main()
