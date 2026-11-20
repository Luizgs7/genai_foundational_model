"""F2-H3 — ranqueadores, MAP@12 e bootstrap (recomendação do próximo artigo).

Braços (todos sobre os mesmos candidatos de f2h3_candidatos.py):
  R0 populares da semana anterior
  R1 recompra (mais recente primeiro) + populares
  R2 LightGBM LambdaRank com as features tradicionais
  R3 backbone sozinho: ordena pelo log da probabilidade do artigo como próxima compra
     (lp_total menos log do nº de artigos com o mesmo perfil de tokens)
  R4 híbrido: R2 + features do backbone
  R5 LightGBM só com as features do backbone

Fase `val`: treina nas semanas de treino da validação, escolhe num_leaves e o nº
de árvores pelo MAP@12 da semana de validação. Fase `test`: congela essas
escolhas, treina nas semanas anteriores ao teste e avalia a semana de teste uma
vez. 3 sementes; cada semente s usa o backbone final_s<s>. Listas com menos de
12 itens são completadas pelos populares. IC 95% da diferença de MAP@12 por
bootstrap Poisson por cliente (média das sementes por cliente).

Uso: python3 pipeline/common/f2h3_ranquear.py configs/hm_clientes.yaml --fase val|test [--sementes 0 1 2]
Gera runs/<nome>/f2h3/resultado_<fase>.json (+ escolhas_val.json na fase val)
"""

import argparse
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, "pipeline/common")
from schema import carregar_config, run_dir  # noqa: E402

SEMANAS = {"val": (["2020-02-20", "2020-02-27"], "2020-03-05"),
           "test": (["2020-09-02", "2020-09-09"], "2020-09-16")}
K = 12
GRADE_FOLHAS = [31, 127]
MAX_ARVORES = 1500
B_BOOT = 1000


def ler_semana(d, semente, outros, rep="v1"):
    c = pl.read_parquet(os.path.join(d, "candidatos.parquet"))
    c = c.with_columns(pl.col(pl.Float64).cast(pl.Float32))
    if rep == "v2":
        # F3-H6: pontuação do motor v2 (um backbone só; f3h6_pontuar_v2.py). Artigo fora do catálogo do v2: nulo.
        v = pl.read_parquet(os.path.join(d, "v2.parquet")).select("cliente", "artigo", pl.col("logp").alias("v2_lp"))
        c = c.join(v, on=["cliente", "artigo"], how="left")
        c = c.with_columns((pl.col("v2_lp") - pl.col("v2_lp").max().over("cliente")).alias("v2_lp_rel"),
                           pl.col("v2_lp").rank("ordinal", descending=True).over("cliente").alias("v2_rank"))
        return c.sort("cliente", "artigo")
    fm = pl.read_parquet(os.path.join(d, f"fm_s{semente}.parquet"))
    af = pl.read_parquet(os.path.join(d, "artigos_fm.parquet"))
    perfil = [f"t{k}" for k in (1, 2, 3, 4, 5, 7, 8)]
    n_perfil = af.group_by(perfil).len().rename({"len": "fm_n_perfil"})
    af = af.join(n_perfil, on=perfil).select("artigo", "fm_n_perfil", (pl.col("t7") == outros).cast(pl.Int8).alias("fm_outros"))
    c = c.join(fm, on=["cliente", "artigo"], how="left").join(af, on="artigo", how="left")
    c = c.with_columns((pl.col("lp_total") - pl.col("fm_n_perfil").cast(pl.Float32).log()).alias("fm_lp_norm"))
    c = c.with_columns((pl.col("fm_lp_norm") - pl.col("fm_lp_norm").max().over("cliente")).alias("fm_lp_norm_rel"),
                       (pl.col("lp_total") - pl.col("lp_total").max().over("cliente")).alias("fm_lp_total_rel"),
                       pl.col("fm_lp_norm").rank("ordinal", descending=True).over("cliente").alias("fm_rank"))
    return c.sort("cliente", "artigo")


def colunas(c):
    fm = [x for x in c.columns if x.startswith(("lp_", "fm_", "v2_"))]
    trad = [x for x in c.columns if x not in fm and x not in ("cliente", "artigo", "rotulo")]
    return {"R2": trad, "R4": trad + fm, "R5": fm}


def map12(listas, alvos_por_cli, clientes):
    """AP@12 por cliente (ordem de `clientes`)."""
    ap = np.zeros(len(clientes))
    for i, cli in enumerate(clientes):
        reais = alvos_por_cli[cli]
        acertos, s = 0, 0.0
        for k, art in enumerate(listas.get(cli, [])[:K]):
            if art in reais:
                acertos += 1
                s += acertos / (k + 1)
        ap[i] = s / min(len(reais), K)
    return ap


def listas_de_score(c, score, populares):
    """top-12 por cliente pelo score (desc), completado pelos populares."""
    df = c.select("cliente", "artigo").with_columns(pl.Series("s", score)).filter(pl.col("s").is_not_nan() & pl.col("s").is_not_null())
    top = df.sort(["cliente", "s"], descending=[False, True]).group_by("cliente", maintain_order=True).head(K)
    out = {}
    for cli, arts in top.group_by("cliente").agg(pl.col("artigo")).iter_rows():
        out[cli] = list(arts)
    return completar(out, populares)


def completar(listas, populares, clientes=None):
    for cli in (clientes if clientes is not None else list(listas)):
        l = listas.get(cli, [])
        if len(l) < K:
            vistos = set(l)
            l = l + [p for p in populares[:K + len(l)] if p not in vistos][:K - len(l)]
        listas[cli] = l
    return listas


def treinar(X, y, grupos, folhas, n_arv, semente, Xv=None, yv=None, gv=None):
    m = lgb.LGBMRanker(objective="lambdarank", n_estimators=n_arv, learning_rate=0.05, num_leaves=folhas,
                       min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                       random_state=semente, verbose=-1, lambdarank_truncation_level=20)
    kw = {}
    if Xv is not None:
        kw = dict(eval_set=[(Xv, yv)], eval_group=[gv], eval_at=[K],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    m.fit(X, y, group=grupos, **kw)
    return m


def preparar_treino(base, semanas, semente, outros, rep="v1"):
    partes = []
    for w in semanas:
        c = ler_semana(os.path.join(base, w), semente, outros, rep)
        com_pos = c.group_by("cliente").agg(pl.col("rotulo").max().alias("_p")).filter(pl.col("_p") > 0).select("cliente")
        partes.append(c.join(com_pos, on="cliente", how="semi").with_columns(pl.lit(w).alias("_w")))
    t = pl.concat(partes, how="diagonal_relaxed").sort("_w", "cliente", "artigo")
    grupos = t.group_by(["_w", "cliente"], maintain_order=True).len()["len"].to_numpy()
    return t, grupos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--fase", required=True, choices=["val", "test"])
    ap.add_argument("--sementes", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--rep", default="v1", choices=["v1", "v2"],
                    help="v2 (F3-H6): features do motor v2 em vez das do backbone v1; grava *_v2.json")
    a = ap.parse_args()
    suf = "" if a.rep == "v1" else "_" + a.rep
    pontuacao = "fm_lp_norm" if a.rep == "v1" else "v2_lp"
    t0 = time.time()
    config = carregar_config(a.config)
    base = run_dir(config, "f2h3")
    vocab = json.load(open(os.path.join(run_dir(config, "vocabulario"), "vocab.json"), encoding="utf-8"))
    outros = vocab["ARTIGO_OUTROS"]
    sem_treino, sem_alvo = SEMANAS[a.fase]
    d_alvo = os.path.join(base, sem_alvo)
    alvos = pl.read_parquet(os.path.join(d_alvo, "alvos.parquet"))
    alvos_por_cli = {cli: set(arts) for cli, arts in alvos.group_by("cliente").agg(pl.col("artigo")).iter_rows()}
    clientes = sorted(alvos_por_cli)
    populares = pl.read_parquet(os.path.join(d_alvo, "populares.parquet")).sort("pop_rank")["artigo"].to_list()
    escolhas_path = os.path.join(base, f"escolhas_val{suf}.json")
    escolhas = json.load(open(escolhas_path)) if a.fase == "test" else {}

    ap_por = {}
    # braços sem treino
    c0 = ler_semana(d_alvo, a.sementes[0], outros, a.rep)
    ap_por["R0"] = map12(completar({}, populares, clientes), alvos_por_cli, clientes)
    s1 = np.where(c0["src_rep"].to_numpy() == 1, 1e6 - c0["par_dias_desde_ult"].fill_null(1e5).to_numpy(), np.nan)
    ap_por["R1"] = map12(listas_de_score(c0, s1, populares), alvos_por_cli, clientes)
    teto = float(c0["rotulo"].sum() / alvos.height)
    print(f"R0 {ap_por['R0'].mean():.5f}  R1 {ap_por['R1'].mean():.5f}  cobertura-teto {teto:.4f}", flush=True)
    del c0

    por_semente = {}
    for s in a.sementes:
        c = ler_semana(d_alvo, s, outros, a.rep)
        r = {"R3": map12(listas_de_score(c, c[pontuacao].to_numpy(), populares), alvos_por_cli, clientes)}
        cols = colunas(c)
        t, grupos = preparar_treino(base, sem_treino, s, outros, a.rep)
        gv = c.group_by("cliente", maintain_order=True).len()["len"].to_numpy()
        for braco in ("R2", "R4", "R5"):
            X = t.select(cols[braco]).to_numpy().astype(np.float32)
            y = t["rotulo"].to_numpy()
            Xv = c.select(cols[braco]).to_numpy().astype(np.float32)
            if a.fase == "val":
                melhor = None
                for folhas in GRADE_FOLHAS:
                    m = treinar(X, y, grupos, folhas, MAX_ARVORES, s, Xv, c["rotulo"].to_numpy(), gv)
                    apv = map12(listas_de_score(c, m.predict(Xv), populares), alvos_por_cli, clientes)
                    print(f"  s{s} {braco} folhas={folhas} árvores={m.best_iteration_} MAP@12={apv.mean():.5f}", flush=True)
                    if melhor is None or apv.mean() > melhor[0]:
                        melhor = (apv.mean(), folhas, int(m.best_iteration_), apv)
                escolhas.setdefault(braco, {})[str(s)] = {"num_leaves": melhor[1], "n_estimators": melhor[2]}
                r[braco] = melhor[3]
            else:
                e = escolhas[braco][str(s)]
                m = treinar(X, y, grupos, e["num_leaves"], e["n_estimators"], s)
                r[braco] = map12(listas_de_score(c, m.predict(Xv), populares), alvos_por_cli, clientes)
                if braco == "R4":
                    imp = dict(zip(cols[braco], m.booster_.feature_importance("gain")))
                    tot = sum(imp.values())
                    r["_imp_fm_R4"] = sum(v for k, v in imp.items() if k.startswith(("lp_", "fm_", "v2_"))) / tot
            print(f"s{s} {braco} MAP@12={r[braco].mean():.5f} ({time.time() - t0:.0f}s)", flush=True)
        por_semente[s] = r
        del t, X, Xv, c

    if a.fase == "val":
        json.dump(escolhas, open(escolhas_path, "w"), indent=1)
    bracos = ["R2", "R3", "R4", "R5"]
    for b in bracos:
        ap_por[b] = np.mean([por_semente[s][b] for s in a.sementes], axis=0)
    res = {"fase": a.fase, "semana_alvo": sem_alvo, "semanas_treino": sem_treino, "n_clientes": len(clientes),
           "cobertura_teto": teto, "map12": {b: float(v.mean()) for b, v in ap_por.items()},
           "map12_por_semente": {b: [float(por_semente[s][b].mean()) for s in a.sementes] for b in bracos},
           "participacao_ganho_fm_R4": [float(por_semente[s].get("_imp_fm_R4", np.nan)) for s in a.sementes],
           "deltas_vs_R2": {}}
    rng = np.random.RandomState(0)
    W = rng.poisson(1.0, (B_BOOT, len(clientes))).astype(np.float64)
    for b in ["R0", "R1", "R3", "R4", "R5"]:
        dlt = ap_por[b] - ap_por["R2"]
        boot = (W @ dlt) / W.sum(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        res["deltas_vs_R2"][b] = {"delta": float(dlt.mean()), "relativo": float(dlt.mean() / ap_por["R2"].mean()),
                                  "ic95": [float(lo), float(hi)], "significativo": bool(lo > 0 or hi < 0)}
    res["rep"] = a.rep
    with open(os.path.join(base, f"resultado_{a.fase}{suf}.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res["map12"], indent=1))
    print(json.dumps(res["deltas_vs_R2"], indent=1))


if __name__ == "__main__":
    main()
