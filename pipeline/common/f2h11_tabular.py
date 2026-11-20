"""F2-H11 — modelo tabular pré-treinado (TabPFN, TabICL) como régua, contra o LightGBM.

Sobre as features tradicionais (97 colunas, ou 129 com o embedding PCA-32 no
modo híbrido), na amostra de clientes da F2-H10. Para cada tarefa e tamanho de
contexto N (âncoras de treino sorteadas da amostra, semente fixa):

  tabpfn  TabPFNClassifier: o treino é o contexto; sem otimização de hiperparâmetros
  tabicl  TabICLClassifier: idem, pesos abertos
  lgbm    LightGBM nas MESMAS N linhas, hiperparâmetros congelados da H10 e
          parada antecipada numa amostra fixa da validação

Prevê as âncoras de teste da amostra e grava no formato do GBDT
(runs/<nome>/gbdt/f2h11_<modelo>_<feat>_n<N>/predicoes_test_<tarefa>_seed0.parquet),
avaliado depois por f2h10_avaliar.py contra o mesmo GBDT tradicional da F2-H10.

Entrada: runs/<nome>/f2h11/amostra_features.parquet (gerado por --preparar, local).
Uso: python3 pipeline/common/f2h11_tabular.py configs/hm_clientes.yaml --preparar   (local)
     python3 pipeline/common/f2h11_tabular.py configs/hm_clientes.yaml --modelo tabpfn --n 50000 [--hibrido]
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, "pipeline/common")
from schema import carregar_config, run_dir  # noqa: E402

TAREFAS = ["churn", "proxima_online", "next_category", "ltv_90d"]
N_VAL = 50_000
LOTE_PRED = 20_000


def preparar(config):
    fdir = run_dir(config, "features_tradicionais")
    info = json.load(open(os.path.join(fdir, "features_info.json"), encoding="utf-8"))
    amostra = pl.read_parquet(os.path.join(run_dir(config, "f2h10"), "textos.parquet"), columns=["entidade_id"]).unique()
    cols = ["evento_id", "entidade_id", "split"] + info["features"] + [f"{t}_rotulo" for t in TAREFAS]
    f = pl.scan_parquet(os.path.join(fdir, "features.parquet")).select(cols).join(amostra.lazy(), on="entidade_id", how="semi").collect()
    z = np.load(os.path.join(run_dir(config, "caches"), "pca32_h3.npz"))
    pca = pl.DataFrame({"evento_id": z["evento_id"]}).with_columns(
        [pl.Series(f"emb_pca_{i}", z["E"][:, i]) for i in range(z["E"].shape[1])])
    f = f.join(pca, on="evento_id", how="left")
    out = run_dir(config, "f2h11")
    os.makedirs(out, exist_ok=True)
    f.write_parquet(os.path.join(out, "amostra_features.parquet"))
    json.dump({"features": info["features"], "categoricas": info["categoricas"],
               "emb": [f"emb_pca_{i}" for i in range(z["E"].shape[1])]}, open(os.path.join(out, "info.json"), "w"), indent=1)
    print(f.group_by("split").len(), f"; PCA ausente: {f['emb_pca_0'].null_count()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--preparar", action="store_true")
    ap.add_argument("--modelo", choices=["tabpfn", "tabicl", "lgbm"])
    ap.add_argument("--n", type=int, default=50_000)
    ap.add_argument("--hibrido", action="store_true")
    ap.add_argument("--semente", type=int, default=0)
    ap.add_argument("--tarefas", default=",".join(TAREFAS))
    a = ap.parse_args()
    config = carregar_config(a.config)
    if a.preparar:
        return preparar(config)
    t0 = time.time()
    d = run_dir(config, "f2h11")
    info = json.load(open(os.path.join(d, "info.json")))
    feats = info["features"] + (info["emb"] if a.hibrido else [])
    df = pl.read_parquet(os.path.join(d, "amostra_features.parquet"))
    nome = f"f2h11_{a.modelo}_{'hibrido' if a.hibrido else 'trad'}_n{a.n}"
    out = os.path.join(run_dir(config, "gbdt"), nome)
    os.makedirs(out, exist_ok=True)
    rel = {"modelo": a.modelo, "n": a.n, "features": len(feats), "tarefas": {}}
    params_h10 = json.load(open(os.path.join(run_dir(config, "gbdt"), "h10_trad_lgbm", "relatorio.json")))["tarefas"]
    te = df.filter(pl.col("split") == "test")
    Xte = te.select(feats).to_numpy().astype(np.float32)
    parcial = os.path.join(out, "relatorio_parcial.json")
    if os.path.exists(parcial):
        rel = json.load(open(parcial))
    for t in a.tarefas.split(","):
        if t in rel["tarefas"] and os.path.exists(os.path.join(out, f"predicoes_test_{t}_seed0.parquet")):
            print(f"{nome} {t}: já feita, pulando", flush=True)
            continue
        t1 = time.time()
        tr = df.filter((pl.col("split") == "train") & pl.col(f"{t}_rotulo").is_not_null())
        rng = np.random.RandomState(a.semente)
        idx = np.sort(rng.choice(tr.height, min(a.n, tr.height), replace=False))
        tr = tr[idx]
        y_raw = tr[f"{t}_rotulo"].to_numpy()
        binaria = set(np.unique(y_raw.astype(str))) <= {"0.0", "1.0"}
        classes = sorted(np.unique(y_raw.astype(str)), key=str) if not binaria else None
        y = y_raw.astype(float).astype(int) if binaria else np.array([classes.index(v) for v in y_raw.astype(str)])
        X = tr.select(feats).to_numpy().astype(np.float32)
        if a.modelo == "lgbm":
            import lightgbm as lgb
            va = df.filter((pl.col("split") == "val") & pl.col(f"{t}_rotulo").is_not_null())
            va = va[np.sort(np.random.RandomState(0).choice(va.height, min(N_VAL, va.height), replace=False))]
            yv_raw = va[f"{t}_rotulo"].to_numpy()
            yv = yv_raw.astype(float).astype(int) if binaria else np.array([classes.index(v) if v in classes else -1 for v in yv_raw.astype(str)])
            ok = yv >= 0
            p = dict(params_h10[t]["melhores_hiperparametros"])
            p.update({"objective": "binary" if binaria else "multiclass", "verbose": -1, "seed": a.semente,
                      "bagging_freq": 1, "num_threads": 8})
            if not binaria:
                p["num_class"] = len(classes)
            m = lgb.train(p, lgb.Dataset(X, y), num_boost_round=3000,
                          valid_sets=[lgb.Dataset(va.select(feats).to_numpy().astype(np.float32)[ok], yv[ok])],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
            prever = lambda Z: m.predict(Z, num_iteration=m.best_iteration)  # noqa: E731
            extra = {"n_arvores": int(m.best_iteration)}
        else:
            if a.modelo == "tabpfn":
                from tabpfn import TabPFNClassifier
                m = TabPFNClassifier(device="cuda", random_state=a.semente, ignore_pretraining_limits=True)
            else:
                from tabicl import TabICLClassifier
                m = TabICLClassifier(device="cuda", random_state=a.semente)
            m.fit(X, y)
            prever = m.predict_proba
            extra = {}
        partes = [prever(Xte[i:i + LOTE_PRED]) for i in range(0, len(Xte), LOTE_PRED)]
        P = np.concatenate(partes)
        pred = te.select("evento_id", "entidade_id").to_pandas()
        if binaria:
            pred["score"] = P if P.ndim == 1 else P[:, 1]
        else:
            for k, c in enumerate(classes):
                pred[f"p_{c}"] = P[:, k]
        pred.to_parquet(os.path.join(out, f"predicoes_test_{t}_seed0.parquet"), index=False)
        rel["tarefas"][t] = {"n_contexto": int(len(y)), "segundos": round(time.time() - t1, 1), **extra}
        print(f"{nome} {t}: {rel['tarefas'][t]}", flush=True)
        json.dump(rel, open(parcial, "w"), indent=1)  # retomada por tarefa (o de 300 mil leva horas por tarefa)
    rel["segundos"] = round(time.time() - t0, 1)
    json.dump(rel, open(os.path.join(out, "relatorio.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
