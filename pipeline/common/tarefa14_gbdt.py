"""GBDT (LightGBM / XGBoost) sobre as âncoras downstream — H5/H6/H7 do
experimento H&M. Mesmas âncoras, splits e rótulos da fusão DCNv2, para a
comparação ser justa.

Conjuntos de features (`--features`):
  trad          features tradicionais (features_tradicionais.py) — baseline H5
  emb           só o embedding do modelo fundacional (bruto)          — H6
  emb_pca       só o embedding, reduzido por PCA (k=--pca_k, ajustado no treino) — H6
  hibrido       tradicionais + embedding PCA                           — H7
  hibrido_bruto tradicionais + embedding bruto                         — H7
`--sem_cadastro` remove os campos de cadastro (H9).

Protocolo:
  1. HPO (Optuna TPE, `--n_trials`) numa amostra do treino
     (`--amostra_hpo`) com early stopping numa amostra da validação;
     objetivo = AUC (binária) ou log loss (multiclasse) na validação.
  2. Modelo final com os melhores hiperparâmetros no treino inteiro,
     early stopping na validação inteira, repetido para cada semente
     (`--seeds`) -> média ± desvio na validação.
  3. Teste só com `--avaliar_teste` (H10: avaliação única).
Grava métricas, hiperparâmetros, importância (gain, com a participação das
features de embedding) e predições de val/teste para bootstrap.

Uso: python3 pipeline/common/tarefa14_gbdt.py <config.yaml> --features trad --modelo lgbm
         [--tarefas churn,next_category] [--n_trials 40] [--amostra_hpo 500000]
         [--seeds 0,1,2] [--emb fusao] [--pca_k 32] [--saida <nome>] [--sem_cadastro] [--avaliar_teste] [--params_de <saida>]
Gera runs/<nome>/gbdt/<saida>/{relatorio.json, predicoes_<split>_<tarefa>.parquet}.
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
from schema import caminho_tabela, carregar_config, ler_tabela, mem, run_dir, tarefas_ativas  # noqa: E402
from tarefa8_rotulos import accuracy, auc_roc, f1_macro  # noqa: E402

CADASTRO_SUFIXOS = ("_cod", "_freq")
# hiperparâmetros fixos da sonda das ablações (--n_trials 0): mesmos pra todas
PARAMS_FIXOS = {
    "lgbm": {"learning_rate": 0.1, "num_leaves": 63, "min_child_samples": 200, "feature_fraction": 0.8,
             "bagging_fraction": 0.8, "lambda_l1": 0.0, "lambda_l2": 1.0},
    "xgb": {"eta": 0.1, "max_depth": 7, "min_child_weight": 10.0, "subsample": 0.8, "colsample_bytree": 0.8,
            "lambda": 1.0, "alpha": 0.0},
}


def campos_cadastro(config):
    return set(config.get("campos_estaticos_numericos", [])) | \
        {f"{c}_cod" for c in config.get("campos_estaticos_categoricos", [])} | \
        {f"{c}_freq" for c in config.get("features_tradicionais", {}).get("estaticos_frequencia", [])}


def carregar(config, args, splits):
    fdir = run_dir(config, "features_tradicionais")
    with open(os.path.join(fdir, "features_info.json"), encoding="utf-8") as f:
        info = json.load(f)
    feats = pl.read_parquet(os.path.join(fdir, "features.parquet")).filter(pl.col("split").is_in(splits))
    nomes = [c for c in info["features"]]
    categoricas = [c for c in info["categoricas"]]
    if args.sem_cadastro:
        fora = campos_cadastro(config)
        nomes = [c for c in nomes if c not in fora]
        categoricas = [c for c in categoricas if c not in fora]

    usar_trad = args.features in ("trad", "hibrido", "hibrido_bruto")
    usar_emb = args.features in ("emb", "emb_pca", "hibrido", "hibrido_bruto")
    X_trad = feats.select(nomes).to_numpy().astype(np.float32) if usar_trad else np.zeros((feats.height, 0), np.float32)
    meta = feats.select(["evento_id", "entidade_id", "split"] + [c for c in feats.columns if c.endswith(("_rotulo", "_baseline"))]).to_pandas()
    del feats
    nomes_X = nomes if usar_trad else []
    categoricas = categoricas if usar_trad else []

    if usar_emb:
        edir = run_dir(config, args.emb)
        emb = np.load(os.path.join(edir, "embeddings_eventos.npy"), mmap_mode="r")
        idx = ler_tabela(caminho_tabela(config, args.emb, "embeddings_eventos_index"), ["evento_id", "row_idx"])
        pos = pd.Series(idx["row_idx"].to_numpy(), index=idx["evento_id"].to_numpy())
        linhas = pos.reindex(meta["evento_id"].to_numpy())
        if linhas.isna().any():
            raise ValueError(f"{int(linhas.isna().sum())} âncoras sem embedding em {edir}")
        linhas = linhas.to_numpy(dtype=np.int64)
        ordem = np.argsort(linhas)
        E = np.empty((len(linhas), emb.shape[1]), dtype=np.float32)
        E[ordem] = np.asarray(emb[linhas[ordem]], dtype=np.float32)   # leitura sequencial do memmap
        if args.features in ("emb_pca", "hibrido"):
            from sklearn.decomposition import PCA

            treino = (meta["split"] == "train").to_numpy()
            amostra = np.flatnonzero(treino)
            amostra = np.random.RandomState(0).choice(amostra, min(len(amostra), 500_000), replace=False)
            pca = PCA(n_components=args.pca_k, svd_solver="randomized", random_state=0).fit(E[amostra])
            E = pca.transform(E).astype(np.float32)
            print(f"PCA k={args.pca_k}: variância explicada {pca.explained_variance_ratio_.sum():.3f}")
            nomes_e = [f"emb_pca_{i}" for i in range(args.pca_k)]
        else:
            nomes_e = [f"emb_{i}" for i in range(E.shape[1])]
        X = np.hstack([X_trad, E])
        nomes_X = nomes_X + nomes_e
        del E
    else:
        X = X_trad
    return X, nomes_X, categoricas, meta


def tipo_tarefa(y):
    v = pd.Series(y).dropna()
    if set(v.unique()) <= {0.0, 1.0}:
        return "binaria", None
    classes = sorted(v.unique(), key=str)
    return "multiclasse", {c: i for i, c in enumerate(classes)}


def metricas(tipo, y, p, classes_inv=None):
    if tipo == "binaria":
        from sklearn.metrics import average_precision_score

        return {"auc": auc_roc(pd.Series(y), pd.Series(p)), "pr_auc": float(average_precision_score(y, p))}
    from sklearn.metrics import log_loss

    pred = p.argmax(axis=1)
    return {"accuracy": accuracy(pd.Series(y), pd.Series(pred)), "f1_macro": f1_macro(pd.Series(y), pd.Series(pred)),
            "log_loss": float(log_loss(y, np.clip(p, 1e-7, 1), labels=np.arange(p.shape[1])))}


# ---------------------------------------------------------------- modelos

def treinar(modelo, tipo, n_classes, params, X_tr, y_tr, X_va, y_va, cat_idx, seed, threads):
    if modelo == "lgbm":
        import lightgbm as lgb

        p = {"objective": "binary" if tipo == "binaria" else "multiclass", "verbosity": -1, "seed": seed,
             "num_threads": threads, "max_bin": 255, "bagging_freq": 1, **params}
        if tipo != "binaria":
            p["num_class"] = n_classes
        dtr = lgb.Dataset(X_tr, y_tr, categorical_feature=cat_idx, free_raw_data=True)
        dva = lgb.Dataset(X_va, y_va, reference=dtr, categorical_feature=cat_idx)
        m = lgb.train(p, dtr, num_boost_round=5000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        prever = lambda X: m.predict(X, num_iteration=m.best_iteration)  # noqa: E731
        importancia = m.feature_importance("gain")
        return prever, m.best_iteration, importancia
    import xgboost as xgb

    p = {"objective": "binary:logistic" if tipo == "binaria" else "multi:softprob",
         "eval_metric": "auc" if tipo == "binaria" else "mlogloss", "tree_method": "hist", "max_bin": 256,
         "seed": seed, "nthread": threads, **params}
    if tipo != "binaria":
        p["num_class"] = n_classes
    ft = ["c" if i in set(cat_idx) else "q" for i in range(X_tr.shape[1])]
    dtr = xgb.DMatrix(X_tr, y_tr, feature_types=ft, enable_categorical=True)
    dva = xgb.DMatrix(X_va, y_va, feature_types=ft, enable_categorical=True)
    m = xgb.train(p, dtr, num_boost_round=5000, evals=[(dva, "val")], early_stopping_rounds=100, verbose_eval=False)
    prever = lambda X: m.predict(xgb.DMatrix(X, feature_types=ft, enable_categorical=True),  # noqa: E731
                                  iteration_range=(0, m.best_iteration + 1))
    g = m.get_score(importance_type="total_gain")
    importancia = np.array([g.get(f"f{i}", 0.0) for i in range(X_tr.shape[1])])
    return prever, m.best_iteration, importancia


def espaco(trial, modelo):
    if modelo == "lgbm":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 2000, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        }
    return {
        "eta": trial.suggest_float("eta", 0.02, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 200.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "lambda": trial.suggest_float("lambda", 1e-8, 10.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 10.0, log=True),
    }


# ---------------------------------------------------------------- main

def main(args):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    t0 = time.time()
    config = carregar_config(args.config)
    splits = ["train", "val"] + (["test"] if args.avaliar_teste else [])
    X, nomes, categoricas, meta = carregar(config, args, splits)
    cat_idx = [nomes.index(c) for c in categoricas if c in nomes]
    mem("dados carregados")
    print(f"features={args.features} ({len(nomes)} colunas), modelo={args.modelo}, âncoras={len(meta):,}")

    saida = args.saida or f"{args.features}_{args.modelo}" + ("_sem_cadastro" if args.sem_cadastro else "")
    out_dir = os.path.join(run_dir(config, "gbdt"), saida)
    os.makedirs(out_dir, exist_ok=True)
    tarefas = [t.get("nome", t["tipo"]) for t in tarefas_ativas(config)]
    if args.tarefas:
        tarefas = [t for t in tarefas if t in args.tarefas.split(",")]

    rel_path = os.path.join(out_dir, "relatorio.json")
    relatorio = json.load(open(rel_path, encoding="utf-8")) if os.path.exists(rel_path) else {}
    relatorio.update({"features": args.features, "modelo": args.modelo, "n_features": len(nomes),
                      "sem_cadastro": args.sem_cadastro, "emb": args.emb, "pca_k": args.pca_k})
    relatorio.setdefault("tarefas", {})
    split = meta["split"].to_numpy()
    rng = np.random.RandomState(0)

    for nome in tarefas:
        t1 = time.time()
        y_bruto = meta[f"{nome}_rotulo"]
        tipo, mapa = tipo_tarefa(y_bruto)
        y = y_bruto.map(mapa).to_numpy(dtype=float) if mapa else y_bruto.to_numpy(dtype=float)
        n_classes = len(mapa) if mapa else 1
        valido = ~np.isnan(y)
        itr = np.flatnonzero(valido & (split == "train"))
        iva = np.flatnonzero(valido & (split == "val"))
        print(f"\n=== {nome} ({tipo}{'' if not mapa else f', {n_classes} classes'}): treino={len(itr):,} val={len(iva):,}")

        # 1) HPO em amostras
        itr_h = rng.choice(itr, min(len(itr), args.amostra_hpo), replace=False)
        iva_h = rng.choice(iva, min(len(iva), args.amostra_hpo // 2), replace=False)
        Xtr_h, ytr_h, Xva_h, yva_h = X[itr_h], y[itr_h], X[iva_h], y[iva_h]

        def objetivo(trial):
            params = espaco(trial, args.modelo)
            prever, _, _ = treinar(args.modelo, tipo, n_classes, params, Xtr_h, ytr_h, Xva_h, yva_h, cat_idx, 0, args.threads)
            m = metricas(tipo, yva_h, prever(Xva_h))
            return m["auc"] if tipo == "binaria" else m["log_loss"]

        if args.params_de:
            # H10: hiperparâmetros congelados (escolhidos na validação por outra execução)
            origem = json.load(open(os.path.join(run_dir(config, "gbdt"), args.params_de, "relatorio.json"), encoding="utf-8"))
            melhores, melhor_valor = origem["tarefas"][nome]["melhores_hiperparametros"], origem["tarefas"][nome]["hpo_melhor_valor"]
            print(f"  hiperparâmetros congelados de {args.params_de}")
        elif args.n_trials > 0:
            estudo = optuna.create_study(direction="maximize" if tipo == "binaria" else "minimize",
                                         sampler=optuna.samplers.TPESampler(seed=0))
            estudo.optimize(objetivo, n_trials=args.n_trials)
            melhores, melhor_valor = estudo.best_params, estudo.best_value
            print(f"  HPO: melhor {'AUC' if tipo == 'binaria' else 'log loss'}={melhor_valor:.4f} "
                  f"({args.n_trials} tentativas, {time.time()-t1:.0f}s)")
        else:
            melhores, melhor_valor = PARAMS_FIXOS[args.modelo], None   # sonda (ablações H8): sem HPO
        del Xtr_h, Xva_h

        # 2) final em todo o treino, várias sementes
        Xtr, ytr, Xva, yva = X[itr], y[itr], X[iva], y[iva]
        resultados, imp_total = [], None
        for seed in [int(s) for s in args.seeds.split(",")]:
            prever, n_arv, imp = treinar(args.modelo, tipo, n_classes, melhores, Xtr, ytr, Xva, yva,
                                         cat_idx, seed, args.threads)
            p_va = prever(Xva)
            r = {"seed": seed, "n_arvores": int(n_arv), "val": metricas(tipo, yva, p_va)}
            pred = pd.DataFrame({"evento_id": meta["evento_id"].to_numpy()[iva], "entidade_id": meta["entidade_id"].to_numpy()[iva]})
            if tipo == "binaria":
                pred["score"] = p_va
            else:
                for c, i in mapa.items():
                    pred[f"p_{c}"] = p_va[:, i]
            pred.to_parquet(os.path.join(out_dir, f"predicoes_val_{nome}_seed{seed}.parquet"), index=False)
            if args.avaliar_teste:
                ite = np.flatnonzero(valido & (split == "test"))
                p_te = prever(X[ite])
                r["test"] = metricas(tipo, y[ite], p_te)
                pt = pd.DataFrame({"evento_id": meta["evento_id"].to_numpy()[ite], "entidade_id": meta["entidade_id"].to_numpy()[ite]})
                if tipo == "binaria":
                    pt["score"] = p_te
                else:
                    for c, i in mapa.items():
                        pt[f"p_{c}"] = p_te[:, i]
                pt.to_parquet(os.path.join(out_dir, f"predicoes_test_{nome}_seed{seed}.parquet"), index=False)
            resultados.append(r)
            imp_total = imp if imp_total is None else imp_total + imp
            print(f"  seed {seed}: {r['val']}" + (f" | TESTE {r['test']}" if "test" in r else "") + f"  ({n_arv} árvores)")
        del Xtr, Xva

        chaves = list(resultados[0]["val"])
        resumo = {k: {"media": float(np.mean([r["val"][k] for r in resultados])),
                      "desvio": float(np.std([r["val"][k] for r in resultados]))} for k in chaves}
        imp_norm = imp_total / max(imp_total.sum(), 1e-12)
        top = sorted(zip(nomes, imp_norm), key=lambda x: -x[1])[:25]
        share_emb = float(sum(v for n, v in zip(nomes, imp_norm) if n.startswith("emb_")))
        base = None
        if tipo == "binaria":
            base = auc_roc(meta[f"{nome}_rotulo"].iloc[iva], meta[f"{nome}_baseline"].iloc[iva])
        relatorio["tarefas"][nome] = {
            "tipo": tipo, "classes": list(map(str, mapa)) if mapa else None,
            "n_treino": int(len(itr)), "n_val": int(len(iva)),
            "melhores_hiperparametros": melhores, "hpo_melhor_valor": melhor_valor,
            "val_resumo": resumo, "baseline_causal_val_auc": base, "por_semente": resultados,
            "participacao_importancia_embedding": share_emb,
            "top_importancia": [[n, float(v)] for n, v in top],
            "tempo_segundos": time.time() - t1,
        }
        with open(rel_path, "w", encoding="utf-8") as f:
            json.dump(relatorio, f, ensure_ascii=False, indent=2, default=str)
        print(f"  val: " + "  ".join(f"{k}={v['media']:.4f}±{v['desvio']:.4f}" for k, v in resumo.items())
              + (f"  (baseline causal AUC={base:.4f})" if base is not None else "")
              + f"  | importância do embedding: {share_emb:.1%}")

    print(f"\nGravado {rel_path} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--features", required=True, choices=["trad", "emb", "emb_pca", "hibrido", "hibrido_bruto"])
    ap.add_argument("--modelo", required=True, choices=["lgbm", "xgb"])
    ap.add_argument("--tarefas", default=None)
    ap.add_argument("--n_trials", type=int, default=40)
    ap.add_argument("--amostra_hpo", type=int, default=500_000)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--emb", default="fusao")
    ap.add_argument("--pca_k", type=int, default=32)
    ap.add_argument("--saida", default=None)
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--sem_cadastro", action="store_true")
    ap.add_argument("--avaliar_teste", action="store_true")
    ap.add_argument("--params_de", default=None, help="saida de outra execução cujos melhores hiperparâmetros são reusados (sem HPO)")
    main(ap.parse_args())
