"""Comparação entre modelos com intervalo de confiança — H7/H9/H10 do
experimento H&M.

Lê as predições gravadas pela fusão (`tarefa13_treinar_fusao.py`,
predicoes_<split>.parquet) e pelo GBDT (`tarefa14_gbdt.py`,
predicoes_<split>_<tarefa>_seed<N>.parquet), junta com os rótulos pelas
MESMAS âncoras (evento_id) e calcula:
  - a métrica de cada modelo (AUC para binárias, accuracy para multiclasse);
  - o delta B - A com IC 95% por bootstrap POR CLIENTE (Poisson bootstrap:
    cada cliente recebe peso ~ Poisson(1) em cada réplica — respeita a
    correlação entre âncoras do mesmo cliente, que o bootstrap por linha
    ignoraria e daria IC estreito demais).
Com várias sementes, as predições são promediadas (ensemble) antes do
bootstrap; a média ± desvio por semente fica nos relatórios de cada modelo.

Uso: python3 pipeline/common/avaliacao.py <config.yaml> --tarefa churn --split val \\
        --a gbdt/trad_lgbm --b gbdt/hibrido_lgbm [--b fusao] [--B 300]
(caminhos relativos a runs/<nome>/; tipo detectado pelos arquivos presentes)
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "pipeline/common")
from schema import caminho_tabela, carregar_config, ler_tabela, run_dir  # noqa: E402


def carregar_predicoes(base_dir, tarefa, split):
    """-> DataFrame evento_id, entidade_id e `score` (binária) ou colunas p_<classe>."""
    gbdt = sorted(glob.glob(os.path.join(base_dir, f"predicoes_{split}_{tarefa}_seed*.parquet")))
    if gbdt:
        dfs = [pd.read_parquet(f) for f in gbdt]
        out = dfs[0][["evento_id", "entidade_id"]].copy()
        cols = [c for c in dfs[0].columns if c == "score" or c.startswith("p_")]
        for c in cols:
            out[c] = np.mean([d[c].to_numpy() for d in dfs], axis=0)
        return out, len(dfs)
    fus = os.path.join(base_dir, f"predicoes_{split}.parquet")
    if os.path.exists(fus):
        d = pd.read_parquet(fus)
        out = d[["evento_id", "entidade_id"]].copy()
        if f"{tarefa}_score" in d.columns:
            out["score"] = d[f"{tarefa}_score"].to_numpy()
        else:
            pref = f"{tarefa}_p_"
            for c in d.columns:
                if c.startswith(pref):
                    out["p_" + c[len(pref):]] = d[c].to_numpy()
        return out, 1
    raise FileNotFoundError(f"sem predições de {tarefa}/{split} em {base_dir}")


def auc_ponderada(y, s, w):
    """AUC com pesos por linha (empates contam meio)."""
    ordem = np.argsort(s, kind="mergesort")
    y, s, w = y[ordem], s[ordem], w[ordem]
    wp, wn = w * (y == 1), w * (y == 0)
    # agrupa empates de score
    novo = np.r_[True, s[1:] != s[:-1]]
    grupo = np.cumsum(novo) - 1
    pos_g = np.bincount(grupo, weights=wp)
    neg_g = np.bincount(grupo, weights=wn)
    neg_antes = np.r_[0.0, np.cumsum(neg_g)[:-1]]
    num = np.sum(pos_g * (neg_antes + 0.5 * neg_g))
    den = pos_g.sum() * neg_g.sum()
    return num / den if den > 0 else np.nan


def metrica(tipo, y, pred, w):
    if tipo == "binaria":
        return auc_ponderada(y, pred, w)
    return float(np.sum(w * (pred == y)) / np.sum(w))


def comparar(config, tarefa, split, dir_a, dir_b, B=300, seed=0):
    rot = ler_tabela(caminho_tabela(config, "rotulos_downstream", "rotulos"), ["evento_id", f"{tarefa}_rotulo"])
    pa, na = carregar_predicoes(dir_a, tarefa, split)
    pb, nb = carregar_predicoes(dir_b, tarefa, split)
    df = pa.merge(pb, on=["evento_id", "entidade_id"], suffixes=("_a", "_b")).merge(rot, on="evento_id")
    df = df[df[f"{tarefa}_rotulo"].notna()].reset_index(drop=True)
    y_bruto = df[f"{tarefa}_rotulo"]
    if "score_a" in df.columns:
        tipo = "binaria"
        y = y_bruto.to_numpy(dtype=float)
        pred_a, pred_b = df["score_a"].to_numpy(), df["score_b"].to_numpy()
    else:
        tipo = "multiclasse"
        classes = sorted({c[2:-2] for c in df.columns if c.startswith("p_") and c.endswith("_a")}, key=str)
        pa_m = np.stack([df[f"p_{c}_a"].to_numpy() for c in classes], axis=1)
        pb_m = np.stack([df[f"p_{c}_b"].to_numpy() for c in classes], axis=1)
        mapa = {c: i for i, c in enumerate(classes)}
        y = y_bruto.astype(str).map(mapa).to_numpy(dtype=float)
        pred_a, pred_b = pa_m.argmax(axis=1).astype(float), pb_m.argmax(axis=1).astype(float)

    w1 = np.ones(len(df))
    m_a, m_b = metrica(tipo, y, pred_a, w1), metrica(tipo, y, pred_b, w1)
    cod_cliente = pd.factorize(df["entidade_id"])[0]
    n_cli = cod_cliente.max() + 1
    rng = np.random.RandomState(seed)
    deltas = np.empty(B)
    for b in range(B):
        w = rng.poisson(1.0, n_cli)[cod_cliente].astype(float)
        deltas[b] = metrica(tipo, y, pred_b, w) - metrica(tipo, y, pred_a, w)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "tarefa": tarefa, "split": split, "tipo": tipo, "metrica": "auc" if tipo == "binaria" else "accuracy",
        "n_ancoras": int(len(df)), "n_clientes": int(n_cli), "sementes_a": na, "sementes_b": nb,
        "a": dir_a, "b": dir_b, "metrica_a": float(m_a), "metrica_b": float(m_b),
        "delta": float(m_b - m_a), "ic95": [float(lo), float(hi)], "B": B,
        "significativo": bool(lo > 0 or hi < 0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--tarefa", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True, action="append")
    ap.add_argument("--B", type=int, default=300)
    ap.add_argument("--saida", default=None, help="json onde acrescentar o resultado")
    args = ap.parse_args()
    config = carregar_config(args.config)
    base = run_dir(config, "")
    resultados = []
    for b in args.b:
        r = comparar(config, args.tarefa, args.split, os.path.join(base, args.a), os.path.join(base, b), args.B)
        resultados.append(r)
        print(f"{r['tarefa']} [{r['split']}] {r['metrica']}: A={r['metrica_a']:.4f} ({args.a})  B={r['metrica_b']:.4f} ({b})  "
              f"delta={r['delta']:+.4f}  IC95=[{r['ic95'][0]:+.4f}, {r['ic95'][1]:+.4f}]"
              f"{'  *' if r['significativo'] else ''}  (n={r['n_ancoras']:,}, {r['n_clientes']:,} clientes)")
    if args.saida:
        existentes = json.load(open(args.saida)) if os.path.exists(args.saida) else []
        with open(args.saida, "w", encoding="utf-8") as f:
            json.dump(existentes + resultados, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
