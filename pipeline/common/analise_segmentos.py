"""F2-H7 — desempenho por faixa de histórico do cliente (experimento H&M, fase 2).

Pergunta: em clientes com pouco histórico (1 a 3 cestas até a âncora), onde as
features de RFM têm pouca informação, o embedding do modelo fundacional supera
o GBDT tradicional?

Usa as predições de TESTE já existentes da H10 (nenhum modelo é treinado de
novo) e, para cada faixa de `n_cestas` (cestas até a âncora, inclusive) e cada
tarefa, calcula a métrica de cada modelo e o delta B − A com IC 95% por
bootstrap Poisson por cliente (mesmo procedimento de avaliacao.py), dentro da faixa.

Uso: python3 pipeline/common/analise_segmentos.py configs/hm_clientes.yaml \\
        --a gbdt/h10_trad_lgbm --b "caches/h10_fusao_d_completo_s*" --b gbdt/h10_hibrido_lgbm [--B 300]
Gera runs/<nome>/f2_h7_segmentos.json
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, "pipeline/common")
from avaliacao import carregar_predicoes, metrica  # noqa: E402
from schema import caminho_tabela, carregar_config, ler_tabela, run_dir  # noqa: E402

FAIXAS = [(1, 1, "1 cesta (primeira compra)"), (2, 3, "2 a 3 cestas"), (4, 10, "4 a 10 cestas"), (11, 10**9, "11 ou mais")]
TAREFAS = ["churn", "proxima_online", "next_category", "ltv_90d"]


def preparar(pred, tipo):
    """-> vetor de predição (score ou classe) e lista de classes (multiclasse)."""
    if tipo == "binaria":
        return pred["score"].to_numpy(), None
    classes = sorted([c[2:] for c in pred.columns if c.startswith("p_")], key=str)
    m = np.stack([pred[f"p_{c}"].to_numpy() for c in classes], axis=1)
    return m.argmax(axis=1).astype(float), classes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True, action="append")
    ap.add_argument("--B", type=int, default=300)
    args = ap.parse_args()
    config = carregar_config(args.config)
    base = run_dir(config, "")
    seg = pl.read_parquet(os.path.join(run_dir(config, "features_tradicionais"), "features.parquet"),
                          columns=["evento_id", "split", "n_cestas"]).filter(pl.col("split") == "test").to_pandas()
    seg = seg[["evento_id", "n_cestas"]]
    out = {"a": args.a, "b": args.b, "faixas": {}}
    for t in TAREFAS:
        rot = ler_tabela(caminho_tabela(config, "rotulos_downstream", "rotulos"), ["evento_id", f"{t}_rotulo"])
        pa, _ = carregar_predicoes(os.path.join(base, args.a), t, "test")
        tipo = "binaria" if "score" in pa.columns else "multiclasse"
        df = pa[["evento_id", "entidade_id"]].merge(rot, on="evento_id").merge(seg, on="evento_id")
        df = df[df[f"{t}_rotulo"].notna()].reset_index(drop=True)
        ya, classes = preparar(pa.set_index("evento_id").loc[df["evento_id"]].reset_index(), tipo)
        if tipo == "binaria":
            y = df[f"{t}_rotulo"].to_numpy(dtype=float)
        else:
            mapa = {c: i for i, c in enumerate(classes)}
            y = df[f"{t}_rotulo"].astype(str).map(mapa).to_numpy(dtype=float)
        preds = {args.a: ya}
        for b in args.b:
            pb, _ = carregar_predicoes(os.path.join(base, b), t, "test")
            preds[b], _ = preparar(pb.set_index("evento_id").loc[df["evento_id"]].reset_index(), tipo)
        cod = pd.factorize(df["entidade_id"])[0]
        for lo, hi, nome in FAIXAS:
            sel = ((df["n_cestas"] >= lo) & (df["n_cestas"] <= hi)).to_numpy()
            ys, cs = y[sel], cod[sel]
            cs = pd.factorize(cs)[0]
            w1 = np.ones(len(ys))
            r = {"n_ancoras": int(sel.sum()), "n_clientes": int(cs.max() + 1) if len(cs) else 0,
                 "taxa_positiva": float(ys.mean()) if tipo == "binaria" else None,
                 "metrica": "auc" if tipo == "binaria" else "accuracy",
                 "modelos": {m: float(metrica(tipo, ys, p[sel], w1)) for m, p in preds.items()}, "deltas": {}}
            rng = np.random.RandomState(0)
            for b in args.b:
                d = np.empty(args.B)
                for k in range(args.B):
                    w = rng.poisson(1.0, cs.max() + 1)[cs].astype(float)
                    d[k] = metrica(tipo, ys, preds[b][sel], w) - metrica(tipo, ys, ya[sel], w)
                lo95, hi95 = np.percentile(d, [2.5, 97.5])
                r["deltas"][b] = {"delta": r["modelos"][b] - r["modelos"][args.a], "ic95": [float(lo95), float(hi95)],
                                  "significativo": bool(lo95 > 0 or hi95 < 0)}
            out["faixas"].setdefault(nome, {})[t] = r
            print(f"{t:15s} {nome:26s} n={r['n_ancoras']:>9,}  " + "  ".join(
                f"{os.path.basename(b.rstrip('*'))}: {r['deltas'][b]['delta']:+.4f} [{r['deltas'][b]['ic95'][0]:+.4f}, {r['deltas'][b]['ic95'][1]:+.4f}]"
                for b in args.b), flush=True)
    with open(os.path.join(base, "f2_h7_segmentos.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
