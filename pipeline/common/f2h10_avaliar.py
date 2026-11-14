"""F2-H10 — avaliação dos braços na amostra de clientes (teste, uma vez).

Restringe as predições de TESTE de cada braço às âncoras dos clientes da
amostra (os de runs/<nome>/f2h10/textos.parquet), calcula a métrica de cada
tarefa (AUC nas binárias, accuracy nas multiclasse) e as diferenças contra os
braços de referência com IC 95% por bootstrap Poisson por cliente. O mesmo
peso por cliente vale para as 4 tarefas em cada réplica, o que dá também o IC
da diferença na MÉDIA das 4 tarefas (critério principal da F2-H10).

As âncoras são as comuns a todos os braços EM CADA TAREFA. (Uma primeira versão
cruzava também entre tarefas; como "próxima categoria" e "próxima online" só
têm rótulo quando existe cesta seguinte, isso tirava do churn justamente os
clientes que não voltaram. Corrigido em 2026-11-14, antes de qualquer
resultado ser lido.)

Uso: python3 pipeline/common/f2h10_avaliar.py configs/hm_clientes.yaml \\
        --braco T=gbdt/f2h10_trad --braco "E0=f2h10_fusao_e0_s*" ... --ref E0 --ref T [--B 500]
Gera runs/<nome>/<--saida> (padrão f2h10/resultado_test.json)
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

TAREFAS = ["churn", "proxima_online", "next_category", "ltv_90d"]


def vetor(pred, tipo, mapa=None):
    """score (binária) ou classe prevista codificada por `mapa` (rótulo em texto -> inteiro)."""
    if tipo == "binaria":
        return pred["score"].to_numpy()
    cl = [c[2:] for c in pred.columns if c.startswith("p_")]
    m = np.stack([pred[f"p_{c}"].to_numpy() for c in cl], axis=1)
    return np.array([mapa.setdefault(cl[k], len(mapa)) for k in m.argmax(axis=1)], dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--braco", action="append", required=True, help="NOME=subpasta (glob com * junta sementes)")
    ap.add_argument("--ref", action="append", required=True)
    ap.add_argument("--B", type=int, default=500)
    ap.add_argument("--saida", default="f2h10/resultado_test.json", help="relativo a runs/<nome>")
    a = ap.parse_args()
    config = carregar_config(a.config)
    base = run_dir(config, "")
    bracos = dict(b.split("=", 1) for b in a.braco)
    amostra = pl.read_parquet(os.path.join(run_dir(config, "f2h10"), "textos.parquet"), columns=["entidade_id"]).unique()
    am = set(amostra["entidade_id"].to_list())
    rot = ler_tabela(caminho_tabela(config, "rotulos_downstream", "rotulos"), None)
    rot = rot[(rot["split"] == "test") & rot["entidade_id"].isin(am)]

    # âncoras comuns a todos os braços, tarefa a tarefa
    preds = {t: {} for t in TAREFAS}
    comuns = {t: None for t in TAREFAS}
    for nome, sub in bracos.items():
        for t in TAREFAS:
            p, n = carregar_predicoes(os.path.join(base, sub), t, "test")
            p = p[p["entidade_id"].isin(am)]
            preds[t][nome] = (p.set_index("evento_id"), n)
            ids = set(p["evento_id"])
            comuns[t] = ids if comuns[t] is None else comuns[t] & ids
    todas = set().union(*comuns.values())
    print(f"amostra: {len(am):,} clientes; âncoras de teste comuns a todos os braços, por tarefa: "
          + ", ".join(f"{t}={len(comuns[t]):,}" for t in TAREFAS), flush=True)

    rng = np.random.RandomState(0)
    res = {"bracos": bracos, "sementes": {}, "tarefas": {}, "media": {}, "deltas": {}}
    ap_boot = {}
    for t in TAREFAS:
        df = rot[rot["evento_id"].isin(comuns[t]) & rot[f"{t}_rotulo"].notna()][["evento_id", "entidade_id", f"{t}_rotulo"]].sort_values("evento_id")
        tipo = "binaria" if "score" in next(iter(preds[t].values()))[0].columns else "multiclasse"
        res["tarefas"][t] = {"metrica": "auc" if tipo == "binaria" else "accuracy", "n_ancoras": len(df), "modelos": {}}
        v, mapa = {}, {}
        if tipo == "binaria":
            y = df[f"{t}_rotulo"].to_numpy(dtype=float)
        else:
            y = np.array([mapa.setdefault(c, len(mapa)) for c in df[f"{t}_rotulo"].astype(str)], dtype=float)
        for nome in bracos:
            p, n = preds[t][nome]
            v[nome] = vetor(p.loc[df["evento_id"]].reset_index(), tipo, mapa)
            res["sementes"][nome] = n
        w1 = np.ones(len(y))
        for nome in bracos:
            res["tarefas"][t]["modelos"][nome] = float(metrica(tipo, y, v[nome], w1))
        ap_boot[t] = (tipo, y, v, None)

    # bootstrap conjunto: mesmo peso por cliente nas 4 tarefas (clientes indexados pela 1ª tarefa)
    ents_glob = pd.factorize(rot[rot["evento_id"].isin(todas)]["entidade_id"])[1]
    n_cli = int(len(ents_glob))
    pos_ent = {e: i for i, e in enumerate(ents_glob)}
    for t in TAREFAS:
        df_ent = rot[rot["evento_id"].isin(comuns[t]) & rot[f"{t}_rotulo"].notna()].sort_values("evento_id")["entidade_id"]
        tipo, y, v, _ = ap_boot[t]
        ap_boot[t] = (tipo, y, v, df_ent.map(pos_ent).to_numpy())
    m_boot = {nome: np.zeros((a.B, len(TAREFAS))) for nome in bracos}
    for k in range(a.B):
        wc = rng.poisson(1.0, len(ents_glob)).astype(float)
        for j, t in enumerate(TAREFAS):
            tipo, y, v, idx = ap_boot[t]
            w = wc[idx]
            for nome in bracos:
                m_boot[nome][k, j] = metrica(tipo, y, v[nome], w)
    for nome in bracos:
        res["media"][nome] = float(np.mean([res["tarefas"][t]["modelos"][nome] for t in TAREFAS]))
    for r in a.ref:
        for nome in bracos:
            if nome == r:
                continue
            d = {}
            for j, t in enumerate(TAREFAS):
                b = m_boot[nome][:, j] - m_boot[r][:, j]
                lo, hi = np.percentile(b, [2.5, 97.5])
                d[t] = {"delta": res["tarefas"][t]["modelos"][nome] - res["tarefas"][t]["modelos"][r],
                        "ic95": [float(lo), float(hi)], "significativo": bool(lo > 0 or hi < 0)}
            b = m_boot[nome].mean(1) - m_boot[r].mean(1)
            lo, hi = np.percentile(b, [2.5, 97.5])
            d["media"] = {"delta": res["media"][nome] - res["media"][r], "ic95": [float(lo), float(hi)],
                          "significativo": bool(lo > 0 or hi < 0)}
            res["deltas"][f"{nome}-{r}"] = d
    res["n_clientes"] = n_cli
    destino = os.path.join(base, a.saida)
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    with open(destino, "w") as f:
        json.dump(res, f, indent=1)
    for nome in bracos:
        print(f"{nome:6s} " + "  ".join(f"{t}={res['tarefas'][t]['modelos'][nome]:.4f}" for t in TAREFAS) + f"  média={res['media'][nome]:.4f}")
    for k, d in res["deltas"].items():
        print(f"{k:10s} média {d['media']['delta']:+.4f} [{d['media']['ic95'][0]:+.4f}, {d['media']['ic95'][1]:+.4f}]")


if __name__ == "__main__":
    main()
