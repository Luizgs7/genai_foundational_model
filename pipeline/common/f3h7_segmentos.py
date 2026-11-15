"""F3-H7/F3-H8 — os segmentos de cada espaço recuperam os segmentos plantados?

Só na base sintética, que tem gabarito (`gabarito/clientes_latentes.parquet`,
coluna `segmento`). Para cada cliente, o estado na sua última âncora de um
split; K-means com K = número de segmentos plantados; ARI e NMI contra o
gabarito, média de 3 sementes do K-means.

Espaços:
  rfm       intervalo desde a cesta anterior, cestas e gasto em 365 dias, tempo de cliente
  trad      todas as features tradicionais (inclui as fontes extras, se houver)
  E:<pasta> vetor do motor na âncora (embeddings_eventos.npy da pasta)
  H:<pasta> trad + vetor do motor em PCA-32

A forma de padronizar (z-score por coluna ou bruto) é escolhida na âncora de
VALIDAÇÃO, pelo ARI, e o resultado reportado é o da âncora de TESTE (lição da
F3-H2: a forma de padronizar muda muito o resultado de um espaço).

Uso: python3 pipeline/common/f3h7_segmentos.py <config.yaml> --espacos rfm trad E:emb_v2 H:emb_v2 [--saida f3h7_segmentos]
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import polars as pl
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

sys.path.insert(0, "pipeline/common")
from schema import carregar_config, run_dir  # noqa: E402

RFM = ["gap_ultimo", "cestas_365d", "valor_365d", "tempo_cliente_dias"]


def ultimas_ancoras(feats, split):
    f = feats.filter(pl.col("split") == split).sort(["entidade_id", "evento_id"])
    return f.group_by("entidade_id", maintain_order=True).last()


def embeddings(config, pasta, eventos):
    d = run_dir(config, pasta)
    emb = np.load(os.path.join(d, "embeddings_eventos.npy"), mmap_mode="r")
    idx = pd.read_parquet(os.path.join(d, "embeddings_eventos_index.parquet"), columns=["evento_id", "row_idx"])
    linhas = pd.Series(idx["row_idx"].to_numpy(), index=idx["evento_id"].to_numpy()).reindex(eventos)
    if linhas.isna().any():
        raise ValueError(f"{int(linhas.isna().sum())} âncoras sem vetor em {d}")
    return np.asarray(emb[linhas.to_numpy(np.int64)], dtype=np.float32)


def preparar(X, forma, ref=None):
    X = np.asarray(X, np.float64)
    med = np.nanmedian(X if ref is None else ref, axis=0)
    X = np.where(np.isnan(X), med, X)
    if forma == "z":
        R = X if ref is None else np.where(np.isnan(ref), med, ref)
        X = (X - R.mean(0)) / np.maximum(R.std(0), 1e-9)
    return X


def agrupar(X, y, K, sementes=(0, 1, 2)):
    ari, nmi = [], []
    for s in sementes:
        rot = KMeans(K, n_init=10, random_state=s).fit_predict(X)
        ari.append(adjusted_rand_score(y, rot)); nmi.append(normalized_mutual_info_score(y, rot))
    return float(np.mean(ari)), float(np.mean(nmi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--espacos", nargs="+", default=["rfm", "trad"])
    ap.add_argument("--saida", default="f3h7_segmentos")
    a = ap.parse_args()
    config = carregar_config(a.config)
    fdir = run_dir(config, "features_tradicionais")
    info = json.load(open(os.path.join(fdir, "features_info.json"), encoding="utf-8"))
    nomes = [c for c in info["features"] if not c.endswith("_baseline_feat")]
    feats = pl.read_parquet(os.path.join(fdir, "features.parquet"))
    pasta_dados = os.path.dirname(config["v2"]["catalogo"]["tabela"]) if "catalogo" in config.get("v2", {}) else None
    lat = pl.read_parquet(os.path.join(pasta_dados, "gabarito", "clientes_latentes.parquet"), columns=["cliente_id", "segmento"])
    K = lat["segmento"].n_unique()
    res = {"K": K, "espacos": {}}
    for esp in a.espacos:
        por_split = {}
        for split in ("val", "test"):
            anc = ultimas_ancoras(feats, split).join(lat.rename({"cliente_id": "entidade_id"}), on="entidade_id", how="inner")
            y = anc["segmento"].to_numpy()
            ev = anc["evento_id"].to_numpy()
            if esp == "rfm":
                X = anc.select(RFM).to_numpy()
            elif esp == "trad":
                X = anc.select(nomes).to_numpy()
            elif esp.startswith("E:"):
                X = embeddings(config, esp[2:], ev)
            elif esp.startswith("H:"):
                E = embeddings(config, esp[2:], ev)
                E = PCA(32, random_state=0).fit_transform(E - E.mean(0))
                X = np.hstack([preparar(anc.select(nomes).to_numpy(), "z"), E])
            else:
                raise ValueError(esp)
            por_split[split] = {forma: agrupar(preparar(X, forma), y, K) for forma in ("z", "bruto")}
            por_split[split]["n_clientes"] = len(y)
        forma = max(("z", "bruto"), key=lambda f: por_split["val"][f][0])
        ari, nmi = por_split["test"][forma]
        res["espacos"][esp] = {"forma_escolhida_na_validacao": forma, "ari_teste": ari, "nmi_teste": nmi,
                               "n_clientes_teste": por_split["test"]["n_clientes"], "detalhe": por_split}
        print(f"{esp:28s} forma={forma:5s} ARI teste={ari:.4f}  NMI teste={nmi:.4f}  (val: z {por_split['val']['z'][0]:.4f}, "
              f"bruto {por_split['val']['bruto'][0]:.4f})", flush=True)
    out = run_dir(config, "f3h7")
    os.makedirs(out, exist_ok=True)
    json.dump(res, open(os.path.join(out, a.saida + ".json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
