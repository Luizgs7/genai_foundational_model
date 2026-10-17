"""Extrai, para UM cliente da H&M, o estado dele em cada etapa do pipeline —
insumo do relatório visual "rastro ponta a ponta" (mesmo formato do
relatório do cliente 860.703.096-50 da base sintética).

Etapas de dados (sempre disponíveis): compras brutas, splits, cestas,
discretização, tokens, janelas de treino, rótulos e features tradicionais.
Etapas de modelo (quando os artefatos existem): embedding da âncora e
predições da fusão / GBDT nas âncoras do cliente.

Uso: python3 pipeline/relatorios/rastro_cliente_hm_dados.py <config.yaml> <entidade_id>
Gera runs/<nome>/relatorios/rastro_<entidade_id>.json
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, "pipeline/common")
from schema import carregar_config, run_dir  # noqa: E402
from sequencias_io import carregar_sequencias  # noqa: E402


def main(config_path, ent):
    config = carregar_config(config_path)
    ent = int(ent)
    R = lambda *p: os.path.join(run_dir(config, ""), *p)  # noqa: E731
    out = {"entidade_id": ent}

    mapa = pl.scan_parquet("dados_hm/mapa_clientes.parquet").filter(pl.col("cliente") == ent).collect()
    out["customer_id"] = mapa["customer_id"][0] if mapa.height else None

    ev = pl.scan_parquet(R("eventos_canonicos.parquet")).filter(pl.col("entidade_id") == ent).collect().to_pandas()
    ev = ev.sort_values(["data_evento", "evento_id"]).reset_index(drop=True)
    sp = pl.scan_parquet(R("splits", "splits.parquet")).filter(pl.col("entidade_id") == ent).collect().to_pandas()
    ds = pl.scan_parquet(R("discretizacao", "discretizado.parquet")).filter(pl.col("entidade_id") == ent).collect().to_pandas()
    ev = ev.merge(sp[["evento_id", "split"]], on="evento_id").merge(ds[["evento_id", "preco_bucket"]], on="evento_id")
    art = pd.read_parquet("dados_hm/articles.parquet", columns=["article_id", "prod_name", "department_name"])
    ev = ev.merge(art.rename(columns={"article_id": "artigo"}), on="artigo", how="left")

    datas = pd.read_parquet(R("splits", "splits.parquet"), columns=["data_evento", "split"])
    cortes = {s: str(datas.loc[datas["split"] == s, "data_evento"].min().date()) for s in ["val", "test"]}
    out["cortes"] = cortes
    out["periodo"] = [str(datas["data_evento"].min().date()), str(datas["data_evento"].max().date())]
    del datas

    est = ev.iloc[0]
    out["cadastro"] = {c: (None if pd.isna(est[c]) else (str(est[c]) if not isinstance(est[c], (int, float, np.floating, np.integer)) else float(est[c])))
                       for c in ["idade", "clube", "newsletter", "cep_top", "fn", "active"]}

    with open(R("discretizacao", "buckets.json")) as f:
        buckets = json.load(f)["preco"]["edges"]
    with open(R("vocabulario", "vocab.json")) as f:
        vocab = json.load(f)
    with open(R("vocabulario", "top_n_mapa.json")) as f:
        topn = json.load(f)["artigo"]["mapa"]
    inv = {v: k for k, v in vocab.items()}

    compras = []
    for _, r in ev.iterrows():
        compras.append({
            "evento_id": int(r["evento_id"]), "data": str(r["data_evento"].date()), "split": str(r["split"]),
            "artigo": int(r["artigo"]), "nome": r["prod_name"], "departamento": r["department_name"],
            "indice": str(r["indice"]), "grupo_produto": str(r["grupo_produto"]), "tipo_produto": str(r["tipo_produto"]),
            "cor": str(r["cor"]), "linha_vestuario": str(r["linha_vestuario"]), "canal": str(r["canal"]),
            "preco": float(r["preco"]), "preco_bucket": int(r["preco_bucket"]),
            "faixas_grupo": buckets.get(str(r["grupo_produto"])),
            "artigo_no_top2000": str(r["artigo"]) in topn,
        })
    out["compras"] = compras

    # tokens
    seqs = carregar_sequencias(config)
    i = seqs.indice(ent)
    d = seqs.como_dict(i)
    out["tokens"] = {"seq_full": [inv[t] for t in d["seq_full"]],
                     "len_full": len(d["seq_full"]),
                     "len_train": len(d["seq_train"]) if d["seq_train"] else 0,
                     "len_train_val": len(d["seq_train_val"]) if d["seq_train_val"] else 0}
    pos = pl.scan_parquet(R("serializacao", "posicoes_evento.parquet")).filter(pl.col("entidade_id") == ent).collect().to_pandas()
    out["posicoes"] = {int(a): int(b) for a, b in zip(pos["evento_id"], pos["pos_full"])}
    out["vocab_size"] = len(vocab)
    out["n_positions"] = config["arquitetura"]["n_positions"]

    # rótulos (cestas)
    rot = pl.scan_parquet(R("rotulos_downstream", "rotulos.parquet")).filter(pl.col("entidade_id") == ent).collect().to_pandas()
    rot = rot.astype({c: object for c in rot.columns if isinstance(rot[c].dtype, pd.CategoricalDtype)})
    out["cestas"] = json.loads(rot.to_json(orient="records"))

    # features tradicionais
    ft = pl.scan_parquet(R("features_tradicionais", "features.parquet")).filter(pl.col("entidade_id") == ent).collect().to_pandas()
    ft = ft.astype({c: object for c in ft.columns if isinstance(ft[c].dtype, pd.CategoricalDtype)})
    out["features"] = json.loads(ft.to_json(orient="records"))
    with open(R("features_tradicionais", "features_info.json")) as f:
        out["features_info"] = {"n_features": json.load(f)["n_features"]}

    # embeddings da âncora (se a extração já rodou)
    idx_path = R("fusao", "embeddings_eventos_index.parquet")
    if os.path.exists(idx_path):
        idx = pl.scan_parquet(idx_path).filter(pl.col("entidade_id") == ent).collect().to_pandas()
        emb = np.load(R("fusao", "embeddings_eventos.npy"), mmap_mode="r")
        out["embeddings"] = {int(e): np.asarray(emb[r], dtype=np.float32).round(4).tolist() for e, r in zip(idx["evento_id"], idx["row_idx"])}

    # predições disponíveis (val/test) de fusão e GBDT
    preds = {}
    for f in glob.glob(R("**", "predicoes_*.parquet"), recursive=True):
        p = pl.scan_parquet(f).filter(pl.col("entidade_id") == ent).collect().to_pandas()
        if len(p):
            p = p.astype({c: object for c in p.columns if isinstance(p[c].dtype, pd.CategoricalDtype)})
            preds[os.path.relpath(f, run_dir(config, ""))] = json.loads(p.to_json(orient="records"))
    out["predicoes"] = preds

    os.makedirs(R("relatorios"), exist_ok=True)
    path = R("relatorios", f"rastro_{ent}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=str)
    print(f"Gravado {path}: {len(compras)} compras, {len(out['cestas'])} cestas, {out['tokens']['len_full']} tokens, "
          f"{len(out['features'])} âncoras com features, {len(preds)} arquivos de predição")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
