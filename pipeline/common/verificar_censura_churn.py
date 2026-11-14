"""Verificação: quanto a censura unilateral do rótulo de churn no fim do teste
(âncoras dos últimos 90 dias só entram com rótulo 0) altera as AUCs publicadas?

A receita de churn marcava como determinável toda âncora com cesta seguinte,
mesmo com a janela de 90 dias ainda aberta no fim do dataset. Essas âncoras só
podem ter rótulo 0: é uma seleção pelo próprio desfecho.

Uso: python3 pipeline/common/verificar_censura_churn.py
Gera runs/hm_clientes/f3h2/verificacao_censura_churn.json"""
import glob, json, sys
import numpy as np, polars as pl
sys.path.insert(0, "pipeline/common")
from avaliacao import metrica

R = "runs/hm_clientes/"
rot = pl.read_parquet(R + "rotulos_downstream/rotulos.parquet", columns=["evento_id", "entidade_id", "split", "churn_rotulo"]).filter(
    (pl.col("split") == "test") & pl.col("churn_rotulo").is_not_null())
datas = pl.read_parquet(R + "eventos_canonicos.parquet", columns=["evento_id", "data_evento"])
fim = datas["data_evento"].max()
rot = rot.join(datas, on="evento_id", how="left").with_columns(((pl.lit(fim) - pl.col("data_evento")).dt.total_days()).alias("dias_ate_fim"))
del datas
print("fim do dataset:", fim, "| âncoras de teste com rótulo de churn:", rot.height)
print(rot.with_columns((pl.col("dias_ate_fim") >= 90).alias("janela_fechada")).group_by("janela_fechada").agg(
    pl.len().alias("n"), pl.col("churn_rotulo").mean().alias("taxa_churn")).sort("janela_fechada"))
modelos = {"T  GBDT tradicional (LightGBM)": "gbdt/h10_trad_lgbm/predicoes_test_churn_seed*.parquet",
           "H  GBDT híbrido (LightGBM)": "gbdt/h10_hibrido_lgbm/predicoes_test_churn_seed*.parquet",
           "GBDT só embedding (PCA-32)": "gbdt/h10_emb_pca_lgbm/predicoes_test_churn_seed*.parquet",
           "E  fusão completa (backbone H3)": "caches/h10_fusao_d_completo_s*/predicoes_test*.parquet"}
res = {}
for nome, pad in modelos.items():
    arqs = sorted(glob.glob(R + pad))
    if not arqs:
        print(nome, ": sem predições locais"); continue
    ps = []
    for a in arqs:
        p = pl.read_parquet(a)
        col = "score" if "score" in p.columns else ("churn_score" if "churn_score" in p.columns else None)
        if col is None:
            cand = [c for c in p.columns if "churn" in c and c not in ("churn_rotulo",)]
            col = cand[0]
        ps.append(p.select(["evento_id", pl.col(col).alias(f"s{len(ps)}")]))
    p = ps[0]
    for q in ps[1:]:
        p = p.join(q, on="evento_id", how="inner")
    d = rot.join(p, on="evento_id", how="inner")
    s = d.select([c for c in d.columns if c.startswith("s") and c[1:].isdigit()]).to_numpy().mean(1)
    y = d["churn_rotulo"].to_numpy().astype(float)
    limpo = (d["dias_ate_fim"] >= 90).to_numpy()
    res[nome] = {"n": int(len(y)), "auc_publicada": float(metrica("binaria", y, s, np.ones(len(y)))),
                 "n_limpo": int(limpo.sum()), "auc_janela_fechada": float(metrica("binaria", y[limpo], s[limpo], np.ones(int(limpo.sum()))))}
    print(f"{nome:34s} n={len(y):,}  AUC como publicada={res[nome]['auc_publicada']:.4f}   AUC só com janela fechada (n={int(limpo.sum()):,})={res[nome]['auc_janela_fechada']:.4f}")
json.dump(res, open(R + "f3h2/verificacao_censura_churn.json", "w"), indent=1, ensure_ascii=False)
