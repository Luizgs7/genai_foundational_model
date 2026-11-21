"""Diagnóstico exploratório da F3-H2 (fora do critério): os segmentos de cada
espaço separam clientes ou separam o CALENDÁRIO? Mede a parcela da incerteza do
mês da âncora explicada pelo segmento (K = 8, semente 0).

Uso: python3 pipeline/common/f3h2_diagnostico_calendario.py val|test
Gera runs/hm_clientes/f3h2/diagnostico_calendario_<fase>.json
F3-H6: CACHE=caches/pca32_v2.npz SAIDA=f3h6_cluster usam o vetor do v2 e gravam em runs/hm_clientes/f3h6_cluster/
(o espaço E é então posto na escala por bloco, a forma escolhida na validação da F3-H6)."""
import json, os, sys
import numpy as np, polars as pl
sys.path.insert(0, "pipeline/common")
import f3h2_clusterizacao as m
from schema import carregar_config, run_dir
from sklearn.cluster import KMeans

config = carregar_config("configs/hm_clientes.yaml")
fdir = run_dir(config, "features_tradicionais")
info = json.load(open(os.path.join(fdir, "features_info.json")))
arq = os.path.join(fdir, "features.parquet")
CACHE = os.environ.get("CACHE", "caches/pca32_h3.npz")
SAIDA = os.environ.get("SAIDA", "f3h2")
z = np.load(os.path.join(run_dir(config, ""), CACHE))
ev, E = z["evento_id"], z["E"]
pos = pl.DataFrame({"evento_id": ev, "_p": np.arange(len(ev))})
rfm = ["gap_ultimo", "cestas_365d", "valor_365d", "tempo_cliente_dias"]
cat = [c for c in info["categoricas"] if c in info["features"]]
num = [c for c in info["features"] if c not in cat]
saida = {}
for fase in sys.argv[1:]:
    idx = pl.read_parquet(arq, columns=["evento_id", "entidade_id", "split"])
    rng = np.random.RandomState(0)
    esc = {}
    for sp in ("train", fase):
        sub = idx.filter(pl.col("split") == sp)
        sel = m.uma_por_cliente(sub["entidade_id"].to_numpy(), rng)
        if sp == "train" and len(sel) > m.N_TREINO:
            sel = np.sort(rng.choice(sel, m.N_TREINO, replace=False))
        esc[sp] = sub[sel]
    del idx
    def carregar(sel):
        df = pl.scan_parquet(arq).join(sel.select("evento_id").lazy(), on="evento_id", how="semi").collect()
        return sel.select("evento_id").join(df, on="evento_id", how="left")
    tr, av = carregar(esc["train"]), carregar(esc[fase])
    emb = [E[d.select("evento_id").join(pos, on="evento_id", how="left")["_p"].to_numpy()].astype(np.float32) for d in (tr, av)]
    esp = {"T1": m.padronizar(*m.espaco_rfm(tr, av, rfm)), "E": m.padronizar(*emb) if SAIDA == "f3h2" else m.escala_bloco(*emb)}
    t2 = m.espaco_agregados(tr, av, num, cat)
    esp["T2"] = m.escala_bloco(t2[0], t2[1])
    mes = av["mes"].to_numpy().astype(np.int64)
    meses, y = np.unique(mes, return_inverse=True)
    churn = av["churn_rotulo"].to_numpy()
    okc = ~np.isnan(churn)
    ci = np.where(okc, churn, 0).astype(np.int64)
    r = np.random.RandomState(1000)
    perm = r.permutation(tr.height)
    saida[fase] = {"taxa_churn_por_mes": {int(k): round(float(np.nanmean(churn[mes == k])), 3) for k in meses}}
    # quanto do churn o próprio mês explica (teto de um "segmento = calendário")
    saida[fase]["U_churn_dado_mes"] = round(m.incerteza_explicada((y * 2 + ci)[okc], len(meses), 2), 4)
    for n, (xtr, xav) in esp.items():
        seg = KMeans(n_clusters=8, n_init=3, random_state=0).fit(xtr[perm[:m.N_AJUSTE]]).predict(xav)
        saida[fase][n] = {"U_mes_dado_segmento": round(m.incerteza_explicada(seg * len(meses) + y, 8, len(meses)), 4),
                          "U_churn_dado_segmento": round(m.incerteza_explicada((seg * 2 + ci)[okc], 8, 2), 4)}
    print(fase, json.dumps(saida[fase], ensure_ascii=False), flush=True)
json.dump(saida, open(f"runs/hm_clientes/{SAIDA}/diagnostico_calendario_{sys.argv[1]}.json", "w"), indent=1, ensure_ascii=False)
