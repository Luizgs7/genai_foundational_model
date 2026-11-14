"""F3-H2 — clusterização: os segmentos formados a partir da representação única
explicam o comportamento futuro tão bem quanto os segmentos tradicionais?

Desenho registrado em PLANO_FASE3.md antes de rodar:

  - uma âncora por cliente, sorteada com semente fixa; o K-means é ajustado nas
    âncoras de TREINO e aplicado às do split avaliado;
  - espaços comparados, todos padronizados (média 0 e desvio 1 por dimensão,
    com estatísticas do treino):
      T1  RFM clássico: intervalo desde a cesta anterior, frequência e gasto na
          maior janela (365 dias) e tempo de casa;
      T2  os agregados tradicionais (97 features) reduzidos a 32 componentes;
      E   o embedding do backbone reduzido a 32 componentes (cache do PCA);
      H   T2 e E juntos;
  - K = 8 (principal), K = 4 e 16 (sensibilidade), 3 sementes;
  - desfechos futuros da âncora: os rótulos das tarefas downstream ativas;
  - índice principal: média, sobre os desfechos, da parcela da incerteza do
    desfecho explicada pelo segmento, I(segmento; desfecho) / H(desfecho);
  - estabilidade: ARI entre dois ajustes em metades disjuntas dos clientes de
    treino, aplicados ao mesmo split avaliado;
  - IC 95% por bootstrap Poisson por cliente (uma âncora por cliente).

Critério: EMPATE se o índice de E ou de H for pelo menos 95% do melhor entre T1
e T2 (limite inferior do IC de índice_S − 0,95·índice_T acima de zero) e o ARI
não ficar mais de 0,05 abaixo. FORMA FORTE: acima do melhor T com IC excluindo
o zero.

Refinamento feito na validação, antes do teste (registrado no plano): cada
espaço reduzido por PCA tem duas formas de padronização, e a forma de cada
espaço é ESCOLHIDA NA VALIDAÇÃO pelo índice principal (K = 8):
  - "padronizado": média 0 e desvio 1 em cada dimensão;
  - "escala_bloco": o bloco inteiro dividido por um único fator, o que preserva
    a geometria do PCA (padronizar dimensão a dimensão dá peso igual a
    componentes de pouca variância e, na validação, derrubou o T2).
O T1 é sempre padronizado por dimensão. O teste usa as escolhas gravadas em
resultado_val.json (--fase test as lê de lá).

O código é desenvolvido com --fase val; --fase test roda uma vez.

Uso: python3 pipeline/common/f3h2_clusterizacao.py configs/hm_clientes.yaml --fase val|test
        [--emb_cache caches/pca32_h3.npz] [--emb_extra NOME=caches/outro.npz ...] [--B 300]
Gera runs/<nome>/f3h2/resultado_<fase>.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, "pipeline/common")
from schema import carregar_config, run_dir, tarefas_ativas  # noqa: E402

KS = [8, 4, 16]
SEMENTES = [0, 1, 2]
N_AJUSTE = 300_000      # âncoras de treino usadas em cada ajuste do K-means
N_TREINO = 2 * N_AJUSTE  # clientes de treino carregados (duas metades disjuntas para a estabilidade)
N_COMP = 32


def uma_por_cliente(ent, rng):
    """índices de uma linha sorteada por entidade."""
    perm = rng.permutation(len(ent))
    _, primeiro = np.unique(ent[perm], return_index=True)
    return np.sort(perm[primeiro])


def incerteza_explicada(cod, n_seg, n_cls, w=None):
    """U = I(segmento; desfecho) / H(desfecho) a partir do código conjunto seg*n_cls + classe."""
    c = np.bincount(cod, weights=w, minlength=n_seg * n_cls).reshape(n_seg, n_cls).astype(np.float64)
    p = c / c.sum()
    ps, py = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    ok = p > 0
    info = float((p[ok] * np.log(p[ok] / (ps @ py)[ok])).sum())
    hy = float(-(py[py > 0] * np.log(py[py > 0])).sum())
    return info / hy if hy > 0 else 0.0


def padronizar(tr, av):
    m, s = tr.mean(0), tr.std(0)
    s[s == 0] = 1.0
    return ((tr - m) / s).astype(np.float32), ((av - m) / s).astype(np.float32)


def escala_bloco(tr, av):
    """centra e divide o bloco inteiro por um único fator (variância média 1): preserva a geometria do PCA."""
    m = tr.mean(0)
    f = float(np.sqrt(((tr - m) ** 2).mean()))
    return ((tr - m) / f).astype(np.float32), ((av - m) / f).astype(np.float32)


def espaco_agregados(tr, av, numericas, categoricas):
    """agregados tradicionais -> 32 componentes (transformações ajustadas só no treino)."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import QuantileTransformer
    qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal", subsample=200_000, random_state=0)
    xtr = qt.fit_transform(tr.select(numericas).to_numpy().astype(np.float64)).astype(np.float32)
    xav = qt.transform(av.select(numericas).to_numpy().astype(np.float64)).astype(np.float32)
    xtr, xav = np.nan_to_num(xtr, nan=0.0), np.nan_to_num(xav, nan=0.0)
    blocos_tr, blocos_av = [xtr], [xav]
    for c in categoricas:   # one-hot com os níveis do treino (até 30; o resto vira zero)
        vtr, vav = tr[c].to_numpy(), av[c].to_numpy()
        niveis, cont = np.unique(vtr[~np.isnan(vtr.astype(float))], return_counts=True)
        niveis = niveis[np.argsort(-cont)][:30]
        blocos_tr.append((vtr[:, None] == niveis[None, :]).astype(np.float32))
        blocos_av.append((vav[:, None] == niveis[None, :]).astype(np.float32))
    xtr, xav = np.hstack(blocos_tr), np.hstack(blocos_av)
    m, s = xtr.mean(0), xtr.std(0)
    s[s == 0] = 1.0
    xtr, xav = (xtr - m) / s, (xav - m) / s
    pca = PCA(n_components=N_COMP, svd_solver="randomized", random_state=0).fit(xtr)
    return pca.transform(xtr).astype(np.float32), pca.transform(xav).astype(np.float32), float(pca.explained_variance_ratio_.sum())


def espaco_rfm(tr, av, colunas):
    xtr = np.log1p(np.clip(tr.select(colunas).to_numpy().astype(np.float64), 0, None))
    xav = np.log1p(np.clip(av.select(colunas).to_numpy().astype(np.float64), 0, None))
    med = np.nanmedian(xtr, axis=0)
    for j in range(xtr.shape[1]):
        xtr[np.isnan(xtr[:, j]), j] = med[j]
        xav[np.isnan(xav[:, j]), j] = med[j]
    return xtr.astype(np.float32), xav.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--fase", required=True, choices=["val", "test"])
    ap.add_argument("--emb_cache", default="caches/pca32_h3.npz", help="npz com evento_id e E (PCA do embedding), relativo a runs/<nome>")
    ap.add_argument("--emb_extra", action="append", default=[], help="NOME=arquivo.npz: outro embedding reduzido, vira o espaço E_NOME")
    ap.add_argument("--B", type=int, default=300)
    a = ap.parse_args()
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    t0 = time.time()
    config = carregar_config(a.config)
    base = run_dir(config, "")
    fdir = run_dir(config, "features_tradicionais")
    info = json.load(open(os.path.join(fdir, "features_info.json"), encoding="utf-8"))
    tarefas = [t.get("nome", t["tipo"]) for t in tarefas_ativas(config)]
    rot_cols = [f"{t}_rotulo" for t in tarefas]
    janela = max(config["features_tradicionais"]["janelas_dias"])
    rfm = ["gap_ultimo", f"cestas_{janela}d", f"valor_{janela}d", "tempo_cliente_dias"]
    assert all(c in info["features"] for c in rfm), rfm

    # uma âncora por cliente em cada split
    arq = os.path.join(fdir, "features.parquet")
    idx = pl.read_parquet(arq, columns=["evento_id", "entidade_id", "split"])
    rng = np.random.RandomState(0)
    escolha = {}
    for sp in ("train", a.fase):
        sub = idx.filter(pl.col("split") == sp)
        sel = uma_por_cliente(sub["entidade_id"].to_numpy(), rng)
        n_cli = len(sel)
        if sp == "train" and n_cli > N_TREINO:   # teto de memória: o ajuste usa no máximo N_TREINO clientes
            sel = np.sort(rng.choice(sel, N_TREINO, replace=False))
        escolha[sp] = sub[sel]
        print(f"{sp}: {sub.height:,} âncoras, {n_cli:,} clientes ({len(sel):,} usados)", flush=True)
    del idx

    def carregar(sel):
        df = pl.scan_parquet(arq).join(sel.select("evento_id").lazy(), on="evento_id", how="semi").collect()
        return sel.select("evento_id").join(df, on="evento_id", how="left")   # mesma ordem de `sel`
    tr, av = carregar(escolha["train"]), carregar(escolha[a.fase])
    assert tr.height == escolha["train"].height and av.height == escolha[a.fase].height

    categoricas = [c for c in info["categoricas"] if c in info["features"]]
    numericas = [c for c in info["features"] if c not in categoricas]
    t1_tr, t1_av = espaco_rfm(tr, av, rfm)
    t2_tr, t2_av, var_t2 = espaco_agregados(tr, av, numericas, categoricas)
    print(f"T2: {len(numericas)} numéricas + {len(categoricas)} categóricas -> {N_COMP} componentes "
          f"(variância explicada {var_t2:.3f}) ({time.time() - t0:.0f}s)", flush=True)

    def embedding(arquivo):
        z = np.load(os.path.join(base, arquivo))
        ev, E = z["evento_id"], z["E"]
        pos = pl.DataFrame({"evento_id": ev, "_p": np.arange(len(ev))})
        saidas = []
        for d in (tr, av):
            p = d.select("evento_id").join(pos, on="evento_id", how="left")["_p"]
            assert p.null_count() == 0, f"{arquivo}: {p.null_count()} âncoras sem embedding"
            saidas.append(E[p.to_numpy()].astype(np.float32))
        return saidas
    brutos = {"T1": (t1_tr, t1_av), "T2": (t2_tr, t2_av)}
    brutos["E"] = tuple(embedding(a.emb_cache))
    for extra in a.emb_extra:
        nome, arquivo = extra.split("=", 1)
        brutos[f"E_{nome}"] = tuple(embedding(arquivo))

    # espaços finais: padronizado (principal) e escala por bloco (sensibilidade)
    def montar(norm_pca):
        esp = {"T1": padronizar(*brutos["T1"])}
        for n in brutos:
            if n != "T1":
                esp[n] = norm_pca(*brutos[n])
        for n in [k for k in brutos if k.startswith("E")]:
            h = "H" + n[1:]
            esp[h] = (np.hstack([esp["T2"][0], esp[n][0]]), np.hstack([esp["T2"][1], esp[n][1]]))
        return esp
    variantes = {"padronizado": montar(padronizar), "escala_bloco": montar(escala_bloco)}

    # desfechos do split avaliado
    desfechos = {}
    for t, c in zip(tarefas, rot_cols):
        v = av[c]
        ok = v.is_not_null().to_numpy()
        if v.dtype in (pl.Float32, pl.Float64):
            ok &= ~np.isnan(v.fill_null(np.nan).to_numpy())
        cls, y = np.unique(v.filter(pl.Series(ok)).cast(pl.Utf8).to_numpy(), return_inverse=True)
        desfechos[t] = (ok, y.astype(np.int64), len(cls))
        print(f"desfecho {t}: {int(ok.sum()):,} âncoras com rótulo, {len(cls)} classes", flush=True)

    n_tr = t1_tr.shape[0]
    res = {"fase": a.fase, "n_clientes_treino": n_tr, "n_clientes_avaliados": av.height, "desfechos": tarefas,
           "rfm": rfm, "variancia_explicada_T2": var_t2, "variantes": {}}
    atribuicoes = {}
    for nome_v, esp in variantes.items():
        res["variantes"][nome_v] = {}
        for K in KS:
            linha = {}
            for n, (xtr, xav) in esp.items():
                us, aris, tamanhos = [], [], []
                for s in SEMENTES:
                    r = np.random.RandomState(1000 + s)
                    perm = r.permutation(n_tr)
                    km = KMeans(n_clusters=K, n_init=3, random_state=s).fit(xtr[perm[:N_AJUSTE]])
                    seg = km.predict(xav)
                    atribuicoes[(nome_v, K, n, s)] = seg
                    us.append({t: incerteza_explicada(seg[ok] * nc + y, K, nc) for t, (ok, y, nc) in desfechos.items()})
                    tamanhos.append(np.sort(np.bincount(seg, minlength=K) / len(seg)).tolist())
                    meio = n_tr // 2
                    ka = KMeans(n_clusters=K, n_init=3, random_state=s).fit(xtr[perm[:meio][:N_AJUSTE]])
                    kb = KMeans(n_clusters=K, n_init=3, random_state=s).fit(xtr[perm[meio:][:N_AJUSTE]])
                    aris.append(float(adjusted_rand_score(ka.predict(xav), kb.predict(xav))))
                por_desfecho = {t: float(np.mean([u[t] for u in us])) for t in desfechos}
                linha[n] = {"indice": float(np.mean(list(por_desfecho.values()))), "por_desfecho": por_desfecho,
                            "indice_por_semente": [float(np.mean(list(u.values()))) for u in us],
                            "ari": float(np.mean(aris)), "ari_por_semente": aris,
                            "menor_e_maior_segmento": [float(np.mean([t_[0] for t_ in tamanhos])), float(np.mean([t_[-1] for t_ in tamanhos]))]}
                print(f"[{nome_v}] K={K} {n:8s} índice={linha[n]['indice']:.4f} ARI={linha[n]['ari']:.3f}  "
                      + "  ".join(f"{t}={v:.4f}" for t, v in por_desfecho.items()) + f"  ({time.time() - t0:.0f}s)", flush=True)
            res["variantes"][nome_v][f"K{K}"] = linha

    # forma de padronização de cada espaço: escolhida na validação pelo índice principal (K = 8)
    K = KS[0]
    nomes = list(variantes["padronizado"].keys())
    out = run_dir(config, "f3h2")
    if a.fase == "val":
        escolhas = {n: max(variantes, key=lambda v: res["variantes"][v][f"K{K}"][n]["indice"]) for n in nomes}
    else:
        escolhas = json.load(open(os.path.join(out, "resultado_val.json"), encoding="utf-8"))["escolha_variante"]
        assert set(escolhas) == set(nomes), (escolhas, nomes)
    res["escolha_variante"] = escolhas
    print("forma escolhida na validação:", escolhas, flush=True)
    principal = {n: res["variantes"][escolhas[n]][f"K{K}"][n] for n in nomes}
    res["principal"] = {n: {**principal[n], "variante": escolhas[n]} for n in nomes}
    res["sensibilidade_K"] = {f"K{k}": {n: {"indice": res["variantes"][escolhas[n]][f"K{k}"][n]["indice"],
                                           "ari": res["variantes"][escolhas[n]][f"K{k}"][n]["ari"]} for n in nomes} for k in KS}

    # bootstrap por cliente (K principal, forma escolhida): índice médio das sementes
    rngb = np.random.RandomState(0)
    boot = {n: np.zeros(a.B) for n in nomes}
    codigos = {(n, s, t): atribuicoes[(escolhas[n], K, n, s)][ok] * nc + y
               for n in nomes for s in SEMENTES for t, (ok, y, nc) in desfechos.items()}
    for b in range(a.B):
        w = rngb.poisson(1.0, av.height).astype(np.float64)
        for n in nomes:
            boot[n][b] = np.mean([np.mean([incerteza_explicada(codigos[(n, s, t)], K, nc, w[ok])
                                           for t, (ok, y, nc) in desfechos.items()]) for s in SEMENTES])
    melhor_t = max(("T1", "T2"), key=lambda n: principal[n]["indice"])
    res["melhor_T"] = melhor_t
    res["ic95"] = {n: [float(x) for x in np.percentile(boot[n], [2.5, 97.5])] for n in nomes}
    res["veredito"] = {}
    for n in nomes:
        if n in ("T1", "T2"):
            continue
        d = boot[n] - boot[melhor_t]
        d95 = boot[n] - 0.95 * boot[melhor_t]
        dif_ari = principal[n]["ari"] - principal[melhor_t]["ari"]
        v = {"indice": principal[n]["indice"], "indice_melhor_T": principal[melhor_t]["indice"],
             "razao": principal[n]["indice"] / principal[melhor_t]["indice"],
             "delta": principal[n]["indice"] - principal[melhor_t]["indice"],
             "ic95_delta": [float(x) for x in np.percentile(d, [2.5, 97.5])],
             "ic95_delta_margem": [float(x) for x in np.percentile(d95, [2.5, 97.5])],
             "ari": principal[n]["ari"], "ari_melhor_T": principal[melhor_t]["ari"], "delta_ari": dif_ari}
        v["empate_indice"] = bool(v["ic95_delta_margem"][0] > 0)
        v["estabilidade_ok"] = bool(dif_ari >= -0.05)
        v["empate"] = bool(v["empate_indice"] and v["estabilidade_ok"])
        v["forma_forte"] = bool(v["ic95_delta"][0] > 0 and v["estabilidade_ok"])
        res["veredito"][n] = v
        print(f"VEREDITO {n} vs {melhor_t}: índice {v['indice']:.4f} vs {v['indice_melhor_T']:.4f} (razão {v['razao']:.3f}); "
              f"IC do delta [{v['ic95_delta'][0]:+.4f}, {v['ic95_delta'][1]:+.4f}]; ARI {v['ari']:.3f} vs {v['ari_melhor_T']:.3f}; "
              f"empate={v['empate']} forte={v['forma_forte']}", flush=True)
    res["segundos"] = round(time.time() - t0, 1)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f"resultado_{a.fase}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(f"gravado em {out}/resultado_{a.fase}.json ({res['segundos']}s)")


if __name__ == "__main__":
    main()
