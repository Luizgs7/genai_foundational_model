"""Features tradicionais (feature engineering de ML clássico) por âncora — H5
do experimento H&M. É a régua contra a qual o modelo fundacional precisa
provar valor: RFM, janelas de tempo, composição de categorias, recompra,
última cesta, calendário e cadastro, todas CAUSAIS (só eventos até a data
da âncora, inclusive a cesta da própria âncora — o rótulo é sempre sobre a
PRÓXIMA cesta).

Configuração (seção `features_tradicionais:` do YAML):
    campo_valor: preco               # valor monetário do item
    campos_binarios: [online]        # participação (média) por janela
    campos_composicao: [indice, grupo_produto]   # participação por categoria
    max_categorias_composicao: 12    # top-K categorias do TREINO por campo (resto = outros)
    campo_item: artigo               # recompra do mesmo item
    campo_tipo: tipo_produto         # variedade de tipos
    campo_grupo_preco: grupo_produto # preço relativo = preço / preço médio do grupo no treino
    janelas_dias: [7, 30, 90, 180, 365]
    estaticos_frequencia: [cep]      # frequency encoding (nº de entidades com o mesmo valor)

Pressupõe `ancora: cesta` (uma âncora por entidade+dia, no último item).
Estatísticas calibráveis (top-K de categorias, preço médio por grupo) vêm
SÓ do split de treino.

Uso: python3 pipeline/common/features_tradicionais.py <config.yaml> [--teste_vazamento N]
Gera runs/<nome>/features_tradicionais/{features.parquet, features_info.json}.
`--teste_vazamento N`: recalcula N âncoras de val/teste usando só os eventos
até a data de cada uma e confere que as features batem.
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
from schema import (caminho_tabela, carregar_config, gravar_tabela, juntar_por_evento, ler_fonte,  # noqa: E402
                    ler_tabela, mem, run_dir, tarefas_ativas)

DIA_BASE = np.datetime64("1970-01-01", "D")


def carregar_eventos(config, ft):
    cols_cat = list(dict.fromkeys(ft["campos_composicao"] + [ft["campo_tipo"], ft["campo_grupo_preco"]]))
    cols = ["evento_id", "entidade_id", "data_evento", ft["campo_valor"], ft["campo_item"]] + ft["campos_binarios"] + cols_cat
    df = ler_fonte(config, list(dict.fromkeys(cols)))
    splits = ler_tabela(caminho_tabela(config, "splits", "splits"), ["evento_id", "split"])
    df = juntar_por_evento(df, splits, ["split"])
    del splits
    categorias = {}
    dados = {
        "evento_id": df["evento_id"].to_numpy(),
        "ent": df["entidade_id"].to_numpy(),
        "dia": (df["data_evento"].to_numpy().astype("datetime64[D]") - DIA_BASE).astype(np.int32),
        "valor": df[ft["campo_valor"]].to_numpy(dtype=np.float32),
        "item": df[ft["campo_item"]].to_numpy(),
        "treino": (df["split"].astype(str) == "train").to_numpy() if not hasattr(df["split"], "cat")
        else (df["split"].cat.codes.to_numpy() == list(df["split"].cat.categories).index("train")),
    }
    for b in ft["campos_binarios"]:
        dados[f"bin_{b}"] = df[b].to_numpy(dtype=np.float32)
    for c in cols_cat:
        cat = df[c].astype("category")
        dados[f"cat_{c}"] = cat.cat.codes.to_numpy().astype(np.int16)
        categorias[c] = list(cat.cat.categories)
    del df
    ev = pl.DataFrame(dados).sort(["ent", "dia", "evento_id"])
    return ev, categorias


def calibrar(ev, categorias, ft):
    """Estatísticas só do treino: top-K categorias por campo e preço médio por grupo."""
    treino = ev["treino"].to_numpy()
    calib = {"top": {}, "preco_medio_grupo": {}}
    for c in ft["campos_composicao"]:
        cont = np.bincount(ev[f"cat_{c}"].to_numpy()[treino].astype(np.int64) + 1)[1:]  # +1: código -1 (nulo)
        calib["top"][c] = [int(k) for k in np.argsort(-cont, kind="stable")[:ft["max_categorias_composicao"]] if cont[k] > 0]
    g = ev[f"cat_{ft['campo_grupo_preco']}"].to_numpy()[treino].astype(np.int64)
    v = ev["valor"].to_numpy()[treino].astype(np.float64)
    soma, n = np.bincount(g + 1, weights=v)[1:], np.bincount(g + 1)[1:]
    calib["preco_medio_grupo"] = {int(k): float(soma[k] / n[k]) for k in range(len(n)) if n[k] > 0}
    calib["preco_medio_global"] = float(v.mean())
    calib["categorias"] = categorias
    return calib


def primeira_ocorrencia(ent, valor):
    """True na 1ª vez (em ordem cronológica — `ev` já está ordenado) que cada
    par (entidade, valor) aparece. Ordenação estável em vez de tabela hash
    (memória em 30M+ linhas)."""
    ordem = np.lexsort((valor, ent))  # estável: preserva a ordem cronológica dentro do par
    e, v = ent[ordem], valor[ordem]
    primeiro = np.r_[True, (e[1:] != e[:-1]) | (v[1:] != v[:-1])]
    saida = np.empty(len(ent), dtype=bool)
    saida[ordem] = primeiro
    return saida


def cestas(ev, calib, ft):
    """Agrega itens por (entidade, dia). `ev` está ordenado por entidade/dia,
    então cada cesta é um trecho contíguo: somas por reduceat e moda por
    bincount (sem group_by com hash — memória em 30M+ linhas)."""
    ent = ev["ent"].to_numpy()
    dia = ev["dia"].to_numpy()
    nova = np.r_[True, (ent[1:] != ent[:-1]) | (dia[1:] != dia[:-1])]
    ini = np.flatnonzero(nova)
    n_c = len(ini)
    ultimo = np.r_[ini[1:] - 1, len(ent) - 1]
    cesta = np.cumsum(nova) - 1
    soma = lambda x: np.add.reduceat(np.asarray(x, dtype=np.float64), ini)  # noqa: E731

    g = ev[f"cat_{ft['campo_grupo_preco']}"].to_numpy().astype(np.int64)
    mapa = calib["preco_medio_grupo"]
    lookup = np.full(max(int(g.max()), max(mapa, default=0)) + 2, calib["preco_medio_global"])
    for k, v in mapa.items():
        lookup[k] = v
    valor = ev["valor"].to_numpy()
    preco_rel = valor / lookup[g]

    K = int(g.max()) + 1
    cont = np.bincount(cesta * K + np.maximum(g, 0), minlength=n_c * K).reshape(n_c, K)
    dados = {
        "ent": ent[ini], "dia": dia[ini], "evento_id": ev["evento_id"].to_numpy()[ultimo],
        "itens": np.diff(np.r_[ini, len(ent)]).astype(np.float64),
        "valor": soma(valor),
        "preco_max": np.maximum.reduceat(valor, ini),
        "preco_rel_soma": soma(preco_rel),
        "itens_novos": soma(primeira_ocorrencia(ent, ev["item"].to_numpy())),
        "tipos_novos": soma(primeira_ocorrencia(ent, ev[f"cat_{ft['campo_tipo']}"].to_numpy())),
        "grupo_dominante": cont.argmax(axis=1),
    }
    del cont
    for bcol in ft["campos_binarios"]:
        dados[f"{bcol}_n"] = soma(ev[f"bin_{bcol}"].to_numpy())
    for c in ft["campos_composicao"]:
        codigos = ev[f"cat_{c}"].to_numpy()
        for k in calib["top"][c]:
            dados[f"{c}_{k}_n"] = soma(codigos == k)
    return pl.DataFrame(dados)


def calcular_features(b, ft, manter=None):
    """Features por cesta. `manter` (máscara booleana sobre as cestas) limita
    o que é GUARDADO (float32) — os acumulados são calculados na tabela
    inteira, porque dependem de todas as cestas anteriores."""
    ent = b["ent"].to_numpy().astype(np.int64)
    dia = b["dia"].to_numpy().astype(np.int64)
    n = len(b)
    nova_ent = np.r_[True, ent[1:] != ent[:-1]]
    inicio = np.flatnonzero(nova_ent)
    idx_inicio = np.repeat(inicio, np.diff(np.r_[inicio, n]))      # índice da 1ª cesta da entidade

    def acum(x):
        """soma acumulada por entidade (inclusiva)"""
        c = np.cumsum(x, dtype=np.float64)
        base = np.r_[0.0, c][idx_inicio]
        return c - base

    chave = ent * 100_000 + dia
    C = {}

    def janela(x, W):
        """soma em (dia-W, dia] da mesma entidade"""
        if id(x) not in C:
            C.clear()  # só um array em cache por vez (memória)
            C[id(x)] = (x, np.r_[0.0, np.cumsum(x, dtype=np.float64)])
        c = C[id(x)][1]
        j = np.searchsorted(chave, chave - W + 1, side="left")
        return c[np.arange(1, n + 1)] - c[j]

    class Saida(dict):
        """guarda cada feature já em float32 e só nas cestas mantidas"""
        def __setitem__(self, k, v):
            v = np.asarray(v)
            super().__setitem__(k, (v[manter] if manter is not None else v).astype(np.float32))

    f = Saida()
    itens = b["itens"].to_numpy().astype(np.float64)
    valor = b["valor"].to_numpy().astype(np.float64)
    um = np.ones(n)

    n_cestas, n_itens, valor_total = acum(um), acum(itens), acum(valor)
    f["n_cestas"], f["n_itens"], f["valor_total"] = n_cestas, n_itens, valor_total
    f["valor_medio_cesta"] = valor_total / n_cestas
    f["itens_medio_cesta"] = n_itens / n_cestas
    f["preco_medio_item"] = valor_total / n_itens
    f["tempo_cliente_dias"] = (dia - dia[idx_inicio]).astype(np.float64)

    gap = np.where(nova_ent, np.nan, np.r_[np.nan, np.diff(dia)]).astype(np.float64)
    tem_gap = ~np.isnan(gap)
    g0 = np.nan_to_num(gap)
    n_gaps = acum(tem_gap.astype(np.float64))
    soma_gap = acum(g0)
    soma_gap2 = acum(g0 ** 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        gap_medio = np.where(n_gaps > 0, soma_gap / n_gaps, np.nan)
        f["gap_ultimo"] = gap
        f["gap_medio"] = gap_medio
        f["gap_desvio"] = np.where(n_gaps > 1, np.sqrt(np.maximum(soma_gap2 / n_gaps - gap_medio ** 2, 0)), np.nan)
        f["gap_ultimo_sobre_medio"] = gap / gap_medio
    gmax = pl.DataFrame({"e": ent, "g": g0}).select(pl.col("g").cum_max().over("e"))["g"].to_numpy()
    f["gap_max"] = np.where(n_gaps > 0, gmax, np.nan)

    cj = {}
    for W in ft["janelas_dias"]:
        cj[W] = janela(um, W)
        f[f"cestas_{W}d"] = cj[W]
        f[f"itens_{W}d"] = janela(itens, W)
        f[f"valor_{W}d"] = janela(valor, W)
    itens_90d = janela(itens, 90)
    with np.errstate(invalid="ignore", divide="ignore"):
        if 30 in cj and 90 in cj:
            f["tendencia_30_90"] = cj[30] / (cj[90] / 3.0)
        if 90 in cj and 365 in cj:
            f["tendencia_90_365"] = cj[90] / (cj[365] / 4.0)
    del cj

    for bcol in ft["campos_binarios"]:
        x = b[f"{bcol}_n"].to_numpy().astype(np.float64)
        f[f"{bcol}_share"] = acum(x) / n_itens
        with np.errstate(invalid="ignore", divide="ignore"):
            f[f"{bcol}_share_90d"] = janela(x, 90) / itens_90d
        f[f"{bcol}_ultima_cesta"] = x / itens

    for c in ft["campos_composicao"]:
        for col in [x for x in b.columns if x.startswith(f"{c}_") and x.endswith("_n")]:
            k = col[len(c) + 1:-2]
            x = b[col].to_numpy().astype(np.float64)
            f[f"{c}_{k}_share"] = acum(x) / n_itens
            with np.errstate(invalid="ignore", divide="ignore"):
                f[f"{c}_{k}_share_90d"] = janela(x, 90) / itens_90d
            C.pop(id(x), None)
    mem("acumulados")

    pmax = pl.DataFrame({"e": ent, "p": b["preco_max"].to_numpy()}).select(pl.col("p").cum_max().over("e"))["p"].to_numpy()
    f["preco_max"] = pmax.astype(np.float64)
    f["preco_relativo_medio"] = acum(b["preco_rel_soma"].to_numpy().astype(np.float64)) / n_itens
    itens_distintos = acum(b["itens_novos"].to_numpy().astype(np.float64))
    f["itens_distintos"] = itens_distintos
    f["taxa_recompra_item"] = 1.0 - itens_distintos / n_itens
    f["tipos_distintos"] = acum(b["tipos_novos"].to_numpy().astype(np.float64))

    f["ultima_cesta_itens"] = itens
    f["ultima_cesta_valor"] = valor
    f["ultima_cesta_preco_relativo"] = b["preco_rel_soma"].to_numpy() / itens
    f["ultima_cesta_grupo_dominante"] = b["grupo_dominante"].to_numpy().astype(np.float64)  # categórica (código)

    datas = DIA_BASE + dia.astype("timedelta64[D]")
    f["mes"] = (datas.astype("datetime64[M]").astype(np.int64) % 12 + 1).astype(np.float64)
    f["dia_semana"] = ((dia + 3) % 7).astype(np.float64)  # 1970-01-01 foi quinta -> 0 = segunda

    sel = (lambda x: x[manter]) if manter is not None else (lambda x: x)
    out = pl.DataFrame(dict(f))
    out = out.with_columns(pl.Series("evento_id", sel(b["evento_id"].to_numpy())),
                           pl.Series("entidade_id", sel(b["ent"].to_numpy())))
    return out


def anexar_estaticos(feats, config, ft):
    """Cadastro da entidade (numéricos como estão, categóricos como código) +
    frequency encoding dos campos de alta cardinalidade."""
    cat_est = config.get("campos_estaticos_categoricos", [])
    num_est = config.get("campos_estaticos_numericos", [])
    freq = ft.get("estaticos_frequencia", [])
    cols = list(dict.fromkeys(["entidade_id"] + cat_est + num_est + freq))
    est = ler_fonte(config, cols).drop_duplicates("entidade_id").reset_index(drop=True)
    dados = {"entidade_id": est["entidade_id"].to_numpy()}
    for c in num_est:
        dados[c] = est[c].to_numpy(dtype=np.float32)
    for c in cat_est:
        dados[f"{c}_cod"] = est[c].astype("category").cat.codes.to_numpy().astype(np.float32)
    for c in freq:
        dados[f"{c}_freq"] = est[c].map(est[c].value_counts()).to_numpy(dtype=np.float32)
    return feats.join(pl.DataFrame(dados), on="entidade_id", how="left")


def anexar_rotulos(feats, config):
    """Rótulos, split, flag de uso downstream e baselines causais numéricos
    (os mesmos que a fusão recebe como feature)."""
    rot = ler_tabela(caminho_tabela(config, "rotulos_downstream", "rotulos"))
    if "usar_downstream" in rot.columns:
        rot = rot[rot["usar_downstream"].to_numpy()].reset_index(drop=True)
    keep = ["evento_id", "split"] + (["usar_downstream"] if "usar_downstream" in rot.columns else [])
    for t in tarefas_ativas(config):
        nome = t.get("nome", t["tipo"])
        keep.append(f"{nome}_rotulo")
        if pd.api.types.is_numeric_dtype(rot[f"{nome}_baseline"]):
            rot[f"{nome}_baseline_feat"] = rot[f"{nome}_baseline"].astype(np.float32)
            keep.append(f"{nome}_baseline_feat")
        keep.append(f"{nome}_baseline")
    rot = rot[keep]
    for c in rot.columns:
        if rot[c].dtype == object:
            rot[c] = rot[c].astype("category")  # texto -> category (memória); polars lê como Categorical
    return feats.join(pl.from_pandas(rot), on="evento_id", how="inner")


def teste_vazamento(ev, calib, ft, feats, n, seed=0):
    """Recalcula âncoras de val/teste só com eventos até a data delas."""
    alvo = feats.filter(pl.col("split").is_in(["val", "test"])).sample(n, seed=seed)
    diffs = 0
    colunas = [c for c in feats.columns if c not in ("evento_id", "entidade_id") and feats[c].dtype in (pl.Float32, pl.Float64)
               and not c.endswith("_baseline_feat") and c not in ("idade", "fn", "active") and not c.endswith(("_cod", "_freq"))]
    for row in alvo.iter_rows(named=True):
        dia_anc = ev.filter(pl.col("evento_id") == row["evento_id"])["dia"][0]
        sub = ev.filter((pl.col("ent") == row["entidade_id"]) & (pl.col("dia") <= dia_anc))
        recalc = calcular_features(cestas(sub, calib, ft), ft).filter(pl.col("evento_id") == row["evento_id"])
        for c in colunas:
            a, b = recalc[c][0], row[c]
            if not ((a is None and b is None) or (a is not None and b is not None and (np.isclose(a, b, rtol=1e-4, equal_nan=True)))):
                diffs += 1
                print(f"  DIFERENÇA evento {row['evento_id']} {c}: completo={b} truncado={a}")
    print(f"Teste de vazamento: {n} âncoras, {len(colunas)} features cada, {diffs} diferenças")
    return diffs == 0


def main(config_path, n_teste_vazamento=0):
    t0 = time.time()
    config = carregar_config(config_path)
    ft = config["features_tradicionais"]
    ev, categorias = carregar_eventos(config, ft)
    mem("eventos")
    calib = calibrar(ev, categorias, ft)
    mem("calibração")
    if n_teste_vazamento:
        # subconjunto de clientes (as features de um cliente só dependem dos
        # eventos dele; a calibração acima usou a base inteira)
        rng = np.random.RandomState(0)
        clientes = rng.choice(np.unique(ev["ent"].to_numpy()), 5000, replace=False)
        ev = ev.filter(pl.col("ent").is_in(clientes))
        sub = calcular_features(cestas(ev, calib, ft), ft)
        rot = ler_tabela(caminho_tabela(config, "rotulos_downstream", "rotulos"), ["evento_id", "split"])
        sub = sub.join(pl.from_pandas(rot.astype({"split": str})), on="evento_id", how="inner")
        ok = teste_vazamento(ev, calib, ft, sub, n_teste_vazamento)
        sys.exit(0 if ok else 1)
    b = cestas(ev, calib, ft)
    mem("cestas agregadas")
    del ev
    ancoras_path = os.path.join(run_dir(config, "rotulos_downstream"), "ancoras.parquet")
    manter = None
    if os.path.exists(ancoras_path):
        manter = np.isin(b["evento_id"].to_numpy(), pd.read_parquet(ancoras_path)["evento_id"].to_numpy())
    feats = calcular_features(b, ft, manter)
    del b
    mem("features")
    feats = anexar_rotulos(feats, config)
    feats = anexar_estaticos(feats, config, ft)
    mem("rótulos e cadastro")

    if "usar_downstream" in feats.columns:
        feats = feats.filter(pl.col("usar_downstream")).drop("usar_downstream")
    out_dir = run_dir(config, "features_tradicionais")
    os.makedirs(out_dir, exist_ok=True)
    feats.write_parquet(os.path.join(out_dir, "features.parquet"))
    nomes_feat = [c for c in feats.columns if c not in ("evento_id", "entidade_id", "split")
                  and not c.endswith(("_rotulo", "_baseline"))]
    info = {
        "n_ancoras": feats.height, "por_split": dict(feats.group_by("split").len().iter_rows()),
        "n_features": len(nomes_feat), "features": nomes_feat,
        "categoricas": [c for c in nomes_feat if c.endswith("_cod") or c in ("ultima_cesta_grupo_dominante", "mes", "dia_semana")],
        "calibracao": {"top": calib["top"], "preco_medio_grupo": calib["preco_medio_grupo"]},
        "tempo_segundos": time.time() - t0,
    }
    with open(os.path.join(out_dir, "features_info.json"), "w", encoding="utf-8") as fh:
        json.dump(info, fh, ensure_ascii=False, indent=2, default=str)
    print(f"Gravado {out_dir}/features.parquet: {feats.height:,} âncoras x {len(nomes_feat)} features "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--teste_vazamento", type=int, default=0)
    a = ap.parse_args()
    main(a.config, a.teste_vazamento)
