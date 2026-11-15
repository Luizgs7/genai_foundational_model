"""Motor v2 — serialização: a linha do tempo de eventos compostos de cada entidade.
Desenho em MOTOR_V2.md (seções 3 e 4).

Lê as fontes declaradas em `v2:` no YAML (v2_esquema.py) e grava em runs/<nome>/v2/:

  itens.parquet      catálogo: item, idx (1..n; 0 = sem item), atributos codificados, visto_treino
  pos_*.npy          uma linha por POSIÇÃO (pedido, sessão de navegação), ordenadas por entidade e tempo:
                       ent, t_min (minutos desde 1970), tipo (fonte), split (0 treino, 1 val, 2 teste),
                       n_lin, valor, cab (campos do cabeçalho), estado (valor vigente das fontes de estado),
                       lin_ini (início das linhas da posição), id_ancora (id da última linha, para os rótulos do v1)
  lin_*.npy          uma linha por LINHA de evento: item (idx no catálogo) e campos (faixas / categorias)
  ent_*.npy          id e início das posições de cada entidade
  alvo_*.npy         por posição: índice do próximo pedido (alvo_prox), contagem e gasto de pedidos nos
                     30/90/180 dias seguintes. As MÁSCARAS de censura dependem da data de corte e são
                     calculadas no treino (v2_modelo.alvos_do_lote), nunca aqui.
  meta.json          vocabulários, limites das faixas, cardinalidades, datas de corte, contagens

Códigos: 0 = ausente, 1 = valor que não apareceu no treino, 2.. = valores vistos no treino.
Faixas numéricas: quantis calculados só com linhas de treino.

Uso: python3 pipeline/common/v2_serializacao.py <config.yaml> [--dados v2] [--janela N]
         [--sem_fonte NOME ...] [--sem_campo FONTE.CAMPO ...]
  --dados       pasta de saída em runs/<nome>/ (variantes da ablação da F3-H7 lado a lado)
  --sem_fonte   retira uma fonte inteira (não pode ser a fonte-alvo)
  --sem_campo   retira um campo de uma fonte (linha, cabeçalho ou estado)
  --janela      sobrepõe o tamanho da janela do config
"""

import argparse
import json
import os
import sys
import time
from datetime import timedelta

import numpy as np
import polars as pl

sys.path.insert(0, "pipeline/common")
from schema import carregar_config, run_dir  # noqa: E402
from v2_esquema import FAIXAS_CONTAGEM, FAIXAS_INTERVALO_DIAS, FAIXAS_N_LINHAS, esquema  # noqa: E402

N_FAIXAS = 10
DESLOC_ENT = 26        # chave ordenável = (entidade << 26) | minutos desde 1970 (cabe até 2097)


def unicos(xs):
    return list(dict.fromkeys(x for x in xs if x))


def vocabulario(serie_treino):
    vals = serie_treino.drop_nulls().cast(pl.Utf8).unique().sort().to_list()
    return {v: i + 2 for i, v in enumerate(vals)}


def codificar(serie, voc):
    s = serie.cast(pl.Utf8)
    nulo = s.is_null().to_numpy()
    if not voc:
        return np.where(nulo, 0, 1).astype(np.int32)
    ids = s.fill_null("").replace_strict(list(voc.keys()), list(voc.values()), default=1, return_dtype=pl.Int32)
    return np.where(nulo, 0, ids.to_numpy()).astype(np.int32)


def limites_quantil(valores, n=N_FAIXAS):
    v = np.asarray(valores, dtype=np.float64)
    v = v[~np.isnan(v)]
    if not len(v):
        return []
    return np.unique(np.quantile(v, np.linspace(0, 1, n + 1)[1:-1])).tolist()


def em_faixas(valores, limites):
    v = np.asarray(valores, dtype=np.float64)
    ids = np.searchsorted(np.asarray(limites, dtype=np.float64), v, side="right") + 1
    return np.where(np.isnan(v), 0, ids).astype(np.int32)


def ler_tabela_fonte(f, cols):
    """lê as colunas pedidas; com `filtro`, lê a tabela inteira antes (o filtro pode usar outras colunas)."""
    if f["filtro"]:
        return pl.read_parquet(f["tabela"]).filter(pl.sql_expr(f["filtro"])).select(cols)
    return pl.read_parquet(f["tabela"], columns=cols)


def minutos(expr):
    return (expr.cast(pl.Datetime("ms")).dt.epoch("ms") // 60000).cast(pl.Int64)


def variante(v2, sem_fonte, sem_campo, janela):
    """aplica a ablação pedida ao esquema: retira fontes e campos, troca a janela."""
    alvo = v2["fonte_alvo"]
    if alvo in sem_fonte:
        raise ValueError(f"a fonte-alvo ({alvo}) não pode ser retirada")
    nomes = {f["nome"] for f in v2["fontes"]}
    for n in sem_fonte:
        if n not in nomes:
            raise ValueError(f"fonte desconhecida: {n}")
    v2["fontes"] = [f for f in v2["fontes"] if f["nome"] not in sem_fonte]
    for fc in sem_campo:
        fonte, campo = fc.split(".", 1)
        f = next((x for x in v2["fontes"] if x["nome"] == fonte), None)
        achou = False
        listas = [f[k] for k in ("numericos_linha", "categoricos_linha", "numericos", "categoricos")] if f else []
        if f and isinstance(f.get("cabecalho"), dict):
            listas += [f["cabecalho"].get("categoricos", []), f["cabecalho"].get("numericos", [])]
        for lista in listas:
            if campo in lista:
                lista.remove(campo); achou = True
        if not achou:
            raise ValueError(f"campo desconhecido: {fc}")
    if janela:
        v2["janela"] = janela
    return v2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--dados", default="v2")
    ap.add_argument("--sem_fonte", nargs="*", default=[])
    ap.add_argument("--sem_campo", nargs="*", default=[])
    ap.add_argument("--janela", type=int, default=None)
    a = ap.parse_args()
    t0 = time.time()
    config = carregar_config(a.config)
    v2 = variante(esquema(config), a.sem_fonte, a.sem_campo, a.janela)
    out = run_dir(config, a.dados)
    os.makedirs(out, exist_ok=True)
    fontes_pos = [f for f in v2["fontes"] if f["natureza"] in ("itens", "evento")]
    fontes_est = [f for f in v2["fontes"] if f["natureza"] == "estado"]
    alvo = next(f for f in v2["fontes"] if f.get("alvo"))
    meta = {"janela": v2["janela"], "janelas_dias": v2["janelas_dias"], "fontes": [f["nome"] for f in fontes_pos],
            "fonte_alvo": alvo["nome"], "campos_linha": [], "campos_cab": [], "campos_estado": [], "vocab": {}, "limites": {}}

    # ------------------------------------------------------------ leitura das fontes com posição
    def ler(f):
        grupo = [g for g in (f.get("agrupar_por") or []) if g != "_dia"]
        cab = f.get("cabecalho") or {}
        cab_derivado = ((cab.get("categoricos") or []) + (cab.get("numericos") or [])) if cab.get("derivado") else []
        chave_cab = (cab.get("chave") or []) if cab.get("tabela") else []
        cols = unicos([f["entidade"], f["data"]] + grupo + [f["item"]] + f["numericos_linha"] + f["categoricos_linha"]
                      + f["valor_linha"] + [f["id_linha"]] + cab_derivado + chave_cab)
        df = ler_tabela_fonte(f, cols)
        df = df.rename({f["entidade"]: "_ent", f["data"]: "_t"}).with_columns(minutos(pl.col("_t")).alias("_tmin"))
        if "_dia" in (f.get("agrupar_por") or []):
            df = df.with_columns((pl.col("_tmin") // 1440).alias("_dia"))
        return df

    dados = {f["nome"]: ler(f) for f in fontes_pos}
    da = dados[alvo["nome"]]
    # entidades = as que têm ao menos um evento da fonte-alvo
    ent_ids = da["_ent"].unique().sort()
    n_ent = ent_ids.len()
    mapa_ent = pl.DataFrame({"_ent": ent_ids, "_e": np.arange(n_ent, dtype=np.int32)})
    # datas de corte: a mesma regra do v1 (tarefa7_splits), sobre a fonte-alvo
    fim = da["_t"].max()
    corte_teste = fim - timedelta(days=config["splits"]["test_window_days"])
    corte_val = corte_teste - timedelta(days=config["splits"]["val_window_days"])
    m_val = int(pl.Series([corte_val]).cast(pl.Datetime("ms")).dt.epoch("ms")[0] // 60000)
    m_teste = int(pl.Series([corte_teste]).cast(pl.Datetime("ms")).dt.epoch("ms")[0] // 60000)
    m_fim = int(pl.Series([fim]).cast(pl.Datetime("ms")).dt.epoch("ms")[0] // 60000)
    meta.update({"corte_val": str(corte_val), "corte_teste": str(corte_teste), "fim": str(fim),
                 "m_val": m_val, "m_teste": m_teste, "m_fim": m_fim, "n_entidades": int(n_ent)})
    print(f"{n_ent:,} entidades; treino até {corte_val}, validação até {corte_teste}, fim {fim}", flush=True)

    # ------------------------------------------------------------ catálogo
    cat = v2["catalogo"]
    if cat.get("tabela"):
        c = pl.read_parquet(cat["tabela"], columns=unicos([cat["item"]] + cat["categoricos"] + cat["numericos"]))
    else:   # derivado da fonte-alvo: os atributos do item são colunas da própria tabela de linhas
        c = (pl.scan_parquet(alvo["tabela"]).select(unicos([cat["item"]] + cat["categoricos"] + cat["numericos"]))
             .group_by(cat["item"]).agg(pl.all().first()).collect())
    c = c.rename({cat["item"]: "_item"}).unique("_item").sort("_item")
    # itens que aparecem nas fontes e não estão no catálogo entram sem atributos
    tipo_item = c["_item"].dtype
    vistos = pl.concat([dados[f["nome"]].select(pl.col(f["item"]).cast(tipo_item).alias("_item")) for f in fontes_pos if f["item"]]).drop_nulls().unique()
    c = pl.concat([c, vistos.join(c, on="_item", how="anti")], how="diagonal").sort("_item")
    n_itens = c.height
    c = c.with_columns(pl.Series("_idx", np.arange(1, n_itens + 1, dtype=np.int32)))
    treino_alvo = da.filter(pl.col("_tmin") < m_val)
    vistos_treino = treino_alvo.select(pl.col(alvo["item"]).cast(tipo_item).alias("_item")).drop_nulls().unique()
    c = c.join(vistos_treino.with_columns(pl.lit(True).alias("visto_treino")), on="_item", how="left").with_columns(pl.col("visto_treino").fill_null(False))
    atrib = np.zeros((n_itens + 1, len(cat["categoricos"]) + len(cat["numericos"])), dtype=np.int32)
    cards_atrib = []
    ctr = c.filter(pl.col("visto_treino"))
    for j, a in enumerate(cat["categoricos"]):
        voc = vocabulario(ctr[a])
        atrib[1:, j] = codificar(c[a], voc)
        meta["vocab"][f"item.{a}"] = voc; cards_atrib.append(len(voc) + 2)
    for j, a in enumerate(cat["numericos"]):
        lim = limites_quantil(ctr[a].cast(pl.Float64).to_numpy())
        atrib[1:, len(cat["categoricos"]) + j] = em_faixas(c[a].cast(pl.Float64).fill_null(np.nan).to_numpy(), lim)
        meta["limites"][f"item.{a}"] = lim; cards_atrib.append(len(lim) + 2)
    np.save(os.path.join(out, "item_atrib.npy"), atrib)
    np.save(os.path.join(out, "item_visto_treino.npy"), np.r_[False, c["visto_treino"].to_numpy()])
    c.select(["_item", "_idx", "visto_treino"]).rename({"_item": "item", "_idx": "idx"}).write_parquet(os.path.join(out, "itens.parquet"))
    mapa_item = c.select(["_item", "_idx"])
    meta.update({"n_itens": int(n_itens), "atributos_item": cat["categoricos"] + cat["numericos"], "cards_atrib": cards_atrib,
                 "itens_vistos_no_treino": int(c["visto_treino"].sum())})
    print(f"catálogo: {n_itens:,} itens ({meta['itens_vistos_no_treino']:,} vistos no treino), {len(cards_atrib)} atributos ({time.time() - t0:.0f}s)", flush=True)

    # ------------------------------------------------------------ posições e linhas de cada fonte
    pos_partes, lin_partes = [], []
    for tipo, f in enumerate(fontes_pos):
        d = dados.pop(f["nome"]).join(mapa_ent, on="_ent", how="inner")
        if f["natureza"] == "itens":
            grupo = f.get("agrupar_por") or []
            chaves = [g for g in grupo if g != "entidade_id" and g != f["entidade"]]
            d = d.sort(["_e", "_tmin"] + chaves)
            muda = (pl.col("_e") != pl.col("_e").shift(1)).fill_null(True)
            for g in chaves:
                muda = muda | (pl.col(g) != pl.col(g).shift(1)).fill_null(True)
        else:       # evento: sessão = eventos da mesma entidade a menos de `sessao_minutos` um do outro
            d = d.sort(["_e", "_tmin"])
            muda = ((pl.col("_e") != pl.col("_e").shift(1)) | ((pl.col("_tmin") - pl.col("_tmin").shift(1)) > f["sessao_minutos"])).fill_null(True)
        d = d.with_columns((muda.cast(pl.Int64).cum_sum() - 1).alias("_g"))
        # item e campos de linha
        if f["item"]:
            d = (d.with_columns(pl.col(f["item"]).cast(tipo_item).alias("_item"))
                 .join(mapa_item, on="_item", how="left", maintain_order="left").with_columns(pl.col("_idx").fill_null(0)))
        else:
            d = d.with_columns(pl.lit(0, dtype=pl.Int32).alias("_idx"))
        treino = d.filter(pl.col("_tmin") < m_val)
        campos = {}
        for a in f["numericos_linha"]:
            lim = limites_quantil(treino[a].cast(pl.Float64).fill_null(np.nan).to_numpy())
            campos[f"{f['nome']}.{a}"] = (em_faixas(d[a].cast(pl.Float64).fill_null(np.nan).to_numpy(), lim), len(lim) + 2)
            meta["limites"][f"{f['nome']}.{a}"] = lim
        for a in f["categoricos_linha"]:
            voc = vocabulario(treino[a])
            campos[f"{f['nome']}.{a}"] = (codificar(d[a], voc), len(voc) + 2)
            meta["vocab"][f"{f['nome']}.{a}"] = voc
        valor = None
        if f["valor_linha"]:
            e = pl.lit(1.0)
            for a in f["valor_linha"]:
                e = e * pl.col(a).cast(pl.Float64)
            d = d.with_columns(e.alias("_valor"))
            valor = "_valor"
        ags = [pl.col("_e").first(), pl.col("_tmin").min(), pl.len().alias("_n")]
        ags.append(pl.col(valor).sum().alias("_valor") if valor else pl.lit(0.0).alias("_valor"))
        ags.append(pl.col(f["id_linha"]).max().alias("_id") if f["id_linha"] else pl.lit(-1, dtype=pl.Int64).alias("_id"))
        cab = f.get("cabecalho") or {}
        cab_cols = (cab.get("categoricos") or []) + (cab.get("numericos") or [])
        if cab.get("derivado"):
            ags += [pl.col(a).first() for a in cab_cols]
        chave_cab = cab.get("chave") or []
        if cab.get("tabela"):
            ags += [pl.col(k).first() for k in chave_cab]
        p = d.group_by("_g").agg(ags)
        if cab.get("tabela"):
            p = p.join(pl.read_parquet(cab["tabela"], columns=unicos(chave_cab + cab_cols)), on=chave_cab, how="left")
        p = p.sort("_g")            # índice da posição dentro da fonte = _g
        assert d["_g"].is_sorted() and p.height == int(d["_g"].max()) + 1   # as linhas já estão na ordem das posições
        ptr = p.filter(pl.col("_tmin") < m_val)
        cabs = {}
        for a in cab.get("categoricos") or []:
            voc = vocabulario(ptr[a])
            cabs[f"{f['nome']}.{a}"] = (codificar(p[a], voc), len(voc) + 2)
            meta["vocab"][f"cab.{f['nome']}.{a}"] = voc
        for a in cab.get("numericos") or []:
            lim = limites_quantil(ptr[a].cast(pl.Float64).fill_null(np.nan).to_numpy())
            cabs[f"{f['nome']}.{a}"] = (em_faixas(p[a].cast(pl.Float64).fill_null(np.nan).to_numpy(), lim), len(lim) + 2)
            meta["limites"][f"cab.{f['nome']}.{a}"] = lim
        pos_partes.append({"tipo": tipo, "e": p["_e"].to_numpy(), "t": p["_tmin"].to_numpy(), "n": p["_n"].to_numpy(),
                           "valor": p["_valor"].to_numpy().astype(np.float32), "id": p["_id"].to_numpy(), "cab": cabs})
        lin_partes.append({"g": d["_g"].to_numpy(), "item": d["_idx"].to_numpy().astype(np.int32), "campos": campos})
        print(f"fonte {f['nome']}: {p.height:,} posições, {d.height:,} linhas ({time.time() - t0:.0f}s)", flush=True)
        del d, p, treino, ptr

    # ------------------------------------------------------------ linha do tempo única
    n_por = [len(x["e"]) for x in pos_partes]
    e = np.concatenate([x["e"] for x in pos_partes]); t = np.concatenate([x["t"] for x in pos_partes])
    tipo = np.concatenate([np.full(n, x["tipo"], np.int8) for n, x in zip(n_por, pos_partes)])
    ordem = np.lexsort((tipo, t, e))                       # entidade, tempo, tipo
    inv = np.empty(len(ordem), np.int64); inv[ordem] = np.arange(len(ordem))
    n_pos = len(ordem)
    e, t, tipo = e[ordem], t[ordem], tipo[ordem]
    n_lin = np.concatenate([x["n"] for x in pos_partes])[ordem].astype(np.int32)
    valor = np.concatenate([x["valor"] for x in pos_partes])[ordem]
    id_anc = np.concatenate([x["id"] for x in pos_partes])[ordem].astype(np.int64)
    nomes_cab = unicos([k for x in pos_partes for k in x["cab"]])
    cab = np.zeros((n_pos, max(len(nomes_cab), 1)), np.int32)
    base = 0
    for n, x in zip(n_por, pos_partes):
        for k, (ids, _) in x["cab"].items():
            cab[inv[base:base + n], nomes_cab.index(k)] = ids
        base += n
    cards_cab = [next(x["cab"][k][1] for x in pos_partes if k in x["cab"]) for k in nomes_cab]
    # linhas: reindexadas pela posição global e ordenadas por ela
    nomes_lin = unicos([k for x in lin_partes for k in x["campos"]])
    pos_da_linha = np.concatenate([inv[b + x["g"]] for b, x in zip(np.r_[0, np.cumsum(n_por)[:-1]], lin_partes)])
    ordem_l = np.argsort(pos_da_linha, kind="stable")
    lin_item = np.concatenate([x["item"] for x in lin_partes])[ordem_l]
    lin_campos = np.zeros((len(ordem_l), max(len(nomes_lin), 1)), np.int16)
    dest = np.empty(len(ordem_l), np.int64); dest[ordem_l] = np.arange(len(ordem_l))
    base = 0
    for x in lin_partes:
        n = len(x["g"])
        for k, (ids, _) in x["campos"].items():
            lin_campos[dest[base:base + n], nomes_lin.index(k)] = ids
        base += n
    cards_lin = [next(x["campos"][k][1] for x in lin_partes if k in x["campos"]) for k in nomes_lin]
    lin_ini = np.r_[0, np.cumsum(n_lin)].astype(np.int64)
    assert lin_ini[-1] == len(lin_item)
    split = np.where(t < m_val, 0, np.where(t < m_teste, 1, 2)).astype(np.int8)
    ent_ini = np.r_[0, np.cumsum(np.bincount(e, minlength=n_ent))].astype(np.int64)
    del lin_partes, pos_partes

    # ------------------------------------------------------------ estado vigente em cada posição
    nomes_est, cards_est, col_est = [], [], []
    pos_df = pl.DataFrame({"_e": e, "_tmin": t}).with_row_index("_p")
    for f in fontes_est:
        cols = unicos([f["entidade"], f["data"]] + f["numericos"] + f["categoricos"])
        s = ler_tabela_fonte(f, cols)
        s = (s.rename({f["entidade"]: "_ent"}).with_columns(minutos(pl.col(f["data"])).alias("_tmin"))
             .join(mapa_ent, on="_ent", how="inner").sort(["_e", "_tmin"]))
        st = s.filter(pl.col("_tmin") < m_val)
        j = pos_df.sort(["_e", "_tmin"]).join_asof(s.select(["_e", "_tmin"] + f["numericos"] + f["categoricos"]),
                                                   on="_tmin", by="_e", strategy="backward").sort("_p")
        for a in f["numericos"]:
            lim = limites_quantil(st[a].cast(pl.Float64).fill_null(np.nan).to_numpy())
            col_est.append(em_faixas(j[a].cast(pl.Float64).fill_null(np.nan).to_numpy(), lim))
            nomes_est.append(f"{f['nome']}.{a}"); cards_est.append(len(lim) + 2); meta["limites"][f"estado.{f['nome']}.{a}"] = lim
        for a in f["categoricos"]:
            voc = vocabulario(st[a])
            col_est.append(codificar(j[a], voc))
            nomes_est.append(f"{f['nome']}.{a}"); cards_est.append(len(voc) + 2); meta["vocab"][f"estado.{f['nome']}.{a}"] = voc
        print(f"estado {f['nome']}: {s.height:,} registros anexados às posições ({time.time() - t0:.0f}s)", flush=True)
    estado = np.stack(col_est, axis=1).astype(np.int16) if col_est else np.zeros((n_pos, 1), np.int16)

    # ------------------------------------------------------------ alvos (sem máscara de censura: ela depende do corte)
    tipo_alvo = [f["nome"] for f in fontes_pos].index(alvo["nome"])
    eh_alvo = tipo == tipo_alvo
    ia = np.flatnonzero(eh_alvo)
    prox_incl = np.minimum.accumulate(np.where(eh_alvo, np.arange(n_pos), n_pos)[::-1])[::-1]
    prox = np.r_[prox_incl[1:], n_pos]
    prox = np.where((prox < n_pos) & (e[np.minimum(prox, n_pos - 1)] == e), prox, -1).astype(np.int64)
    chave = (e.astype(np.int64) << DESLOC_ENT) | t
    chave_a = chave[ia]
    soma_a = np.r_[0.0, np.cumsum(valor[ia].astype(np.float64))]
    lo = np.searchsorted(chave_a, chave, side="right")
    for w in v2["janelas_dias"]:
        hi = np.searchsorted(chave_a, chave + w * 1440, side="right")
        np.save(os.path.join(out, f"alvo_cont_{w}.npy"), (hi - lo).astype(np.int32))
        np.save(os.path.join(out, f"alvo_gasto_{w}.npy"), (soma_a[hi] - soma_a[lo]).astype(np.float32))
        tr = (t + w * 1440 < m_val) & (hi > lo)
        meta["limites"][f"alvo.gasto_{w}"] = limites_quantil((soma_a[hi] - soma_a[lo])[tr], 9)
    meta["limites"]["alvo.valor"] = limites_quantil(valor[eh_alvo & (t < m_val)])

    for nome, arr in (("pos_ent", e.astype(np.int32)), ("pos_t_min", t.astype(np.int64)), ("pos_tipo", tipo), ("pos_split", split),
                      ("pos_n_lin", n_lin), ("pos_valor", valor), ("pos_cab", cab.astype(np.int16)), ("pos_estado", estado),
                      ("pos_lin_ini", lin_ini), ("pos_id_ancora", id_anc), ("lin_item", lin_item), ("lin_campos", lin_campos),
                      ("ent_ini", ent_ini), ("alvo_prox", prox)):
        np.save(os.path.join(out, nome + ".npy"), arr)
    pl.DataFrame({"entidade": ent_ids}).write_parquet(os.path.join(out, "entidades.parquet"))
    meta.update({"campos_linha": nomes_lin, "cards_linha": cards_lin, "campos_cab": nomes_cab, "cards_cab": cards_cab,
                 "campos_estado": nomes_est, "cards_estado": cards_est, "n_tipos": len(fontes_pos), "tipo_alvo": int(tipo_alvo),
                 "n_posicoes": int(n_pos), "n_linhas": int(len(lin_item)), "n_posicoes_alvo": int(eh_alvo.sum()),
                 "faixas_intervalo_dias": FAIXAS_INTERVALO_DIAS, "faixas_n_linhas": FAIXAS_N_LINHAS, "faixas_contagem": FAIXAS_CONTAGEM,
                 "posicoes_por_split": {s: int((split == k).sum()) for k, s in enumerate(("train", "val", "test"))},
                 "posicoes_por_entidade": {f"p{q}": float(np.percentile(np.diff(ent_ini), q)) for q in (50, 90, 99, 100)},
                 "linhas_por_posicao": {f"p{q}": float(np.percentile(n_lin, q)) for q in (50, 90, 99, 100)},
                 "segundos": round(time.time() - t0, 1)})
    json.dump(meta, open(os.path.join(out, "meta.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps({k: meta[k] for k in ("n_entidades", "n_itens", "n_posicoes", "n_linhas", "n_posicoes_alvo", "posicoes_por_split",
                                           "posicoes_por_entidade", "linhas_por_posicao", "campos_linha", "campos_cab", "campos_estado", "segundos")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
