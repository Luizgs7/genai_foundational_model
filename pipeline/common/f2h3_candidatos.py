"""F2-H3 — candidatos, features e tokens para a recomendação do próximo artigo.

Tarefa (competição H&M no Kaggle): para cada cliente que compra na semana-alvo
[ini, ini+7), uma lista de 12 artigos, avaliada por MAP@12. Todos os braços
ranqueiam os MESMOS candidatos, gerados só com dados anteriores a `ini`:

  rep  artigos comprados pelo cliente nas 8 semanas anteriores
  pop  os 100 mais vendidos na semana anterior
  var  outras cores (mesmo product_code) dos artigos de `rep`, entre as mais vendidas em 4 semanas
  co   artigos mais comprados junto (mesma cesta, 4 semanas) com os da última cesta do cliente
  faixa os 50 mais vendidos na semana anterior entre clientes da mesma faixa etária (décadas)

Para cada semana grava em runs/<nome>/f2h3/<ini>/:
  candidatos.parquet  cliente, artigo, rotulo, flags de origem e features tradicionais
  alvos.parquet       cliente, artigo comprados na semana (para o MAP@12, inclui clientes sem candidatos)
  populares.parquet   ranking de popularidade da semana anterior (lista do R0 e fallback)
  clientes_fm.parquet cliente -> índice da entidade, tamanho do prefixo (hist. até ini) e token de canal
  artigos_fm.parquet  artigo -> 9 tokens do bloco [EVT, índice, grupo, tipo, cor, linha, canal, artigo, preço]
                      copiados da venda mais recente antes de `ini` (mesma tokenização do pré-treino)

Semanas de treino do ranqueador usam uma amostra fixa de clientes (`--max_clientes`).

Uso: python3 pipeline/common/f2h3_candidatos.py configs/hm_clientes.yaml --semana 2020-03-05 [--max_clientes 40000]
"""

import argparse
import datetime as dt
import json
import os
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, "pipeline/common")
from schema import carregar_config, run_dir  # noqa: E402

N_POP = 100
N_VAR = 5
N_CO = 5
N_FAIXA = 50
DIAS_REP = 56


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--semana", required=True, help="início da semana-alvo (AAAA-MM-DD)")
    ap.add_argument("--max_clientes", type=int, default=0, help="0 = todos os compradores da semana")
    ap.add_argument("--semente", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    config = carregar_config(a.config)
    ini = dt.date.fromisoformat(a.semana)
    fim = ini + dt.timedelta(days=7)
    out = os.path.join(run_dir(config, "f2h3"), a.semana)
    os.makedirs(out, exist_ok=True)

    ev = (pl.scan_parquet(config["fonte_canonica"])
          .select("evento_id", "entidade_id", "data_evento", "indice", "grupo_produto", "tipo_produto", "cor",
                  "artigo", "preco", "online", "idade")
          .with_columns(pl.col("data_evento").dt.date().alias("d"))
          .rename({"entidade_id": "cliente"}))
    hist_all = ev.filter(pl.col("d") < ini)

    alvos = ev.filter((pl.col("d") >= ini) & (pl.col("d") < fim)).select("cliente", "artigo").unique().collect()
    clientes = alvos.select("cliente").unique().sort("cliente")
    if a.max_clientes and clientes.height > a.max_clientes:
        rng = np.random.RandomState(a.semente)
        clientes = clientes[np.sort(rng.choice(clientes.height, a.max_clientes, replace=False))]
        alvos = alvos.join(clientes, on="cliente", how="semi")
    alvos.write_parquet(os.path.join(out, "alvos.parquet"))
    print(f"semana {ini}..{fim - dt.timedelta(days=1)}: {clientes.height:,} clientes, {alvos.height:,} pares-alvo", flush=True)

    # ---------------------------------------------------------------- artigos (todos os clientes)
    rec = hist_all.filter(pl.col("d") >= ini - dt.timedelta(days=28))
    dias = (pl.lit(ini) - pl.col("d")).dt.total_days()
    art = (rec.group_by("artigo").agg(
        pl.len().alias("art_vendas_4s"),
        (dias < 7).sum().alias("art_vendas_1s"),
        (dias < 14).sum().alias("art_vendas_2s"),
        pl.col("cliente").n_unique().alias("art_clientes_4s"),
        pl.col("preco").mean().alias("art_preco_4s"),
        pl.col("online").mean().alias("art_frac_online"),
        pl.col("idade").mean().alias("art_idade_media_cli"),
    ).with_columns((pl.col("art_vendas_1s") / pl.col("art_vendas_4s")).alias("art_tendencia")).collect())
    estreia = hist_all.group_by("artigo").agg(pl.col("d").min().alias("_d0")).collect()
    art = art.join(estreia, on="artigo", how="left").with_columns(
        (pl.lit(ini) - pl.col("_d0")).dt.total_days().alias("art_dias_desde_estreia")).drop("_d0")
    pop = (art.filter(pl.col("art_vendas_1s") > 0).sort(["art_vendas_1s", "art_vendas_4s", "artigo"], descending=[True, True, False])
           .with_row_index("pop_rank").select("artigo", "pop_rank"))
    pop.write_parquet(os.path.join(out, "populares.parquet"))
    art = art.join(pop, on="artigo", how="left")

    # ---------------------------------------------------------------- histórico dos clientes da semana
    h = hist_all.join(clientes.lazy(), on="cliente", how="semi").collect()
    hd = h.with_columns((pl.lit(ini) - pl.col("d")).dt.total_days().alias("dias"))
    cli = hd.group_by("cliente").agg(
        pl.len().alias("cli_itens"),
        pl.col("d").n_unique().alias("cli_cestas"),
        pl.col("dias").min().alias("cli_dias_desde_ult"),
        (pl.col("dias") < DIAS_REP).sum().alias("cli_itens_8s"),
        pl.col("preco").mean().alias("cli_preco_medio"),
        pl.col("online").mean().alias("cli_frac_online"),
        pl.col("idade").last().alias("cli_idade"),
    )

    # popularidade na faixa etária do cliente (semana anterior)
    faixa = (pl.col("idade") // 10).fill_null(-1).cast(pl.Int16)
    pop_faixa = (rec.filter((pl.lit(ini) - pl.col("d")).dt.total_days() < 7).with_columns(faixa.alias("faixa"))
                 .group_by("faixa", "artigo").len().rename({"len": "art_vendas_1s_faixa"}).collect())
    cli = cli.with_columns((pl.col("cli_idade") // 10).fill_null(-1).cast(pl.Int16).alias("faixa"))

    # candidatos
    rep = (hd.filter(pl.col("dias") < DIAS_REP).group_by("cliente", "artigo")
           .agg(pl.col("dias").min().alias("_d")).select("cliente", "artigo").with_columns(pl.lit(1, pl.Int8).alias("src_rep")))
    pop_top = pop.filter(pl.col("pop_rank") < N_POP).select("artigo")
    c_pop = clientes.join(pop_top, how="cross").with_columns(pl.lit(1, pl.Int8).alias("src_pop"))
    rec_prod = rec.with_columns((pl.col("artigo") // 1000).alias("prod")).group_by("prod", "artigo").len().collect()
    var_top = (rec_prod.sort("len", descending=True).group_by("prod", maintain_order=True).head(N_VAR)
               .select("prod", "artigo"))
    c_var = (rep.with_columns((pl.col("artigo") // 1000).alias("prod")).select("cliente", "prod").unique()
             .join(var_top, on="prod").select("cliente", "artigo").unique().with_columns(pl.lit(1, pl.Int8).alias("src_var")))
    cestas = rec.select("cliente", "d", "artigo").unique().collect()
    pares = (cestas.join(cestas, on=["cliente", "d"], suffix="_b").filter(pl.col("artigo") != pl.col("artigo_b"))
             .group_by("artigo", "artigo_b").len())
    co_top = pares.sort("len", descending=True).group_by("artigo", maintain_order=True).head(N_CO).select("artigo", "artigo_b")
    del pares, cestas
    ult = hd.filter(pl.col("dias") < DIAS_REP)
    ult = ult.filter(pl.col("d") == pl.col("d").max().over("cliente")).select("cliente", "artigo").unique()
    c_co = (ult.join(co_top, on="artigo").select("cliente", pl.col("artigo_b").alias("artigo")).unique()
            .with_columns(pl.lit(1, pl.Int8).alias("src_co")))
    faixa_top = (pop_faixa.sort("art_vendas_1s_faixa", "artigo", descending=[True, False])
                 .group_by("faixa", maintain_order=True).head(N_FAIXA).select("faixa", "artigo"))
    c_faixa = (clientes.join(cli.select("cliente", "faixa"), on="cliente", how="left").with_columns(pl.col("faixa").fill_null(-1))
               .join(faixa_top, on="faixa").select("cliente", "artigo").with_columns(pl.lit(1, pl.Int8).alias("src_faixa")))
    origens = (rep, c_pop, c_var, c_co, c_faixa)
    cand = pl.concat([t.select("cliente", "artigo") for t in origens]).unique()
    for t in origens:
        cand = cand.join(t, on=["cliente", "artigo"], how="left")
    cand = cand.with_columns(pl.col("^src_.*$").fill_null(0))

    # features do par cliente-artigo (histórico inteiro do cliente)
    par = hd.group_by("cliente", "artigo").agg(pl.len().alias("par_compras"), pl.col("dias").min().alias("par_dias_desde_ult"))
    hd = hd.with_columns((pl.col("artigo") // 1000).alias("prod"))
    par_prod = hd.group_by("cliente", "prod").agg(pl.len().alias("par_prod_compras"), pl.col("dias").min().alias("par_prod_dias"))
    attrs = (hist_all.filter(pl.col("d") >= ini - dt.timedelta(days=365)).group_by("artigo")
             .agg(pl.col("tipo_produto").last(), pl.col("grupo_produto").last(), pl.col("indice").last(), pl.col("cor").last())
             .collect())
    tot = hd.group_by("cliente").len().rename({"len": "_n"})
    afin = []
    for campo in ("tipo_produto", "grupo_produto", "indice", "cor"):
        afin.append((campo, hd.group_by("cliente", campo).len().join(tot, on="cliente")
                     .select("cliente", campo, (pl.col("len") / pl.col("_n")).alias(f"afin_{campo}"))))
    cand = (cand.join(par, on=["cliente", "artigo"], how="left")
            .with_columns((pl.col("artigo") // 1000).alias("prod"))
            .join(par_prod, on=["cliente", "prod"], how="left")
            .join(art, on="artigo", how="left")
            .join(cli, on="cliente", how="left")
            .join(attrs, on="artigo", how="left"))
    for campo, t in afin:
        cand = cand.join(t, on=["cliente", campo], how="left").with_columns(pl.col(f"afin_{campo}").fill_null(0.0))
    cand = (cand.join(pop_faixa, on=["faixa", "artigo"], how="left")
            .with_columns(pl.col("art_vendas_1s_faixa").fill_null(0),
                          pl.col("par_compras").fill_null(0), pl.col("par_prod_compras").fill_null(0),
                          (pl.col("art_preco_4s") / pl.col("cli_preco_medio")).alias("x_preco_rel"),
                          (pl.col("art_idade_media_cli") - pl.col("cli_idade")).abs().alias("x_dif_idade"))
            .drop("tipo_produto", "grupo_produto", "indice", "cor", "prod", "faixa"))
    cand = cand.join(alvos.with_columns(pl.lit(1, pl.Int8).alias("rotulo")), on=["cliente", "artigo"], how="left")
    cand = cand.with_columns(pl.col("rotulo").fill_null(0)).sort("cliente", "artigo")
    cand.write_parquet(os.path.join(out, "candidatos.parquet"))
    n_pos = int(cand["rotulo"].sum())
    print(f"candidatos: {cand.height:,} ({cand.height / clientes.height:.0f}/cliente); "
          f"cobertura dos pares-alvo: {n_pos / alvos.height:.4f}", flush=True)

    # ---------------------------------------------------------------- tokens para o backbone
    ser = run_dir(config, "serializacao")
    ents = pl.read_parquet(os.path.join(ser, "entidades.parquet"), columns=["entidade_id", "offset"]).with_row_index("ent_idx")
    pos = pl.scan_parquet(os.path.join(ser, "posicoes_evento.parquet"))
    tokens = np.load(os.path.join(ser, "tokens.npy"), mmap_mode="r")
    vocab = json.load(open(os.path.join(run_dir(config, "vocabulario"), "vocab.json"), encoding="utf-8"))
    evt = vocab["EVT"]

    def inicio_bloco(off, pf):
        """índice global do token EVT do bloco que termina em `pf` (10 tokens sem recência, 11 com)."""
        g = off + pf
        com_rec = tokens[g - 10] == evt
        return np.where(com_rec, g - 10, g - 9)

    # clientes: prefixo = seq_full[: pos_full do último evento antes de ini + 1]
    ult_ev = (h.sort("d", "evento_id").group_by("cliente").agg(pl.col("evento_id").last())
              .join(pos.select("evento_id", "pos_full").collect(), on="evento_id")
              .join(ents.rename({"entidade_id": "cliente"}), on="cliente"))
    off = ult_ev["offset"].to_numpy().astype(np.int64)
    pf = ult_ev["pos_full"].to_numpy().astype(np.int64)
    canal = tokens[inicio_bloco(off, pf) + 6]
    clientes_fm = ult_ev.select("cliente", "ent_idx", (pl.col("pos_full") + 1).alias("prefixo")).with_columns(
        pl.Series("tok_canal", canal.astype(np.int32)))
    clientes_fm.write_parquet(os.path.join(out, "clientes_fm.parquet"))

    # artigos: bloco da venda mais recente antes de ini
    arts = cand.select("artigo").unique()
    ref = (hist_all.join(arts.lazy(), on="artigo", how="semi").select("artigo", "cliente", "d", "evento_id")
           .sort("d", "evento_id").group_by("artigo").agg(pl.col("evento_id").last(), pl.col("cliente").last())
           .join(pos.select("evento_id", "pos_full"), on="evento_id")
           .collect().join(ents.rename({"entidade_id": "cliente"}), on="cliente"))
    ini_b = inicio_bloco(ref["offset"].to_numpy().astype(np.int64), ref["pos_full"].to_numpy().astype(np.int64))
    blocos = np.stack([tokens[ini_b + k] for k in range(9)], axis=1).astype(np.int32)
    assert (blocos[:, 0] == evt).all()
    artigos_fm = ref.select("artigo").with_columns([pl.Series(f"t{k}", blocos[:, k]) for k in range(9)])
    artigos_fm.write_parquet(os.path.join(out, "artigos_fm.parquet"))
    print(f"tokens: {clientes_fm.height:,} clientes com histórico, {artigos_fm.height:,} artigos ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
