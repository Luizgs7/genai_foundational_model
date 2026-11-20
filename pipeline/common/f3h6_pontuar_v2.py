"""F3-H6 — pontuação dos candidatos da F2-H3 pelo motor v2 (feature do ranqueador híbrido).

Para cada semana em runs/<nome>/f2h3/<semana>/ e cada cliente com candidatos, o estado depois da
última posição antes da semana pontua o catálogo inteiro; grava a log-probabilidade de cada artigo
candidato ser comprado no próximo pedido (`f2h3/<semana>/v2.parquet`: cliente, artigo, logp).

Mesmo protocolo do braço R4 da F2-H3: um backbone congelado para todas as semanas (aqui o
`v2_pretreino` da F3-H3, com eventos até o corte de validação), lido com o histórico até a véspera
de cada semana. Artigo candidato fora do catálogo do v2 fica sem linha (nulo no ranqueador).

Uso: python3 pipeline/common/f3h6_pontuar_v2.py <config.yaml> [--pretreino v2_pretreino] [--semanas ...]
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import polars as pl
import torch

sys.path.insert(0, "pipeline/common")
from schema import carregar_config, run_dir  # noqa: E402
from v2_extrair import carregar_modelo  # noqa: E402
from v2_modelo import DadosV2, montar_lote  # noqa: E402
from v2_pretreino import lotes_por_orcamento  # noqa: E402

SEMANAS = ["2020-02-20", "2020-02-27", "2020-03-05", "2020-09-02", "2020-09-09", "2020-09-16"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--pretreino", default="v2_pretreino")
    ap.add_argument("--dados", default="v2")
    ap.add_argument("--semanas", nargs="+", default=SEMANAS)
    ap.add_argument("--orcamento", type=int, default=120000)
    a = ap.parse_args()
    config = carregar_config(a.config)
    d = DadosV2(run_dir(config, a.dados))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    modelo, ck = carregar_modelo(config, d, a.pretreino, device)
    ent_id = pl.read_parquet(os.path.join(d.pasta, "entidades.parquet"))["entidade"].to_numpy()
    idx_de = {int(c): i for i, c in enumerate(ent_id)}
    itens = pl.read_parquet(os.path.join(d.pasta, "itens.parquet")).select(pl.col("item").cast(pl.Int64).alias("artigo"),
                                                                           pl.col("idx").cast(pl.Int64))
    rel = {"pretreino": a.pretreino, "corte_do_backbone": str(np.datetime64(ck["corte_min"], "m")), "semanas": {}}
    for sem in a.semanas:
        t0 = time.time()
        pasta = os.path.join(run_dir(config, "f2h3"), sem)
        corte = int(np.datetime64(sem, "m").astype(np.int64))
        if ck["corte_min"] > corte:
            sys.exit(f"o backbone viu eventos depois do início de {sem}")
        cand = (pl.read_parquet(os.path.join(pasta, "candidatos.parquet"), columns=["cliente", "artigo"])
                .with_columns(pl.col("artigo").cast(pl.Int64)).join(itens, on="artigo", how="inner"))
        cand = cand.with_columns(pl.col("cliente").cast(pl.Int64).replace_strict(idx_de, default=-1, return_dtype=pl.Int64).alias("ent"))
        cand = cand.filter(pl.col("ent") >= 0)
        por_ent = {e: (g["artigo"].to_numpy(), g["idx"].to_numpy()) for (e,), g in cand.partition_by("ent", as_dict=True).items()}
        jan = {}
        for ini, fim in d.janelas(corte, so_entidades=sorted(por_ent)):
            e = int(d.ent[ini])
            if e not in jan or fim > jan[e][1]:
                jan[e] = (ini, fim)
        partes = []
        with torch.no_grad():
            for jl in lotes_por_orcamento(d, list(jan.values()), a.orcamento):
                lote = montar_lote(d, jl, corte, com_alvos=False, device=device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                    h = modelo.estados(lote)
                ult = lote["valido"].sum(1) - 1
                lp = modelo.pontuar_itens(h[torch.arange(len(jl), device=h.device), ult].float())
                for r, (ini, _) in enumerate(jl):
                    e = int(d.ent[ini])
                    arts, idx = por_ent[e]
                    v = lp[r, torch.from_numpy(idx).to(lp.device)].cpu().numpy()
                    partes.append(pl.DataFrame({"cliente": np.full(len(arts), ent_id[e], np.int64), "artigo": arts,
                                                "logp": v.astype(np.float32)}))
        out = pl.concat(partes)
        tipos = pl.read_parquet(os.path.join(pasta, "candidatos.parquet"), n_rows=1).schema
        out = out.with_columns(pl.col("cliente").cast(tipos["cliente"]), pl.col("artigo").cast(tipos["artigo"]))
        out.write_parquet(os.path.join(pasta, "v2.parquet"))
        n_cli = cand["cliente"].n_unique()
        rel["semanas"][sem] = {"pares": out.height, "clientes": int(n_cli), "clientes_com_estado": len(jan),
                               "pares_candidatos_no_catalogo": cand.height, "segundos": round(time.time() - t0, 1)}
        print(sem, json.dumps(rel["semanas"][sem]), flush=True)
    os.makedirs(run_dir(config, "f3h6"), exist_ok=True)
    json.dump(rel, open(os.path.join(run_dir(config, "f3h6"), "pontuar_v2.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
