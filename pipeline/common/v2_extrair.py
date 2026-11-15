"""Motor v2 — extração: vetor do cliente nas âncoras e recomendação direta do modelo.

Vetores (padrão): o estado do cliente logo depois de cada pedido-âncora, no mesmo
formato do v1 (embeddings_eventos.npy + embeddings_eventos_index.parquet), para
a fusão, o GBDT e a avaliação existentes. As âncoras são as de
rotulos_downstream/ancoras.parquet, se existir; senão, todos os pedidos. Como a
atenção é causal, o estado de uma âncora só depende do que veio antes dela.
Históricos mais longos que a janela usam janelas sobrepostas (passo de meia
janela): cada âncora tem pelo menos meia janela de contexto.

Recomendação (--recomendar AAAA-MM-DD): para cada entidade, os K itens mais
prováveis no próximo pedido, a partir da última posição anterior à data.

Uso: python3 pipeline/common/v2_extrair.py <config.yaml> [--pretreino v2_pretreino] [--saida v2_fusao]
     python3 pipeline/common/v2_extrair.py <config.yaml> --recomendar 2020-09-16 [--k 12] [--saida v2_rec]
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
from v2_modelo import DadosV2, ModeloV2, montar_lote  # noqa: E402
from v2_pretreino import lotes_por_orcamento  # noqa: E402


def carregar_modelo(config, d, pretreino, device):
    ck = torch.load(os.path.join(run_dir(config, pretreino), "melhor.pt"), map_location=device, weights_only=False)
    modelo = ModeloV2(d, **ck["arquitetura"]).to(device)
    modelo.load_state_dict(ck["modelo"])
    return modelo.eval(), ck


def janelas_sobrepostas(d, so_entidades=None, t_max=None):
    """-> lista de (ini, fim, aproveita_de): os estados valem de `aproveita_de` até `fim`."""
    J = d.meta["janela"]
    saida = []
    ents = range(len(d.ent_ini) - 1) if so_entidades is None else so_entidades
    for i in ents:
        a, b = int(d.ent_ini[i]), int(d.ent_ini[i + 1])
        if t_max is not None:
            b = a + int(np.searchsorted(d.t[a:b], t_max, side="left"))
        if b <= a:
            continue
        if b - a <= J:
            saida.append((a, b, a)); continue
        fim = b
        while True:
            ini = max(a, fim - J)
            saida.append((ini, fim, a if ini == a else fim - J // 2))
            if ini == a:
                break
            fim -= J // 2
    return saida


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--pretreino", default="v2_pretreino")
    ap.add_argument("--dados", default="v2", help="pasta da serialização em runs/<nome>/")
    ap.add_argument("--saida", default=None)
    ap.add_argument("--recomendar", default=None, help="AAAA-MM-DD: recomenda a partir da última posição anterior à data")
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--orcamento", type=int, default=120000)
    ap.add_argument("--dispositivo", default="auto", choices=["auto", "cpu", "cuda"])
    a = ap.parse_args()
    t0 = time.time()
    config = carregar_config(a.config)
    d = DadosV2(run_dir(config, a.dados))
    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.dispositivo == "auto" else a.dispositivo
    amp = device == "cuda"
    modelo, ck = carregar_modelo(config, d, a.pretreino, device)
    ent_id = pl.read_parquet(os.path.join(d.pasta, "entidades.parquet"))["entidade"].to_numpy()
    itens = pl.read_parquet(os.path.join(d.pasta, "itens.parquet"))

    if a.recomendar:
        out = run_dir(config, a.saida or "v2_rec")
        os.makedirs(out, exist_ok=True)
        corte = int(np.datetime64(a.recomendar, "m").astype(np.int64))
        # última janela de cada entidade antes da data
        jan = {}
        for ini, fim in d.janelas(corte):
            e = int(d.ent[ini])
            if e not in jan or fim > jan[e][1]:
                jan[e] = (ini, fim)
        item_de = np.r_[-1, itens.sort("idx")["item"].to_numpy()]
        ents, top, pont = [], [], []
        with torch.no_grad():
            for jl in lotes_por_orcamento(d, list(jan.values()), a.orcamento):
                lote = montar_lote(d, jl, corte, com_alvos=False, device=device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    h = modelo.estados(lote)
                ult = lote["valido"].sum(1) - 1
                lp = modelo.pontuar_itens(h[torch.arange(len(jl), device=h.device), ult].float())
                v, i = lp.topk(a.k, dim=-1)
                ents.append(np.array([int(d.ent[x[0]]) for x in jl])); top.append(i.cpu().numpy()); pont.append(v.cpu().numpy())
        ents, top, pont = np.concatenate(ents), np.concatenate(top), np.concatenate(pont)
        pl.DataFrame({"entidade_id": np.repeat(ent_id[ents], a.k), "posicao": np.tile(np.arange(1, a.k + 1), len(ents)),
                      "item": item_de[top.ravel()], "log_prob": pont.ravel().astype(np.float32)}).write_parquet(
            os.path.join(out, f"recomendacoes_{a.recomendar}.parquet"))
        json.dump({"data": a.recomendar, "k": a.k, "n_entidades": int(len(ents)), "corte_do_pretreino_min": int(ck["corte_min"]),
                   "segundos": round(time.time() - t0, 1)}, open(os.path.join(out, f"relatorio_{a.recomendar}.json"), "w"), indent=1)
        print(f"{len(ents):,} entidades, {a.k} itens cada -> {out} ({time.time() - t0:.0f}s)")
        return

    out = run_dir(config, a.saida or "v2_fusao")
    os.makedirs(out, exist_ok=True)
    alvo = np.asarray(d.tipo) == d.meta["tipo_alvo"]
    arq_anc = os.path.join(run_dir(config, "rotulos_downstream"), "ancoras.parquet")
    if os.path.exists(arq_anc):
        anc = pl.read_parquet(arq_anc)["evento_id"].to_numpy()
        eh_anc = alvo & np.isin(np.asarray(d.id_ancora), anc)
        print(f"âncoras do v1: {len(anc):,}; encontradas entre os pedidos: {int(eh_anc.sum()):,}", flush=True)
    else:
        eh_anc = alvo
    pos_anc = np.flatnonzero(eh_anc)
    linha = np.full(len(d.ent), -1, np.int64)
    linha[pos_anc] = np.arange(len(pos_anc))
    dim = ck["arquitetura"]["n_embd"]
    emb = np.lib.format.open_memmap(os.path.join(out, "embeddings_eventos.npy"), mode="w+", dtype=np.float16, shape=(len(pos_anc), dim))
    jan = janelas_sobrepostas(d)
    de = {(i, f): ap_ for i, f, ap_ in jan}
    feitas = 0
    with torch.no_grad():
        for jl in lotes_por_orcamento(d, [(i, f) for i, f, _ in jan], a.orcamento):
            lote = montar_lote(d, jl, d.meta["m_fim"] + 1, com_alvos=False, device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                h = modelo.estados(lote).float().cpu().numpy()
            for r, (i, f) in enumerate(jl):
                p = np.arange(de[(i, f)], f)
                p = p[linha[p] >= 0]
                emb[linha[p]] = h[r, p - i].astype(np.float16)
                feitas += len(p)
    emb.flush()
    pl.DataFrame({"evento_id": np.asarray(d.id_ancora)[pos_anc], "entidade_id": ent_id[np.asarray(d.ent)[pos_anc]].astype(np.int32),
                  "row_idx": np.arange(len(pos_anc))}).write_parquet(os.path.join(out, "embeddings_eventos_index.parquet"))
    json.dump({"n_ancoras": int(len(pos_anc)), "dim": int(dim), "pretreino": a.pretreino, "segundos": round(time.time() - t0, 1)},
              open(os.path.join(out, "v2_extracao.json"), "w"), indent=1)
    print(f"{len(pos_anc):,} âncoras ({feitas:,} gravadas), dim {dim} -> {out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
