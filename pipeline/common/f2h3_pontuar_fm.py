"""F2-H3 — pontuação dos candidatos pelo backbone (modelo fundacional sozinho).

Para cada (cliente, artigo candidato), escreve o bloco do artigo depois do
histórico do cliente até a véspera da semana-alvo e lê, do backbone CONGELADO,
o log da probabilidade de cada token do bloco:

  entrada  [EVT, índice, grupo, tipo, cor, linha, canal, artigo]
  alvo     [índice, grupo, tipo, cor, linha, canal, artigo, preço]

O canal é o da última compra do cliente (fixo entre candidatos, por isso fora
da soma). Todos os candidatos de um cliente vão numa passada só: a sequência é
[histórico | bloco 1 | bloco 2 | ...], cada bloco com as posições que teria logo
depois do histórico e uma máscara que o deixa ver o histórico e a si mesmo,
nunca os outros blocos. Clientes com mais de `--cand_por_passada` candidatos
ocupam várias linhas do lote (o histórico se repete).

Entradas: runs/<nome>/f2h3/<semana>/{candidatos,clientes_fm,artigos_fm}.parquet
(gerados por f2h3_candidatos.py) e o checkpoint `<pretreino>/checkpoints/melhor.pt`.
Saída: runs/<nome>/f2h3/<semana>/fm_<rotulo>.parquet (cliente, artigo, lp_*).

Uso: python3 pipeline/common/f2h3_pontuar_fm.py configs/hm_clientes.yaml --semana 2020-03-05 \\
        --arch_de final_s0/config.yaml --pretreino final_s0/pretreino --rotulo s0
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

sys.path.insert(0, "pipeline/common")
from modelo_decoder import arquitetura_efetiva, construir_modelo  # noqa: E402
from schema import carregar_config, run_dir  # noqa: E402
from sequencias_io import carregar_sequencias  # noqa: E402

CAMPOS = ["indice", "grupo", "tipo", "cor", "linha", "canal", "artigo", "preco"]
BLOCO = 8
TOKENS_POR_LOTE = 49152


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--semana", required=True)
    ap.add_argument("--arch_de", required=True, help="config.yaml (relativo a runs/<nome>) com a arquitetura do backbone")
    ap.add_argument("--pretreino", required=True)
    ap.add_argument("--rotulo", required=True)
    ap.add_argument("--cand_por_passada", type=int, default=64)
    ap.add_argument("--max_clientes", type=int, default=0, help="só para teste rápido")
    a = ap.parse_args()
    t0 = time.time()
    config = carregar_config(a.config)
    base = run_dir(config, "")
    arch = arquitetura_efetiva(carregar_config(os.path.join(base, a.arch_de))["arquitetura"])
    n_pos = arch["n_positions"]
    max_prefixo = n_pos - BLOCO
    d = os.path.join(run_dir(config, "f2h3"), a.semana)

    vocab = json.load(open(os.path.join(run_dir(config, "vocabulario"), "vocab.json"), encoding="utf-8"))
    pad = vocab["PAD"]
    seqs = carregar_sequencias(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = construir_modelo(arch, vocab).to(device)
    model.carregar_pesos(torch.load(os.path.join(base, a.pretreino, "checkpoints", "melhor.pt"), map_location=device))
    model.eval()
    nucleo = model.nucleo

    cf = pl.read_parquet(os.path.join(d, "clientes_fm.parquet")).sort("cliente")
    if a.max_clientes:
        cf = cf.head(a.max_clientes)
    af = pl.read_parquet(os.path.join(d, "artigos_fm.parquet"))
    cand = (pl.read_parquet(os.path.join(d, "candidatos.parquet"), columns=["cliente", "artigo"])
            .join(cf.select("cliente"), on="cliente", how="semi")
            .join(af, on="artigo", how="inner").sort("cliente", "artigo"))
    blocos = cand.select([f"t{k}" for k in range(9)]).to_numpy().astype(np.int64)  # (N, 9)
    cli_c = cand["cliente"].to_numpy()
    cli_ids = cf["cliente"].to_numpy()
    ini_c = np.searchsorted(cli_c, cli_ids, side="left")
    fim_c = np.searchsorted(cli_c, cli_ids, side="right")
    ent = cf["ent_idx"].to_numpy()
    pref = cf["prefixo"].to_numpy()
    canal = cf["tok_canal"].to_numpy()
    print(f"{len(cli_ids):,} clientes, {len(cand):,} pares cliente-artigo", flush=True)

    # unidades de trabalho: (cliente, fatia de candidatos)
    unid = []
    for i in range(len(cli_ids)):
        for s in range(ini_c[i], fim_c[i], a.cand_por_passada):
            unid.append((i, s, min(s + a.cand_por_passada, fim_c[i])))
    tam = np.array([min(pref[i], max_prefixo) + (f - s) * BLOCO for i, s, f in unid])
    ordem = np.argsort(tam, kind="stable")
    lp = np.full((len(cand), len(CAMPOS)), np.nan, dtype=np.float32)
    autocast = torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda")

    def rodar(lote):
        T = int(max(tam[u] for u in lote))
        B = len(lote)
        ids = np.full((B, T), pad, dtype=np.int64)
        posi = np.zeros((B, T), dtype=np.int64)
        masc = np.zeros((B, T, T), dtype=bool)
        alvo_pos, alvo_tok, alvo_lin = [], [], []
        for r, u in enumerate(lote):
            i, s, f = unid[u]
            P = int(pref[i]); L = min(P, max_prefixo)
            ids[r, :L] = seqs.full(ent[i])[P - L:P]
            posi[r, :L] = np.arange(L)
            masc[r, :L, :L] = np.tril(np.ones((L, L), dtype=bool))
            b = blocos[s:f].copy()
            b[:, 6] = canal[i]
            k = f - s
            blk = b[:, :BLOCO].reshape(-1)
            ids[r, L:L + k * BLOCO] = blk
            posi[r, L:L + k * BLOCO] = np.tile(np.arange(L, L + BLOCO), k)
            tri = np.tril(np.ones((BLOCO, BLOCO), dtype=bool))
            for j in range(k):
                o = L + j * BLOCO
                masc[r, o:o + BLOCO, :L] = True
                masc[r, o:o + BLOCO, o:o + BLOCO] = tri
            masc[r, L + k * BLOCO:, :] = False
            masc[r, np.arange(L + k * BLOCO, T), np.arange(L + k * BLOCO, T)] = True  # padding vê a si mesmo (evita NaN)
            alvo_pos.append(np.arange(L, L + k * BLOCO) + r * T)
            alvo_tok.append(b[:, 1:9].reshape(-1))
            alvo_lin.append(np.arange(s, f))
        ids_t = torch.from_numpy(ids).to(device)
        pos_t = torch.from_numpy(posi).to(device)
        m_t = torch.from_numpy(masc).to(device).unsqueeze(1)
        with torch.no_grad(), autocast:
            h = nucleo.estados(ids_t, pos_t, m_t)
            hp = h.reshape(-1, h.shape[-1])[torch.from_numpy(np.concatenate(alvo_pos)).to(device)]
            logp = F.log_softmax(F.linear(hp, nucleo.wte.weight).float(), dim=-1)
            tok = torch.from_numpy(np.concatenate(alvo_tok)).to(device)
            v = logp.gather(1, tok[:, None])[:, 0].reshape(-1, BLOCO).cpu().numpy()
        lp[np.concatenate(alvo_lin)] = v

    atual, maior, feitos, t_ult = [], 0, 0, time.time()
    for u in ordem:
        novo = max(maior, tam[u])
        if atual and (len(atual) + 1) * novo > TOKENS_POR_LOTE:
            rodar(atual); feitos += len(atual)
            atual, novo = [], tam[u]
            if time.time() - t_ult > 60:
                print(f"  {feitos:,}/{len(unid):,} unidades ({time.time() - t0:.0f}s)", flush=True); t_ult = time.time()
        atual.append(u)
        maior = novo
    if atual:
        rodar(atual)

    soma = np.delete(lp, CAMPOS.index("canal"), axis=1).sum(axis=1)
    out = cand.select("cliente", "artigo").with_columns(
        [pl.Series(f"lp_{c}", lp[:, k]) for k, c in enumerate(CAMPOS) if c != "canal"] + [pl.Series("lp_total", soma)])
    out.write_parquet(os.path.join(d, f"fm_{a.rotulo}.parquet"))
    print(f"gravado fm_{a.rotulo}.parquet: {out.height:,} pares, NaN={int(np.isnan(soma).sum())} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
