"""F3-H8 — forma: cobertura do contexto e custo do pré-treino em função do tamanho do pedido.

Cobertura (da serialização do v2):
  - linhas serializadas contra linhas da fonte canônica (nenhuma linha cortada);
  - posições por entidade contra a janela (entidades cujo histórico inteiro cabe numa janela).

Custo: um passo de treino (estados + perdas + backward, bf16) num lote fixo de janelas reais, em que
todas as posições recebem exatamente L linhas (itens e campos sorteados dos reais) e o próximo pedido
também tem L itens. Só L varia: se o custo por pedido for linear no tamanho do pedido, o tempo é
a + b·L e a inclinação log-log, para L grande, fica perto de 1 (ou abaixo).

Uso: python3 pipeline/common/f3h8_custo.py <config.yaml> [--dados v2] [--pretreino pre_v2]
        [--janelas 16 --T 256] [--L 1,2,4,...] [--saida custo]
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
from v2_modelo import FAIXAS_N_LINHAS, DadosV2, ModeloV2, faixa, montar_lote  # noqa: E402


def cobertura(config, d):
    m = d.meta
    ent_ini = np.asarray(d.ent_ini)
    n_pos = np.diff(ent_ini)
    J = m["janela"]
    v2 = config["v2"]
    fonte = next(f for f in v2["fontes"] if f.get("alvo"))
    canon = pl.scan_parquet(config["fonte_canonica"]).select(pl.len()).collect().item()
    tipo = np.asarray(d.tipo)
    n_lin = np.asarray(d.n_lin)
    # posições de pedido: as que têm próximo-pedido como alvo são do tipo da fonte-alvo (tipo 0 na serialização)
    lin_pedido = int(n_lin[tipo == 0].sum())
    return {"janela": J, "n_entidades": int(len(n_pos)), "posicoes_max": int(n_pos.max()),
            "posicoes_p99": float(np.percentile(n_pos, 99)),
            "entidades_acima_da_janela": int((n_pos > J).sum()),
            "cobertura_entidades": float((n_pos <= J).mean()),
            "linhas_por_posicao_max": int(n_lin.max()),
            "linhas_fonte_canonica": int(canon), "linhas_pedido_serializadas": lin_pedido,
            "fonte_alvo": fonte["nome"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--dados", default="v2")
    ap.add_argument("--pretreino", default="pre_v2")
    ap.add_argument("--janelas", type=int, default=8)
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--L", default="1,2,4,8,16,32,64,128,256,512,900")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--saida", default="custo")
    a = ap.parse_args()
    config = carregar_config(a.config)
    d = DadosV2(run_dir(config, a.dados))
    res = {"cobertura": cobertura(config, d)}
    print(json.dumps(res["cobertura"], ensure_ascii=False), flush=True)

    device = "cuda"
    ck = torch.load(os.path.join(run_dir(config, a.pretreino), "melhor.pt"), map_location=device, weights_only=False)
    modelo = ModeloV2(d, **ck["arquitetura"]).to(device)
    modelo.load_state_dict(ck["modelo"])
    modelo.train()
    rng = np.random.RandomState(0)
    corte = d.meta["m_val"]
    jan = [(i, f) for i, f in d.janelas(corte) if f - i >= a.T]
    jan = [(f - a.T, f) for i, f in (jan[k] for k in rng.choice(len(jan), a.janelas, replace=False))]
    base = montar_lote(d, jan, corte, device=device)
    plano = torch.from_numpy(base["plano"]).to(device)
    n_pos, n_alvo = len(plano), len(base["item_onde"])
    vistos = torch.from_numpy(np.flatnonzero(np.asarray(d.item_visto) > 0.5)).to(device)
    campos = base["lin_campos"]
    BT = base["B"] * base["T"]

    def lote_com(L):
        lt = dict(base)
        lt["lin_pos"] = plano.repeat_interleave(L)
        lt["lin_item"] = vistos[torch.randint(len(vistos), (n_pos * L,), device=device)]
        lt["lin_campos"] = campos[torch.randint(len(campos), (n_pos * L,), device=device)]
        ln = torch.zeros(BT, device=device)
        ln[plano] = float(L)
        lt["lin_n"] = ln.view(base["B"], base["T"])
        nl = torch.zeros(BT, dtype=torch.long, device=device)
        nl[plano] = int(1 + faixa(np.array([L]), FAIXAS_N_LINHAS)[0])
        lt["nlin"] = nl.view(base["B"], base["T"])
        lt["item_pos"] = torch.arange(n_alvo, device=device).repeat_interleave(L)
        lt["item_alvo"] = vistos[torch.randint(len(vistos), (n_alvo * L,), device=device)]
        return lt

    def passo(lt):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = modelo.estados(lt)
        p = modelo.perdas(h, lt)
        modelo.zero_grad(set_to_none=True)
        p["total"].backward()

    linhas = []
    for L in [int(x) for x in a.L.split(",")]:
        lt = lote_com(L)
        for _ in range(2):
            passo(lt)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        ts = []
        for _ in range(a.reps):
            t0 = time.perf_counter(); passo(lt); torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
        s = float(np.median(ts))
        r = {"L": L, "posicoes": n_pos, "linhas": n_pos * L, "segundos": s,
             "ms_por_mil_linhas": 1000 * s / (n_pos * L) * 1000, "pico_mb": torch.cuda.max_memory_allocated() / 2**20}
        linhas.append(r)
        print(json.dumps(r), flush=True)
        del lt
    Ls = np.array([r["L"] for r in linhas], float)
    S = np.array([r["segundos"] for r in linhas])
    b, a0 = np.polyfit(Ls, S, 1)
    pred = a0 + b * Ls
    grande = Ls >= 32
    incl = float(np.polyfit(np.log(Ls[grande]), np.log(S[grande]), 1)[0]) if grande.sum() >= 2 else None
    res["custo"] = {"janelas": a.janelas, "T": a.T, "medidas": linhas,
                    "ajuste_linear": {"a_segundos": float(a0), "b_segundos_por_linha_por_posicao": float(b),
                                      "r2": float(1 - ((S - pred) ** 2).sum() / ((S - S.mean()) ** 2).sum()),
                                      "erro_relativo_max": float(np.max(np.abs(S - pred) / S))},
                    "inclinacao_loglog_L>=32": incl}
    rel = os.path.join(run_dir(config, a.pretreino), "v2_relatorio.json")
    if os.path.exists(rel):
        r = json.load(open(rel))
        res["pre_treino"] = {k: r[k] for k in ("segundos", "posicoes_por_segundo") if k in r} | {"epocas": len(r.get("historico", []))}
    print(json.dumps(res["custo"]["ajuste_linear"]), "inclinação log-log:", incl, flush=True)
    out = run_dir(config, "f3h8")
    os.makedirs(out, exist_ok=True)
    json.dump(res, open(os.path.join(out, a.saida + ".json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
