"""Motor v2 — pré-treino do backbone de eventos compostos. Desenho em MOTOR_V2.md.

Treina com as posições anteriores à data de corte (padrão: o fim do split de
treino). Os alvos de cada posição são o próximo pedido e as janelas futuras,
com a regra de censura (nada depois do corte). A validação mede as mesmas
perdas nas posições do período de validação, com o corte no fim dele.

  - lotes por orçamento de posições + linhas, com janelas de tamanho parecido;
  - AdamW, aquecimento linear e decaimento em cosseno; bf16 na GPU;
  - checkpoint a cada época e a cada N passos; --retomar continua do último;
  - grava config, métricas por época e o melhor checkpoint pela perda de validação.

Uso: python3 pipeline/common/v2_pretreino.py <config.yaml> [--saida v2_pretreino] [--corte AAAA-MM-DD]
        [--n_embd 256 --n_layer 6 --n_head 4 --n_kv_heads 2 --d_item 64] [--epocas 3] [--lr 1e-3]
        [--orcamento 60000] [--seed 0] [--retomar] [--dispositivo auto|cpu|cuda]
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "pipeline/common")
from schema import carregar_config, run_dir  # noqa: E402
from v2_modelo import IGNORAR, DadosV2, ModeloV2, montar_lote  # noqa: E402


def lotes_por_orcamento(d, janelas, orcamento, rng=None):
    """agrupa janelas de tamanho parecido; custo de um lote = B x T (posições com preenchimento) + linhas."""
    tam = np.array([f - i for i, f in janelas])
    lin = np.array([int(d.lin_ini[f] - d.lin_ini[i]) for i, f in janelas])
    ordem = np.argsort(tam, kind="stable")
    lotes, atual, maior, linhas = [], [], 0, 0
    for j in ordem:
        novo = max(maior, int(tam[j]))
        if atual and (len(atual) + 1) * novo + linhas + int(lin[j]) > orcamento:
            lotes.append(atual)
            atual, novo, linhas = [], int(tam[j]), 0
        atual.append(janelas[j]); maior = novo; linhas += int(lin[j])
    if atual:
        lotes.append(atual)
    if rng is not None:
        rng.shuffle(lotes)
    return lotes


def restringir(lote, d, t_min):
    """mantém os alvos só das posições com t >= t_min (validação: as posições do período avaliado)."""
    t = np.asarray(d.t)[lote["pos"]]
    fora = np.zeros(lote["B"] * lote["T"], bool)
    fora[lote["plano"][t < t_min]] = True
    fora = torch.from_numpy(fora.reshape(lote["B"], lote["T"])).to(lote["valido"].device)
    for k in lote["alvo"]:
        lote["alvo"][k] = lote["alvo"][k].masked_fill(fora, IGNORAR)
    lote["censura_faixa"] = lote["censura_faixa"].masked_fill(fora, -1)
    mantem = ~fora.reshape(-1)[lote["item_onde"]]
    novo_idx = torch.cumsum(mantem.long(), 0) - 1
    pares = mantem[lote["item_pos"]]
    lote["item_alvo"] = lote["item_alvo"][pares]
    lote["item_pos"] = novo_idx[lote["item_pos"][pares]]
    lote["item_onde"] = lote["item_onde"][mantem]
    return lote


def avaliar(modelo, d, lotes, corte, t_min, device, amp):
    modelo.eval()
    soma, n = {}, 0
    with torch.no_grad():
        for jl in lotes:
            lote = restringir(montar_lote(d, jl, corte, device=device), d, t_min)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                h = modelo.estados(lote)
            for k, v in modelo.perdas(h, lote).items():
                soma[k] = soma.get(k, 0.0) + float(v)
            n += 1
    modelo.train()
    return {k: v / max(n, 1) for k, v in soma.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--saida", default="v2_pretreino")
    ap.add_argument("--dados", default="v2", help="pasta da serialização em runs/<nome>/")
    ap.add_argument("--corte", default=None, help="AAAA-MM-DD; padrão: fim do split de treino")
    ap.add_argument("--n_embd", type=int, default=256); ap.add_argument("--n_layer", type=int, default=6)
    ap.add_argument("--n_head", type=int, default=4); ap.add_argument("--n_kv_heads", type=int, default=2)
    ap.add_argument("--d_item", type=int, default=64); ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epocas", type=int, default=3); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--orcamento", type=int, default=60000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val_entidades", type=int, default=20000, help="entidades na validação por época")
    ap.add_argument("--checkpoint_passos", type=int, default=500)
    ap.add_argument("--retomar", action="store_true")
    ap.add_argument("--alvos_atributos", action="store_true", help="F3-H3b: composição do próximo pedido por atributo do catálogo")
    ap.add_argument("--trava_gpu", default=os.environ.get("TRAVA_GPU", "/content/trava_gpu"),
                    help="com janela > 1024 (muita memória), espera enquanto este arquivo existir (outro treino grande na GPU)")
    ap.add_argument("--dispositivo", default="auto", choices=["auto", "cpu", "cuda"])
    a = ap.parse_args()
    config = carregar_config(a.config)
    d = DadosV2(run_dir(config, a.dados))
    m = d.meta
    if m["janela"] > 1024:
        while os.path.exists(a.trava_gpu):
            print(f"esperando a GPU ({a.trava_gpu} existe)", flush=True); time.sleep(60)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.dispositivo == "auto" else a.dispositivo
    amp = device == "cuda"
    out = run_dir(config, a.saida)
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(a.seed)
    rng = np.random.RandomState(a.seed)
    if a.corte:
        corte = int(np.datetime64(a.corte, "m").astype(np.int64))
        corte_val, t_val = None, None                  # com corte próprio não há período de validação reservado
    else:
        corte, corte_val, t_val = m["m_val"], m["m_teste"], m["m_val"]

    arq = {k: getattr(a, k) for k in ("n_embd", "n_layer", "n_head", "n_kv_heads", "d_item", "dropout", "alvos_atributos")}
    modelo = ModeloV2(d, **arq).to(device)
    jan = d.janelas(corte)
    lotes = lotes_por_orcamento(d, jan, a.orcamento)
    total = len(lotes) * a.epocas
    opt = torch.optim.AdamW(modelo.parameters(), lr=a.lr, weight_decay=0.05)
    aquec = min(200, max(1, total // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / aquec) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(total, 1)))))
    lotes_val = []
    if corte_val is not None:
        ents = np.unique(np.asarray(d.ent)[(np.asarray(d.t) >= t_val) & (np.asarray(d.t) < corte_val)])
        if len(ents) > a.val_entidades:
            ents = np.sort(np.random.RandomState(0).choice(ents, a.val_entidades, replace=False))
        # só a janela mais recente de cada entidade (a que contém o período de validação)
        ultimas = {}
        for ini, fim in d.janelas(corte_val, so_entidades=ents):
            e = int(d.ent[ini])
            if e not in ultimas or fim > ultimas[e][1]:
                ultimas[e] = (ini, fim)
        lotes_val = lotes_por_orcamento(d, list(ultimas.values()), a.orcamento)
    n_pos = sum(f - i for i, f in jan)
    print(f"{modelo.n_parametros():,} parâmetros; {len(jan):,} janelas de treino ({n_pos:,} posições) em {len(lotes):,} lotes; "
          f"{len(lotes_val):,} lotes de validação; dispositivo {device}", flush=True)

    estado = {"passo": 0, "epoca": 0, "historico": [], "melhor": float("inf"), "segundos": 0.0}
    ult = os.path.join(out, "ultimo.pt")
    if a.retomar and os.path.exists(ult):
        ck = torch.load(ult, map_location=device, weights_only=False)
        modelo.load_state_dict(ck["modelo"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        rng.set_state(ck["rng"]); estado = ck["estado"]
        print(f"retomando da época {estado['epoca']}, passo {estado['passo']}", flush=True)

    def salvar(nome):
        torch.save({"modelo": modelo.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "rng": rng.get_state(),
                    "estado": estado, "arquitetura": arq, "corte_min": corte}, os.path.join(out, nome + ".tmp"))
        os.replace(os.path.join(out, nome + ".tmp"), os.path.join(out, nome))

    json.dump({"arquitetura": arq, "corte_min": corte, "epocas": a.epocas, "lr": a.lr, "orcamento": a.orcamento, "seed": a.seed,
               "n_parametros": modelo.n_parametros(), "n_janelas": len(jan), "n_posicoes_treino": int(n_pos)},
              open(os.path.join(out, "v2_config.json"), "w"), indent=1)
    modelo.train()
    t0 = time.time() - estado["segundos"]
    while estado["epoca"] < a.epocas:
        # a ordem dos lotes da época sai do gerador no estado em que a época começou: numa retomada no meio
        # da época, o mesmo estado refaz a mesma ordem e os lotes já vistos são pulados
        if estado.get("epoca_do_rng") == estado["epoca"]:
            rng.set_state(estado["rng_epoca"])
        else:
            estado["rng_epoca"], estado["epoca_do_rng"] = rng.get_state(), estado["epoca"]
        ordem = rng.permutation(len(lotes))
        feitos = estado["passo"] - estado["epoca"] * len(lotes)           # passos já dados nesta época (retomada)
        acum, n_acum = {}, 0
        for k in ordem[feitos:]:
            lote = montar_lote(d, lotes[k], corte, device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                h = modelo.estados(lote)
            perdas = modelo.perdas(h, lote)
            opt.zero_grad(set_to_none=True)
            perdas["total"].backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            opt.step(); sched.step()
            estado["passo"] += 1
            for nome, v in perdas.items():
                acum[nome] = acum.get(nome, 0.0) + float(v.detach())
            n_acum += 1
            if estado["passo"] % 50 == 0:
                print(json.dumps({"passo": estado["passo"], "total": round(acum["total"] / n_acum, 4),
                                  "itens": round(acum.get("itens", 0.0) / n_acum, 4), "s": round(time.time() - t0, 1)}), flush=True)
            if estado["passo"] % a.checkpoint_passos == 0:
                estado["segundos"] = time.time() - t0
                salvar("ultimo.pt")
        estado["epoca"] += 1
        estado["segundos"] = time.time() - t0
        reg = {"epoca": estado["epoca"], "passo": estado["passo"], "treino": {k: v / max(n_acum, 1) for k, v in acum.items()},
               "segundos": round(estado["segundos"], 1)}
        if lotes_val:
            reg["val"] = avaliar(modelo, d, lotes_val, corte_val, t_val, device, amp)
            # critério: soma das perdas de validação, sem as janelas futuras que não cabem no período de
            # validação (com o corte no fim dele, só fecham para as posições do primeiro dia: viram ruído)
            dias_val = (corte_val - t_val) / 1440.0
            fora = {f"{p}_{w}" for w in m["janelas_dias"] if w >= dias_val for p in ("cont", "gasto")}
            reg["criterio"] = sum(v for k, v in reg["val"].items() if k != "total" and k not in fora)
            reg["criterio_sem"] = sorted(fora)
        estado["historico"].append(reg)
        criterio = reg["criterio"] if lotes_val else reg["treino"]["total"]
        print(json.dumps({"epoca": reg["epoca"], "treino": round(reg["treino"]["total"], 4), "criterio": round(criterio, 4),
                          "val": {k: round(v, 4) for k, v in reg.get("val", {}).items()}, "s": reg["segundos"]}), flush=True)
        torch.save({"modelo": modelo.state_dict(), "arquitetura": arq, "corte_min": corte, "epoca": estado["epoca"]},
                   os.path.join(out, f"epoca_{estado['epoca']}.pt"))      # só os pesos, para escolher a época depois
        if criterio < estado["melhor"]:
            estado["melhor"] = criterio
            salvar("melhor.pt")
        salvar("ultimo.pt")
        json.dump({"historico": estado["historico"], "melhor_criterio": estado["melhor"], "segundos": estado["segundos"],
                   "posicoes_por_segundo": round(n_pos * estado["epoca"] / max(estado["segundos"], 1e-9), 1)},
                  open(os.path.join(out, "v2_relatorio.json"), "w"), indent=1)
    print(f"fim: {estado['epoca']} épocas em {estado['segundos']:.0f}s; melhor critério {estado['melhor']:.4f}")


if __name__ == "__main__":
    main()
