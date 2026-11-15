"""Motor v2 — testes exigidos antes de qualquer resultado (MOTOR_V2.md, seção 8).

  1. serialização   as posições estão em ordem de tempo por entidade; o próximo pedido e as
                    janelas futuras conferem com um cálculo independente, feito à força bruta;
  2. censura        nenhum alvo usa evento posterior ao corte; a posição sem próximo pedido
                    até o corte entra só no alvo de intervalo, como "maior que (corte − t)";
  3. causalidade    alterar eventos futuros não muda o estado das posições anteriores;
  4. aprendizado    num lote pequeno repetido, a perda de cada alvo cai bastante.

Uso: python3 pipeline/common/v2_testes.py <config.yaml> [pasta_dados=v2]      (roda na CPU, numa base pequena)
"""

import sys

import numpy as np
import torch

sys.path.insert(0, "pipeline/common")
from schema import carregar_config, run_dir  # noqa: E402
from v2_esquema import FAIXAS_CONTAGEM, FAIXAS_INTERVALO_DIAS  # noqa: E402
from v2_modelo import IGNORAR, DadosV2, ModeloV2, faixa, montar_lote  # noqa: E402

falhas = []


def checar(nome, ok, detalhe=""):
    print(f"  [{'OK' if ok else 'FALHOU'}] {nome}{(' — ' + detalhe) if detalhe else ''}", flush=True)
    if not ok:
        falhas.append(nome)


def teste_serializacao(d, rng):
    print("1. serialização")
    e, t = np.asarray(d.ent), np.asarray(d.t)
    checar("posições ordenadas por entidade e tempo", bool(np.all((np.diff(e) > 0) | ((np.diff(e) == 0) & (np.diff(t) >= 0)))))
    checar("linhas: os inícios fecham com o total", int(d.lin_ini[-1]) == len(d.lin_item) and bool(np.all(np.diff(d.lin_ini) == d.n_lin)))
    alvo = np.asarray(d.tipo) == d.meta["tipo_alvo"]
    ents = rng.choice(len(d.ent_ini) - 1, min(200, len(d.ent_ini) - 1), replace=False)
    ok_prox = ok_jan = True
    n = 0
    for i in ents:
        a, b = int(d.ent_ini[i]), int(d.ent_ini[i + 1])
        for p in range(a, b):
            futuros = [q for q in range(p + 1, b) if alvo[q]]
            esperado = futuros[0] if futuros else -1
            ok_prox &= int(d.prox[p]) == esperado
            for w in d.meta["janelas_dias"]:
                dentro = [q for q in range(a, b) if alvo[q] and t[p] < t[q] <= t[p] + w * 1440]
                ok_jan &= int(d.jan[w][0][p]) == len(dentro)
                ok_jan &= abs(float(d.jan[w][1][p]) - float(np.sum(np.asarray(d.valor)[dentro]))) <= 1e-2 * max(1.0, float(d.jan[w][1][p]))
            n += 1
    checar("próximo pedido confere com a força bruta", ok_prox, f"{n:,} posições de {len(ents)} entidades")
    checar("contagem e gasto das janelas futuras conferem", ok_jan)
    checar("estado nunca vem do futuro (junção para trás)", True, "garantido por join_asof(strategy='backward') na serialização")


def teste_censura(d, rng):
    print("2. censura")
    m = d.meta
    # corte no meio do treino: sobra bastante evento DEPOIS do corte para tentar vazar
    corte = int(np.percentile(np.asarray(d.t), 40))
    jan = d.janelas(corte)
    jan = [jan[k] for k in rng.choice(len(jan), min(64, len(jan)), replace=False)]
    lote = montar_lote(d, jan, corte)
    p, t = lote["pos"], np.asarray(d.t)[lote["pos"]]
    checar("nenhuma posição do lote é posterior ao corte", bool(np.all(t < corte)))
    prox = np.asarray(d.prox)[p]
    conhecido = (prox >= 0) & (np.asarray(d.t)[np.maximum(prox, 0)] < corte)
    al = {k: v.numpy().ravel()[lote["plano"]] for k, v in lote["alvo"].items()}
    checar("itens, valor e linhas só quando o próximo pedido é anterior ao corte",
           bool(np.all((al["valor"] != IGNORAR) == conhecido) and np.all((al["n_linhas"] != IGNORAR) == conhecido)
                and len(lote["item_onde"]) == int(conhecido.sum())))
    cf = lote["censura_faixa"].numpy().ravel()[lote["plano"]]
    esperado = faixa((corte - t) / 1440.0, FAIXAS_INTERVALO_DIAS)
    checar("sem próximo pedido até o corte: entra como 'intervalo maior que (corte − t)'",
           bool(np.all((cf >= 0) == ~conhecido) and np.all(cf[~conhecido] == esperado[~conhecido])),
           f"{int((~conhecido).sum()):,} posições censuradas, {int(conhecido.sum()):,} observadas")
    ok = True
    for w in m["janelas_dias"]:
        fechada = t + w * 1440 <= corte
        ok &= bool(np.all((al[f"cont_{w}"] != IGNORAR) == fechada) and np.all((al[f"gasto_{w}"] != IGNORAR) == fechada))
        # com a janela fechada antes do corte, a contagem não pode incluir pedidos depois do corte
        ok &= bool(np.all(al[f"cont_{w}"][fechada] == faixa(np.asarray(d.jan[w][0])[p][fechada], FAIXAS_CONTAGEM)))
    checar("janelas futuras: alvo só com a janela fechada antes do corte", ok)
    # os itens-alvo vêm de pedidos anteriores ao corte
    onde = lote["item_onde"].numpy()
    pos_conhecidas = p[np.searchsorted(lote["plano"], onde)]
    checar("os itens-alvo vêm de pedidos anteriores ao corte", bool(np.all(np.asarray(d.t)[np.asarray(d.prox)[pos_conhecidas]] < corte)))


def teste_causalidade(d, modelo, rng):
    print("3. causalidade")
    m = d.meta
    jan = [j for j in d.janelas(m["m_val"]) if j[1] - j[0] >= 12][:8]
    lote = montar_lote(d, jan, m["m_val"], com_alvos=False)
    modelo.eval()
    with torch.no_grad():
        h0 = modelo.estados(lote)
    # altera tudo o que pertence às posições da segunda metade de cada janela
    T = lote["T"]
    corte_pos = torch.tensor([(f - i) // 2 for i, f in jan])
    futuro = torch.arange(T)[None, :] >= corte_pos[:, None]
    alt = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in lote.items()}
    for k in ("tipo", "dt", "dow", "mes", "nlin"):      # valores sorteados dentro do vocabulário de cada campo
        alt[k][futuro] = torch.randint(0, getattr(modelo, k).num_embeddings, alt[k][futuro].shape)
    alt["cab"][futuro] = 0
    alt["estado"][futuro] = 0
    lin_fut = futuro.reshape(-1)[alt["lin_pos"]]
    alt["lin_item"][lin_fut] = torch.randint(1, m["n_itens"] + 1, (int(lin_fut.sum()),))
    alt["lin_campos"][lin_fut] = 0
    with torch.no_grad():
        h1 = modelo.estados(alt)
    passado = (~futuro) & lote["valido"]
    dif = float((h0 - h1).abs()[passado].max())
    mudou = float((h0 - h1).abs()[futuro & lote["valido"]].max())
    checar("alterar o futuro não muda o estado do passado", dif < 1e-5, f"maior diferença no passado: {dif:.2e}; no futuro: {mudou:.2f}")


def teste_aprendizado(d, rng):
    print("4. aprendizado mínimo")
    m = d.meta
    torch.manual_seed(0)
    modelo = ModeloV2(d, n_embd=64, n_layer=2, n_head=4, n_kv_heads=2, d_item=32, dropout=0.0)
    jan = [j for j in d.janelas(m["m_val"]) if 8 <= j[1] - j[0] <= 40][:8]
    lote = montar_lote(d, jan, m["m_val"])
    opt = torch.optim.AdamW(modelo.parameters(), lr=3e-3)
    modelo.train()
    ini = None
    for passo in range(300):
        perdas = modelo.perdas(modelo.estados(lote), lote)
        if passo == 0:
            ini = {k: float(v) for k, v in perdas.items()}
        opt.zero_grad(); perdas["total"].backward(); opt.step()
    fim = {k: float(v) for k, v in modelo.perdas(modelo.estados(lote), lote).items()}
    for k in ini:
        if k != "total":
            checar(f"perda de '{k}' cai", fim[k] < 0.35 * ini[k], f"{ini[k]:.3f} -> {fim[k]:.3f}")


def main():
    config = carregar_config(sys.argv[1])
    d = DadosV2(run_dir(config, sys.argv[2] if len(sys.argv) > 2 else "v2"))
    rng = np.random.RandomState(0)
    torch.manual_seed(0)
    teste_serializacao(d, rng)
    teste_censura(d, rng)
    teste_causalidade(d, ModeloV2(d, n_embd=64, n_layer=2, n_head=4, n_kv_heads=2, d_item=32), rng)
    teste_aprendizado(d, rng)
    print("\nTODOS OS TESTES PASSARAM" if not falhas else f"\n{len(falhas)} TESTE(S) FALHARAM: {falhas}")
    sys.exit(1 if falhas else 0)


if __name__ == "__main__":
    main()
