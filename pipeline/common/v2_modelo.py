"""Motor v2 — dados em lote, modelo e perdas. Desenho em MOTOR_V2.md (seções 3 a 5).

Dados (`DadosV2`): os arrays de v2_serializacao.py, lidos por memmap.

Lote (`montar_lote`): uma lista de janelas (início, fim) de posições da mesma
entidade vira tensores (B, T) preenchidos à direita, mais as LINHAS de todos os
eventos do lote num único tensor, com o índice da posição a que pertencem.
Os ALVOS e as máscaras de censura são calculados aqui, a partir da data de corte:
nenhum alvo usa evento posterior ao corte (MOTOR_V2.md, 4.1).

Modelo (`ModeloV2`): vetor de cada posição = tipo do evento + cabeçalho + estado
vigente + tempo + média dos vetores das linhas; blocos causais do decoder do v1
(RoPE, SwiGLU, atenção agrupada); saídas: itens do próximo pedido (probabilidade
sobre o catálogo inteiro, com os mesmos vetores de item da entrada), intervalo
até o próximo pedido (com censura), valor e número de linhas do próximo pedido,
contagem e gasto em janelas futuras.
"""

import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from modelo_decoder import Bloco, RoPE, norma
from v2_esquema import FAIXAS_CONTAGEM, FAIXAS_INTERVALO_DIAS, FAIXAS_N_LINHAS

IGNORAR = -100
FAIXAS_DT_ENTRADA = [0.02, 0.25, 1, 2, 3, 5, 7, 10, 14, 21, 30, 45, 60, 90, 120, 180, 365]   # dias desde a posição anterior
P_SEM_ID = 0.10          # no treino, o vetor próprio do item é desligado nesta fração das linhas


class DadosV2:
    def __init__(self, pasta):
        self.pasta = pasta
        self.meta = json.load(open(os.path.join(pasta, "meta.json"), encoding="utf-8"))
        m = lambda n: np.load(os.path.join(pasta, n + ".npy"), mmap_mode="r")  # noqa: E731
        self.ent, self.t, self.tipo, self.split = m("pos_ent"), m("pos_t_min"), m("pos_tipo"), m("pos_split")
        self.n_lin, self.valor, self.cab, self.estado = m("pos_n_lin"), m("pos_valor"), m("pos_cab"), m("pos_estado")
        self.lin_ini, self.id_ancora = m("pos_lin_ini"), m("pos_id_ancora")
        self.lin_item, self.lin_campos = m("lin_item"), m("lin_campos")
        self.ent_ini, self.prox = m("ent_ini"), m("alvo_prox")
        self.jan = {w: (m(f"alvo_cont_{w}"), m(f"alvo_gasto_{w}")) for w in self.meta["janelas_dias"]}
        self.item_atrib, self.item_visto = m("item_atrib"), m("item_visto_treino")
        self.lim_valor = np.asarray(self.meta["limites"]["alvo.valor"], dtype=np.float64)
        self.lim_gasto = {w: np.asarray(self.meta["limites"][f"alvo.gasto_{w}"], dtype=np.float64) for w in self.meta["janelas_dias"]}

    def classes(self):
        """número de classes de cada alvo."""
        c = {"intervalo": len(FAIXAS_INTERVALO_DIAS) + 1, "valor": len(self.lim_valor) + 1, "n_linhas": len(FAIXAS_N_LINHAS) + 1}
        for w in self.meta["janelas_dias"]:
            c[f"cont_{w}"] = len(FAIXAS_CONTAGEM) + 1
            c[f"gasto_{w}"] = len(self.lim_gasto[w]) + 2          # 0 = sem gasto
        return c

    def janelas(self, corte_min, min_pos=1, so_entidades=None):
        """janelas de posições anteriores a `corte_min`, por entidade, sem sobreposição e alinhadas ao fim do histórico."""
        J = self.meta["janela"]
        saida = []
        ents = range(len(self.ent_ini) - 1) if so_entidades is None else so_entidades
        for i in ents:
            a, b = int(self.ent_ini[i]), int(self.ent_ini[i + 1])
            b = a + int(np.searchsorted(self.t[a:b], corte_min, side="left"))
            if b - a < min_pos:
                continue
            fim = b
            while fim > a:
                ini = max(a, fim - J)
                saida.append((ini, fim))
                fim = ini
        return saida


def faixa(valores, limites):
    """índice da faixa: 0 para valores <= limites[0], ..., len(limites) para valores > limites[-1]."""
    return np.searchsorted(np.asarray(limites, dtype=np.float64), valores, side="left")


def montar_lote(d, janelas, corte_min, com_alvos=True, device="cpu"):
    B, T = len(janelas), max(f - i for i, f in janelas)
    idx = np.full((B, T), -1, np.int64)
    for r, (i, f) in enumerate(janelas):
        idx[r, :f - i] = np.arange(i, f)
    valido = idx >= 0
    p = idx[valido]                                   # posições reais, na ordem (linha do lote, tempo)
    plano = np.flatnonzero(valido.ravel())            # índice no tensor (B*T)
    t = d.t[p]
    # tempo desde a posição anterior da MESMA janela (a primeira de cada janela fica com a faixa 0)
    ant = np.r_[0, t[:-1]]
    primeira = np.r_[True, plano[1:] // T != plano[:-1] // T]
    dt_dias = np.where(primeira, np.nan, (t - ant) / 1440.0)
    dt_id = np.where(np.isnan(dt_dias), 0, 1 + faixa(np.nan_to_num(dt_dias), FAIXAS_DT_ENTRADA))
    dias = t // 1440
    dow = ((dias + 4) % 7) + 1                         # 1970-01-01 foi quinta-feira
    mes = (dias.astype("datetime64[D]").astype("datetime64[M]").astype(np.int64) % 12) + 1

    def cheio(valores, dtype=np.int64):
        out = np.zeros((B * T,) + valores.shape[1:], dtype)
        out[plano] = valores
        return torch.from_numpy(out.reshape((B, T) + valores.shape[1:])).to(device)

    lote = {"valido": torch.from_numpy(valido).to(device), "tipo": cheio(d.tipo[p].astype(np.int64)),
            "cab": cheio(d.cab[p].astype(np.int64)), "estado": cheio(d.estado[p].astype(np.int64)),
            "dt": cheio(dt_id.astype(np.int64)), "dow": cheio(dow.astype(np.int64)), "mes": cheio(mes.astype(np.int64)),
            "nlin": cheio(1 + faixa(d.n_lin[p], FAIXAS_N_LINHAS).astype(np.int64)), "pos": p, "plano": plano, "B": B, "T": T}
    # linhas de todas as posições do lote
    li, lf = d.lin_ini[p], d.lin_ini[p + 1]
    n = (lf - li).astype(np.int64)
    sel = np.repeat(li - np.r_[0, np.cumsum(n)[:-1]], n) + np.arange(int(n.sum()))
    lote["lin_item"] = torch.from_numpy(np.asarray(d.lin_item[sel]).astype(np.int64)).to(device)
    lote["lin_campos"] = torch.from_numpy(np.asarray(d.lin_campos[sel]).astype(np.int64)).to(device)
    lote["lin_pos"] = torch.from_numpy(np.repeat(plano, n)).to(device)
    lote["lin_n"] = cheio(n.astype(np.float32), np.float32)
    if not com_alvos:
        return lote

    # ---- alvos, com a regra de censura (MOTOR_V2.md, 4.1): nada depois de `corte_min`
    prox = d.prox[p]
    conhecido = (prox >= 0) & (d.t[np.maximum(prox, 0)] < corte_min)
    pc = np.maximum(prox, 0)
    dias_ate = (d.t[pc] - t) / 1440.0
    alvo = {"intervalo": np.where(conhecido, faixa(dias_ate, FAIXAS_INTERVALO_DIAS), IGNORAR),
            "valor": np.where(conhecido, faixa(d.valor[pc], d.lim_valor), IGNORAR),
            "n_linhas": np.where(conhecido, faixa(d.n_lin[pc], FAIXAS_N_LINHAS), IGNORAR)}
    # posição sem próximo pedido até o corte: só se sabe que o intervalo é MAIOR que (corte - t)
    censura_dias = np.where(conhecido, np.nan, (corte_min - t) / 1440.0)
    lote["censura_faixa"] = cheio(np.where(conhecido, -1, faixa(np.nan_to_num(censura_dias), FAIXAS_INTERVALO_DIAS)).astype(np.int64))
    for w in d.meta["janelas_dias"]:
        fechada = t + w * 1440 <= corte_min
        cont, gasto = d.jan[w][0][p], d.jan[w][1][p]
        alvo[f"cont_{w}"] = np.where(fechada, faixa(cont, FAIXAS_CONTAGEM), IGNORAR)
        alvo[f"gasto_{w}"] = np.where(fechada, np.where(gasto > 0, 1 + faixa(gasto, d.lim_gasto[w]), 0), IGNORAR)
    lote["alvo"] = {k: cheio(v.astype(np.int64)) if False else torch.from_numpy(_espalhar(v.astype(np.int64), plano, B * T, IGNORAR).reshape(B, T)).to(device)
                    for k, v in alvo.items()}
    # itens do próximo pedido: pares (posição no lote, item) só para as posições com próximo pedido conhecido
    k = np.flatnonzero(conhecido)
    ai, af = d.lin_ini[pc[k]], d.lin_ini[pc[k] + 1]
    na = (af - ai).astype(np.int64)
    sel = np.repeat(ai - np.r_[0, np.cumsum(na)[:-1]], na) + np.arange(int(na.sum()))
    lote["item_pos"] = torch.from_numpy(np.repeat(np.arange(len(k)), na)).to(device)      # índice entre as posições conhecidas
    lote["item_alvo"] = torch.from_numpy(np.asarray(d.lin_item[sel]).astype(np.int64)).to(device)
    lote["item_onde"] = torch.from_numpy(plano[k]).to(device)                              # posição (B*T) de cada conhecida
    return lote


def _espalhar(valores, plano, tamanho, preenchimento):
    out = np.full(tamanho, preenchimento, valores.dtype)
    out[plano] = valores
    return out


class ModeloV2(nn.Module):
    def __init__(self, d, n_embd=256, n_layer=6, n_head=4, n_kv_heads=2, d_item=64, dropout=0.1, ffn="swiglu", norm="layernorm",
                 alvos_atributos=False):
        super().__init__()
        m = d.meta
        # F3-H3b: alvos de composição do próximo pedido — para cada atributo do catálogo, a distribuição dos
        # valores nas linhas do próximo pedido (o item muda de coleção; os atributos permanecem)
        self.alvos_atributos = alvos_atributos
        self.n_itens = m["n_itens"]
        self.janelas_dias = m["janelas_dias"]
        self.item_id = nn.Embedding(self.n_itens + 1, d_item, padding_idx=0)
        self.item_atr = nn.ModuleList([nn.Embedding(c, d_item, padding_idx=0) for c in m["cards_atrib"]])
        self.register_buffer("atrib", torch.from_numpy(np.asarray(d.item_atrib).astype(np.int64)), persistent=False)
        self.register_buffer("visto", torch.from_numpy(np.asarray(d.item_visto).astype(np.float32)), persistent=False)
        self.item_para_d = nn.Linear(d_item, n_embd, bias=False)
        self.lin = nn.ModuleList([nn.Embedding(c, n_embd, padding_idx=0) for c in (m["cards_linha"] or [1])])
        self.tipo = nn.Embedding(m["n_tipos"], n_embd)
        self.cab = nn.ModuleList([nn.Embedding(c, n_embd, padding_idx=0) for c in (m["cards_cab"] or [1])])
        self.est = nn.ModuleList([nn.Embedding(c, n_embd, padding_idx=0) for c in (m["cards_estado"] or [1])])
        self.dt = nn.Embedding(len(FAIXAS_DT_ENTRADA) + 2, n_embd)
        self.mes = nn.Embedding(13, n_embd)
        self.dow = nn.Embedding(8, n_embd)
        self.nlin = nn.Embedding(len(FAIXAS_N_LINHAS) + 2, n_embd)
        self.drop = nn.Dropout(dropout)
        rope = RoPE(n_embd // n_head, m["janela"])
        self.blocos = nn.ModuleList([Bloco(n_embd, n_head, n_kv_heads, norm, ffn, dropout, True, rope) for _ in range(n_layer)])
        self.ln_f = norma(norm, n_embd, True)
        self.saida_item = nn.Linear(n_embd, d_item, bias=False)
        self.vies_item = nn.Parameter(torch.zeros(self.n_itens + 1))
        self.vies_novo = nn.Parameter(torch.zeros(1))      # viés de um item sem vetor próprio (novo ou desligado)
        self.cabecas = nn.ModuleDict({k: nn.Linear(n_embd, c) for k, c in d.classes().items()})
        self.cab_atrib = nn.ModuleList([nn.Linear(n_embd, c) for c in m["cards_atrib"]]) if alvos_atributos else None
        self.apply(lambda mod: self._init(mod, n_layer))

    @staticmethod
    def _init(mod, n_layer):
        if isinstance(mod, nn.Linear):
            std = 0.02 / math.sqrt(2 * n_layer) if getattr(mod, "escala_residual", False) else 0.02
            nn.init.normal_(mod.weight, 0.0, std)
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)
        elif isinstance(mod, nn.Embedding):
            nn.init.normal_(mod.weight, 0.0, 0.02)
            if mod.padding_idx is not None:
                with torch.no_grad():
                    mod.weight[mod.padding_idx].zero_()

    def vetores_itens(self):
        """(n_itens + 1, d_item): vetor próprio (só para itens vistos no treino) + atributos; a linha 0 é 'sem item'."""
        v = self.item_id.weight * self.visto[:, None]
        for j, emb in enumerate(self.item_atr):
            v = v + emb(self.atrib[:, j])
        return v

    def estados(self, lote):
        """estado do cliente depois de cada posição: (B, T, n_embd)."""
        B, T = lote["B"], lote["T"]
        it = lote["lin_item"]
        e_id = self.item_id(it) * self.visto[it][:, None]
        if self.training:                                   # desliga o vetor próprio em parte das linhas
            e_id = e_id * (torch.rand(len(it), 1, device=it.device) >= P_SEM_ID)
        v = e_id
        for j, emb in enumerate(self.item_atr):
            v = v + emb(self.atrib[it, j])
        lin = self.item_para_d(v)
        for j, emb in enumerate(self.lin):
            lin = lin + emb(lote["lin_campos"][:, j])
        x = torch.zeros(B * T, lin.shape[1], device=lin.device, dtype=lin.dtype).index_add_(0, lote["lin_pos"], lin)
        x = x.view(B, T, -1) / lote["lin_n"].clamp(min=1.0)[..., None].to(lin.dtype)
        x = x + self.tipo(lote["tipo"]) + self.dt(lote["dt"]) + self.mes(lote["mes"]) + self.dow(lote["dow"]) + self.nlin(lote["nlin"])
        for j, emb in enumerate(self.cab):
            x = x + emb(lote["cab"][..., j])
        for j, emb in enumerate(self.est):
            x = x + emb(lote["estado"][..., j])
        x = self.drop(x)
        for bloco in self.blocos:
            x = bloco(x)
        return self.ln_f(x)

    def saida_catalogo(self):
        """vetores, vieses e itens fora do softmax, na SAÍDA.

        Item visto no treino: vetor próprio + atributos e viés próprio. Item novo (não visto no treino):
        só atributos e o viés compartilhado `vies_novo`. No treino, (a) os itens novos ficam fora do
        softmax — ainda não existem no período de treino; tratá-los como "nunca comprados" ensina o
        modelo a não recomendar lançamentos — e (b) uma fração P_SEM_ID dos itens vistos recebe a
        representação de item novo, para o modelo aprender a pontuar pelos atributos e o `vies_novo`."""
        V = self.vetores_itens()
        novo = self.visto < 0.5
        b = torch.where(novo, self.vies_novo, self.vies_item)
        fora = torch.zeros_like(novo)
        fora[0] = True                                          # 'sem item'
        if self.training:
            frio = (torch.rand(len(novo), device=V.device) < P_SEM_ID) & ~novo
            V = V - self.item_id.weight * frio[:, None].to(V.dtype)
            b = torch.where(frio, self.vies_novo, b)
            fora = fora | novo
        return V, b, fora

    def pontuar_itens(self, h, bloco=2048):
        """log-probabilidade de cada item do catálogo para cada estado em `h` (N, n_embd) -> (N, n_itens + 1)."""
        V, b, fora = self.saida_catalogo()
        partes = []
        for i in range(0, h.shape[0], bloco):
            lg = (self.saida_item(h[i:i + bloco]) @ V.T.to(h.dtype) + b.to(h.dtype)).float()
            partes.append(F.log_softmax(lg.masked_fill(fora[None, :], float("-inf")), dim=-1))
        return torch.cat(partes) if partes else h.new_zeros((0, self.n_itens + 1))

    def _logp_alvos(self, hq, V, vies, fora, pos, alvo):
        """log p(alvo) para os pares (posição local, item) de um bloco de estados `hq`."""
        lg = (self.saida_item(hq) @ V.T.to(hq.dtype) + vies.to(hq.dtype)).float()
        lg = lg.masked_fill(fora[None, :], float("-inf"))
        return lg[pos, alvo] - torch.logsumexp(lg, dim=-1)[pos]

    def perdas(self, h, lote, bloco_itens=2048):
        """dicionário de perdas (cada uma dividida pelo log do número de classes) e a soma."""
        saida = {}
        hp = h.reshape(-1, h.shape[-1])
        # itens do próximo pedido: média, por posição, de -log p(item), com o softmax sobre o catálogo inteiro.
        # Em blocos de posições e com recomputação no backward: só a matriz (bloco x catálogo) de um bloco
        # fica na memória (com 105 mil itens, a matriz do lote inteiro passaria de 10 GB).
        if len(lote["item_onde"]):
            hq = hp[lote["item_onde"]]
            V, b, fora = self.saida_catalogo()
            ip, ia = lote["item_pos"], lote["item_alvo"]
            if self.training:                       # alvo fora do softmax (só com corte posterior ao treino): descartado
                ok = ~fora[ia]
                ip, ia = ip[ok], ia[ok]
            g = torch.empty(len(ip), device=hq.device)
            for i in range(0, hq.shape[0], bloco_itens):
                sel = torch.nonzero((ip >= i) & (ip < i + bloco_itens), as_tuple=True)[0]
                if not len(sel):
                    continue
                args = (hq[i:i + bloco_itens], V, b, fora, ip[sel] - i, ia[sel])
                g[sel] = (checkpoint(self._logp_alvos, *args, use_reentrant=False) if torch.is_grad_enabled()
                          else self._logp_alvos(*args))
            n = torch.zeros(len(lote["item_onde"]), device=g.device).index_add_(0, ip, torch.ones_like(g))
            por_pos = torch.zeros(len(lote["item_onde"]), device=g.device).index_add_(0, ip, -g) / n.clamp(min=1)
            saida["itens"] = por_pos.mean() / math.log(self.n_itens)
            if self.cab_atrib is not None:
                # entropia cruzada contra a distribuição empírica de cada atributo nas linhas do próximo pedido:
                # média, por posição, de -log p(valor do atributo de cada linha)
                for j, cab in enumerate(self.cab_atrib):
                    lsm = F.log_softmax(cab(hq).float(), dim=-1)
                    ga = lsm[ip, self.atrib[ia, j]]
                    pp = torch.zeros(len(lote["item_onde"]), device=g.device).index_add_(0, ip, -ga) / n.clamp(min=1)
                    saida[f"atrib_{j}"] = pp[n > 0].mean() / math.log(max(cab.out_features, 2))
        for k, cab in self.cabecas.items():
            lg = cab(h).float()
            alvo = lote["alvo"][k]
            if k == "intervalo":
                # observadas: entropia cruzada; censuradas: -log P(intervalo na faixa da censura ou depois)
                lsm = F.log_softmax(lg, dim=-1)
                obs = alvo != IGNORAR
                perda_obs = -lsm.gather(-1, alvo.clamp(min=0)[..., None])[..., 0]
                cf = lote["censura_faixa"]
                cens = (cf >= 0) & lote["valido"]
                faixas = torch.arange(lg.shape[-1], device=lg.device)
                sobrev = torch.logsumexp(lsm.masked_fill(faixas[None, None, :] < cf.clamp(min=0)[..., None], float("-inf")), dim=-1)
                total = torch.where(obs, perda_obs, torch.zeros_like(perda_obs)).sum() + torch.where(cens, -sobrev, torch.zeros_like(sobrev)).sum()
                saida[k] = total / (obs.sum() + cens.sum()).clamp(min=1) / math.log(lg.shape[-1])
            elif (alvo != IGNORAR).any():
                saida[k] = F.cross_entropy(lg.view(-1, lg.shape[-1]), alvo.view(-1), ignore_index=IGNORAR) / math.log(lg.shape[-1])
        saida["total"] = sum(saida.values())
        return saida

    def n_parametros(self):
        return sum(p.numel() for p in self.parameters())
