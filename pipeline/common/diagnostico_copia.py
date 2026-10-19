"""Diagnóstico da hipótese de "cópia intra-cesta" (experimento H&M, H8).

O RoPE derrubou a loss de pré-treino de ~1,6 para ~0,9 sem ganho downstream.
Hipótese: itens da mesma cesta repetem data, canal, grupo, faixa de preço…,
e uma posição relativa explícita deixa o modelo copiar "o mesmo campo do item
anterior" quase de graça. Se for isso, a diferença de loss entre um modelo
NoPE e um RoPE deve se concentrar nos tokens que REPETEM o valor do mesmo
campo no evento anterior.

Para cada modelo (mesma amostra de entidades de validação, mesmos alvos da
avaliação do pré-treino), mede a loss por token e a decompõe por:
  - campo do token-alvo (prefixo: GRUPO_PRODUTO, PRECO_BUCKET, RECENCIA, ...)
  - repete: o alvo é igual ao último token do mesmo campo antes dele.
    Em "outros" ficam EVT, EOS e os campos sem ocorrência anterior.

Uso: python3 pipeline/common/diagnostico_copia.py configs/hm_clientes.yaml \\
        --modelos nope_h3=pretreino rope_final=final_s0/pretreino [--n_entidades 3000]
Gera runs/<nome>/relatorios/diagnostico_copia.json
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "pipeline/common")
from modelo_decoder import arquitetura_efetiva, construir_modelo  # noqa: E402
from schema import carregar_config, run_dir  # noqa: E402
from sequencias_io import carregar_sequencias  # noqa: E402
from tarefa_pretreino import exemplos_de_avaliacao, montar_lote_avaliacao  # noqa: E402

CAMPOS = ["GRUPO_PRODUTO", "TIPO_PRODUTO", "LINHA_VESTUARIO", "PRECO_BUCKET", "INDICE", "COR", "CANAL",
          "ARTIGO", "RECENCIA", "MES"]


def campo_de(tok):
    for p in CAMPOS:
        if tok.startswith(p):
            return p
    return "OUTRO"


def marcar_repeticao(seq, campo_id, n_campos):
    """repete[p] = seq[p] é igual ao último token do mesmo campo antes de p."""
    ultimo = np.full(n_campos, -1, dtype=np.int64)
    rep = np.zeros(len(seq), dtype=bool)
    for p, t in enumerate(seq):
        c = campo_id[t]
        if c < 0:
            continue
        rep[p] = ultimo[c] == t
        ultimo[c] = t
    return rep


@torch.no_grad()
def perdas_por_token(model, ids, alvo, device, tokens_por_lote=16384):
    """-> lista (por exemplo) de arrays: perda no alvo de cada posição p (NaN onde não é alvo)."""
    out = []
    L = ids.shape[1]
    passo = max(1, tokens_por_lote // L)
    for b in range(0, len(ids), passo):
        x = torch.from_numpy(ids[b:b + passo]).to(device)
        logits = model(x, (x != PAD).long()).float()
        perd = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1),
                               reduction="none").view(x.shape[0], -1).cpu().numpy()
        for r in range(x.shape[0]):
            v = np.full(L, np.nan, dtype=np.float64)
            v[1:] = np.where(alvo[b + r, 1:], perd[r], np.nan)
            out.append(v)
    return out


def main():
    global PAD
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--modelos", nargs="+", required=True, help="nome=pasta_do_pretreino (relativa a runs/<nome>/)")
    ap.add_argument("--n_entidades", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--saida", default="diagnostico_copia.json")
    args = ap.parse_args()

    config = carregar_config(args.config)
    vocab = json.load(open(f"{run_dir(config, 'vocabulario')}/vocab.json", encoding="utf-8"))
    PAD = vocab["PAD"]
    id2tok = {v: k for k, v in vocab.items()}
    campo_id = np.array([CAMPOS.index(campo_de(id2tok[i])) if campo_de(id2tok[i]) in CAMPOS else -1
                         for i in range(len(vocab))])
    seqs = carregar_sequencias(config)
    val_ex, _ = exemplos_de_avaliacao(seqs, "val")
    sel = np.random.RandomState(args.seed).choice(len(val_ex), min(args.n_entidades, len(val_ex)), replace=False)
    exemplos = [val_ex[k] for k in sorted(sel)]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    resultado = {"n_entidades": len(exemplos), "modelos": {}}
    grupos = None
    for spec in args.modelos:
        nome, pasta = spec.split("=", 1)
        pre_dir = run_dir(config, pasta)
        arch = arquitetura_efetiva(json.load(open(os.path.join(pre_dir, "pretreino_config.json")))["arquitetura"])
        model = construir_modelo(arch, vocab).to(device)
        model.carregar_pesos(torch.load(os.path.join(pre_dir, "checkpoints", "melhor.pt"), map_location=device))
        model.eval()
        ids, alvo = montar_lote_avaliacao(seqs, exemplos, "val", arch["n_positions"], PAD)
        perdas = perdas_por_token(model, ids, alvo, device)
        if grupos is None:   # mesma marcação para todos os modelos (mesmos alvos)
            campos_alvo, repete, todas = [], [], []
            for r in range(len(ids)):
                m = alvo[r].copy()
                m[0] = False
                seq = ids[r][ids[r] != PAD]
                rep = marcar_repeticao(seq, campo_id, len(CAMPOS))
                pos = np.flatnonzero(m[:len(seq)])
                campos_alvo.append(np.array([campo_de(id2tok[t]) for t in seq[pos]]))
                repete.append(rep[pos])
                todas.append(pos)
            grupos = {"campo": np.concatenate(campos_alvo), "repete": np.concatenate(repete)}
        perda = np.concatenate([perdas[r][todas[r]] for r in range(len(ids))])
        assert not np.isnan(perda).any()
        c, rep = grupos["campo"], grupos["repete"]
        por_campo = {}
        for k in CAMPOS + ["OUTRO"]:
            for flag, rot in ((True, "repete"), (False, "nao_repete")):
                sel_k = (c == k) & (rep == flag)
                if sel_k.sum():
                    por_campo.setdefault(k, {})[rot] = {"n": int(sel_k.sum()), "loss": float(perda[sel_k].mean())}
        resultado["modelos"][nome] = {
            "pasta": pasta, "arquitetura": arch,
            "loss_media": float(perda.mean()), "n_tokens": int(len(perda)),
            "loss_repete": float(perda[rep].mean()), "loss_nao_repete": float(perda[~rep].mean()),
            "frac_tokens_repete": float(rep.mean()),
            "soma_loss_repete": float(perda[rep].sum()), "soma_loss_nao_repete": float(perda[~rep].sum()),
            "por_campo": por_campo,
        }
        print(f"{nome:14s} loss={perda.mean():.4f}  repete={perda[rep].mean():.4f} ({rep.mean():.1%} dos tokens)  "
              f"nao_repete={perda[~rep].mean():.4f}")
        del model
        torch.cuda.empty_cache()

    # decomposição da diferença entre o 1º modelo (referência) e os demais
    nomes = list(resultado["modelos"])
    ref = resultado["modelos"][nomes[0]]
    n = ref["n_tokens"]
    for nome in nomes[1:]:
        m = resultado["modelos"][nome]
        d_rep = (ref["soma_loss_repete"] - m["soma_loss_repete"]) / n
        d_nrep = (ref["soma_loss_nao_repete"] - m["soma_loss_nao_repete"]) / n
        m["queda_vs_" + nomes[0]] = {"total": d_rep + d_nrep, "de_tokens_que_repetem": d_rep,
                                     "de_tokens_que_nao_repetem": d_nrep,
                                     "frac_da_queda_em_repeticao": d_rep / (d_rep + d_nrep) if d_rep + d_nrep else None}
        print(f"queda {nomes[0]}→{nome}: {d_rep + d_nrep:+.4f} = repete {d_rep:+.4f} + não repete {d_nrep:+.4f}")
    os.makedirs(run_dir(config, "relatorios"), exist_ok=True)
    out = os.path.join(run_dir(config, "relatorios"), args.saida)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=1)
    print(f"Gravado {out}")


if __name__ == "__main__":
    main()
