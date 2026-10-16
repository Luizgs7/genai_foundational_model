"""Motor genérico — Extração de embeddings do backbone (Tarefa 13 generalizada).

Para cada evento-âncora, o embedding é o hidden state da última camada na
posição do ÚLTIMO token do bloco daquele evento (índice explícito gravado
pela Tarefa 3 em `posicoes_evento.parquet`). Como a atenção é causal, esse
vetor só depende do histórico até o próprio evento — sem vazamento.
Backbone CONGELADO (checkpoint do pré-treino), só inferência.

Uma passada por entidade: janela = últimos `n_positions` tokens de
`seq_full[:-1]`; eventos cuja posição caiu fora da janela (entidades de
cauda longa) ficam sem embedding — mesma limitação já documentada.

Escala (experimento H&M):
  - lotes de várias entidades (preenchidas à direita, ordenadas por tamanho,
    orçamento de tokens) em vez de uma entidade por forward;
  - grava direto num `.npy` em memmap (fp16 por padrão em runs grandes,
    `--dtype float32` pro comportamento legado);
  - âncoras opcionais: se existir `rotulos_downstream/ancoras.parquet`
    (coluna evento_id, gerada pela Tarefa 8), extrai só esses eventos.

Uso: python3 pipeline/common/tarefa13_extrair_embeddings.py <config.yaml>
         [--pretreino <subpasta>] [--saida <subpasta>] [--dtype float16|float32]

Gera runs/<nome>/<saida>/{embeddings_eventos.npy, embeddings_eventos_index.parquet}
(saida padrão: fusao).
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "pipeline/common")
from modelo_decoder import arquitetura_efetiva, construir_modelo  # noqa: E402
from schema import caminho_tabela, carregar_config, gravar_tabela, ler_tabela, run_dir  # noqa: E402
from sequencias_io import carregar_sequencias  # noqa: E402

TOKENS_POR_LOTE = 65536


def main(config_path, pretreino="pretreino", saida="fusao", dtype="float32", arquitetura=None):
    t0 = time.time()
    config = carregar_config(config_path)
    arch = arquitetura_efetiva(arquitetura or config.get("arquitetura", {}))
    n_positions = arch["n_positions"]

    with open(f"{run_dir(config, 'vocabulario')}/vocab.json", encoding="utf-8") as f:
        vocab = json.load(f)
    pad_id = vocab["PAD"]
    seqs = carregar_sequencias(config)
    posicoes = ler_tabela(caminho_tabela(config, "serializacao", "posicoes_evento"))

    ancoras_path = os.path.join(run_dir(config, "rotulos_downstream"), "ancoras.parquet")
    if os.path.exists(ancoras_path):
        ancoras = pd.read_parquet(ancoras_path, columns=["evento_id"])
        posicoes = posicoes[posicoes["evento_id"].isin(ancoras["evento_id"])]
        print(f"Âncoras restritas por {ancoras_path}: {len(posicoes):,} eventos")

    # posição de cada âncora dentro da janela da sua entidade
    idx_ent = posicoes["entidade_id"].map({str(e): i for i, e in enumerate(seqs.ids)} if seqs.ids.dtype == object
                                          else pd.Series(np.arange(len(seqs.ids)), index=seqs.ids))
    if idx_ent.isna().any():
        raise ValueError("posicoes_evento tem entidades ausentes nas sequências")
    idx_ent = idx_ent.to_numpy(dtype=np.int64)
    total = seqs.len_full[idx_ent] - 1               # tamanho de seq_full[:-1]
    offset_janela = np.maximum(0, total - n_positions)
    pos_full = posicoes["pos_full"].to_numpy(dtype=np.int64)
    dentro = (pos_full >= offset_janela) & (pos_full < total)
    n_truncados = int((~dentro).sum())

    pos_ok = posicoes[dentro].reset_index(drop=True)
    idx_ent_ok = idx_ent[dentro]
    local_ok = pos_full[dentro] - offset_janela[dentro]
    row_idx = np.arange(len(pos_ok))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = construir_modelo(arch, vocab).to(device)
    ckpt = f"{run_dir(config, pretreino)}/checkpoints/melhor.pt"
    model.carregar_pesos(torch.load(ckpt, map_location=device))
    model.eval()

    out_dir = run_dir(config, saida)
    os.makedirs(out_dir, exist_ok=True)
    emb = np.lib.format.open_memmap(os.path.join(out_dir, "embeddings_eventos.npy"), mode="w+",
                                    dtype=np.dtype(dtype), shape=(len(pos_ok), arch["n_embd"]))

    # agrupa as âncoras por entidade e processa entidades em lotes por orçamento
    ordem_anc = np.argsort(idx_ent_ok, kind="stable")
    ents, inicio = np.unique(idx_ent_ok[ordem_anc], return_index=True)
    fim = np.r_[inicio[1:], len(ordem_anc)]
    tam = np.minimum(seqs.len_full[ents] - 1, n_positions)
    ordem_ents = np.argsort(tam, kind="stable")

    autocast = torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda")

    def rodar(lote):
        janelas = []
        for k in lote:
            s = seqs.full(ents[k])[:-1]
            janelas.append(s[max(0, len(s) - n_positions):])
        T = max(len(s) for s in janelas)
        ids = np.full((len(janelas), T), pad_id, dtype=np.int64)
        for r, s in enumerate(janelas):
            ids[r, :len(s)] = s
        ids_t = torch.from_numpy(ids).to(device)
        with torch.no_grad(), autocast:
            h = model.estados(ids_t, (ids_t != pad_id).long())
        h = h.float()
        linhas, lin_lote, cols = [], [], []
        for r, k in enumerate(lote):
            a = ordem_anc[inicio[k]:fim[k]]
            linhas.append(row_idx[a]); lin_lote.append(np.full(len(a), r)); cols.append(local_ok[a])
        linhas, lin_lote, cols = np.concatenate(linhas), np.concatenate(lin_lote), np.concatenate(cols)
        vals = h[torch.from_numpy(lin_lote).to(device), torch.from_numpy(cols).to(device)].cpu().numpy()
        emb[linhas] = vals.astype(emb.dtype)

    atual, maior, feitos = [], 0, 0
    for k in ordem_ents:
        novo_maior = max(maior, tam[k])
        if atual and (len(atual) + 1) * novo_maior > TOKENS_POR_LOTE:
            rodar(atual); feitos += len(atual)
            atual, novo_maior = [], tam[k]
        atual.append(k)
        maior = novo_maior
    if atual:
        rodar(atual)
    emb.flush()

    index = pd.DataFrame({"evento_id": pos_ok["evento_id"], "entidade_id": pos_ok["entidade_id"], "row_idx": row_idx})
    gravar_tabela(index, config, saida, "embeddings_eventos_index")
    legado = os.path.join(out_dir, "embeddings_eventos_index.csv")
    if os.path.exists(legado):
        os.remove(legado)

    print(f"embeddings extraidos: {emb.shape[0]:,} (dim={emb.shape[1]}, {emb.dtype})")
    print(f"eventos descartados por truncamento (n_positions={n_positions}): {n_truncados:,}")
    print(f"Gravado em {out_dir}/ ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--pretreino", default="pretreino")
    ap.add_argument("--saida", default="fusao")
    ap.add_argument("--dtype", default="float32", choices=["float16", "float32"])
    a = ap.parse_args()
    main(a.config, a.pretreino, a.saida, a.dtype)
