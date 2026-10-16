"""Motor genérico — Extração de embeddings do backbone (Tarefa 13 generalizada).

Para cada evento-âncora, o embedding é o hidden state da última camada na
posição do ÚLTIMO token do bloco daquele evento (índice explícito gravado
pela Tarefa 3 em `posicoes_evento.parquet`). Como a atenção é causal, esse
vetor só depende do histórico até o próprio evento — sem vazamento.
Backbone CONGELADO (checkpoint do pré-treino), só inferência.

Janela principal por entidade = últimos `n_positions` tokens de
`seq_full[:-1]`. Âncoras anteriores a ela (entidades de cauda longa)
recebem janelas extras terminando nelas, então TODAS as âncoras têm
embedding — o mesmo conjunto de exemplos em qualquer n_positions, o que
mantém justas as comparações downstream (H&M). O legado descartava essas
âncoras (`--sem_janelas_extras` reproduz isso).

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


def main(config_path, pretreino="pretreino", saida="fusao", dtype="float32", arquitetura=None, janelas_extras=True):
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

    # janelas: a 1ª termina no fim da sequência (como no legado); âncoras que
    # ficam antes dela (entidades de cauda longa) ganham janelas extras, cada
    # uma terminando na âncora mais recente ainda não coberta. Com
    # `--sem_janelas_extras` volta ao legado (âncora fora da janela é descartada).
    idx_ent = posicoes["entidade_id"].map({str(e): i for i, e in enumerate(seqs.ids)} if seqs.ids.dtype == object
                                          else pd.Series(np.arange(len(seqs.ids)), index=seqs.ids))
    if idx_ent.isna().any():
        raise ValueError("posicoes_evento tem entidades ausentes nas sequências")
    posicoes = posicoes.assign(_ent=idx_ent.to_numpy(dtype=np.int64)).sort_values(["_ent", "pos_full"]).reset_index(drop=True)
    ent_a = posicoes["_ent"].to_numpy()
    pos_a = posicoes["pos_full"].to_numpy(dtype=np.int64)
    total_a = seqs.len_full[ent_a] - 1                   # tamanho de seq_full[:-1]

    fim_janela = total_a.copy()                           # por padrão, a janela principal
    ok = pos_a < total_a
    fora = ok & (pos_a < total_a - n_positions)
    if not janelas_extras:
        ok &= ~fora
    else:
        # laço só nas entidades com alguma âncora fora da janela principal
        inicio_ent = np.r_[0, np.flatnonzero(np.diff(ent_a)) + 1]
        fim_ent = np.r_[inicio_ent[1:], len(ent_a)]
        tem_fora = np.add.reduceat(fora.astype(np.int64), inicio_ent) > 0
        for a, b in zip(inicio_ent[tem_fora], fim_ent[tem_fora]):
            f = total_a[a]
            for k in range(b - 1, a - 1, -1):
                if pos_a[k] < f - n_positions:            # não coube: nova janela terminando nela
                    f = pos_a[k] + 1
                fim_janela[k] = f
    n_truncados = int((~ok).sum())
    posicoes, ent_a, pos_a, fim_janela = posicoes[ok].reset_index(drop=True), ent_a[ok], pos_a[ok], fim_janela[ok]
    chave = ent_a * (int(fim_janela.max()) + 1) + fim_janela
    chaves_unicas, janela_id = np.unique(chave, return_inverse=True)
    janelas = pd.DataFrame({"ent": chaves_unicas // (int(fim_janela.max()) + 1),
                            "fim": chaves_unicas % (int(fim_janela.max()) + 1)})
    ini_janela = np.maximum(0, janelas["fim"].to_numpy() - n_positions)
    local = pos_a - ini_janela[janela_id]
    row_idx = np.arange(len(posicoes))
    print(f"Âncoras: {len(posicoes):,} em {len(janelas):,} janelas "
          f"({len(janelas) - janelas['ent'].nunique():,} janelas extras); descartadas: {n_truncados:,}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = construir_modelo(arch, vocab).to(device)
    ckpt = f"{run_dir(config, pretreino)}/checkpoints/melhor.pt"
    model.carregar_pesos(torch.load(ckpt, map_location=device))
    model.eval()

    out_dir = run_dir(config, saida)
    os.makedirs(out_dir, exist_ok=True)
    emb = np.lib.format.open_memmap(os.path.join(out_dir, "embeddings_eventos.npy"), mode="w+",
                                    dtype=np.dtype(dtype), shape=(len(posicoes), arch["n_embd"]))

    # agrupa as âncoras por janela e processa janelas em lotes por orçamento
    ordem_anc = np.argsort(janela_id, kind="stable")
    jan_ord = janela_id[ordem_anc]
    inicio = np.searchsorted(jan_ord, np.arange(len(janelas)), side="left")
    fim = np.searchsorted(jan_ord, np.arange(len(janelas)), side="right")
    ents = janelas["ent"].to_numpy()
    fins = janelas["fim"].to_numpy()
    tam = fins - ini_janela
    ordem_jan = np.argsort(tam, kind="stable")

    autocast = torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda")

    def rodar(lote):
        seqs_lote = []
        for k in lote:
            s = seqs.full(ents[k])
            seqs_lote.append(s[ini_janela[k]:fins[k]])
        T = max(len(s) for s in seqs_lote)
        ids = np.full((len(seqs_lote), T), pad_id, dtype=np.int64)
        for r, s in enumerate(seqs_lote):
            ids[r, :len(s)] = s
        ids_t = torch.from_numpy(ids).to(device)
        with torch.no_grad(), autocast:
            h = model.estados(ids_t, (ids_t != pad_id).long())
        h = h.float()
        linhas, lin_lote, cols = [], [], []
        for r, k in enumerate(lote):
            a = ordem_anc[inicio[k]:fim[k]]
            linhas.append(row_idx[a]); lin_lote.append(np.full(len(a), r)); cols.append(local[a])
        linhas, lin_lote, cols = np.concatenate(linhas), np.concatenate(lin_lote), np.concatenate(cols)
        vals = h[torch.from_numpy(lin_lote).to(device), torch.from_numpy(cols).to(device)].cpu().numpy()
        emb[linhas] = vals.astype(emb.dtype)

    atual, maior, feitos, t_ult = [], 0, 0, time.time()
    for k in ordem_jan:
        novo_maior = max(maior, tam[k])
        if atual and (len(atual) + 1) * novo_maior > TOKENS_POR_LOTE:
            rodar(atual); feitos += len(atual)
            atual, novo_maior = [], tam[k]
            if time.time() - t_ult > 60:
                print(f"  {feitos:,}/{len(janelas):,} janelas ({time.time()-t0:.0f}s)", flush=True); t_ult = time.time()
        atual.append(k)
        maior = novo_maior
    if atual:
        rodar(atual)
    emb.flush()

    index = pd.DataFrame({"evento_id": posicoes["evento_id"], "entidade_id": posicoes["entidade_id"], "row_idx": row_idx})
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
    ap.add_argument("--sem_janelas_extras", action="store_true", help="legado: descarta âncoras fora da última janela")
    a = ap.parse_args()
    main(a.config, a.pretreino, a.saida, a.dtype, janelas_extras=not a.sem_janelas_extras)
