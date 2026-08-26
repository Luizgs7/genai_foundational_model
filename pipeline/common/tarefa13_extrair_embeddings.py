"""Motor genérico — Extração de embeddings do backbone (Tarefa 13 generalizada).

Versão schema-agnóstica de `pipeline/fusao/tarefa13_extrair_embeddings.py`
(branch `luiz_g`). Mudança estrutural desta generalização: a posição de
âncora de cada evento vem do índice explícito gravado por
`pipeline/common/tarefa3_serializacao.py` (`posicoes_evento.csv`), não de
uma fórmula aritmética fixa (que só valia pro número fixo de campos por
bloco da base sintética) — ver PLANO_PRODUTO.md.

Requer GPU (attn_implementation=flash_attention_2). Backbone: checkpoint
de `pipeline/common/tarefa_pretreino.py`, CONGELADO — só inferência.

Uso: python3 pipeline/common/tarefa13_extrair_embeddings.py <config.yaml>

Gera runs/<nome>/fusao/{embeddings_eventos.npy, embeddings_eventos_index.csv}.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from transformers import GPT2Config, GPT2LMHeadModel

sys.path.insert(0, "pipeline/common")
from schema import carregar_config, run_dir  # noqa: E402


def main(config_path):
    config = carregar_config(config_path)
    arch = config.get("arquitetura", {})
    n_embd, n_layer, n_head = arch.get("n_embd", 128), arch.get("n_layer", 4), arch.get("n_head", 4)
    n_positions = arch.get("n_positions", 512)
    attn_implementation = arch.get("attn_implementation", "flash_attention_2")

    with open(f"{run_dir(config, 'vocabulario')}/vocab.json", encoding="utf-8") as f:
        vocab = json.load(f)
    with open(f"{run_dir(config, 'serializacao')}/sequencias.json", encoding="utf-8") as f:
        seqs = json.load(f)
    posicoes = pd.read_csv(f"{run_dir(config, 'serializacao')}/posicoes_evento.csv")

    ckpt = f"{run_dir(config, 'pretreino')}/checkpoints/melhor.pt"
    device = "cuda"
    gpt2_config = GPT2Config(
        vocab_size=len(vocab), n_positions=n_positions, n_embd=n_embd, n_layer=n_layer, n_head=n_head,
        bos_token_id=vocab["BOS"], eos_token_id=vocab["EOS"], attn_implementation=attn_implementation,
    )
    model = GPT2LMHeadModel(gpt2_config).to(dtype=torch.bfloat16, device=device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    backbone = model.transformer

    embeddings = []
    linhas_index = []
    n_truncados = 0

    for entidade_id, grupo in posicoes.groupby("entidade_id", sort=False):
        s = seqs.get(str(entidade_id)) or seqs.get(entidade_id)
        if s is None:
            continue
        seq_full = s["seq_full"]
        conteudo = seq_full[:-1]
        total = len(conteudo)
        offset = max(0, total - n_positions)
        janela = conteudo[offset:]

        ids_t = torch.tensor([janela], dtype=torch.long, device=device)
        with torch.no_grad():
            out = backbone(input_ids=ids_t)
        hidden = out.last_hidden_state[0].float().cpu().numpy()

        for _, row in grupo.iterrows():
            pos_full = int(row["pos_full"])
            if pos_full < offset or pos_full >= total:
                n_truncados += 1
                continue
            local_pos = pos_full - offset
            embeddings.append(hidden[local_pos])
            linhas_index.append({
                "evento_id": row["evento_id"], "entidade_id": entidade_id, "row_idx": len(embeddings) - 1,
            })

    emb_arr = np.stack(embeddings).astype(np.float32)
    out_dir = run_dir(config, "fusao")
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "embeddings_eventos.npy"), emb_arr)
    pd.DataFrame(linhas_index).to_csv(os.path.join(out_dir, "embeddings_eventos_index.csv"), index=False)

    print(f"embeddings extraidos: {emb_arr.shape[0]} (dim={emb_arr.shape[1]})")
    print(f"eventos descartados por truncamento (n_positions={n_positions}): {n_truncados}")
    print(f"Gravado em {out_dir}/")


if __name__ == "__main__":
    main(sys.argv[1])
