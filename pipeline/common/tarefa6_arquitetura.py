"""Motor genérico — Arquitetura e instanciação do modelo (Tarefa 6 generalizada).

Versão schema-agnóstica de `pipeline/arquitetura/tarefa6_arquitetura.py`
(branch `luiz_g`) — mesma lógica (n_positions derivado do p99 real de
tokens, validação de batching sem GPU, instanciação real condicional a
torch/transformers estarem disponíveis), parametrizada via config.

Uso: python3 pipeline/common/tarefa6_arquitetura.py <config.yaml>
"""

import json
import math
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "pipeline/common")
from schema import carregar_config, run_dir  # noqa: E402


def decidir_n_positions(seqs, margem_blocos=64):
    comprimentos = pd.Series([len(s["seq_full"]) for s in seqs.values()])
    p99 = comprimentos.quantile(0.99)
    n_positions = int(math.ceil(p99 / margem_blocos) * margem_blocos)
    print("--- Dimensionamento de n_positions (a partir de seq_full) ---")
    print(comprimentos.describe(percentiles=[0.5, 0.9, 0.99]))
    print(f"\nn_positions escolhido: {n_positions} (cobre p99={p99:.0f} tokens; "
          f"máx observado={comprimentos.max()} tokens fica fora dessa janela)")
    return n_positions


def montar_batch(seqs, pad_id, n_positions, batch_size=8, seed=0):
    rng = np.random.default_rng(seed)
    candidatos = [s["seq_train"] for s in seqs.values() if s["seq_train"]]
    idx = rng.choice(len(candidatos), size=min(batch_size, len(candidatos)), replace=False)
    escolhidos = [candidatos[i] for i in idx]

    input_ids = np.full((len(escolhidos), n_positions), pad_id, dtype=np.int64)
    attention_mask = np.zeros((len(escolhidos), n_positions), dtype=np.int64)
    for i, seq in enumerate(escolhidos):
        seq = seq[:n_positions]
        input_ids[i, : len(seq)] = seq
        attention_mask[i, : len(seq)] = 1

    labels = input_ids.copy()
    labels[attention_mask == 0] = -100
    return input_ids, attention_mask, labels


def parametros_estimados(vocab_size, n_embd, n_layer):
    emb = vocab_size * n_embd
    por_camada = 12 * n_embd**2
    return emb + n_layer * por_camada


def main(config_path):
    config = carregar_config(config_path)
    arch = config.get("arquitetura", {})
    n_embd, n_layer, n_head = arch.get("n_embd", 128), arch.get("n_layer", 4), arch.get("n_head", 4)

    with open(f"{run_dir(config, 'vocabulario')}/vocab.json", encoding="utf-8") as f:
        vocab = json.load(f)
    with open(f"{run_dir(config, 'serializacao')}/sequencias.json", encoding="utf-8") as f:
        seqs = json.load(f)

    vocab_size = len(vocab)
    pad_id = vocab["PAD"]
    print(f"vocab_size = {vocab_size}")
    n_positions = arch.get("n_positions") or decidir_n_positions(seqs)

    total_params = parametros_estimados(vocab_size, n_embd, n_layer)
    print(f"\nConfig: n_embd={n_embd}, n_layer={n_layer}, n_head={n_head}, n_positions={n_positions}")
    print(f"Parâmetros estimados: ~{total_params/1e6:.2f}M")

    print("\n--- Validação local (sem GPU): batching/padding/máscara ---")
    input_ids, attention_mask, labels = montar_batch(seqs, pad_id, n_positions, batch_size=8)
    assert (labels[attention_mask == 0] == -100).all()
    assert (attention_mask.sum(axis=1) > 0).all()
    print(f"OK: input_ids.shape={input_ids.shape}, padding mascarado corretamente.")

    print("\n--- Instanciação real do modelo (requer torch/transformers + GPU) ---")
    try:
        import torch
        from transformers import GPT2Config, GPT2LMHeadModel
    except ImportError as e:
        print(f"PULADO: {e}. Rodar no Colab (GPU) pra validar instanciação real.")
        return

    gpt2_config = GPT2Config(
        vocab_size=vocab_size, n_positions=n_positions, n_embd=n_embd, n_layer=n_layer, n_head=n_head,
        bos_token_id=vocab["BOS"], eos_token_id=vocab["EOS"],
        attn_implementation=arch.get("attn_implementation", "flash_attention_2"),
    )
    model = GPT2LMHeadModel(gpt2_config).to(dtype=torch.bfloat16, device="cuda")
    with torch.no_grad():
        model.transformer.wpe.weight.zero_()
    model.transformer.wpe.weight.requires_grad_(False)

    n_params_real = sum(p.numel() for p in model.parameters())
    print(f"Modelo instanciado. Parâmetros reais: {n_params_real/1e6:.2f}M")

    batch = {
        "input_ids": torch.from_numpy(input_ids).to("cuda"),
        "attention_mask": torch.from_numpy(attention_mask).to("cuda"),
        "labels": torch.from_numpy(labels).to("cuda"),
    }
    out = model(**batch)
    print(f"Forward pass OK. logits.shape={tuple(out.logits.shape)}, loss={out.loss.item():.4f}")
    print("\nOK.")


if __name__ == "__main__":
    main(sys.argv[1])
