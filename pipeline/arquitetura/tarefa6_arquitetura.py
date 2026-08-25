"""Tarefa 6 — Arquitetura e instanciação do modelo (Transformer causal, NoPE, FlashAttention2).

Ver .claude/skills/causal-transformer-nope-flashattn/SKILL.md para o racional.

Ambiente local não tem GPU/torch/transformers (checado em 2026-08-21) — a parte
de instanciação/forward/smoke-test do modelo (Estágio 3 do ARQUITETURA.md) só
roda em GPU (Colab, ~16GB VRAM), pois FlashAttention2 exige CUDA + bf16/fp16.
Este script portanto:
  1. Sempre roda (não depende de torch): deriva n_positions dos dados reais
     (Tarefa 3, sequencias.json) e valida a lógica de padding/batching/máscara
     causal com numpy sobre um batch real de clientes.
  2. Roda condicionalmente (se torch/transformers estiverem instalados):
     instancia o modelo de fato, neutraliza NoPE, ativa FlashAttention2 e roda
     forward pass + smoke test de 2 passos de treino num batch real — a parte
     que cobre o DoD completo da Tarefa 6 (pendente de execução no Colab).
"""

import json
import math

import numpy as np
import pandas as pd

VOCAB_JSON = "pipeline/vocabulario/vocab.json"
SEQUENCIAS_JSON = "pipeline/serializacao/sequencias.json"

N_EMBD = 384
N_LAYER = 8
N_HEAD = 8


def carregar_vocab():
    with open(VOCAB_JSON, encoding="utf-8") as f:
        return json.load(f)


def carregar_sequencias():
    with open(SEQUENCIAS_JSON, encoding="utf-8") as f:
        return json.load(f)


def decidir_n_positions(seqs, margem_blocos=64):
    """n_positions dimensionado pelo p99 do comprimento em TOKENS (não em
    eventos) de seq_full, arredondado para cima em blocos de `margem_blocos`.

    O ARQUITETURA.md original propunha n_positions=128 citando p99=40/máx=88
    *eventos*/cliente — mas cada evento vira ~12 tokens (EVT + 10 campos +
    mês, ±recência), então 128 cobriria só ~10 eventos. O número certo tem
    que vir da distribuição de comprimento em tokens (medida na Tarefa 3),
    não em eventos.
    """
    comprimentos = pd.Series([len(s["seq_full"]) for s in seqs.values()])
    p99 = comprimentos.quantile(0.99)
    maximo = comprimentos.max()
    n_positions = int(math.ceil(p99 / margem_blocos) * margem_blocos)
    print("--- Dimensionamento de n_positions (a partir de seq_full, Tarefa 3) ---")
    print(comprimentos.describe(percentiles=[0.5, 0.9, 0.99]))
    print(
        f"\nn_positions escolhido: {n_positions} (cobre p99={p99:.0f} tokens; "
        f"máx observado={maximo} tokens fica fora dessa janela — tratado via "
        f"truncamento/currículo de janela + extrapolação NoPE, ver ARQUITETURA.md)"
    )
    return n_positions


def montar_batch(seqs, pad_id, n_positions, batch_size=8, seed=0):
    """Batch real a partir de sequências de clientes (seq_train, quando
    existente), truncado/paddado para n_positions. Retorna input_ids,
    attention_mask e labels (para NTP, com -100 nas posições de PAD)."""
    rng = np.random.default_rng(seed)
    candidatos = [s["seq_train"] for s in seqs.values() if s["seq_train"]]
    idx = rng.choice(len(candidatos), size=batch_size, replace=False)
    escolhidos = [candidatos[i] for i in idx]

    input_ids = np.full((batch_size, n_positions), pad_id, dtype=np.int64)
    attention_mask = np.zeros((batch_size, n_positions), dtype=np.int64)
    for i, seq in enumerate(escolhidos):
        seq = seq[:n_positions]
        input_ids[i, : len(seq)] = seq
        attention_mask[i, : len(seq)] = 1

    labels = input_ids.copy()
    labels[attention_mask == 0] = -100
    return input_ids, attention_mask, labels


def parametros_estimados(vocab_size, n_embd, n_layer):
    """Contagem aproximada de parâmetros de um GPT-2-like (embedding de
    token, tied com lm_head, + blocos transformer), sem precisar instanciar
    o modelo. Por camada: ~12*n_embd^2 (proj. QKVO + MLP 4x, fórmula padrão
    GPT-2, ignorando bias/LayerNorm por serem desprezíveis nessa escala)."""
    emb = vocab_size * n_embd
    por_camada = 12 * n_embd**2
    return emb + n_layer * por_camada


def main():
    vocab = carregar_vocab()
    vocab_size = len(vocab)
    pad_id = vocab["PAD"]
    seqs = carregar_sequencias()

    print(f"vocab_size = {vocab_size} (Tarefa 5)")
    n_positions = decidir_n_positions(seqs)

    total_params = parametros_estimados(vocab_size, N_EMBD, N_LAYER)
    print(
        f"\nConfig proposta: n_embd={N_EMBD}, n_layer={N_LAYER}, n_head={N_HEAD}, "
        f"n_positions={n_positions}"
    )
    print(f"Parâmetros estimados (fórmula, sem instanciar): ~{total_params/1e6:.1f}M")

    print("\n--- Validação local (sem GPU): lógica de batching/padding/máscara ---")
    input_ids, attention_mask, labels = montar_batch(seqs, pad_id, n_positions, batch_size=8)
    print(f"input_ids.shape={input_ids.shape}, attention_mask.shape={attention_mask.shape}")
    assert input_ids.shape == (8, n_positions)
    assert (labels[attention_mask == 0] == -100).all()
    assert (attention_mask.sum(axis=1) > 0).all()
    print("OK: shapes corretos, padding mascarado em labels com -100, nenhuma sequência vazia.")

    print("\n--- Instanciação real do modelo (requer torch/transformers + GPU) ---")
    try:
        import torch
        from transformers import GPT2Config, GPT2LMHeadModel
    except ImportError as e:
        print(
            f"PULADO: {e}. Ambiente sem torch/transformers — rodar este bloco no "
            f"Colab (GPU, ~16GB VRAM) para validar instanciação real, forward pass "
            f"e smoke test de treino (DoD completo da Tarefa 6)."
        )
        return

    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=n_positions,
        n_embd=N_EMBD,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        bos_token_id=vocab["BOS"],
        eos_token_id=vocab["EOS"],
        attn_implementation="flash_attention_2",
    )
    model = GPT2LMHeadModel(config)
    model = model.to(dtype=torch.bfloat16, device="cuda")

    # NoPE: neutraliza a matriz de position embeddings — fica sempre zero e
    # não aprendida (congelada), em vez de somar um vetor de posição ao token.
    with torch.no_grad():
        model.transformer.wpe.weight.zero_()
    model.transformer.wpe.weight.requires_grad_(False)

    n_params_real = sum(p.numel() for p in model.parameters())
    print(f"Modelo instanciado. Parâmetros reais: {n_params_real/1e6:.1f}M")

    batch = {
        "input_ids": torch.from_numpy(input_ids).to("cuda"),
        "attention_mask": torch.from_numpy(attention_mask).to("cuda"),
        "labels": torch.from_numpy(labels).to("cuda"),
    }
    out = model(**batch)
    print(f"Forward pass OK. logits.shape={tuple(out.logits.shape)}, loss={out.loss.item():.4f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for step in range(2):
        optimizer.zero_grad()
        out = model(**batch)
        out.loss.backward()
        optimizer.step()
        print(f"Smoke test treino - passo {step + 1}: loss={out.loss.item():.4f}")

    print("\nOK: smoke test de treino completo sem erro/OOM.")


if __name__ == "__main__":
    main()
