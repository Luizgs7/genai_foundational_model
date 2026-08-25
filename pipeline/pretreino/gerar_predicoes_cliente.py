"""Gera o rastro de previsões token a token do backbone treinado para um
cliente exemplo, nos períodos de validação e teste.

Reusa a mesma lógica de avaliação temporal de tarefa_pretreino.py
(rotular_alvo, boundary de val/teste) para produzir, para um único CPF,
contexto/alvo real/previsão top-1/top-5/confiança/loss em cada posição —
insumo da Etapa 6 de pipeline/relatorios/rastro_cliente_860.703.096-50.html.

Requer GPU (mesmo motivo de tarefa_pretreino.py: attn_implementation=
flash_attention_2). Roda a partir do checkpoint já treinado em
pipeline/pretreino/checkpoints/melhor.pt — não retreina nada.
"""

import json
import math

import torch
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel

VOCAB_JSON = "pipeline/vocabulario/vocab.json"
SEQUENCIAS_JSON = "pipeline/serializacao/sequencias.json"
CKPT = "pipeline/pretreino/checkpoints/melhor.pt"
OUT_JSON = "pipeline/pretreino/cliente_860_predicoes.json"
CPF_ALVO = "860.703.096-50"

N_EMBD, N_LAYER, N_HEAD, N_POSITIONS = 128, 4, 4, 512
TOP_K = 5


def rotular_alvo(seq, boundary, n_positions):
    """Mesma função de tarefa_pretreino.py — mantida idêntica para garantir
    que os índices de alvo aqui casam exatamente com os usados no treino."""
    input_ids = seq[:-1]
    labels = seq[1:]
    L = len(seq)
    mask = [boundary <= (i + 1) <= L - 2 for i in range(L - 1)]
    if len(input_ids) > n_positions:
        input_ids = input_ids[-n_positions:]
        labels = labels[-n_positions:]
        mask = mask[-n_positions:]
    return input_ids, labels, mask


def main():
    vocab = json.load(open(VOCAB_JSON, encoding="utf-8"))
    id_to_token = {v: k for k, v in vocab.items()}
    seqs = json.load(open(SEQUENCIAS_JSON, encoding="utf-8"))
    s = seqs[CPF_ALVO]
    st, stv, sf = s["seq_train"], s["seq_train_val"], s["seq_full"]

    boundary_train = (len(st) - 1) if st else 0
    boundary_train_val = (len(stv) - 1) if stv else boundary_train

    device = "cuda"
    config = GPT2Config(
        vocab_size=len(vocab),
        n_positions=N_POSITIONS,
        n_embd=N_EMBD,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        bos_token_id=vocab["BOS"],
        eos_token_id=vocab["EOS"],
        attn_implementation="flash_attention_2",
    )
    model = GPT2LMHeadModel(config).to(dtype=torch.bfloat16, device=device)
    model.load_state_dict(torch.load(CKPT, map_location=device))
    model.eval()

    def prever(seq, boundary, split_label):
        input_ids, labels, mask = rotular_alvo(seq, boundary, N_POSITIONS)
        ids_t = torch.tensor([input_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(input_ids=ids_t)
        logits = out.logits[0].float()
        probs = F.softmax(logits, dim=-1)
        registros = []
        for i, m in enumerate(mask):
            if not m:
                continue
            alvo = labels[i]
            p_alvo = probs[i, alvo].item()
            loss_i = -math.log(max(p_alvo, 1e-12))
            topk = torch.topk(probs[i], TOP_K)
            top_tokens = [id_to_token.get(idx.item(), f"<{idx.item()}>") for idx in topk.indices]
            top_probs = [round(p.item(), 4) for p in topk.values]
            registros.append({
                "split": split_label,
                "contexto_token": id_to_token.get(input_ids[i], f"<{input_ids[i]}>"),
                "alvo_real": id_to_token.get(alvo, f"<{alvo}>"),
                "top5_previsto": top_tokens,
                "top5_prob": top_probs,
                "acertou_top1": top_tokens[0] == id_to_token.get(alvo, f"<{alvo}>"),
                "prob_alvo": round(p_alvo, 4),
                "loss_token": round(loss_i, 4),
            })
        loss_medio = sum(r["loss_token"] for r in registros) / max(len(registros), 1)
        return registros, loss_medio

    val_regs, val_loss = prever(stv, boundary_train, "val")
    test_regs, test_loss = prever(sf, boundary_train_val, "test")

    resultado = {
        "cpf": CPF_ALVO,
        "checkpoint": CKPT,
        "config_modelo": {"n_embd": N_EMBD, "n_layer": N_LAYER, "n_head": N_HEAD},
        "validacao": {
            "n_alvos": len(val_regs),
            "loss_media": round(val_loss, 4),
            "perplexidade": round(math.exp(val_loss), 4),
            "acertos_top1": sum(r["acertou_top1"] for r in val_regs),
            "previsoes": val_regs,
        },
        "teste": {
            "n_alvos": len(test_regs),
            "loss_media": round(test_loss, 4),
            "perplexidade": round(math.exp(test_loss), 4),
            "acertos_top1": sum(r["acertou_top1"] for r in test_regs),
            "previsoes": test_regs,
        },
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)

    print(f"val:   n={len(val_regs):3d}  loss={val_loss:.4f}  ppl={math.exp(val_loss):.3f}  "
          f"acertos_top1={resultado['validacao']['acertos_top1']}/{len(val_regs)}")
    print(f"teste: n={len(test_regs):3d}  loss={test_loss:.4f}  ppl={math.exp(test_loss):.3f}  "
          f"acertos_top1={resultado['teste']['acertos_top1']}/{len(test_regs)}")
    print(f"Gravado {OUT_JSON}")


if __name__ == "__main__":
    main()
