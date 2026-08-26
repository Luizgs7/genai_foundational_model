"""Gera o rastro de previsões token a token do backbone treinado (NTP) pra
uma entidade específica, nos períodos de validação e teste — insumo do
relatório visual (equivalente genérico de gerar_predicoes_cliente.py da
branch luiz_g). Roda em CPU (attn_implementation=sdpa funciona sem GPU).

Uso: python3 pipeline/common/gerar_predicoes_entidade.py <config.yaml> <entidade_id>
"""

import json
import math
import sys

import torch
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel

sys.path.insert(0, "pipeline/common")
from schema import carregar_config, run_dir  # noqa: E402


def rotular_alvo(seq, boundary, n_positions):
    input_ids = seq[:-1]
    labels = seq[1:]
    L = len(seq)
    mask = [boundary <= (i + 1) <= L - 2 for i in range(L - 1)]
    if len(input_ids) > n_positions:
        input_ids = input_ids[-n_positions:]
        labels = labels[-n_positions:]
        mask = mask[-n_positions:]
    return input_ids, labels, mask


def main(config_path, entidade_id):
    config = carregar_config(config_path)
    arch = config.get("arquitetura", {})
    n_embd, n_layer, n_head = arch.get("n_embd", 128), arch.get("n_layer", 4), arch.get("n_head", 4)
    n_positions = arch.get("n_positions", 512)

    with open(f"{run_dir(config, 'vocabulario')}/vocab.json", encoding="utf-8") as f:
        vocab = json.load(f)
    id_to_token = {v: k for k, v in vocab.items()}
    with open(f"{run_dir(config, 'serializacao')}/sequencias.json", encoding="utf-8") as f:
        seqs = json.load(f)

    s = seqs[entidade_id]
    st, stv, sf = s["seq_train"], s["seq_train_val"], s["seq_full"]
    boundary_train = (len(st) - 1) if st else 0
    boundary_train_val = (len(stv) - 1) if stv else boundary_train

    gpt2_config = GPT2Config(
        vocab_size=len(vocab), n_positions=n_positions, n_embd=n_embd, n_layer=n_layer, n_head=n_head,
        bos_token_id=vocab["BOS"], eos_token_id=vocab["EOS"], attn_implementation="sdpa",
    )
    model = GPT2LMHeadModel(gpt2_config).to(dtype=torch.bfloat16)
    model.load_state_dict(torch.load(f"{run_dir(config, 'pretreino')}/checkpoints/melhor.pt", map_location="cpu"))
    model.eval()

    def prever(seq, boundary, split_label):
        input_ids, labels, mask = rotular_alvo(seq, boundary, n_positions)
        ids_t = torch.tensor([input_ids], dtype=torch.long)
        with torch.no_grad():
            out = model(input_ids=ids_t)
        probs = F.softmax(out.logits[0].float(), dim=-1)
        registros = []
        for i, m in enumerate(mask):
            if not m:
                continue
            alvo = labels[i]
            p_alvo = probs[i, alvo].item()
            topk = torch.topk(probs[i], 1)
            top1 = id_to_token.get(topk.indices[0].item(), "?")
            registros.append({
                "split": split_label,
                "contexto_token": id_to_token.get(input_ids[i], "?"),
                "alvo_real": id_to_token.get(alvo, "?"),
                "top1_previsto": top1,
                "top1_prob": round(topk.values[0].item(), 4),
                "acertou_top1": top1 == id_to_token.get(alvo, "?"),
                "prob_alvo": round(p_alvo, 4),
                "loss_token": round(-math.log(max(p_alvo, 1e-12)), 4),
            })
        loss_medio = sum(r["loss_token"] for r in registros) / max(len(registros), 1)
        return registros, loss_medio

    val_regs, val_loss = prever(stv, boundary_train, "val") if stv and len(stv) > (len(st) if st else 1) else ([], 0)
    test_regs, test_loss = prever(sf, boundary_train_val, "test") if len(sf) > (len(stv) if stv else (len(st) if st else 1)) else ([], 0)

    resultado = {
        "entidade_id": entidade_id,
        "validacao": {"n_alvos": len(val_regs), "loss_media": round(val_loss, 4),
                      "perplexidade": round(math.exp(val_loss), 4) if val_regs else None,
                      "acertos_top1": sum(r["acertou_top1"] for r in val_regs), "previsoes": val_regs},
        "teste": {"n_alvos": len(test_regs), "loss_media": round(test_loss, 4),
                  "perplexidade": round(math.exp(test_loss), 4) if test_regs else None,
                  "acertos_top1": sum(r["acertou_top1"] for r in test_regs), "previsoes": test_regs},
    }
    out_path = f"{run_dir(config, 'relatorios')}/predicoes_{entidade_id[:16]}.json"
    import os
    os.makedirs(run_dir(config, "relatorios"), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)
    print(f"val: n={len(val_regs)} loss={val_loss:.4f} acertos={resultado['validacao']['acertos_top1']}")
    print(f"teste: n={len(test_regs)} loss={test_loss:.4f} acertos={resultado['teste']['acertos_top1']}")
    print(f"Gravado {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
