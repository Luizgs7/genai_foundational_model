"""Tarefa 13 (Estágio 4/5, ARQUITETURA.md) — extrai o embedding sequencial
do backbone treinado (Tarefa 12), na posição de âncora de CADA transação
de cada cliente — não só a última do split, como um teste inicial mostrou
ser tendencioso (ver nota abaixo).

Âncora de uma transação = hidden state do backbone na posição do último
token daquele evento, processado com atenção causal — ou seja, o modelo
só "viu" o histórico até e incluindo aquela compra. Este é exatamente o
"embedding sequencial" do Estágio 4 do ARQUITETURA.md, calculado em tantos
pontos no tempo quanto existirem transações rotuladas (Tarefa 8).

Por que não só a última transação de cada split (tentativa inicial):
escolher a ÚLTIMA transação de um cliente dentro de um split introduz um
viés de seleção forte — ela tende a estar perto da borda do split (pouco
tempo pra "voltar a comprar"), inflando artificialmente a taxa de churn
(medido: 83% no treino, 55% na validação, vs. a taxa real ~29,7%/~31,8%
documentada na Tarefa 8). Usar TODAS as transações rotuladas como âncora
reproduz exatamente as taxas oficiais da Tarefa 8 (validado antes de rodar
isto: 31,8% treino / 31,7% val / 14,0% teste).

Posição do token de âncora: cada evento é serializado como EVT + 9 campos
fixos + [recência, se não for o 1º evento do cliente] + mês — ou seja,
bloco de 11 tokens (1º evento) ou 12 tokens (demais). Logo, pro i-ésimo
evento (0-indexado, ordenado por data_compra/transacao_id — mesma ordem
de pipeline/serializacao/tarefa3_serializacao.py), o índice do último
token desse bloco dentro de `[bos] + blocos...` é `11 + 12*i` (fórmula
verificada batendo com as contagens de tokens do relatório do cliente
860.703.096-50: 373 tokens/31 transações, 301 tokens/25 transações).

Limitação aceita: clientes com histórico longo o suficiente pra estourar
n_positions=512 (~121 clientes, ~6,4% das linhas rotuladas) têm as
transações mais antigas fora da janela de contexto de uma única passada
— essas ficam sem embedding e são excluídas (mesma limitação já
documentada em LIMITACOES.md sobre n_positions).

Requer GPU (attn_implementation=flash_attention_2). Backbone: checkpoint
da Tarefa 12 (pipeline/pretreino/checkpoints/melhor.pt), CONGELADO — só
inferência, sem gradiente.
"""

import numpy as np
import pandas as pd
import torch
from transformers import GPT2Config, GPT2LMHeadModel

VOCAB_JSON = "pipeline/vocabulario/vocab.json"
SEQUENCIAS_JSON = "pipeline/serializacao/sequencias.json"
ROTULOS_CSV = "pipeline/rotulos_downstream/tarefa8_rotulos.csv"
CKPT = "pipeline/pretreino/checkpoints/melhor.pt"

OUT_EMB_NPY = "pipeline/fusao/embeddings_eventos.npy"
OUT_INDEX_CSV = "pipeline/fusao/embeddings_eventos_index.csv"

N_EMBD, N_LAYER, N_HEAD, N_POSITIONS = 128, 4, 4, 512
PRIMEIRO_BLOCO_LEN = 11
DEMAIS_BLOCO_LEN = 12


def pos_ancora(i):
    """Índice (0-indexado) do último token do i-ésimo evento (0-indexado)
    dentro de `[bos] + blocos...`, sem contar o EOS final."""
    return PRIMEIRO_BLOCO_LEN + DEMAIS_BLOCO_LEN * i


def main():
    import json

    vocab = json.load(open(VOCAB_JSON, encoding="utf-8"))
    seqs = json.load(open(SEQUENCIAS_JSON, encoding="utf-8"))

    rotulos = pd.read_csv(ROTULOS_CSV, parse_dates=["data_compra"])
    rotulos = rotulos.sort_values(["cpf", "data_compra", "transacao_id"]).reset_index(drop=True)
    rotulos["evento_idx"] = rotulos.groupby("cpf").cumcount()

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
    backbone = model.transformer

    embeddings = []
    linhas_index = []
    n_truncados = 0

    for cpf, grupo in rotulos.groupby("cpf", sort=False):
        s = seqs.get(cpf)
        if s is None:
            continue
        seq_full = s["seq_full"]
        conteudo = seq_full[:-1]  # remove só o EOS final
        total = len(conteudo)
        offset = max(0, total - N_POSITIONS)
        janela = conteudo[offset:]

        ids_t = torch.tensor([janela], dtype=torch.long, device=device)
        with torch.no_grad():
            out = backbone(input_ids=ids_t)
        hidden = out.last_hidden_state[0].float().cpu().numpy()  # (T, n_embd)

        for _, row in grupo.iterrows():
            pos_full = pos_ancora(int(row["evento_idx"]))
            if pos_full < offset or pos_full >= total:
                n_truncados += 1
                continue
            local_pos = pos_full - offset
            embeddings.append(hidden[local_pos])
            linhas_index.append({
                "transacao_id": row["transacao_id"],
                "cpf": cpf,
                "split": row["split"],
                "row_idx": len(embeddings) - 1,
            })

    emb_arr = np.stack(embeddings).astype(np.float32)
    np.save(OUT_EMB_NPY, emb_arr)
    pd.DataFrame(linhas_index).to_csv(OUT_INDEX_CSV, index=False)

    print(f"embeddings extraidos: {emb_arr.shape[0]} (dim={emb_arr.shape[1]})")
    print(f"eventos descartados por truncamento (n_positions={N_POSITIONS}): {n_truncados}")
    print(f"Gravado {OUT_EMB_NPY} e {OUT_INDEX_CSV}")


if __name__ == "__main__":
    main()
