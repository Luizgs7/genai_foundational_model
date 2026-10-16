"""Fábrica do backbone causal (decoder-only) usado no pré-treino, na extração
de embeddings e nos relatórios.

Dois tipos, escolhidos por `arquitetura.tipo` no YAML:

  - `gpt2_hf` (padrão, legado): `GPT2LMHeadModel` da HuggingFace com o
    embedding posicional zerado e congelado (NoPE) — exatamente o backbone
    dos runs synthetic_agro/olist_sellers. Checkpoints antigos carregam
    direto.
  - `decoder`: decoder próprio, com cada componente configurável — é o que
    viabiliza as ablações de arquitetura (H8 do experimento H&M) mudando um
    fator por vez:
        posicao:     nope | rope | aprendida
        norm:        layernorm | rmsnorm
        ffn:         gelu (4x, igual GPT-2) | swiglu (8/3x — mesmo nº de parâmetros)
        n_kv_heads:  = n_head (MHA) | 1 < n < n_head (GQA) | 1 (MQA)
        dropout:     float (GPT-2 usa 0.1 em embedding, atenção e resíduo)
        bias:        bool (GPT-2 usa bias; Llama não)
    Com posicao=nope, norm=layernorm, ffn=gelu, n_kv_heads=n_head,
    dropout=0.1, bias=true ele é arquiteturalmente o mesmo GPT-2+NoPE (pré-norm,
    GELU tanh, embeddings de entrada/saída compartilhados, init N(0, 0,02)
    com projeções residuais escaladas por 1/sqrt(2*n_layer)).

Interface comum (`ModeloCausal`): `forward(input_ids, attention_mask)` ->
logits; `estados(input_ids, attention_mask)` -> hidden states da última
camada (pós-norm final, igual ao `last_hidden_state` da HF);
`pesos()`/`carregar_pesos()` operam no núcleo (mantém compatibilidade dos
nomes de parâmetro com os checkpoints legados). Os lotes são sempre
preenchidos à DIREITA: com máscara causal, o padding à direita nunca é
visto pelas posições válidas, então o decoder próprio dispensa máscara de
padding.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- decoder próprio

class RoPE(nn.Module):
    def __init__(self, head_dim, n_positions, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        t = torch.arange(n_positions, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, x):  # x: (B, H, T, D)
        T = x.shape[-2]
        cos, sin = self.cos[:T].to(x.dtype), self.sin[:T].to(x.dtype)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        rot = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return rot.flatten(-2)


class Atencao(nn.Module):
    def __init__(self, n_embd, n_head, n_kv_heads, dropout, bias, rope):
        super().__init__()
        assert n_embd % n_head == 0 and n_head % n_kv_heads == 0
        self.n_head, self.n_kv, self.hd = n_head, n_kv_heads, n_embd // n_head
        self.q = nn.Linear(n_embd, n_head * self.hd, bias=bias)
        self.kv = nn.Linear(n_embd, 2 * n_kv_heads * self.hd, bias=bias)
        self.proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.proj.escala_residual = True
        self.dropout = dropout
        self.resid_drop = nn.Dropout(dropout)
        self.rope = rope

    def forward(self, x):
        B, T, C = x.shape
        q = self.q(x).view(B, T, self.n_head, self.hd).transpose(1, 2)
        k, v = self.kv(x).view(B, T, 2, self.n_kv, self.hd).permute(2, 0, 3, 1, 4)
        if self.rope is not None:
            q, k = self.rope(q), self.rope(k)
        if self.n_kv != self.n_head:
            rep = self.n_head // self.n_kv
            k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class FFN(nn.Module):
    def __init__(self, n_embd, tipo, dropout, bias):
        super().__init__()
        self.tipo = tipo
        if tipo == "gelu":
            hidden = 4 * n_embd
            self.fc = nn.Linear(n_embd, hidden, bias=bias)
        elif tipo == "swiglu":
            # 8/3 * d (arredondado pra múltiplo de 8) -> mesmo nº de parâmetros do GELU 4x
            hidden = int(math.ceil((8 * n_embd / 3) / 8) * 8)
            self.fc = nn.Linear(n_embd, 2 * hidden, bias=bias)
        else:
            raise ValueError(f"ffn desconhecida: {tipo}")
        self.proj = nn.Linear(hidden, n_embd, bias=bias)
        self.proj.escala_residual = True
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        if self.tipo == "gelu":
            h = F.gelu(self.fc(x), approximate="tanh")
        else:
            a, b = self.fc(x).chunk(2, dim=-1)
            h = F.silu(a) * b
        return self.drop(self.proj(h))


def norma(tipo, n_embd, bias):
    if tipo == "layernorm":
        return nn.LayerNorm(n_embd, eps=1e-5, bias=bias)
    if tipo == "rmsnorm":
        return nn.RMSNorm(n_embd, eps=1e-5)
    raise ValueError(f"norm desconhecida: {tipo}")


class Bloco(nn.Module):
    def __init__(self, n_embd, n_head, n_kv_heads, norm, ffn, dropout, bias, rope):
        super().__init__()
        self.ln1 = norma(norm, n_embd, bias)
        self.attn = Atencao(n_embd, n_head, n_kv_heads, dropout, bias, rope)
        self.ln2 = norma(norm, n_embd, bias)
        self.mlp = FFN(n_embd, ffn, dropout, bias)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class DecoderCausal(nn.Module):
    def __init__(self, vocab_size, n_positions, n_embd, n_layer, n_head, n_kv_heads=None,
                 posicao="nope", norm="layernorm", ffn="gelu", dropout=0.1, bias=True):
        super().__init__()
        n_kv_heads = n_kv_heads or n_head
        self.n_positions = n_positions
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(n_positions, n_embd) if posicao == "aprendida" else None
        rope = RoPE(n_embd // n_head, n_positions) if posicao == "rope" else None
        if posicao not in ("nope", "rope", "aprendida"):
            raise ValueError(f"posicao desconhecida: {posicao}")
        self.drop = nn.Dropout(dropout)
        self.blocos = nn.ModuleList([Bloco(n_embd, n_head, n_kv_heads, norm, ffn, dropout, bias, rope)
                                     for _ in range(n_layer)])
        self.ln_f = norma(norm, n_embd, bias)
        self.apply(lambda m: self._init(m, n_layer))

    @staticmethod
    def _init(m, n_layer):
        if isinstance(m, nn.Linear):
            std = 0.02 / math.sqrt(2 * n_layer) if getattr(m, "escala_residual", False) else 0.02
            nn.init.normal_(m.weight, 0.0, std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def estados(self, input_ids):
        x = self.wte(input_ids)
        if self.wpe is not None:
            x = x + self.wpe(torch.arange(input_ids.shape[1], device=input_ids.device))
        x = self.drop(x)
        for bloco in self.blocos:
            x = bloco(x)
        return self.ln_f(x)

    def forward(self, input_ids):
        h = self.estados(input_ids)
        return F.linear(h, self.wte.weight)  # embeddings de entrada/saída compartilhados


# ---------------------------------------------------------------- interface comum

class ModeloCausal(nn.Module):
    def __init__(self, nucleo, tipo):
        super().__init__()
        self.nucleo = nucleo
        self.tipo = tipo

    def forward(self, input_ids, attention_mask=None):
        if self.tipo == "gpt2_hf":
            return self.nucleo(input_ids=input_ids, attention_mask=attention_mask).logits
        return self.nucleo(input_ids)

    def estados(self, input_ids, attention_mask=None):
        if self.tipo == "gpt2_hf":
            return self.nucleo.transformer(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.nucleo.estados(input_ids)

    def pesos(self):
        return self.nucleo.state_dict()

    def carregar_pesos(self, state_dict):
        self.nucleo.load_state_dict(state_dict)

    def n_parametros(self):
        total = sum(p.numel() for p in self.parameters())
        treinaveis = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, treinaveis


def arquitetura_efetiva(arch):
    """Arquitetura com os padrões preenchidos (o que de fato é instanciado)."""
    a = {
        "tipo": "gpt2_hf", "n_positions": 512, "n_embd": 128, "n_layer": 4, "n_head": 4,
        "attn_implementation": "sdpa",
        "n_kv_heads": None, "posicao": "nope", "norm": "layernorm", "ffn": "gelu", "dropout": 0.1, "bias": True,
    }
    a.update(arch or {})
    if a["n_kv_heads"] is None:
        a["n_kv_heads"] = a["n_head"]
    return a


def construir_modelo(arch, vocab):
    a = arquitetura_efetiva(arch)
    if a["tipo"] == "gpt2_hf":
        from transformers import GPT2Config, GPT2LMHeadModel

        attn = a["attn_implementation"]
        if attn == "flash_attention_2" and not torch.cuda.is_available():
            attn = "sdpa"
        cfg = GPT2Config(
            vocab_size=len(vocab), n_positions=a["n_positions"], n_embd=a["n_embd"], n_layer=a["n_layer"],
            n_head=a["n_head"], bos_token_id=vocab["BOS"], eos_token_id=vocab["EOS"], attn_implementation=attn,
            resid_pdrop=a["dropout"], embd_pdrop=a["dropout"], attn_pdrop=a["dropout"],
        )
        nucleo = GPT2LMHeadModel(cfg)
        with torch.no_grad():
            nucleo.transformer.wpe.weight.zero_()
        nucleo.transformer.wpe.weight.requires_grad_(False)  # NoPE
        return ModeloCausal(nucleo, "gpt2_hf")
    if a["tipo"] == "decoder":
        nucleo = DecoderCausal(
            vocab_size=len(vocab), n_positions=a["n_positions"], n_embd=a["n_embd"], n_layer=a["n_layer"],
            n_head=a["n_head"], n_kv_heads=a["n_kv_heads"], posicao=a["posicao"], norm=a["norm"], ffn=a["ffn"],
            dropout=a["dropout"], bias=a["bias"],
        )
        return ModeloCausal(nucleo, "decoder")
    raise ValueError(f"arquitetura.tipo desconhecido: {a['tipo']}")
