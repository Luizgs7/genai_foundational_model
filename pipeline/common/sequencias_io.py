"""Leitura das sequências de tokens serializadas pela Tarefa 3.

Formato atual (gerado por `tarefa3_serializacao.py`, escala de dezenas de
milhões de eventos):

  runs/<nome>/serializacao/tokens.npy         todos os `seq_full` concatenados
                                              (int16 ou int32), lido via memmap
  runs/<nome>/serializacao/entidades.parquet  1 linha por entidade: entidade_id,
                                              offset, len_full, n_eventos,
                                              len_blocos_train, len_blocos_train_val
  runs/<nome>/serializacao/posicoes_evento.parquet

`seq_train` e `seq_train_val` não são gravados: como os splits são por data
e os eventos de cada entidade estão em ordem temporal, os blocos de treino
são sempre um prefixo dos blocos de `seq_full` (a Tarefa 3 checa isso por
assert). Então:

  seq_train     = seq_full[: 1 + len_blocos_train]     + [EOS]   (None se len_blocos_train == 0)
  seq_train_val = seq_full[: 1 + len_blocos_train_val] + [EOS]   (None se 0)

exatamente as mesmas listas que o formato legado (`sequencias.json`)
guardava explicitamente. O formato legado continua legível
(`Sequencias.carregar` cai nele se não houver `tokens.npy`).
"""

import json
import os

import numpy as np
import pandas as pd


class Sequencias:
    def __init__(self, tokens, entidades):
        self.tokens = tokens
        self.entidades = entidades.reset_index(drop=True)
        self.ids = self.entidades["entidade_id"].to_numpy()
        self.offset = self.entidades["offset"].to_numpy(dtype=np.int64)
        self.len_full = self.entidades["len_full"].to_numpy(dtype=np.int64)
        self.len_train = self.entidades["len_blocos_train"].to_numpy(dtype=np.int64)
        self.len_train_val = self.entidades["len_blocos_train_val"].to_numpy(dtype=np.int64)
        self._indice = None

    # ------------------------------------------------------------ carga

    @classmethod
    def carregar(cls, serializacao_dir):
        tokens_path = os.path.join(serializacao_dir, "tokens.npy")
        if os.path.exists(tokens_path):
            tokens = np.load(tokens_path, mmap_mode="r")
            entidades = pd.read_parquet(os.path.join(serializacao_dir, "entidades.parquet"))
            return cls(tokens, entidades)
        return cls.de_json(os.path.join(serializacao_dir, "sequencias.json"))

    @classmethod
    def de_json(cls, path):
        """Converte o formato legado (sequencias.json) pra mesma interface."""
        with open(path, encoding="utf-8") as f:
            seqs = json.load(f)
        partes, linhas, offset = [], [], 0
        for entidade_id, s in seqs.items():
            sf = s["seq_full"]
            lt = len(s["seq_train"]) - 2 if s["seq_train"] else 0
            ltv = len(s["seq_train_val"]) - 2 if s["seq_train_val"] else 0
            partes.append(np.asarray(sf, dtype=np.int32))
            linhas.append({"entidade_id": entidade_id, "offset": offset, "len_full": len(sf),
                           "n_eventos": s["n_eventos"], "len_blocos_train": lt, "len_blocos_train_val": ltv})
            offset += len(sf)
        tokens = np.concatenate(partes) if partes else np.zeros(0, dtype=np.int32)
        return cls(tokens, pd.DataFrame(linhas))

    # ------------------------------------------------------------ acesso

    def __len__(self):
        return len(self.ids)

    def indice(self, entidade_id):
        if self._indice is None:
            self._indice = {str(e): i for i, e in enumerate(self.ids)}
        return self._indice.get(str(entidade_id))

    def full(self, i):
        o = self.offset[i]
        return np.asarray(self.tokens[o: o + self.len_full[i]], dtype=np.int64)

    def _prefixo(self, i, n_blocos):
        if n_blocos == 0:
            return None
        o = self.offset[i]
        eos = self.tokens[o + self.len_full[i] - 1]
        return np.concatenate([np.asarray(self.tokens[o: o + 1 + n_blocos], dtype=np.int64), [eos]])

    def train(self, i):
        return self._prefixo(i, self.len_train[i])

    def train_val(self, i):
        return self._prefixo(i, self.len_train_val[i])

    def como_dict(self, i):
        """Mesma estrutura do legado sequencias.json (listas Python)."""
        st, stv = self.train(i), self.train_val(i)
        return {
            "n_eventos": int(self.entidades.at[i, "n_eventos"]),
            "seq_full": self.full(i).tolist(),
            "seq_train": st.tolist() if st is not None else None,
            "seq_train_val": stv.tolist() if stv is not None else None,
        }


def carregar_sequencias(config):
    from schema import run_dir

    return Sequencias.carregar(run_dir(config, "serializacao"))
