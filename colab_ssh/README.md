# SSH compartilhado para o Colab do grupo

Guia para o time conectar, via VSCode Remote-SSH, na mesma sessão do Colab
(mesma conta Google, mesma GPU) — sem precisar de sudo nas máquinas locais e
sem compartilhar senha nenhuma (autenticação só por chave pública).

## Como funciona, resumido

```
Sua máquina  --ssh-->  cloudflared (ProxyCommand)  --túnel-->  Cloudflare  --túnel-->  cloudflared (no Colab)  -->  sshd  -->  container do Colab (GPU)
```

O Colab não tem IP público acessível de fora. O `cloudflared` cria um túnel
de saída (o Colab "liga" pra Cloudflare, não o contrário) e expõe uma URL
tipo `https://algo-aleatorio.trycloudflare.com`. Do seu lado, o SSH usa essa
URL como se fosse um host normal, mas troca a conexão TCP direta por um
`ProxyCommand` que fala com esse túnel.

**A URL muda toda vez que a sessão do Colab reinicia** (é um túnel efêmero,
sem conta Cloudflare por trás) — sempre que alguém reiniciar o runtime, tem
que rodar a célula de novo e reenviar a URL nova pro grupo.

## Passo 1 — cada pessoa, uma vez, na sua máquina

Pré-requisito: VSCode com a extensão **Remote - SSH** instalada.

```bash
git clone https://github.com/Luizgs7/genai_foundational_model.git
cd genai_foundational_model
bash colab_ssh/setup_local.sh
```

O script (sem sudo):
- baixa o binário do `cloudflared` (detecta Linux/macOS, x86/arm) em `~/.local/bin`;
- gera uma chave SSH dedicada só pra isso (`~/.ssh/colab_gcm`) — não mexe nas
  suas outras chaves;
- adiciona um bloco `Host *.trycloudflare.com` no seu `~/.ssh/config`.

No final ele imprime sua **chave pública**. Copie essa linha.

## Passo 2 — entrar na lista de chaves autorizadas

Abra `colab_ssh/authorized_keys` neste repo, adicione a linha que o script
imprimiu (uma chave por linha) e suba isso (commit + push, ou peça pra
alguém do time incluir). É uma chave pública — não tem problema nenhum isso
estar num repo público.

## Passo 3 — quem estiver com a sessão do Colab aberta

No notebook, numa célula:

```python
!pip -q install colab_ssh
!curl -fsSL -o colab_session_setup.py https://raw.githubusercontent.com/Luizgs7/genai_foundational_model/main/colab_ssh/colab_session_setup.py
!python3 colab_session_setup.py
```

Isso sobe o SSH, busca a lista atualizada de chaves do time direto do
GitHub (`colab_ssh/authorized_keys`), corrige dois problemas que já pegamos
na prática (porta do sshd e `LD_LIBRARY_PATH` da GPU — detalhes na seção
abaixo) e imprime a URL do túnel no final. **Manda essa URL pro grupo.**

> Atualizou sua chave no passo 2 depois que essa célula já rodou? Precisa
> rodar a célula de novo pra puxar a lista atualizada (ela é lida do GitHub
> a cada execução, não fica salva entre sessões).

## Passo 4 — conectar pelo VSCode

`Ctrl/Cmd+Shift+P` → **Remote-SSH: Connect to Host** → cole o hostname que
alguém te passou (ex.: `algo-aleatorio.trycloudflare.com`, só o domínio, sem
`https://`) → o VSCode abre uma nova janela já dentro do container do Colab.

## Problemas conhecidos

| Sintoma | Causa | O que fazer |
|---|---|---|
| `curl`/navegador na URL do túnel retorna **502 Bad Gateway** | `cloudflared` apontando pra uma porta onde não tem nada escutando | Normalmente já é evitado pelo `colab_session_setup.py` (ele detecta a porta certa). Se acontecer mesmo assim, roda `!ss -tlnp \| grep sshd` no Colab pra confirmar a porta e reabre o túnel apontando pra ela. |
| `curl`/navegador na URL do túnel retorna **erro 1033 da Cloudflare** ("Cloudflare Tunnel error") | A URL usada não corresponde a nenhum túnel ativo no momento. Causa mais comum: o pacote `colab_ssh` sobe, por conta própria, um primeiro túnel (apontando pra porta 22, errada) e imprime uma URL bonita e completa de instruções de conexão — só que esse túnel é derrubado poucos segundos depois pelo próprio script (que sobe outro, correto, na porta certa). É fácil copiar essa URL "de instruções" por engano em vez da URL de verdade, impressa isolada no bloco final (`URL do túnel: ...`) | Sempre use a URL do bloco final impresso por `colab_session_setup.py` (só uma linha, entre `====`), ignore qualquer `*.trycloudflare.com` impressa antes disso. Se aparecer mesmo assim, roda a célula do passo 3 de novo e pega só a URL final. |
| `Permission denied (publickey)` mesmo com a chave presente em `colab_ssh/authorized_keys` no seu checkout local | O `colab_session_setup.py` busca a lista de chaves direto de `raw.githubusercontent.com/.../main/colab_ssh/authorized_keys` — se o commit com sua chave só existe local ou numa branch (ex.: sua branch de feature) e não foi de fato enviado pro `main` no GitHub, o Colab nunca vê a chave nova, mesmo que o `git log` local mostre o commit | Confirma que o commit foi **enviado** (`git push`, não só `git commit`) para o branch `main` do remoto (`git log origin/main` deve incluir seu commit) — e só então roda/re-roda a célula do passo 3. |
| Comando `nvidia-smi`/checagem de GPU funciona numa célula do Colab mas falha (`libnvidia-ml.so` não encontrada) quando rodado via `ssh host "comando"` de fora (não interativo) | `~/.bashrc` (onde a correção do `LD_LIBRARY_PATH` fica persistida) só é carregado em shells **interativas** — um `ssh host "comando"` roda uma shell não interativa e não lê o `~/.bashrc` | Exporte a variável explicitamente no próprio comando SSH: `ssh host "export LD_LIBRARY_PATH=/usr/lib64-nvidia:\$LD_LIBRARY_PATH; nvidia-smi"`. |
| `nvidia-smi` diz que não acha `libnvidia-ml.so` / PyTorch relata `cuda.is_available()=False` | Dentro do container do Colab, a lib do driver não está no `LD_LIBRARY_PATH` padrão (mas a GPU está lá — é só path) | Já corrigido pelo `colab_session_setup.py` (deixa isso persistente no `~/.bashrc` do container). Se for testar manualmente: `export LD_LIBRARY_PATH=/usr/lib64-nvidia:$LD_LIBRARY_PATH`. |
| URL do túnel parou de responder do nada | Sessão do Colab desconectou/reiniciou, ou alguém rodou a célula de novo (o que mata o túnel anterior e cria outro) | Roda o passo 3 de novo e pega a URL nova. |

## Nota de segurança

- É **uma sessão compartilhada**: todo mundo com chave autorizada conecta
  como `root` no mesmo container. Qualquer pessoa do grupo consegue ver
  processos e arquivos de quem também estiver conectado ali. Não é lugar
  pra guardar segredo de longo prazo (tokens, credenciais de outros
  serviços) — só para o trabalho do projeto.
- Sua **chave privada** (`~/.ssh/colab_gcm`) nunca sai da sua máquina e
  nunca deve ser commitada. Só a `.pub` (pública) vai pro repo.
- `flash-attn`/dependências pesadas instaladas na sessão somem quando o
  runtime do Colab reinicia (container efêmero) — reinstale se precisar
  (ver `pipeline/requirements.txt`).
