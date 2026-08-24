"""colab_ssh/colab_session_setup.py

Roda DENTRO de uma célula do notebook do Colab (não na sua máquina local).
Sobe SSH + túnel Cloudflare autenticado só por chave pública, usando a lista
de chaves do time (colab_ssh/authorized_keys neste repo).

Uso, numa célula do Colab:

    !pip -q install colab_ssh
    !curl -fsSL -o colab_session_setup.py https://raw.githubusercontent.com/Luizgs7/genai_foundational_model/main/colab_ssh/colab_session_setup.py
    !python3 colab_session_setup.py

Cada vez que essa célula roda, o container é novo (chaves e processos
anteriores somem) e a URL do túnel muda — rode de novo e reenvie a URL nova
pro grupo sempre que o runtime do Colab reiniciar/desconectar.

Correções aplicadas aqui, descobertas na mão nas primeiras vezes que
conectamos (ver colab_ssh/README.md, seção "problemas conhecidos"):
  - o `launch_ssh_cloudflared` do pacote colab_ssh sobe o sshd numa porta que
    NÃO é necessariamente a 22 (foi 2222 nas sessões que testamos) — este
    script detecta a porta real em vez de assumir 22, e sobe o túnel
    cloudflared apontando pra ela.
  - esse mesmo `launch_ssh_cloudflared` também sobe, por conta própria, um
    túnel cloudflared apontando pra porta 22 (a errada) e imprime uma URL
    *.trycloudflare.com com instruções de conexão — bonita, completa, e
    inútil. É fácil copiar essa URL por engano em vez da de verdade (impressa
    só no final deste script); ela costuma dar 502 e, assim que este script
    mata esse túnel logo em seguida, passa a dar erro 1033 da Cloudflare
    ("no active tunnel"). Este script já derruba esse túnel bogus assim que
    ele sobe e avisa pra ignorar qualquer URL impressa antes do final.
  - dentro do container do Colab, o driver da GPU (libnvidia-ml.so) não está
    no LD_LIBRARY_PATH padrão — sem isso, nvidia-smi/torch.cuda relatam "sem
    GPU" mesmo com o dispositivo presente. Este script deixa isso persistente
    no ~/.bashrc — mas isso só é lido em shells interativas; um `ssh host
    "comando"` direto (não interativo) não carrega o ~/.bashrc, então
    precisa exportar a variável manualmente nesse tipo de uso (ver README).
  - a URL do túnel podia sumir mesmo com tudo saudável: um `sleep` fixo
    curto (8s) às vezes checava o log antes da linha da URL ser escrita.
    Este script agora espera ativamente por até 30s.
"""

import re
import subprocess
import sys
import time
import urllib.request

TEAM_KEYS_URL = (
    "https://raw.githubusercontent.com/Luizgs7/genai_foundational_model/"
    "main/colab_ssh/authorized_keys"
)
LOGFILE = "cloudflared_team.log"
METRICS_PORT = "45999"


def sh(cmd, check=True):
    print(f"$ {cmd}")
    return subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)


def main():
    print("1/5 - subindo sshd via colab_ssh...")
    try:
        from colab_ssh import launch_ssh_cloudflared
    except ImportError:
        sh("pip install -q colab_ssh")
        from colab_ssh import launch_ssh_cloudflared
    # a senha não importa: PasswordAuthentication fica desabilitado nessa
    # imagem por padrão — só autenticação por chave pública funciona.
    launch_ssh_cloudflared(password="unused-password-auth-is-disabled")
    # launch_ssh_cloudflared já sobe seu PRÓPRIO túnel cloudflared, apontando
    # pra porta 22 — mas o sshd desse container escuta em outra porta (ver
    # 3/5), então esse túnel é inútil e será derrubado agora mesmo. A URL
    # *.trycloudflare.com impressa acima por essa lib NÃO deve ser usada
    # (foi a causa de mais de um 502/1033 "misterioso" nas primeiras vezes
    # que o time testou isso).
    sh("pkill -f 'cloudflared tunnel'", check=False)
    time.sleep(2)
    print(
        "\n(ignore qualquer URL *.trycloudflare.com impressa acima — era de um "
        "túnel temporário apontando pra porta errada, já foi encerrado. A URL "
        "certa é a impressa no final deste script.)"
    )

    print("\n2/5 - buscando chaves públicas do time...")
    keys = urllib.request.urlopen(TEAM_KEYS_URL, timeout=15).read().decode()
    sh("mkdir -p /root/.ssh && chmod 700 /root/.ssh")
    with open("/root/.ssh/authorized_keys", "w") as f:
        f.write(keys)
    sh("chmod 600 /root/.ssh/authorized_keys")
    n = len([ln for ln in keys.splitlines() if ln.strip() and not ln.strip().startswith("#")])
    print(f"    {n} chave(s) autorizada(s) a partir do repositório.")

    print("\n3/5 - detectando a porta real do sshd...")
    out = sh("ss -tlnp | grep sshd", check=False).stdout
    m = re.search(r":(\d+)\s", out)
    port = m.group(1) if m else "2222"
    print(f"    sshd escutando em :{port}")

    print("\n4/5 - corrigindo LD_LIBRARY_PATH da GPU (bug conhecido do container)...")
    sh("grep -q lib64-nvidia ~/.bashrc || echo 'export LD_LIBRARY_PATH=/usr/lib64-nvidia:$LD_LIBRARY_PATH' >> ~/.bashrc")
    sh("grep -q 'cuda/bin' ~/.bashrc || echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc")
    sh("grep -q CUDA_HOME ~/.bashrc || echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc")

    print("\n5/5 - subindo túnel cloudflared na porta certa...")
    sh("pkill -f 'cloudflared tunnel'", check=False)
    time.sleep(2)
    sh(
        "test -x ./cloudflared || "
        "(curl -fsSL -o ./cloudflared "
        "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 "
        "&& chmod +x ./cloudflared)"
    )
    subprocess.Popen(
        f"./cloudflared tunnel --url ssh://localhost:{port} "
        f"--logfile ./{LOGFILE} --metrics localhost:{METRICS_PORT}",
        shell=True,
    )
    # espera ativamente pela URL em vez de um sleep fixo: um valor fixo curto
    # (testamos 8s) às vezes checava o log um ou dois segundos antes da linha
    # "Your quick Tunnel has been created" ser escrita, e o script reportava
    # "não achei a URL" mesmo com o túnel de pé e saudável.
    url = None
    for _ in range(30):
        time.sleep(1)
        try:
            log = open(LOGFILE).read()
        except FileNotFoundError:
            continue
        url = re.search(r"https://[a-zA-Z0-9.-]+\.trycloudflare\.com", log)
        if url:
            break

    print("\n" + "=" * 60)
    if url:
        print(f" URL do túnel: {url.group(0)}")
        print(" Manda essa URL pro grupo (ela muda toda vez que essa célula roda de novo).")
    else:
        print(f" Não achei a URL em 30s — roda: !tail -30 {LOGFILE}")
    print("=" * 60)


if __name__ == "__main__":
    sys.exit(main())
