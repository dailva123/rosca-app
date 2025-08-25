import requests
import os

# ================================
# Configuração
# ================================
file_path = r"C:\backend\uploads\fototeste.png"

# Porta do servidor (padrão: 8000, mas pode vir do ambiente)
PORT = os.environ.get("API_PORT", "8000")
url = f"http://127.0.0.1:{PORT}/analisar"

# ================================
# Envio da requisição
# ================================
try:
    with open(file_path, "rb") as f:
        files = {"file": f}
        data = {"interna": "false"}  # use "true" para rosca interna
        response = requests.post(url, files=files, data=data)

    print("Status:", response.status_code)

    # Se a resposta for JSON válido
    try:
        resp_json = response.json()
        print("Resposta JSON:", resp_json)

        # Mostrar link do debug se existir
        if "debug" in resp_json:
            debug_url = f"http://127.0.0.1:{PORT}{resp_json['debug']}"
            print("👉 Veja a imagem de análise (debug):", debug_url)

    except Exception:
        print("Resposta bruta:", response.text)

except FileNotFoundError:
    print(f"❌ Arquivo não encontrado: {file_path}")
except requests.exceptions.ConnectionError:
    print(f"❌ Não foi possível conectar em {url}. O servidor está rodando?")
except Exception as e:
    print("❌ Erro inesperado:", str(e))
