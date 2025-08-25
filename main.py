import cv2
import numpy as np
import tempfile
import os
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# ================================
# Servir arquivos estáticos
# ================================
app.mount("/static", StaticFiles(directory="static"), name="static")

# ================================
# Tabelas oficiais simuladas (mm)
# ================================
TABELA_ROSCAS = {
    "BSP": {
        "externa": {"1/8": 9.7, "1/4": 13.2, "3/8": 16.7, "1/2": 20.9, "3/4": 26.4, "1": 33.2},
        "interna": {"1/8": 8.5, "1/4": 11.8, "3/8": 15.3, "1/2": 19.0, "3/4": 24.5, "1": 30.3}
    },
    "NPT": {
        "externa": {"1/8": 10.2, "1/4": 13.7, "3/8": 17.1, "1/2": 21.3, "3/4": 26.7, "1": 33.5},
        "interna": {"1/8": 8.7, "1/4": 11.9, "3/8": 15.5, "1/2": 19.3, "3/4": 24.9, "1": 30.8}
    },
    "UNF": {
        "externa": {"1/4": 6.35, "3/8": 9.53, "1/2": 12.7, "3/4": 19.05, "1": 25.4},
        "interna": {"1/4": 5.8, "3/8": 8.8, "1/2": 12.0, "3/4": 18.3, "1": 24.5}
    }
}

# ================================
# Função para evitar cruzamento de medidas
# ================================
def evitar_cruzamento(medidas, distancia_minima=0.2):
    medidas_ordenadas = sorted(medidas)
    medidas_ajustadas = [medidas_ordenadas[0]]
    for medida in medidas_ordenadas[1:]:
        ultima = medidas_ajustadas[-1]
        if medida - ultima < distancia_minima:
            medida = ultima + distancia_minima
        medidas_ajustadas.append(medida)
    return medidas_ajustadas

# ================================
# Função de decisão
# ================================
def fator_decisao(diametro_medido: float, interna: bool):
    tipo = "interna" if interna else "externa"
    candidatos = []
    for norma, dados in TABELA_ROSCAS.items():
        for bitola, diametro_ref in dados[tipo].items():
            if abs(diametro_medido - diametro_ref) <= 0.3:
                candidatos.append((norma, bitola, diametro_ref))
    if not candidatos:
        return None, None, None, 0.0, tipo
    medidas = [c[2] for c in candidatos]
    _ = evitar_cruzamento(medidas)
    norma, bitola, diametro_ref = candidatos[0]
    return norma, bitola, diametro_ref, 95.0, tipo

# ================================
# Função para medir diâmetro via imagem + DEBUG VISUAL
# ================================
def medir_diametro(imagem_path, interna: bool):
    img = cv2.imread(imagem_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detectar círculos
    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
        param1=100, param2=30, minRadius=10, maxRadius=300
    )

    diametro_px = -1
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        diametro_px = max(circles[:, 2]) * 2
        for (x, y, r) in circles:
            cv2.circle(img, (x, y), r, (0, 255, 0), 2)   # círculo verde
            cv2.rectangle(img, (x-2, y-2), (x+2, y+2), (0, 0, 255), -1) # centro vermelho

    # Detectar cartão (contorno maior)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largura_cartao_px = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 80 < w < 1200 and 40 < h < 800:
            largura_cartao_px = max(largura_cartao_px, w)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # cartão azul

    # Salvar imagem debug
    debug_path = os.path.join("static", "debug_output.png")
    cv2.imwrite(debug_path, img)

    if circles is None or largura_cartao_px == 0:
        return -1

    escala = 86.0 / largura_cartao_px
    return diametro_px * escala

# ================================
# Endpoint para abrir o HTML
# ================================
@app.get("/")
def home():
    return FileResponse("static/index.html")

# ================================
# Endpoint de análise
# ================================
@app.post("/analisar")
async def analisar(file: UploadFile = File(...), interna: str = Form(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(await file.read())
            temp_path = tmp.name

        diametro_medido = medir_diametro(temp_path, interna.lower() == "true")

        if diametro_medido <= 0:
            return JSONResponse(content={
                "erro": "Não foi possível identificar a rosca",
                "debug": "/static/debug_output.png"   # link do debug
            }, status_code=400)

        norma, bitola, diametro_ref, confianca, tipo = fator_decisao(diametro_medido, interna.lower() == "true")

        if not norma:
            return JSONResponse(content={
                "erro": "Nenhuma correspondência encontrada",
                "debug": "/static/debug_output.png"
            }, status_code=400)

        return JSONResponse(content={
            "status": "ok",
            "tipo_rosca": "Rosca interna (fêmea)" if interna.lower() == "true" else "Rosca externa (macho)",
            "diametro_medido_mm": f"{diametro_medido:.2f}",
            "bitola": bitola,
            "norma": norma,
            "confianca": f"{confianca:.1f}%",
            "observacao": "Use cartão padrão (86 mm) como referência.",
            "debug": "/static/debug_output.png"
        })

    except Exception as e:
        return JSONResponse(content={"erro": str(e)}, status_code=500)
