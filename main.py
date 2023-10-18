from fastapi import FastAPI, UploadFile
from google.cloud import vision_v1
from google.oauth2 import service_account
import cv2
import numpy as np

app = FastAPI()

# Substitua 'seu-arquivo-de-credenciais.json' pelo caminho para o seu arquivo de credenciais.
CREDENTIALS_FILE = 'df-flow-agent-ldkc-90f6ad7421a1.json'

# Crie uma instância do cliente da API do Google Cloud Vision.
credentials = service_account.Credentials.from_service_account_file(CREDENTIALS_FILE)
client = vision_v1.ImageAnnotatorClient(credentials=credentials)

@app.post("/process_image/")
async def process_image(file: UploadFile):
    try:
        # Lê o conteúdo da imagem enviada como um arquivo.
        image = await file.read()

        # Cria um objeto de imagem do Google Cloud Vision com o conteúdo da imagem.
        image_content = vision_v1.Image(content=image)

        # Faz a detecção de texto na imagem.
        response = client.text_detection(image=image_content)

        # Extrai o texto e as coordenadas X e Y dos caracteres.
        extracted_text = response.text_annotations[0].description if response.text_annotations else ""
        character_coordinates = [
            {"x": vertex.x, "y": vertex.y}
            for vertex in response.text_annotations[0].bounding_poly.vertices
        ] if response.text_annotations else []

        # Retorna o texto extraído e as coordenadas.
        return {"text": extracted_text, "character_coordinates": character_coordinates}

    except Exception as e:
        return {"error": str(e)}

@app.post("/plot_points/")
async def plot_points(file: UploadFile):
    try:
        # Lê o conteúdo da imagem enviada como um arquivo.
        image = await file.read()

        # Cria um objeto de imagem do Google Cloud Vision com o conteúdo da imagem.
        image_content = vision_v1.Image(content=image)

        # Faz a detecção de texto na imagem.
        response = client.text_detection(image=image_content)

        # Extrai as coordenadas X e Y dos caracteres.
        character_coordinates = response.text_annotations[0].bounding_poly.vertices if response.text_annotations else []

        # Converte a imagem em um formato que pode ser processado pelo OpenCV.
        nparr = np.frombuffer(image, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Desenhe pontos nas coordenadas dos caracteres.
        for vertex in character_coordinates:
            x, y = vertex.x, vertex.y
            cv2.circle(img_np, (x, y), 5, (0, 0, 255), -1)  # Desenha um círculo vermelho em cada ponto

        # Codifica a imagem resultante em bytes para retorná-la.
        _, img_encoded = cv2.imencode('.jpg', img_np)
        img_bytes = img_encoded.tobytes()

        # Retorna a imagem com os pontos plotados.
        return img_bytes

    except Exception as e:
        return {"error": str(e)}
