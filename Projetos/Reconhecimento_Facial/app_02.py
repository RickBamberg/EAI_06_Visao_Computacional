# ================== IMPORTAÇÕES ==================
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from waitress import serve
import numpy as np
import cv2
import pickle
import base64
from deepface import DeepFace
from scipy.spatial.distance import cosine
import threading

# ================== INICIALIZAÇÃO ==================
app = Flask(__name__)
CORS(app)

# ================== CARREGAMENTO DOS MODELOS E DADOS ==================
print("Carregando o detector facial...")
# O detector é carregado no escopo global
detector = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')

print("Construindo o modelo FaceNet via DeepFace...")
# A DeepFace lida com o carregamento do modelo de reconhecimento
facenet_model = DeepFace.build_model('Facenet')

print("Carregando o banco de dados de embeddings...")
# Carregamos nosso "cérebro"
with open('data/embeddings.pickle', 'rb') as f:
    database = pickle.load(f)

known_embeddings = database['embeddings']
known_names = database['names']

# Criamos o "cadeado" para garantir acesso seguro ao detector
detector_lock = threading.Lock()

print("Servidor pronto para receber requisições.")


# ================== ROTA PARA A PÁGINA PRINCIPAL ==================
@app.route('/')
def index():
    return render_template('index.html')


# ================== ROTA PARA O RECONHECIMENTO ==================
@app.route('/reconhecer', methods=['POST'])
def reconhecer():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify([])

    header, encoded = data['image'].split(",", 1)
    if not encoded:
        return jsonify([])
    
    image_data = base64.b64decode(encoded)
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify([])

    # Usa o "cadeado" para garantir acesso exclusivo ao detector
    with detector_lock:
        (h, w) = img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        detector.setInput(blob)
        detections = detector.forward()

    results = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.8:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = img[startY:endY, startX:endX]
            
            if face.size == 0:
                continue

            try:
                embedding_obj = DeepFace.represent(img_path=face, model_name='Facenet', enforce_detection=False)
                embedding = embedding_obj[0]['embedding']
                distances = [cosine(embedding, known_emb) for known_emb in known_embeddings]
                min_distance_index = np.argmin(distances)
                min_distance = distances[min_distance_index]

                # Usamos um limiar que funcionou para você
                if min_distance < 0.5:
                    name = known_names[min_distance_index]
                else:
                    name = "Desconhecido"
                
                results.append({
                    "name": name,
                    "confidence": float(confidence),
                    "box": [int(startX), int(startY), int(endX), int(endY)] # Conversão para int padrão
                })

            except Exception as e:
                continue
    
    return jsonify(results)

# ================== EXECUÇÃO DA APLICAÇÃO ==================
if __name__ == '__main__':
    print("Iniciando o servidor de produção Waitress na porta 5000...")
    serve(app, host='0.0.0.0', port=5000)