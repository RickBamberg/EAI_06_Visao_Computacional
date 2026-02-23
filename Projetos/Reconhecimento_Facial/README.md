# 👤 Reconhecimento Facial com DeepFace

Sistema completo de **reconhecimento facial em tempo real** usando OpenCV + DeepFace (FaceNet). Aplicação Flask com interface web para captura, treinamento e reconhecimento.

---

## 🎯 Objetivo

Sistema end-to-end de reconhecimento facial:
- ✅ Captura de fotos via webcam
- ✅ Detecção e recorte automático de faces
- ✅ Geração de embeddings (FaceNet)
- ✅ Reconhecimento em tempo real
- ✅ Interface web completa (Flask)

**Aplicações**:
- Controle de acesso
- Sistema de presença
- Autenticação biométrica
- Portaria inteligente

---

## 🏗️ Arquitetura do Sistema

```
┌─────────────────────────────────────────┐
│         INTERFACE WEB (Flask)           │
├─────────────────────────────────────────┤
│  Captura  │  Preview  │  Treino  │  ID  │
└─────────────────────────────────────────┘
        ↓           ↓          ↓        ↓
┌─────────────────────────────────────────┐
│           PROCESSAMENTO                 │
├─────────────────────────────────────────┤
│  OpenCV DNN  │  DeepFace  │  FaceNet   │
│  (Detecção)  │  (Pipeline)│ (Embedding)│
└─────────────────────────────────────────┘
        ↓           ↓          ↓        ↓
┌─────────────────────────────────────────┐
│            ARMAZENAMENTO                │
├─────────────────────────────────────────┤
│  data/        │  faces_    │ embeddings │
│  [pessoa]/    │  recortadas│  .pickle   │
└─────────────────────────────────────────┘
```

---

## 📂 Estrutura do Projeto

```
Reconhecimento_Facial/
├── app.py                              # Flask backend (940 linhas)
├── capturar_fotos.py                   # Script CLI de captura
├── deploy.prototxt                     # OpenCV face detector
├── res10_300x300_ssd_iter_140000.caffemodel  # Pesos (15MB)
│
├── data/                               # Dados
│   ├── [Nome_Pessoa]/                  # Fotos originais
│   │   └── 1.jpg, 2.jpg, ...
│   ├── faces_recortadas/               # Faces detectadas
│   │   └── [Nome]_1.jpg, ...
│   └── embeddings.pickle               # Banco de embeddings
│
├── notebook/
│   └── tratamento_imagens.ipynb        # Notebook de treinamento
│
├── templates/                          # Frontend
│   ├── base.html                       # Template base
│   ├── index.html                      # Página inicial
│   ├── capturar.html                   # Captura de fotos
│   ├── preview.html                    # Visualizar fotos
│   ├── tratar.html                     # Treinar modelo
│   └── reconhecer.html                 # Reconhecimento em tempo real
│
└── static/
    └── css/
        └── *.css                       # Estilos
```

---

## 🔄 Fluxo Completo

### 1️⃣ Captura de Fotos

```
Usuário acessa /capturar
    ↓
Insere nome da pessoa
    ↓
Ativa webcam
    ↓
Captura 5 fotos (tecla 'S')
    ↓
Salva em data/[Nome]/
```

**Código (app.py)**:
```python
@app.route('/capturar_foto', methods=['POST'])
def capturar_foto():
    global camera, captura_config
    
    if not captura_config['capturando']:
        return jsonify({'status': 'error', 'message': 'Captura não iniciada'})
    
    # Ler frame da câmera
    ret, frame = camera.read()
    
    # Salvar foto
    nome_pessoa = captura_config['nome_pessoa']
    pasta = f'data/{nome_pessoa}'
    os.makedirs(pasta, exist_ok=True)
    
    count = captura_config['fotos_capturadas'] + 1
    caminho = f'{pasta}/{count}.jpg'
    cv2.imwrite(caminho, frame)
    
    captura_config['fotos_capturadas'] = count
    
    return jsonify({'status': 'success', 'fotos': count})
```

---

### 2️⃣ Treinamento (Detecção + Embeddings)

```
Usuário acessa /tratar
    ↓
Clica "Processar Imagens"
    ↓
┌─────────────────────────────────┐
│  1. Detectar faces (OpenCV DNN) │
│  2. Recortar faces              │
│  3. Salvar em faces_recortadas/ │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│  4. Gerar embeddings (FaceNet)  │
│  5. Salvar embeddings.pickle    │
└─────────────────────────────────┘
    ↓
Sistema pronto para reconhecimento
```

**Detecção de Faces (OpenCV DNN)**:
```python
def detectar_e_recortar_faces():
    # Carregar detector SSD
    detector = cv2.dnn.readNetFromCaffe(
        'deploy.prototxt',
        'res10_300x300_ssd_iter_140000.caffemodel'
    )
    
    # Para cada foto em data/[pessoa]/
    for pessoa in os.listdir('data'):
        pasta_pessoa = f'data/{pessoa}'
        
        for foto in os.listdir(pasta_pessoa):
            img = cv2.imread(f'{pasta_pessoa}/{foto}')
            h, w = img.shape[:2]
            
            # Pré-processar
            blob = cv2.dnn.blobFromImage(
                cv2.resize(img, (300, 300)),
                1.0, (300, 300), (104, 177, 123)
            )
            
            # Detectar
            detector.setInput(blob)
            detections = detector.forward()
            
            # Melhor detecção
            best_idx = np.argmax(detections[0, 0, :, 2])
            confidence = detections[0, 0, best_idx, 2]
            
            if confidence > 0.8:
                # Coordenadas
                box = detections[0, 0, best_idx, 3:7] * [w, h, w, h]
                x1, y1, x2, y2 = box.astype(int)
                
                # Recortar
                face = img[y1:y2, x1:x2]
                
                # Salvar
                cv2.imwrite(
                    f'data/faces_recortadas/{pessoa}_{foto}',
                    face
                )
```

**Geração de Embeddings (FaceNet)**:
```python
def gerar_embeddings():
    from deepface import DeepFace
    
    embeddings = []
    names = []
    
    # Para cada face recortada
    for arquivo in os.listdir('data/faces_recortadas'):
        nome_pessoa = arquivo.split('_')[0]
        caminho = f'data/faces_recortadas/{arquivo}'
        
        # Gerar embedding (vetor 128D)
        embedding_obj = DeepFace.represent(
            img_path=caminho,
            model_name='Facenet',
            enforce_detection=False
        )
        
        embedding = embedding_obj[0]['embedding']
        
        embeddings.append(embedding)
        names.append(nome_pessoa)
    
    # Salvar banco de dados
    with open('data/embeddings.pickle', 'wb') as f:
        pickle.dump({
            'embeddings': np.array(embeddings),
            'names': names
        }, f)
```

---

### 3️⃣ Reconhecimento em Tempo Real

```
Usuário acessa /reconhecer
    ↓
Sistema carrega embeddings.pickle
    ↓
Ativa webcam
    ↓
┌───────────────────────────────┐
│ Loop em tempo real:           │
│  1. Capturar frame            │
│  2. Detectar face             │
│  3. Gerar embedding           │
│  4. Comparar com banco (cosine)│
│  5. Exibir nome + confiança   │
└───────────────────────────────┘
```

**Código de Reconhecimento**:
```python
def reconhecer_frame(frame):
    global known_embeddings, known_names
    
    # 1. Detectar face
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        1.0, (300, 300), (104, 177, 123)
    )
    
    detector.setInput(blob)
    detections = detector.forward()
    
    # 2. Para cada face detectada
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence > 0.5:
            # Coordenadas
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            x1, y1, x2, y2 = box.astype(int)
            
            # Recortar face
            face = frame[y1:y2, x1:x2]
            
            if face.size == 0:
                continue
            
            # 3. Gerar embedding da face
            try:
                embedding_obj = DeepFace.represent(
                    img_path=face,
                    model_name='Facenet',
                    enforce_detection=False
                )
                embedding = embedding_obj[0]['embedding']
                
                # 4. Comparar com banco (distância cosine)
                distancias = [
                    cosine(embedding, known_emb)
                    for known_emb in known_embeddings
                ]
                
                # Melhor match
                min_idx = np.argmin(distancias)
                min_dist = distancias[min_idx]
                
                # Threshold: 0.4 (ajustável)
                if min_dist < 0.4:
                    nome = known_names[min_idx]
                    confianca = (1 - min_dist) * 100
                else:
                    nome = "Desconhecido"
                    confianca = 0
                
                # 5. Desenhar na imagem
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                texto = f"{nome} ({confianca:.1f}%)"
                cv2.putText(frame, texto, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
            except Exception as e:
                print(f"Erro ao processar face: {e}")
    
    return frame
```

---

## 💻 Como Usar

### 1. Instalação

```bash
# Criar ambiente
conda create -n face_rec python=3.9
conda activate face_rec

# Dependências
pip install flask flask-cors opencv-python deepface scipy tqdm

# Baixar modelo OpenCV (se não tiver)
# deploy.prototxt
# res10_300x300_ssd_iter_140000.caffemodel
# Download: https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector
```

### 2. Executar Aplicação

```bash
python app.py
```

Acesse: http://localhost:5000

### 3. Fluxo de Uso

**Passo 1: Capturar Fotos**
1. Menu → Captura
2. Digite nome da pessoa
3. Clique "Iniciar Captura"
4. Pressione 'S' para capturar 5 fotos
5. Clique "Parar Captura"

**Passo 2: Treinar Modelo**
1. Menu → Tratar
2. Clique "Processar Imagens"
3. Aguarde detecção + embeddings (1-2 min)
4. Sistema salva embeddings.pickle

**Passo 3: Reconhecer**
1. Menu → Reconhecer
2. Clique "Iniciar Reconhecimento"
3. Webcam mostra nome + confiança
4. Verde = reconhecido, Vermelho = desconhecido

---

## 📊 Tecnologias Utilizadas

| Componente | Tecnologia | Função |
|------------|-----------|--------|
| **Backend** | Flask | Servidor web |
| **Detecção** | OpenCV DNN (SSD) | Detectar faces |
| **Embedding** | DeepFace (FaceNet) | Vetores 128D |
| **Comparação** | Scipy (cosine) | Similaridade |
| **Frontend** | HTML/CSS/JS + Bootstrap | Interface |
| **Webcam** | OpenCV VideoCapture | Stream de vídeo |

---

## 🎯 Accuracy e Performance

### Métricas

```
Detecção de Faces (OpenCV SSD):
- Accuracy: 95%
- FPS: ~30 (CPU)
- Falsos positivos: <5%

Reconhecimento (FaceNet):
- Accuracy: 97-99% (faces frontais)
- Threshold: 0.4 (cosine distance)
- FPS: ~5-10 (CPU), ~30 (GPU)

End-to-End:
- FPS final: ~5-10 (bottleneck = embedding)
- Latência: ~100-200ms por frame
```

### Otimizações

```python
# 1. Processar apenas a cada N frames
frame_count = 0
if frame_count % 3 == 0:  # Processar 1 a cada 3 frames
    reconhecer_frame(frame)
frame_count += 1

# 2. Reduzir resolução
frame_small = cv2.resize(frame, (320, 240))

# 3. GPU acceleration (se disponível)
# DeepFace usa TensorFlow backend
# Configurar GPU em tensorflow
```

---

## 🔍 Troubleshooting

### Problema 1: Câmera não abre

```python
# Solução: Verificar índice da câmera
camera = cv2.VideoCapture(0)  # Tentar 0, 1, 2...

# Listar câmeras disponíveis
import cv2
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Câmera {i} disponível")
        cap.release()
```

### Problema 2: Modelo não baixa

```python
# DeepFace baixa modelos automaticamente
# Se falhar, baixar manualmente:

# FaceNet weights:
# https://github.com/serengil/deepface_models/releases/download/v1.0/facenet_weights.h5
# Colocar em: ~/.deepface/weights/

# OpenCV detector:
# deploy.prototxt
# res10_300x300_ssd_iter_140000.caffemodel
# Colocar na pasta raiz do projeto
```

### Problema 3: Reconhecimento lento

```python
# Solução: Processar menos frames
# ou usar GPU

# CPU: ~5 FPS
# GPU: ~30 FPS

# Configurar TensorFlow GPU:
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
```

---

## 📈 Próximas Melhorias

- [ ] Adicionar múltiplas faces por frame
- [ ] Salvar log de reconhecimentos
- [ ] Dashboard com estatísticas
- [ ] Modo de treinamento incremental
- [ ] Export para ONNX (deploy otimizado)
- [ ] API REST para integração
- [ ] Suporte a máscaras faciais
- [ ] Age e gender detection

---

## 📖 Recursos

### Documentação
- [DeepFace](https://github.com/serengil/deepface)
- [OpenCV Face Detection](https://docs.opencv.org/4.x/d5/d54/group__objdetect.html)
- [FaceNet Paper](https://arxiv.org/abs/1503.03832)

### Datasets
- [LFW (Labeled Faces in the Wild)](http://vis-www.cs.umass.edu/lfw/)
- [VGGFace2](https://github.com/ox-vgg/vgg_face2)

---

## 📧 Contato

**Autor**: Carlos Henrique Bamberg Marques  
**Email**: rick.bamberg@gmail.com  
**GitHub**: [@RickBamberg](https://github.com/RickBamberg/)

---

## 📄 Licença

MIT License

---

**💡 Dica**: Para melhor accuracy, capture fotos com boa iluminação e ângulos variados!

*Projeto prático do curso "Especialista em IA" - Módulo EAI_06*
