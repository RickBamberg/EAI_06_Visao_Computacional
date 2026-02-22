# AGENT_CONTEXT.md - Fundamentos OpenCV

> **Propósito**: Contexto técnico completo dos notebooks de Fundamentos OpenCV  
> **Última atualização**: Janeiro 2026  
> **Tipo**: Seção educacional com 4 notebooks progressivos

## RESUMO EXECUTIVO

**Objetivo**: Ensinar Visão Computacional com OpenCV  
**Notebooks**: 4 notebooks (básico → YOLOv5)  
**Técnicas**: Operações de imagem, filtros, Haar Cascades, YOLOv5  
**Biblioteca**: OpenCV (cv2) + Ultralytics YOLOv5  
**Diferencial**: Do zero (ler imagem) até state-of-the-art (YOLO)

---

## ESTRUTURA DOS NOTEBOOKS

### Progressão Pedagógica

```
Nível 1: fundamentos_opencv.ipynb
├─ Ler, exibir, salvar
├─ Redimensionar, recortar
├─ Espaços de cor (RGB, Gray, HSV)
└─ Transformações (rotate, flip)

Nível 2: filtros_e_bordas.ipynb
├─ Blur (Gaussian, Median, Bilateral)
├─ Threshold (Binary, Adaptive, Otsu)
├─ Detecção de bordas (Canny, Sobel)
└─ Operações morfológicas (erode, dilate)

Nível 3: deteccao_basica.ipynb
├─ Haar Cascades (teoria)
├─ Detecção de faces
├─ Detecção de olhos
└─ Detecção em vídeo/webcam

Nível 4: deteccao_objetos_yolov5.ipynb
├─ YOLO architecture
├─ YOLOv5 em imagens
├─ YOLOv5 em vídeo
└─ Custom training (overview)
```

---

## NOTEBOOK 1: fundamentos_opencv.ipynb

### Objetivo Pedagógico
Entender manipulação básica de imagens antes de algoritmos complexos.

### Conceitos Core

#### Imagem Digital - Estrutura

```python
import cv2
import numpy as np

# Ler imagem
img = cv2.imread('lena.png')

# Estrutura:
# img.shape = (altura, largura, canais)
# img.shape = (512, 512, 3)
#              ↑     ↑    ↑
#            rows  cols  RGB/BGR

# Cada pixel:
# img[y, x] = [B, G, R]  # OpenCV usa BGR!
# Valores: 0-255 (uint8)

# Acessar pixel:
pixel = img[100, 200]  # linha 100, coluna 200
print(pixel)  # [123, 45, 200] = [Blue, Green, Red]
```

**Por que BGR?**
```python
# Razão histórica:
# - Câmeras antigas (anos 70-80) usavam BGR
# - OpenCV mantém compatibilidade
# - Bibliotecas modernas (matplotlib, PIL) usam RGB

# SEMPRE converter para visualização:
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)  # Correto!
```

#### Espaços de Cor - Quando Usar Cada Um

```python
# 1. RGB (Red, Green, Blue)
# Uso: Visualização, natural para humanos
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 2. Grayscale (Tons de cinza)
# Uso: Processamento, detecção de bordas, menos dados
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# shape: (512, 512) - sem canal de cor
# Valores: 0 (preto) a 255 (branco)

# 3. HSV (Hue, Saturation, Value)
# Uso: Segmentação por cor, mais fácil isolar cores
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# H: 0-180 (cor)
# S: 0-255 (saturação)
# V: 0-255 (brilho)

# Exemplo: Segmentar cor azul
lower_blue = np.array([100, 50, 50])
upper_blue = np.array([130, 255, 255])
mask_blue = cv2.inRange(img_hsv, lower_blue, upper_blue)

# 4. LAB (Lightness, A, B)
# Uso: Ajuste de iluminação, processamento profissional
img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
# L: 0-100 (luminosidade)
# A: -128 a 127 (verde-vermelho)
# B: -128 a 127 (azul-amarelo)
```

#### Redimensionamento - Interpolações

```python
# Métodos de interpolação:

# 1. INTER_NEAREST (mais rápido, pior qualidade)
img_nearest = cv2.resize(img, (200, 200), interpolation=cv2.INTER_NEAREST)

# 2. INTER_LINEAR (padrão, bom balanço)
img_linear = cv2.resize(img, (200, 200), interpolation=cv2.INTER_LINEAR)

# 3. INTER_CUBIC (melhor qualidade, mais lento)
img_cubic = cv2.resize(img, (200, 200), interpolation=cv2.INTER_CUBIC)

# 4. INTER_AREA (melhor para reduzir tamanho)
img_area = cv2.resize(img, (200, 200), interpolation=cv2.INTER_AREA)

# Quando usar:
# - Aumentar imagem: INTER_CUBIC
# - Diminuir imagem: INTER_AREA
# - Tempo real: INTER_LINEAR
```

#### Transformações Geométricas

```python
# Rotação com matriz de transformação
def rotate_image(img, angle):
    """
    Rotaciona imagem em qualquer ângulo
    """
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    
    # Matriz de rotação
    M = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    
    # Aplicar transformação
    rotated = cv2.warpAffine(img, M, (w, h))
    
    return rotated

# Matriz M:
# [cos(θ)  -sin(θ)  tx]
# [sin(θ)   cos(θ)  ty]
```

---

## NOTEBOOK 2: filtros_e_bordas.ipynb

### Objetivo Pedagógico
Entender como extrair features de imagens para processamento posterior.

### Filtros - Matemática e Aplicações

#### Convolução 2D - Base de Todos os Filtros

```python
# Kernel (filtro) aplicado por convolução

# Blur simples (média):
kernel_blur = np.ones((5,5), np.float32) / 25
# [[1/25, 1/25, 1/25, 1/25, 1/25],
#  [1/25, 1/25, 1/25, 1/25, 1/25],
#  [1/25, 1/25, 1/25, 1/25, 1/25],
#  [1/25, 1/25, 1/25, 1/25, 1/25],
#  [1/25, 1/25, 1/25, 1/25, 1/25]]

img_blur = cv2.filter2D(img_gray, -1, kernel_blur)

# Sharpen (realçar):
kernel_sharpen = np.array([
    [ 0, -1,  0],
    [-1,  5, -1],
    [ 0, -1,  0]
])
img_sharp = cv2.filter2D(img_gray, -1, kernel_sharpen)
```

#### Gaussian Blur - Detalhado

```python
# Gaussian Blur usa distribuição normal 2D
# G(x, y) = (1 / 2πσ²) * e^(-(x² + y²) / 2σ²)

img_gaussian = cv2.GaussianBlur(img_gray, (5, 5), sigmaX=0)

# Parâmetros:
# - ksize: (5, 5) = tamanho do kernel (ímpar!)
# - sigmaX: 0 = calculado automaticamente
#   - Se sigmaX = 0, sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
#   - Para ksize=5: sigma ≈ 1.5

# Maior sigma = mais blur
img_gaussian_heavy = cv2.GaussianBlur(img_gray, (11, 11), sigmaX=5)
```

#### Bilateral Filter - Preserva Bordas

```python
# Bilateral Filter = Gaussian + preservação de bordas
# Considera:
# 1. Distância espacial (como Gaussian)
# 2. Similaridade de intensidade (não borra se cores diferentes)

img_bilateral = cv2.bilateralFilter(img_gray, d=9, sigmaColor=75, sigmaSpace=75)

# Parâmetros:
# - d: diâmetro do kernel (9 = 9x9)
# - sigmaColor: 75 = filtro de cor (maior = mais cores similares)
# - sigmaSpace: 75 = filtro espacial (maior = mais blur)

# Ideal para:
# - Redução de ruído preservando bordas
# - Cartoon effects
# - Preprocessing para detecção
```

### Detecção de Bordas - Comparação

#### Canny - Multi-stage Algorithm

```python
# Canny é o melhor detector (5 estágios)

edges = cv2.Canny(img_gray, threshold1=100, threshold2=200)

# Estágios internos:
# 1. Gaussian blur (reduz ruído)
# 2. Sobel (calcula gradientes)
# 3. Non-maximum suppression (afina bordas)
# 4. Double threshold (forte/fraco)
# 5. Edge tracking by hysteresis (conecta bordas)

# Parâmetros:
# - threshold1: 100 (mínimo para borda fraca)
# - threshold2: 200 (mínimo para borda forte)
# - Regra prática: threshold2 = 2-3 × threshold1
```

#### Sobel - Derivadas Direcionais

```python
# Sobel detecta mudanças de intensidade

# Gradiente X (bordas verticais)
sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, dx=1, dy=0, ksize=3)

# Gradiente Y (bordas horizontais)
sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, dx=0, dy=1, ksize=3)

# Magnitude (combina X e Y)
sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

# Direção (ângulo das bordas)
sobel_direction = np.arctan2(sobel_y, sobel_x)

# Kernels Sobel:
# Gx = [[-1, 0, 1],     Gy = [[-1, -2, -1],
#       [-2, 0, 2],            [ 0,  0,  0],
#       [-1, 0, 1]]            [ 1,  2,  1]]
```

### Operações Morfológicas

```python
kernel = np.ones((5,5), np.uint8)

# 1. Erosion (erodir = diminuir objetos brancos)
img_erosion = cv2.erode(img_binary, kernel, iterations=1)
# Uso: Remover pequenos ruídos

# 2. Dilation (dilatar = aumentar objetos brancos)
img_dilation = cv2.dilate(img_binary, kernel, iterations=1)
# Uso: Preencher buracos

# 3. Opening (erosion + dilation)
img_opening = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernel)
# Uso: Remover ruído sem alterar tamanho do objeto

# 4. Closing (dilation + erosion)
img_closing = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernel)
# Uso: Preencher buracos em objetos

# 5. Gradient (dilation - erosion)
img_gradient = cv2.morphologyEx(img_binary, cv2.MORPH_GRADIENT, kernel)
# Uso: Detectar contorno de objetos

# 6. Top Hat (original - opening)
img_tophat = cv2.morphologyEx(img_binary, cv2.MORPH_TOPHAT, kernel)
# Uso: Isolar regiões claras

# 7. Black Hat (closing - original)
img_blackhat = cv2.morphologyEx(img_binary, cv2.MORPH_BLACKHAT, kernel)
# Uso: Isolar regiões escuras
```

---

## NOTEBOOK 3: deteccao_basica.ipynb

### Objetivo Pedagógico
Introduzir detecção de objetos com métodos clássicos (pré-deep learning).

### Haar Cascades - Teoria

#### Algoritmo de Viola-Jones (2001)

```python
# Base teórica:
# 1. Haar Features (padrões retangulares)
# 2. Integral Image (cálculo rápido)
# 3. AdaBoost (seleciona melhores features)
# 4. Cascade (rejeita rapidamente não-faces)

# Haar Features exemplos:
# Edge features:
# [■■|  ]  Detecta bordas verticais
# 
# Line features:
# [■■|  |■■]  Detecta linhas
# 
# Four-rectangle:
# [■■|  ]
# [  |■■]  Detecta padrões diagonais
```

#### Cascade Classifier - Como Funciona

```python
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Arquivo XML contém:
# - 24+ estágios de classificação
# - Cada estágio tem features Haar selecionadas
# - Se passa stage 1 → stage 2 → ... → stage 24 = FACE!
# - Se falha em qualquer stage = NÃO É FACE (rejeita rápido)

# Processo de detecção:
faces = face_cascade.detectMultiScale(
    img_gray,
    scaleFactor=1.1,     # Escala de busca
    minNeighbors=5,      # Vizinhos mínimos
    minSize=(30, 30),    # Tamanho mínimo
    flags=cv2.CASCADE_SCALE_IMAGE
)

# scaleFactor:
# - 1.1 = aumenta 10% cada vez (mais preciso, mais lento)
# - 1.3 = aumenta 30% (mais rápido, menos preciso)

# minNeighbors:
# - Quantas detecções sobrepostas para considerar válido
# - 3-4 = mais detecções (mais falsos positivos)
# - 5-6 = menos detecções (mais preciso)
```

#### Otimização de Detecção

```python
def detect_faces_optimized(img):
    """
    Detecção otimizada de faces
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Equalização de histograma (melhora contraste)
    gray = cv2.equalizeHist(gray)
    
    # 2. Detecção em múltiplas escalas
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    # 3. Non-maximum suppression (remove duplicatas)
    if len(faces) > 0:
        faces = non_max_suppression(faces)
    
    return faces

def non_max_suppression(boxes, overlap_threshold=0.3):
    """
    Remove bounding boxes sobrepostas
    """
    # Implementação simplificada
    # (OpenCV groupRectangles faz isso internamente)
    pass
```

### Detecção em Vídeo - Performance

```python
import time

cap = cv2.VideoCapture(0)
fps_counter = []

while True:
    start_time = time.time()
    
    ret, frame = cap.read()
    if not ret:
        break
    
    # Reduzir tamanho para performance
    frame_small = cv2.resize(frame, (320, 240))
    gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
    
    # Detectar
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    
    # Escalar coordenadas de volta
    scale_x = frame.shape[1] / frame_small.shape[1]
    scale_y = frame.shape[0] / frame_small.shape[0]
    
    for (x, y, w, h) in faces:
        x = int(x * scale_x)
        y = int(y * scale_y)
        w = int(w * scale_x)
        h = int(h * scale_y)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Calcular FPS
    fps = 1 / (time.time() - start_time)
    fps_counter.append(fps)
    
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Face Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"FPS médio: {np.mean(fps_counter):.2f}")
```

---

## NOTEBOOK 4: deteccao_objetos_yolov5.ipynb

### Objetivo Pedagógico
Introduzir deep learning para detecção de objetos (state-of-the-art).

### YOLO - Arquitetura

#### YOLO vs Haar Cascades

| Aspecto | Haar Cascades | YOLOv5 |
|---------|---------------|--------|
| **Ano** | 2001 | 2020 |
| **Método** | Hand-crafted features | Deep Learning |
| **Classes** | 1 (específico) | 80 (COCO) |
| **Accuracy** | ~85% (faces) | ~50% mAP (geral) |
| **Speed** | ~30 FPS (CPU) | ~140 FPS (GPU) |
| **Treino** | Pré-treinado | Transfer learning |

#### YOLO Architecture Overview

```
Input Image (640×640)
    ↓
Backbone (CSPDarknet53)
│   ├─ Conv layers
│   ├─ Residual blocks
│   └─ Feature extraction
    ↓
Neck (PANet)
│   ├─ Feature Pyramid Network
│   └─ Multi-scale fusion
    ↓
Head (Detection)
│   ├─ 3 detection scales
│   │   ├─ 80×80 (small objects)
│   │   ├─ 40×40 (medium objects)
│   │   └─ 20×20 (large objects)
│   └─ Output: [x, y, w, h, conf, class_0, ..., class_79]
    ↓
Post-processing (NMS)
    ↓
Final Detections
```

### YOLOv5 - Código Detalhado

```python
import torch
from PIL import Image

# Carregar modelo
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Configurações
model.conf = 0.5   # Confiança mínima (0-1)
model.iou = 0.45   # IoU threshold para NMS
model.classes = None  # None = todas, [0,2,3] = filtrar classes

# Carregar imagem
img = Image.open('street.jpg')

# Inferência
results = model(img)

# Resultados (múltiplos formatos)
results.print()  # Imprimir no console
results.show()   # Mostrar em janela
results.save()   # Salvar em runs/detect/

# Acessar detecções
detections = results.pandas().xyxy[0]
print(detections)

#    xmin   ymin   xmax   ymax  confidence  class    name
# 0  45.2  120.5  230.8  380.2      0.89      0  person
# 1 150.3   80.1  310.7  290.4      0.78     16     dog
```

### Non-Maximum Suppression (NMS)

```python
def manual_nms(boxes, scores, iou_threshold=0.45):
    """
    NMS manual para entender o processo
    
    boxes: [[x1, y1, x2, y2], ...]
    scores: [conf1, conf2, ...]
    """
    # 1. Ordenar por confiança (maior primeiro)
    indices = np.argsort(scores)[::-1]
    
    keep = []
    
    while len(indices) > 0:
        # 2. Pegar box com maior confiança
        current = indices[0]
        keep.append(current)
        
        if len(indices) == 1:
            break
        
        # 3. Calcular IoU com todos os outros
        current_box = boxes[current]
        other_boxes = boxes[indices[1:]]
        
        ious = compute_iou(current_box, other_boxes)
        
        # 4. Remover boxes com IoU > threshold
        indices = indices[1:][ious < iou_threshold]
    
    return keep

def compute_iou(box1, boxes):
    """
    Intersection over Union
    """
    # Coordenadas de interseção
    x1 = np.maximum(box1[0], boxes[:, 0])
    y1 = np.maximum(box1[1], boxes[:, 1])
    x2 = np.minimum(box1[2], boxes[:, 2])
    y2 = np.minimum(box1[3], boxes[:, 3])
    
    # Área de interseção
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    
    # Áreas individuais
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    
    # União
    union = area1 + area2 - intersection
    
    # IoU
    iou = intersection / union
    
    return iou
```

### YOLOv5 em Vídeo - Otimizado

```python
import cv2
import torch

# Carregar modelo UMA VEZ
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model.conf = 0.5

# GPU se disponível
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

cap = cv2.VideoCapture('video.mp4')
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, fps, (640, 480))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Redimensionar para YOLO input (opcional)
    frame_resized = cv2.resize(frame, (640, 640))
    
    # Detectar
    results = model(frame_resized)
    
    # Renderizar boxes
    frame_result = results.render()[0]
    
    # Resize de volta ao tamanho original
    frame_result = cv2.resize(frame_result, (640, 480))
    
    # Salvar e exibir
    out.write(frame_result)
    cv2.imshow('YOLOv5', frame_result)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
```

---

## COMPARAÇÃO: HAAR vs YOLO

### Tabela Técnica Completa

| Métrica | Haar Cascades | YOLOv5s |
|---------|---------------|---------|
| **Accuracy (mAP)** | N/A (task específica) | 37.4% (COCO) |
| **FPS (CPU)** | ~30 | ~3-5 |
| **FPS (GPU)** | ~30 | ~140 |
| **Tamanho Modelo** | <1 MB | 7.2 MB |
| **Treino Custom** | Difícil | Fácil |
| **Multi-class** | Não | Sim (80 classes) |
| **Quando usar** | Faces/olhos, CPU only | Geral, GPU disponível |

---

## TROUBLESHOOTING COMUM

### Problema 1: Imagem não carrega

```python
img = cv2.imread('imagem.png')
if img is None:
    print("Erro: Arquivo não encontrado ou formato inválido")
    # Soluções:
    # 1. Verificar caminho: os.path.exists('imagem.png')
    # 2. Usar caminho absoluto: cv2.imread('/path/completo/imagem.png')
    # 3. Verificar formato: PNG, JPG, JPEG, BMP
```

### Problema 2: Detecção não funciona

```python
# Haar Cascades:
# 1. Usar imagem em grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. Ajustar parâmetros
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.05,  # Tentar valores menores (mais preciso)
    minNeighbors=3     # Tentar valores menores (mais sensível)
)

# 3. Equalizar histograma
gray = cv2.equalizeHist(gray)
```

### Problema 3: YOLOv5 muito lento

```python
# Soluções:
# 1. Usar modelo menor
model = torch.hub.load('ultralytics/yolov5', 'yolov5n')  # nano

# 2. Reduzir tamanho de entrada
results = model(img, size=320)  # em vez de 640

# 3. Usar GPU
model.to('cuda')

# 4. Batch processing
results = model([img1, img2, img3])  # Mais eficiente
```

---

## TAGS DE BUSCA

`#opencv` `#computer-vision` `#image-processing` `#face-detection` `#haar-cascades` `#yolov5` `#object-detection` `#canny-edge` `#filters` `#morphology`

---

**Versão**: 1.0  
**Compatibilidade**: Python 3.7+, OpenCV 4.5+, YOLOv5  
**Uso recomendado**: Aprendizado de Visão Computacional, projetos de detecção
