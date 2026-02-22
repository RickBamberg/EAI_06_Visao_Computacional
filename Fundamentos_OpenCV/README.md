# 📸 Fundamentos de OpenCV

Introdução completa à **Visão Computacional** com OpenCV. Aprenda desde operações básicas (ler, exibir, salvar imagens) até detecção de objetos com YOLOv5.

---

## 🎯 Objetivo

Dominar fundamentos de processamento de imagens e detecção de objetos:
- ✅ Operações básicas com OpenCV
- ✅ Filtros e detecção de bordas
- ✅ Detecção de faces, olhos, objetos
- ✅ YOLOv5 para detecção em tempo real

**Por que OpenCV?**
- Biblioteca mais usada em Visão Computacional
- +2.500 algoritmos otimizados
- Suporte Python, C++, Java
- Comunidade gigantesca
- Open source e gratuito

---

## 📂 Estrutura dos Notebooks

```
Fundamentos_OpenCV/
├── fundamentos_opencv.ipynb          # Operações básicas
├── filtros_e_bordas.ipynb            # Filtros, Canny, Sobel
├── deteccao_basica.ipynb             # Faces, olhos, Haar Cascades
├── deteccao_objetos_yolov5.ipynb     # YOLOv5 state-of-the-art
├── lena.png                          # Imagem de teste
└── yolov5s.pt                        # Modelo YOLOv5 pré-treinado
```

**Total**: 4 notebooks progressivos (básico → avançado)

---

## 🗺️ Jornada de Aprendizado

### Progressão Recomendada

```
Semana 1: Fundamentos
├── Dia 1-2: fundamentos_opencv.ipynb
│   └─ Ler, exibir, salvar, redimensionar
│
├── Dia 3-4: filtros_e_bordas.ipynb
│   └─ Blur, Canny, Sobel, threshold
│
└── Dia 5-6: deteccao_basica.ipynb
    └─ Haar Cascades (faces, olhos)

Semana 2: Detecção Avançada
└── Dia 1-7: deteccao_objetos_yolov5.ipynb
    └─ YOLOv5 em imagens e vídeo
```

**Tempo estimado**: 2 semanas (dedicação parcial)

---

## 📚 Conteúdo Detalhado

### 1️⃣ fundamentos_opencv.ipynb

**Objetivo**: Operações básicas de manipulação de imagens

#### Tópicos Abordados:

##### Leitura e Exibição
```python
import cv2
import matplotlib.pyplot as plt

# Ler imagem
img = cv2.imread('lena.png')

# OpenCV usa BGR, converter para RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Exibir
plt.imshow(img_rgb)
plt.axis('off')
plt.show()
```

**Por que BGR?**
- Razões históricas (câmeras antigas)
- OpenCV mantém compatibilidade
- **Sempre converter para RGB ao exibir com matplotlib!**

##### Propriedades da Imagem
```python
print(f"Dimensões: {img.shape}")
# Output: (512, 512, 3) → altura × largura × canais

print(f"Tipo: {img.dtype}")
# Output: uint8 (valores 0-255)

print(f"Tamanho: {img.size} pixels")
# Output: 786432 (512 × 512 × 3)
```

##### Redimensionamento
```python
# Redimensionar para 200×200
img_resize = cv2.resize(img_rgb, (200, 200))

# Manter proporção
scale_percent = 50  # 50% do tamanho original
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
img_scaled = cv2.resize(img_rgb, (width, height))
```

##### Recorte (Crop)
```python
# Recortar região: [y1:y2, x1:x2]
img_crop = img_rgb[100:300, 150:350]
```

##### Rotação
```python
# Rotacionar 90° (sentido horário)
img_rotated = cv2.rotate(img_rgb, cv2.ROTATE_90_CLOCKWISE)

# Rotação arbitrária
center = (img.shape[1]//2, img.shape[0]//2)
angle = 45
scale = 1.0
M = cv2.getRotationMatrix2D(center, angle, scale)
img_rotated_45 = cv2.warpAffine(img_rgb, M, (img.shape[1], img.shape[0]))
```

##### Flip (Espelhamento)
```python
# Flip horizontal
img_flip_h = cv2.flip(img_rgb, 1)

# Flip vertical
img_flip_v = cv2.flip(img_rgb, 0)

# Flip ambos
img_flip_both = cv2.flip(img_rgb, -1)
```

##### Conversão de Espaços de Cor
```python
# RGB → Grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# RGB → HSV (Hue, Saturation, Value)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# RGB → LAB
img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
```

**Quando usar cada espaço?**
- **RGB**: Exibição, visualização
- **Grayscale**: Processamento, detecção de bordas
- **HSV**: Segmentação por cor
- **LAB**: Ajuste de iluminação

##### Salvar Imagem
```python
# Salvar (BGR novamente!)
img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
cv2.imwrite('output.png', img_bgr)
```

---

### 2️⃣ filtros_e_bordas.ipynb

**Objetivo**: Aplicar filtros e detectar bordas

#### Tópicos Abordados:

##### Blur (Desfoque)

```python
# Blur simples
img_blur = cv2.blur(img_gray, (5, 5))

# Gaussian Blur (mais natural)
img_gaussian = cv2.GaussianBlur(img_gray, (5, 5), 0)

# Median Blur (remove ruído sal e pimenta)
img_median = cv2.medianBlur(img_gray, 5)

# Bilateral Filter (preserva bordas)
img_bilateral = cv2.bilateralFilter(img_gray, 9, 75, 75)
```

**Quando usar cada blur?**
- **Blur simples**: Rápido, pouco exigente
- **Gaussian**: Mais suave, melhor aparência
- **Median**: Remover ruído pontual
- **Bilateral**: Suavizar preservando bordas

##### Threshold (Binarização)

```python
# Threshold simples
_, img_thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

# Threshold adaptativo (melhor para iluminação variável)
img_adaptive = cv2.adaptiveThreshold(
    img_gray,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11,  # block size
    2    # constant
)

# Otsu's Threshold (automático)
_, img_otsu = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
```

##### Detecção de Bordas - Canny

```python
# Canny Edge Detection (melhor detector)
edges = cv2.Canny(img_gray, 100, 200)

# Parâmetros:
# - threshold1: 100 (mínimo)
# - threshold2: 200 (máximo)
# - Regra: threshold2 = 2 × threshold1
```

**Por que Canny é o melhor?**
- Supressão de não-máximos
- Histerese de threshold
- Bordas finas e conectadas

##### Sobel (Gradientes)

```python
# Sobel X (bordas verticais)
sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)

# Sobel Y (bordas horizontais)
sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)

# Magnitude (combina X e Y)
sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
```

##### Laplacian (Segunda Derivada)

```python
laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
```

##### Operações Morfológicas

```python
kernel = np.ones((5,5), np.uint8)

# Erosão (diminui objetos brancos)
img_erosion = cv2.erode(img_thresh, kernel, iterations=1)

# Dilatação (aumenta objetos brancos)
img_dilation = cv2.dilate(img_thresh, kernel, iterations=1)

# Opening (erosão + dilatação) - remove ruído
img_opening = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel)

# Closing (dilatação + erosão) - preenche buracos
img_closing = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel)
```

---

### 3️⃣ deteccao_basica.ipynb

**Objetivo**: Detectar faces, olhos e objetos com Haar Cascades

#### Haar Cascades

**O que são?**
- Classificadores baseados em features de Haar
- Treinados com AdaBoost
- Rápidos (tempo real em CPU)
- Pré-treinados para faces, olhos, etc.

#### Detecção de Faces

```python
# Carregar classificador
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Detectar faces
faces = face_cascade.detectMultiScale(
    img_gray,
    scaleFactor=1.1,    # Escala de busca
    minNeighbors=5,     # Vizinhos mínimos (qualidade)
    minSize=(30, 30)    # Tamanho mínimo da face
)

# Desenhar retângulos
for (x, y, w, h) in faces:
    cv2.rectangle(img_rgb, (x, y), (x+w, y+h), (255, 0, 0), 2)

print(f"Faces detectadas: {len(faces)}")
```

**Parâmetros Importantes**:
- **scaleFactor**: Menor = mais preciso, mais lento (1.05 - 1.3)
- **minNeighbors**: Maior = menos falsos positivos (3 - 6)
- **minSize**: Tamanho mínimo do objeto

#### Detecção de Olhos

```python
# Carregar classificador de olhos
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml'
)

# Detectar olhos (dentro das faces)
for (x, y, w, h) in faces:
    roi_gray = img_gray[y:y+h, x:x+w]
    roi_color = img_rgb[y:y+h, x:x+w]
    
    eyes = eye_cascade.detectMultiScale(roi_gray)
    
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
```

#### Cascades Disponíveis

```python
# Faces
haarcascade_frontalface_default.xml
haarcascade_frontalface_alt.xml
haarcascade_profileface.xml

# Olhos
haarcascade_eye.xml
haarcascade_eye_tree_eyeglasses.xml

# Corpo
haarcascade_fullbody.xml
haarcascade_upperbody.xml

# Sorrisos
haarcascade_smile.xml
```

#### Detecção em Vídeo

```python
cap = cv2.VideoCapture(0)  # 0 = webcam padrão

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    cv2.imshow('Detecção de Faces', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

### 4️⃣ deteccao_objetos_yolov5.ipynb

**Objetivo**: Detecção de objetos state-of-the-art com YOLOv5

#### YOLOv5 - Estado da Arte

**O que é YOLO?**
- **You Only Look Once** - uma passada pela rede
- Detecção em tempo real (>30 FPS)
- 80 classes (COCO dataset)
- Versões: YOLOv5n, s, m, l, x (nano → extra)

#### Instalação

```python
# Instalar YOLOv5
!pip install ultralytics

# Ou usar repositório oficial
!git clone https://github.com/ultralytics/yolov5
%cd yolov5
!pip install -r requirements.txt
```

#### Detecção em Imagem

```python
import torch
from PIL import Image

# Carregar modelo pré-treinado
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Carregar imagem
img = Image.open('imagem.jpg')

# Detectar
results = model(img)

# Visualizar
results.show()

# Obter dados
detections = results.pandas().xyxy[0]
print(detections)
```

**Saída**:
```
   xmin   ymin   xmax   ymax  confidence  class    name
0   45.0  120.0  230.0  380.0      0.89     0  person
1  150.0   80.0  310.0  290.0      0.78    16     dog
2  320.0  140.0  450.0  340.0      0.65     2     car
```

#### Modelos Disponíveis

| Modelo | Tamanho | mAP | FPS (V100) |
|--------|---------|-----|------------|
| **YOLOv5n** | 1.9 MB | 28.0 | 455 |
| **YOLOv5s** | 7.2 MB | 37.4 | 143 |
| **YOLOv5m** | 21.2 MB | 45.4 | 64 |
| **YOLOv5l** | 46.5 MB | 49.0 | 35 |
| **YOLOv5x** | 86.7 MB | 50.7 | 19 |

**Recomendação**: YOLOv5s para maioria dos casos (balanço speed/accuracy)

#### Detecção em Vídeo

```python
import cv2

# Carregar modelo
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Capturar vídeo
cap = cv2.VideoCapture('video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detectar
    results = model(frame)
    
    # Renderizar
    frame_result = results.render()[0]
    
    cv2.imshow('YOLOv5', frame_result)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

#### Classes Detectadas (COCO)

```python
# 80 classes do COCO dataset
classes = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    # ... (80 total)
]
```

#### Ajustar Confiança

```python
# Filtrar por confiança mínima
model.conf = 0.5  # 50% confiança mínima

# Filtrar por classes específicas
model.classes = [0, 16, 2]  # person, dog, car apenas
```

#### Salvar Resultados

```python
# Salvar imagem com detecções
results.save('output/')

# Salvar coordenadas em CSV
results.save('output/', save_txt=True)

# Salvar em JSON
import json
detections_dict = results.pandas().xyxy[0].to_dict(orient='records')
with open('detections.json', 'w') as f:
    json.dump(detections_dict, f, indent=2)
```

---

## 💻 Instalação e Setup

### Requisitos

```
Python 3.7+
8GB RAM (mínimo)
GPU (opcional, acelera YOLOv5)
```

### Instalação de Dependências

```bash
# Criar ambiente
conda create -n cv_env python=3.9
conda activate cv_env

# Instalar OpenCV
pip install opencv-python opencv-contrib-python

# Matplotlib para visualização
pip install matplotlib

# NumPy
pip install numpy

# Para YOLOv5
pip install ultralytics torch torchvision
```

### Verificar Instalação

```python
import cv2
print(f"OpenCV version: {cv2.__version__}")

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

---

## 🎯 Checklist de Conclusão

### Fundamentos
- [ ] Ler, exibir e salvar imagens
- [ ] Redimensionar e recortar
- [ ] Converter espaços de cor (RGB, Gray, HSV)
- [ ] Aplicar transformações (rotação, flip)

### Filtros
- [ ] Aplicar blur (Gaussian, Median, Bilateral)
- [ ] Threshold e binarização
- [ ] Detectar bordas (Canny, Sobel)
- [ ] Operações morfológicas

### Detecção
- [ ] Detectar faces com Haar Cascades
- [ ] Detectar olhos e outros objetos
- [ ] Implementar detecção em vídeo
- [ ] Usar YOLOv5 em imagens
- [ ] Usar YOLOv5 em vídeo

---

## 📖 Recursos Complementares

### Documentação
- [OpenCV Docs](https://docs.opencv.org/)
- [OpenCV Tutorials](https://docs.opencv.org/master/d9/df8/tutorial_root.html)
- [YOLOv5 Docs](https://docs.ultralytics.com/)

### Cursos
- [OpenCV Course - freeCodeCamp](https://www.youtube.com/watch?v=oXlwWbU8l2o)
- [PyImageSearch](https://www.pyimagesearch.com/)

### Datasets
- [COCO Dataset](https://cocodataset.org/)
- [ImageNet](https://www.image-net.org/)
- [Open Images](https://storage.googleapis.com/openimages/web/index.html)

---

## 🤝 Contribuindo

Encontrou um erro? Tem uma sugestão?

1. Fork o repositório
2. Crie branch
3. Commit mudanças
4. Push para branch
5. Abra Pull Request

---

## 📧 Contato

**Autor**: Carlos Henrique Bamberg Marques  
**Email**: rick.bamberg@gmail.com  
**GitHub**: [@RickBamberg](https://github.com/RickBamberg/)

---

## 📄 Licença

Este projeto está sob a licença MIT.

---

**💡 Dica**: OpenCV + YOLOv5 = poderoso toolkit para Visão Computacional!

*Desenvolvido como parte do curso "Especialista em IA" - Módulo EAI_06*
