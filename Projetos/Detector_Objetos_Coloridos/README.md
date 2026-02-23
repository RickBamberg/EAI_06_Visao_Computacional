# 🎨 Detector de Objetos Coloridos

Sistema de **detecção de objetos por cor em tempo real** usando OpenCV e Flask. Identifica e rastreia objetos de cores específicas através da webcam com interface web interativa.

---

## 🎯 Objetivo

Aplicação web que detecta objetos coloridos em tempo real via webcam usando:
- ✅ Segmentação de cores no espaço HSV
- ✅ Operações morfológicas (noise removal)
- ✅ Detecção de contornos
- ✅ Interface web responsiva (Flask)
- ✅ API REST para configuração dinâmica

**Nível**: ⭐ Fácil  
**Tempo**: 2-3 horas  
**Aplicações**: Reciclagem, robótica, tracking esportivo, QA industrial

---

## 🏗️ Arquitetura

```
┌─────────────────────────────────────────────┐
│           WEBCAM (640x480)                  │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│      BGR → HSV Conversion                   │
│      cv2.cvtColor(COLOR_BGR2HSV)           │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│      Color Segmentation                     │
│      cv2.inRange(hsv, lower, upper)        │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│   Morphological Operations (5x5 kernel)     │
│   ├─ Opening (remove noise)                │
│   └─ Closing (fill holes)                  │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│      Contour Detection                      │
│      cv2.findContours(RETR_EXTERNAL)       │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│   Filter (area > 500px) + Annotation       │
│   ├─ Bounding box                          │
│   ├─ Center point                          │
│   └─ Label + counter                       │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│         MJPEG Streaming (Flask)             │
│         multipart/x-mixed-replace           │
└─────────────────────────────────────────────┘
```

---

## 📂 Estrutura do Projeto

```
Detector_Objetos_Coloridos/
├── app.py                    # Aplicação Flask principal
├── test_opencv.py            # Script de teste/validação
├── requirements.txt          # Dependências Python
├── README.md                 # Este arquivo
├── static/
│   ├── css/
│   │   └── style.css        # Estilos customizados
│   └── js/
│       └── main.js          # Lógica JavaScript
└── templates/
    └── index.html           # Interface web
```

---

## 🚀 Como Usar

### 1. Instalação

```bash
# Clone ou baixe o projeto
git clone <repositorio>
cd Detector_Objetos_Coloridos

# Crie ambiente virtual (recomendado)
python -m venv venv

# Ative o ambiente
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Instale dependências
pip install -r requirements.txt
```

**requirements.txt**:
```
flask==2.3.0
opencv-python==4.8.0
numpy==1.24.3
```

### 2. Testar Instalação

```bash
# Verificar OpenCV e câmera
python test_opencv.py
```

**Output esperado**:
```
🔍 TESTE DE INSTALAÇÃO DO OPENCV
==================================================
✅ OpenCV versão: 4.8.0
✅ Imagem criada com sucesso
✅ Conversão realizada com sucesso
✅ 1 contorno(s) detectado(s)
✅ Câmera acessada com sucesso
✅ Frame capturado: (480, 640, 3)
==================================================
```

### 3. Executar Aplicação

```bash
python app.py
```

**Output**:
```
 * Running on http://0.0.0.0:5000
 * Serving Flask app 'app'
```

### 4. Acessar Interface

Abra o navegador em: **http://localhost:5000**

---

## 🎨 Funcionalidades

### Interface Web

1. **Seleção de Cor**
   - 6 cores disponíveis: Vermelho, Verde, Azul, Amarelo, Laranja, Roxo
   - Troca instantânea sem reiniciar

2. **Filtro de Área Mínima**
   - Slider: 100px - 5000px
   - Filtra ruído e objetos pequenos
   - Default: 500px

3. **Visualização de Máscara**
   - Toggle: ON/OFF
   - Mostra máscara HSV ao lado do vídeo
   - Útil para debug e calibração

4. **Streaming em Tempo Real**
   - MJPEG stream (~30 FPS)
   - Anotações: bounding box, centro, label, área
   - Contador de objetos detectados

---

## 🔧 Configuração de Cores (HSV)

### Cores Pré-configuradas

```python
COLOR_RANGES = {
    'vermelho': [
        ([0, 120, 70], [10, 255, 255]),      # 0-10° (vermelho puro)
        ([170, 120, 70], [180, 255, 255])    # 170-180° (magenta)
    ],
    'verde': [([40, 40, 40], [80, 255, 255])],     # 40-80°
    'azul': [([100, 150, 0], [140, 255, 255])],    # 100-140°
    'amarelo': [([20, 100, 100], [30, 255, 255])], # 20-30°
    'laranja': [([10, 100, 100], [20, 255, 255])], # 10-20°
    'roxo': [([140, 50, 50], [170, 255, 255])]     # 140-170°
}
```

### Adicionar Nova Cor

```python
# 1. Edite app.py
COLOR_RANGES['rosa'] = [
    (np.array([145, 50, 50]), np.array([165, 255, 255]))
]

# 2. Adicione botão em templates/index.html
# <button onclick="setColor('rosa')">Rosa</button>
```

### Calibrar Cores (Iluminação Diferente)

Use ferramenta online HSV color picker ou crie script de calibração:

```python
import cv2
import numpy as np

def calibrate_color():
    cap = cv2.VideoCapture(0)
    
    while True:
        _, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Clique na imagem para ver valores HSV
        cv2.setMouseCallback('frame', lambda event, x, y, flags, param: 
            print(f"HSV: {hsv[y, x]}") if event == cv2.EVENT_LBUTTONDOWN else None
        )
        
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

calibrate_color()
```

---

## 📡 API REST

### Endpoints

#### POST /set_color
Altera cor detectada.

**Request**:
```json
{
  "color": "azul"
}
```

**Response**:
```json
{
  "success": true,
  "color": "azul"
}
```

#### POST /set_min_area
Define área mínima (pixels).

**Request**:
```json
{
  "area": 1000
}
```

**Response**:
```json
{
  "success": true,
  "min_area": 1000
}
```

#### POST /toggle_mask
Liga/desliga visualização da máscara.

**Response**:
```json
{
  "success": true,
  "show_mask": true
}
```

#### GET /get_status
Retorna configuração atual.

**Response**:
```json
{
  "current_color": "vermelho",
  "min_area": 500,
  "show_mask": false,
  "available_colors": ["vermelho", "verde", "azul", "amarelo", "laranja", "roxo"]
}
```

### Exemplo cURL

```bash
# Mudar para verde
curl -X POST http://localhost:5000/set_color \
  -H "Content-Type: application/json" \
  -d '{"color":"verde"}'

# Ajustar área mínima
curl -X POST http://localhost:5000/set_min_area \
  -H "Content-Type: application/json" \
  -d '{"area":1000}'

# Obter status
curl http://localhost:5000/get_status
```

---

## 💡 Aplicações Práticas

### 🏭 Industrial

**Reciclagem Automática**
```python
# Sistema de separação por cor
# Tampas azuis → PET
# Tampas verdes → Vidro
# Tampas amarelas → Metal

if detected_color == 'azul' and area > 1000:
    activate_servo(PET_BIN)  # Abrir caixa PET
```

**Controle de Qualidade**
```python
# Detectar peças defeituosas (cor errada)
expected_color = 'verde'

if detected_color != expected_color:
    log_defect(timestamp, image)
    trigger_alarm()
```

### ⚽ Esportes

**Tracking de Bola**
```python
# Rastrear bola de futebol (branca/amarela)
# Calcular trajetória e velocidade

trajectory = []
for center in detected_centers:
    trajectory.append(center)
    
    if len(trajectory) > 1:
        velocity = calculate_velocity(trajectory[-2], trajectory[-1])
        draw_trajectory(frame, trajectory)
```

### 🤖 Robótica

**Navegação por Cor**
```python
# Seguir objeto vermelho
if detected_color == 'vermelho':
    cx, cy = center
    
    # Centro da imagem
    center_x = frame_width // 2
    
    # Ajustar motores
    if cx < center_x - 50:
        turn_left()
    elif cx > center_x + 50:
        turn_right()
    else:
        move_forward()
```

---

## 🐛 Troubleshooting

### Problema 1: Câmera Não Abre

**Erro**: `VideoCapture not opened`

**Soluções**:
```python
# 1. Tentar índice diferente
camera = cv2.VideoCapture(1)  # Tente 0, 1, 2...

# 2. Backend específico (Linux)
camera = cv2.VideoCapture(0, cv2.CAP_V4L2)

# 3. Verificar permissões (Linux)
# Terminal:
sudo usermod -a -G video $USER
sudo chmod 666 /dev/video0
```

### Problema 2: Detecção Imprecisa

**Muito ruído detectado**:
```python
# Aumentar área mínima
min_area = 1000  # Era 500

# Kernel maior (mais filtragem)
kernel = np.ones((7, 7), np.uint8)

# Valores S e V mais restritivos
'verde': [(np.array([40, 80, 80]), np.array([80, 255, 255]))]
#                        ^    ^
#                    S_min V_min aumentados
```

**Não detecta nada**:
```python
# Reduzir restrições S e V
'azul': [(np.array([100, 50, 50]), np.array([140, 255, 255]))]
#                        ^    ^
#                    S_min V_min reduzidos

# Ampliar range H
'verde': [(np.array([35, 40, 40]), np.array([85, 255, 255]))]
#                    ^                        ^
#                 H expandido (40→35) e (80→85)
```

### Problema 3: Performance Lenta

**FPS baixo**:
```python
# 1. Reduzir resolução
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# 2. Processar a cada N frames
frame_count = 0

def generate_frames():
    global frame_count
    while True:
        frame_count += 1
        if frame_count % 3 == 0:  # Processar 1 a cada 3
            process_frame()

# 3. Kernel menor
kernel = np.ones((3, 3), np.uint8)  # Era 5x5
```

---

## 📈 Melhorias Futuras

- [ ] **Persistência**: Salvar configurações em JSON
- [ ] **Export**: Dados de detecção para CSV/Excel
- [ ] **Calibração Auto**: Auto-ajuste de ranges HSV
- [ ] **Multi-color**: Detectar múltiplas cores simultaneamente
- [ ] **Gravação**: Salvar vídeo com anotações
- [ ] **Trajetória**: Análise de movimento (velocidade, direção)
- [ ] **Dashboard**: Gráficos em tempo real (Chart.js)
- [ ] **Notificações**: Alertas via webhook/email
- [ ] **ML Integration**: Classificação de objetos (YOLO)

---

## 📊 Performance

### Benchmarks (Intel i5, Webcam 720p)

| Resolução | FPS | Latência | CPU |
|-----------|-----|----------|-----|
| 320x240 | 60 | ~16ms | 15% |
| 640x480 | 30 | ~33ms | 30% |
| 1280x720 | 15 | ~66ms | 50% |

**Otimizações Aplicadas**:
- Threading para câmera (Lock)
- MJPEG streaming (baixo overhead)
- Morphological ops otimizadas (kernel 5x5)
- Filtro de área (evita processar ruído)

---

## 📖 Recursos

### Documentação
- [OpenCV Color Segmentation](https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html)
- [Flask Streaming](https://flask.palletsprojects.com/en/2.3.x/patterns/streaming/)
- [Morphological Operations](https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html)

### Ferramentas
- [HSV Color Picker](https://www.cssportal.com/hsv-color-picker/)
- [Contour Detection Tutorial](https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html)

---

## 📧 Contato

**Autor**: Carlos Henrique Bamberg Marques  
**Email**: rick.bamberg@gmail.com  
**GitHub**: [@RickBamberg](https://github.com/RickBamberg/)

---

## 📄 Licença

MIT License - Livre para uso educacional e comercial.

---

**💡 Dica**: Para melhores resultados, use objetos com cores sólidas e evite iluminação muito forte ou muito fraca!

*Projeto do curso "Especialista em IA" - Módulo EAI_06*
