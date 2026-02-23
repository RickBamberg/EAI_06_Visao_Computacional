# AGENT_CONTEXT.md - Detector de Objetos Coloridos

> **Propósito**: Contexto técnico do projeto Flask de detecção de objetos por cor  
> **Última atualização**: Janeiro 2026  
> **Tipo**: Aplicação web Flask + OpenCV + Computer Vision

## RESUMO EXECUTIVO

**Objetivo**: Aplicação web para detecção em tempo real de objetos coloridos via webcam  
**Stack**: Flask + OpenCV + NumPy + HTML/CSS/JS  
**Algoritmo**: Segmentação HSV + Morphological Operations + Contour Detection  
**Features**: 6 cores, filtro de área, visualização de máscara, API REST  
**Dificuldade**: ⭐ Fácil  
**Tempo**: 2-3 horas

---

## ARQUITETURA TÉCNICA

### Pipeline de Detecção

```
┌─────────────────────────────────────────────────────────────┐
│                   WEBCAM CAPTURE                            │
├─────────────────────────────────────────────────────────────┤
│  cv2.VideoCapture(0) → Frame 640x480 BGR                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              COLOR SPACE CONVERSION                         │
├─────────────────────────────────────────────────────────────┤
│  BGR → HSV (Hue, Saturation, Value)                        │
│  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              COLOR SEGMENTATION                             │
├─────────────────────────────────────────────────────────────┤
│  Para cada cor:                                             │
│  ├─ Lower bound: [H_min, S_min, V_min]                     │
│  ├─ Upper bound: [H_max, S_max, V_max]                     │
│  └─ Mask = cv2.inRange(hsv, lower, upper)                  │
│                                                              │
│  Exemplo Vermelho (2 ranges):                               │
│  ├─ Range 1: [0, 120, 70] → [10, 255, 255]                 │
│  └─ Range 2: [170, 120, 70] → [180, 255, 255]              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│          MORPHOLOGICAL OPERATIONS                           │
├─────────────────────────────────────────────────────────────┤
│  Kernel 5x5:                                                │
│  ├─ Opening (erosion → dilation) - Remove ruído            │
│  └─ Closing (dilation → erosion) - Preenche buracos        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│             CONTOUR DETECTION                               │
├─────────────────────────────────────────────────────────────┤
│  cv2.findContours(mask, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)│
│  ├─ RETR_EXTERNAL: Apenas contornos externos               │
│  └─ CHAIN_APPROX_SIMPLE: Compressão de pontos              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│          FILTERING & ANNOTATION                             │
├─────────────────────────────────────────────────────────────┤
│  Para cada contorno:                                        │
│  ├─ Área > min_area? (default 500px)                       │
│  ├─ Bounding box: cv2.boundingRect(contour)                │
│  ├─ Centro: (x + w/2, y + h/2)                             │
│  ├─ Desenhar: retângulo + círculo + labels                 │
│  └─ Contador: objetos detectados                           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│           STREAMING (MJPEG)                                 │
├─────────────────────────────────────────────────────────────┤
│  cv2.imencode('.jpg', frame) → JPEG bytes                   │
│  yield b'--frame\r\n...frame_bytes...\r\n'                  │
│  Response: multipart/x-mixed-replace                        │
└─────────────────────────────────────────────────────────────┘
```

---

## ESPAÇO DE CORES HSV - MATEMÁTICA

### Por Que HSV em Vez de BGR?

```python
# BGR (Blue, Green, Red):
# - Canais correlacionados
# - Sensível a iluminação
# - Difícil segmentar cores

# Exemplo BGR:
# Vermelho claro: (100, 100, 255)
# Vermelho escuro: (0, 0, 128)
# → Ranges muito diferentes!

# HSV (Hue, Saturation, Value):
# - Hue (Matiz): 0-180° - A COR pura
# - Saturation (Saturação): 0-255 - Intensidade da cor
# - Value (Valor): 0-255 - Brilho

# Exemplo HSV:
# Vermelho claro: (0, 128, 255) - Alta saturação, alto brilho
# Vermelho escuro: (0, 255, 128) - Alta saturação, baixo brilho
# → Hue IGUAL (0°)! Fácil segmentar!
```

### Mapeamento de Cores HSV

```python
COLOR_RANGES = {
    'vermelho': [
        # Por que 2 ranges?
        # Vermelho está nas EXTREMIDADES do círculo HSV (0° e 180°)
        (np.array([0, 120, 70]), np.array([10, 255, 255])),    # 0-10°
        (np.array([170, 120, 70]), np.array([180, 255, 255]))  # 170-180°
    ],
    
    'verde': [
        # Verde: 60° no círculo HSV
        # Range: 40-80° (verde amarelado até verde azulado)
        (np.array([40, 40, 40]), np.array([80, 255, 255]))
    ],
    
    'azul': [
        # Azul: 120° no círculo HSV
        # Range: 100-140° (ciano até azul violeta)
        (np.array([100, 150, 0]), np.array([140, 255, 255]))
    ],
    
    'amarelo': [
        # Amarelo: 30° no círculo HSV
        # Range: 20-30° (amarelo puro)
        (np.array([20, 100, 100]), np.array([30, 255, 255]))
    ],
    
    'laranja': [
        # Laranja: 15° no círculo HSV
        # Range: 10-20° (entre vermelho e amarelo)
        (np.array([10, 100, 100]), np.array([20, 255, 255]))
    ],
    
    'roxo': [
        # Roxo/Magenta: 150° no círculo HSV
        # Range: 140-170° (roxo azulado até magenta)
        (np.array([140, 50, 50]), np.array([170, 255, 255]))
    ]
}
```

**Círculo HSV Visual**:
```
         0° = Vermelho
            |
    270° ---|--- 90°
   Magenta  |  Verde
            |
       180° = Ciano
       
Ordem completa:
0° → Vermelho
30° → Laranja
60° → Amarelo
90° → Verde
120° → Ciano
150° → Azul
180° → Roxo/Magenta
```

---

## OPERAÇÕES MORFOLÓGICAS - DETALHADO

### Kernel e Operações

```python
# Kernel: Matriz estruturante
kernel = np.ones((5, 5), np.uint8)

# Visualização do kernel 5x5:
# [1 1 1 1 1]
# [1 1 1 1 1]
# [1 1 1 1 1]
# [1 1 1 1 1]
# [1 1 1 1 1]
```

### Opening (Erosion → Dilation)

```python
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# Objetivo: REMOVER RUÍDO pequeno

# Passo 1: Erosion
# - Pixel mantido apenas se TODOS os vizinhos (kernel) são brancos
# - Encolhe objetos, remove pixels isolados

# Passo 2: Dilation
# - Pixel vira branco se ALGUM vizinho (kernel) é branco
# - Expande objetos de volta ao tamanho original

# Resultado: Ruído pequeno DESAPARECE, objetos grandes MANTÊM tamanho
```

**Exemplo Visual**:
```
Original Mask:
[0 0 0 0 0 0 0]
[0 1 1 1 0 1 0]  ← Objeto + ruído (pixel isolado)
[0 1 1 1 0 0 0]
[0 1 1 1 0 0 0]
[0 0 0 0 0 0 0]

Após Erosion (kernel 3x3):
[0 0 0 0 0 0 0]
[0 0 1 0 0 0 0]  ← Ruído removido, objeto encolhido
[0 0 1 0 0 0 0]
[0 0 0 0 0 0 0]

Após Dilation:
[0 0 0 0 0 0 0]
[0 1 1 1 0 0 0]  ← Objeto restaurado, ruído SUMIU!
[0 1 1 1 0 0 0]
[0 1 1 1 0 0 0]
[0 0 0 0 0 0 0]
```

### Closing (Dilation → Erosion)

```python
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# Objetivo: PREENCHER BURACOS pequenos dentro de objetos

# Passo 1: Dilation
# - Expande objetos, preenche pequenos buracos

# Passo 2: Erosion
# - Encolhe de volta, mantendo buracos preenchidos

# Resultado: Buracos pequenos DESAPARECEM, forma externa MANTÉM
```

---

## DETECÇÃO DE CONTORNOS - CÓDIGO EXPLICADO

```python
def detect_colored_objects(frame, color_name):
    # 1. BGR → HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # 2. Criar máscara (inicialmente vazia)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    
    # 3. Aplicar ranges da cor
    if color_name in COLOR_RANGES:
        for (lower, upper) in COLOR_RANGES[color_name]:
            # inRange: 255 se pixel está no range, 0 caso contrário
            mask_temp = cv2.inRange(hsv, lower, upper)
            
            # bitwise_or: Combina múltiplos ranges (caso vermelho)
            mask = cv2.bitwise_or(mask, mask_temp)
    
    # 4. Operações morfológicas
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # Remove ruído
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Preenche buracos
    
    # 5. Encontrar contornos
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,      # Apenas contornos externos
        cv2.CHAIN_APPROX_SIMPLE # Comprime pontos (economiza memória)
    )
    
    # 6. Processar cada contorno
    output = frame.copy()
    object_count = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Filtrar por área mínima (ruído residual)
        if area > min_area:  # default: 500px
            object_count += 1
            
            # Bounding box (menor retângulo que engloba contorno)
            x, y, w, h = cv2.boundingRect(contour)
            
            # Desenhar retângulo verde
            cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Centro do objeto
            cx, cy = x + w//2, y + h//2
            cv2.circle(output, (cx, cy), 5, (0, 0, 255), -1)
            
            # Anotações
            label = f"{color_name.upper()} #{object_count}"
            cv2.putText(output, label, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            area_text = f"Area: {int(area)}"
            cv2.putText(output, area_text, (x, y+h+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # 7. Contador total
    info_text = f"Objetos {color_name}: {object_count}"
    cv2.putText(output, info_text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return output, object_count
```

### findContours - Parâmetros

```python
contours, hierarchy = cv2.findContours(mask, mode, method)

# MODOS (mode):
# - RETR_EXTERNAL: Apenas contornos EXTERNOS
#   Exemplo: Donut → 1 contorno (círculo externo)
#   
# - RETR_LIST: TODOS os contornos (sem hierarquia)
#   Exemplo: Donut → 2 contornos (externo + buraco)
#   
# - RETR_TREE: Hierarquia completa (pai-filho)
#   Exemplo: Donut → externo (pai), buraco (filho)

# MÉTODOS (method):
# - CHAIN_APPROX_NONE: TODOS os pontos do contorno
#   Exemplo: Quadrado 100x100 → 400 pontos (1 por pixel)
#   
# - CHAIN_APPROX_SIMPLE: Apenas pontos ESSENCIAIS
#   Exemplo: Quadrado 100x100 → 4 pontos (vértices!)
#   Economia: 100x menos memória!
```

---

## FLASK API - ENDPOINTS

### 1. Video Streaming

```python
@app.route('/video_feed')
def video_feed():
    """
    Stream MJPEG (Motion JPEG)
    """
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

def generate_frames():
    """
    Generator que yielda frames em loop
    """
    cam = get_camera()
    
    while True:
        success, frame = cam.read()
        if not success:
            break
        
        # Processar frame
        processed_frame, count = detect_colored_objects(frame, current_color)
        
        # Encodar como JPEG
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()
        
        # Yield no formato MJPEG
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + 
               frame_bytes + 
               b'\r\n')
```

**Formato MJPEG**:
```
--frame
Content-Type: image/jpeg

[JPEG BYTES DO FRAME 1]

--frame
Content-Type: image/jpeg

[JPEG BYTES DO FRAME 2]

--frame
...
```

### 2. Set Color

```python
@app.route('/set_color', methods=['POST'])
def set_color():
    """
    Altera cor detectada dinamicamente
    """
    global current_color
    data = request.json
    color = data.get('color', 'vermelho')
    
    if color in COLOR_RANGES:
        current_color = color
        return jsonify({'success': True, 'color': current_color})
    
    return jsonify({'success': False, 'message': 'Cor inválida'})

# Request:
# POST /set_color
# Body: {"color": "azul"}
#
# Response:
# {"success": true, "color": "azul"}
```

### 3. Set Min Area

```python
@app.route('/set_min_area', methods=['POST'])
def set_min_area():
    """
    Ajusta filtro de área mínima
    """
    global min_area
    data = request.json
    area = data.get('area', 500)
    
    try:
        min_area = int(area)
        return jsonify({'success': True, 'min_area': min_area})
    except ValueError:
        return jsonify({'success': False, 'message': 'Área inválida'})

# Exemplo:
# min_area = 500 → Objetos < 500px ignorados
# min_area = 100 → Detecta objetos menores
# min_area = 2000 → Apenas objetos grandes
```

### 4. Toggle Mask

```python
@app.route('/toggle_mask', methods=['POST'])
def toggle_mask():
    """
    Liga/desliga visualização da máscara HSV
    """
    global show_mask
    show_mask = not show_mask
    return jsonify({'success': True, 'show_mask': show_mask})

# Quando show_mask = True:
# Frame exibido: [Video Original | Máscara HSV]
# Side-by-side usando np.hstack([output, mask_colored])
```

---

## THREADING E CAMERA LOCK

### Por Que Lock?

```python
camera = None
camera_lock = Lock()

def get_camera():
    global camera
    with camera_lock:  # CRITICAL SECTION
        if camera is None or not camera.isOpened():
            camera = cv2.VideoCapture(0)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return camera

# Problema SEM lock:
# - Thread 1: Verifica camera == None → True
# - Thread 2: Verifica camera == None → True (ainda!)
# - Thread 1: Cria VideoCapture(0)
# - Thread 2: Cria VideoCapture(0) ← CONFLITO!
# Resultado: 2 instâncias da mesma câmera → ERRO

# Solução COM lock:
# - Thread 1: Adquire lock, cria câmera
# - Thread 2: ESPERA lock ser liberado
# - Thread 1: Libera lock
# - Thread 2: Adquire lock, vê que camera != None, usa existente
# Resultado: 1 instância compartilhada → OK
```

---

## PERFORMANCE E OTIMIZAÇÕES

### Resolução da Câmera

```python
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Trade-offs:
# 320x240: ~60 FPS, detecção menos precisa
# 640x480: ~30 FPS, boa detecção (PADRÃO)
# 1280x720: ~15 FPS, detecção muito precisa
# 1920x1080: ~5 FPS, overkill
```

### Kernel Size

```python
kernel = np.ones((5, 5), np.uint8)

# Kernel 3x3: Mais rápido, menos filtragem
# Kernel 5x5: Balanço (PADRÃO)
# Kernel 7x7: Mais lento, mais filtragem
# Kernel 11x11: Muito lento, remove objetos pequenos
```

### Processar a Cada N Frames

```python
# Otimização: Processar a cada 3 frames
frame_count = 0

def generate_frames():
    global frame_count
    
    while True:
        frame_count += 1
        
        if frame_count % 3 == 0:  # A cada 3 frames
            processed = detect_colored_objects(frame, current_color)
        else:
            processed = frame  # Frame sem processamento
        
        yield processed

# FPS: 30 → 90 (3x mais rápido!)
# Trade-off: Detecção menos responsiva
```

---

## APLICAÇÕES PRÁTICAS

### 1. Reciclagem Automática

```python
# Detectar tampas de garrafa por cor:
# - Azul → PET
# - Verde → Vidro
# - Amarelo → Metal
# - Vermelho → Plástico especial

# Adicionar atuador (servo motor):
if detected_color == 'azul':
    GPIO.output(PET_BIN, HIGH)  # Abrir caixa PET
```

### 2. Rastreamento de Bola (Futebol)

```python
# Detectar bola branca/amarela
# Calcular trajetória:

previous_center = None

for contour in contours:
    current_center = (cx, cy)
    
    if previous_center:
        # Calcular velocidade
        dx = current_center[0] - previous_center[0]
        dy = current_center[1] - previous_center[1]
        velocity = np.sqrt(dx**2 + dy**2)
        
        # Desenhar trajetória
        cv2.line(frame, previous_center, current_center, (0, 255, 0), 2)
    
    previous_center = current_center
```

### 3. Controle de Qualidade Industrial

```python
# Detectar peças defeituosas (cor diferente):
expected_color = 'verde'  # Peça OK
detected_color = get_dominant_color(contour)

if detected_color != expected_color:
    trigger_alarm()  # Peça defeituosa!
    log_defect(timestamp, image, detected_color)
```

---

## TROUBLESHOOTING

### Problema 1: Câmera Não Funciona

```python
# Erro: VideoCapture não abre

# Solução 1: Tentar índices diferentes
camera = cv2.VideoCapture(0)  # Tente 1, 2, ...

# Solução 2: Backend específico (Linux)
camera = cv2.VideoCapture(0, cv2.CAP_V4L2)

# Solução 3: Verificar permissões (Linux)
# sudo chmod 666 /dev/video0
```

### Problema 2: Detecção Imprecisa

```python
# Muito ruído detectado:
# → Aumentar min_area
min_area = 1000  # Era 500

# → Kernel maior
kernel = np.ones((7, 7), np.uint8)

# Não detecta nada:
# → Calibrar ranges HSV
# Use HSV color picker online
# Ajuste lower/upper bounds
```

### Problema 3: Performance Lenta

```python
# FPS muito baixo:

# 1. Reduzir resolução
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# 2. Processar a cada N frames
if frame_count % 3 == 0:
    process_frame()

# 3. Usar GPU (se disponível)
# Requer: opencv-contrib-python com CUDA
```

---

## TAGS DE BUSCA

`#opencv` `#flask` `#color-detection` `#hsv` `#computer-vision` `#real-time` `#webcam` `#contours` `#morphology`

---

**Versão**: 1.0  
**Compatibilidade**: Python 3.8+, OpenCV 4.5+, Flask 2.0+  
**Uso recomendado**: Educação, prototipagem, projetos pessoais
