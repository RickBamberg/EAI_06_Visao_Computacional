# AGENT_CONTEXT.md - OCR (Optical Character Recognition)

> **Propósito**: Contexto técnico dos notebooks de OCR  
> **Última atualização**: Janeiro 2026  
> **Tipo**: Seção educacional com 7 notebooks

## RESUMO EXECUTIVO

**Objetivo**: Dominar OCR do básico ao avançado  
**Notebooks**: 7 notebooks (Tesseract → PaddleOCR)  
**Engines**: Tesseract, EasyOCR, PaddleOCR  
**Aplicação**: CNH, RG, Notas Fiscais, Documentos  
**Diferencial**: Comparação de 3 engines + projeto completo

---

## ESTRUTURA DOS NOTEBOOKS

```
01_OCR_Basico_Tesseract
├─ Fundamentos Tesseract
├─ Múltiplos idiomas
├─ PSM e OEM modes
└─ Bounding boxes

02_Preprocessamento_Imagens
├─ Grayscale e threshold
├─ Noise removal
├─ Deskew (correção de inclinação)
└─ Pipeline completo

03_OCR_Completo_Extracao_Documentos
├─ Extração de CNH
├─ Extração de RG
├─ Extração de Nota Fiscal
└─ Regex para estruturar dados

04_EasyOCR_Comparacao
├─ Deep Learning OCR
├─ Comparação com Tesseract
└─ GPU acceleration

05_PaddleOCR_Avancado
├─ Estado-da-arte
├─ Layout detection
└─ Table recognition

06_Projeto_Pratico_Dados_Estruturados
├─ Sistema completo
├─ Múltiplas engines
├─ Batch processing
└─ JSON output
```

---

## NOTEBOOK 1: OCR_Basico_Tesseract

### Tesseract - Arquitetura

```
Input Image
    ↓
Adaptive Thresholding
    ↓
Connected Component Analysis
    ↓
Text Line Detection
    ↓
Word Recognition (LSTM Neural Network)
    ↓
Language Model
    ↓
Output Text
```

### PSM (Page Segmentation Modes) - Técnico

```python
# PSM modes completos:
0  = Orientation and script detection (OSD) only
1  = Automatic page segmentation with OSD
2  = Automatic page segmentation, no OSD, no OCR
3  = Fully automatic page segmentation (default)
4  = Single column of text
5  = Single uniform block of vertically aligned text
6  = Single uniform block of text (melhor para documentos)
7  = Single text line (CNH, placas)
8  = Single word
9  = Single word in circle
10 = Single character
11 = Sparse text (texto espalhado)
12 = Sparse text with OSD
13 = Raw line (bypass todas heurísticas)
```

**Quando usar**:
```python
# Documento completo (padrão)
config = '--psm 3'

# Linha única (CNH, RG)
config = '--psm 7'

# Apenas números (placa de carro)
config = '--psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

# Texto esparso (recibos)
config = '--psm 11'
```

### OEM (OCR Engine Modes)

```python
0 = Legacy engine only
1 = Neural nets LSTM engine only (melhor accuracy)
2 = Legacy + LSTM engines
3 = Default (escolhe automaticamente)
```

### Confidence Filtering - Otimização

```python
def extract_high_confidence_text(image, min_conf=60):
    """
    Extrai apenas texto com alta confiança
    """
    dados = pytesseract.image_to_data(image, output_type='dict', lang='por')
    
    texto_filtrado = []
    
    for i in range(len(dados['text'])):
        conf = float(dados['conf'][i])
        text = dados['text'][i]
        
        if conf > min_conf and text.strip():
            texto_filtrado.append(text)
    
    return ' '.join(texto_filtrado)
```

---

## NOTEBOOK 2: Preprocessamento

### Threshold - Matemática

```python
# Binary Threshold:
# dst(x,y) = maxval  se src(x,y) > thresh
#          = 0       caso contrário

# Adaptive Threshold (melhor para documentos):
# thresh(x,y) = mean(região(x,y)) - C
# Adapta threshold localmente!

# Otsu (automático):
# Encontra threshold ótimo minimizando variância intra-classe
```

### Deskew - Algoritmo Completo

```python
def deskew_advanced(image):
    """
    Correção de inclinação usando Hough Lines
    """
    # 1. Detectar bordas
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    
    # 2. Hough Lines (detectar linhas)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    
    if lines is None:
        return image
    
    # 3. Calcular ângulos
    angles = []
    for rho, theta in lines[:, 0]:
        angle = (theta * 180 / np.pi) - 90
        angles.append(angle)
    
    # 4. Mediana dos ângulos (mais robusto)
    median_angle = np.median(angles)
    
    # 5. Rotacionar
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),
                            flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_REPLICATE)
    
    return rotated
```

### Pipeline Otimizado

```python
def preprocess_pipeline(image, doc_type='document'):
    """
    Pipeline otimizado por tipo de documento
    """
    if doc_type == 'document':
        # Documentos limpos
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        return thresh
    
    elif doc_type == 'photo':
        # Fotos de documentos (celular)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        denoised = cv2.bilateralFilter(gray, 5, 50, 50)
        thresh = cv2.adaptiveThreshold(
            denoised, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        deskewed = deskew_advanced(thresh)
        return deskewed
    
    elif doc_type == 'old':
        # Documentos antigos/degradados
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        return thresh
```

---

## NOTEBOOK 3: Extração de Documentos

### Regex Patterns - Completo

```python
# Biblioteca de regex para documentos brasileiros
PATTERNS = {
    'cpf': r'\d{3}\.\d{3}\.\d{3}-\d{2}',
    'cpf_sem_formatacao': r'\d{11}',
    'cnpj': r'\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}',
    'rg': r'\d{1,2}\.\d{3}\.\d{3}-\d{1,2}',
    'cnh': r'\d{11}',
    'data': r'\d{2}/\d{2}/\d{4}',
    'cep': r'\d{5}-\d{3}',
    'telefone': r'\(\d{2}\)\s?\d{4,5}-\d{4}',
    'placa_antiga': r'[A-Z]{3}-\d{4}',
    'placa_mercosul': r'[A-Z]{3}\d[A-Z]\d{2}',
    'email': r'[\w\.-]+@[\w\.-]+\.\w+',
    'valor_monetario': r'R\$\s?[\d.,]+'
}

def extract_by_pattern(text, pattern_name):
    """
    Extrai usando padrão específico
    """
    pattern = PATTERNS.get(pattern_name)
    if pattern:
        matches = re.findall(pattern, text)
        return matches
    return []
```

### CNH - Template Matching

```python
def extract_cnh_structured(image_path):
    """
    Extração estruturada usando posições conhecidas
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # CNH tem regiões conhecidas
    regions = {
        'nome': (100, 150, 400, 50),      # x, y, w, h
        'cpf': (100, 210, 200, 30),
        'data_nasc': (350, 210, 150, 30),
        'numero_cnh': (100, 260, 200, 30)
    }
    
    dados = {}
    
    for campo, (x, y, w, h) in regions.items():
        roi = gray[y:y+h, x:x+w]
        texto = pytesseract.image_to_string(roi, config='--psm 7')
        dados[campo] = texto.strip()
    
    return dados
```

---

## NOTEBOOK 4: EasyOCR

### CRAFT + CRNN Architecture

```
Input Image
    ↓
CRAFT (Text Detection)
│   ├─ Backbone: VGG16
│   ├─ Region Score Map
│   ├─ Affinity Score Map
│   └─ Bounding Boxes
    ↓
CRNN (Text Recognition)
│   ├─ CNN Feature Extraction
│   ├─ RNN (BiLSTM) Sequence
│   ├─ CTC (Connectionist Temporal Classification)
│   └─ Text Output
    ↓
Final Text
```

### GPU Acceleration

```python
import torch

# Verificar GPU
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

# Criar reader com GPU
reader = easyocr.Reader(['pt', 'en'], gpu=True)

# Benchmark
import time

start = time.time()
result_gpu = reader.readtext('documento.png')
tempo_gpu = time.time() - start

reader_cpu = easyocr.Reader(['pt', 'en'], gpu=False)
start = time.time()
result_cpu = reader_cpu.readtext('documento.png')
tempo_cpu = time.time() - start

print(f"GPU: {tempo_gpu:.2f}s")
print(f"CPU: {tempo_cpu:.2f}s")
print(f"Speedup: {tempo_cpu/tempo_gpu:.2f}x")
```

---

## NOTEBOOK 5: PaddleOCR

### PP-OCR Architecture

```
Input Image
    ↓
Text Detection (DB - Differentiable Binarization)
│   ├─ Backbone: MobileNetV3
│   ├─ Head: DBHead
│   └─ Bounding Boxes
    ↓
Text Direction Classification
│   └─ Angle: 0°, 90°, 180°, 270°
    ↓
Text Recognition (CRNN)
│   ├─ Backbone: MobileNetV3
│   ├─ Neck: Sequence Encoder
│   ├─ Head: CTC
│   └─ Text
    ↓
Final Output
```

### Table Recognition - Detalhes

```python
from paddleocr import PPStructure

table_engine = PPStructure(table=True, show_log=False)

# Processar
result = table_engine('tabela.png')

# Resultado contém:
for element in result:
    if element['type'] == 'table':
        # Coordenadas
        bbox = element['bbox']
        
        # HTML da tabela
        html = element['res']['html']
        
        # Células
        cells = element['res']['cell_bbox']
        
        # Converter HTML para DataFrame
        import pandas as pd
        df = pd.read_html(html)[0]
        print(df)
```

---

## COMPARAÇÃO DE ENGINES - TÉCNICA

### Accuracy por Cenário

| Cenário | Tesseract | EasyOCR | PaddleOCR |
|---------|-----------|---------|-----------|
| **Texto limpo** | 90% | 91% | 92% |
| **Documento foto** | 75% | 88% | 92% |
| **Texto rotacionado** | 50% | 85% | 90% |
| **Múltiplas fontes** | 80% | 90% | 93% |
| **Texto manuscrito** | 40% | 60% | 65% |
| **Tabelas** | 60% | N/A | 85% |

### Performance (Tempo de Processamento)

```python
# Benchmark em documento padrão (A4, 300 DPI)

Tesseract (CPU):          0.5s
EasyOCR (CPU):            2.0s
EasyOCR (GPU):            0.3s
PaddleOCR (CPU):          1.0s
PaddleOCR (GPU):          0.2s
```

---

## PROJETO PRÁTICO - ARQUITETURA

### Sistema Completo

```python
class OCRSystem:
    """
    Sistema de produção
    """
    
    def __init__(self):
        self.engines = {
            'tesseract': TesseractEngine(),
            'easyocr': EasyOCREngine(),
            'paddleocr': PaddleOCREngine()
        }
        self.preprocessor = ImagePreprocessor()
    
    def process(self, image_path, strategy='best'):
        """
        Estratégias:
        - 'fast': Tesseract
        - 'accurate': PaddleOCR
        - 'best': Voting entre engines
        """
        img = self.preprocessor.process(image_path)
        
        if strategy == 'fast':
            return self.engines['tesseract'].extract(img)
        
        elif strategy == 'accurate':
            return self.engines['paddleocr'].extract(img)
        
        elif strategy == 'best':
            # Voting
            results = []
            for engine in self.engines.values():
                result = engine.extract(img)
                results.append(result)
            
            return self.vote(results)
    
    def vote(self, results):
        """
        Voting mechanism
        """
        from collections import Counter
        
        # Contar ocorrências de cada palavra
        all_words = []
        for result in results:
            words = result.split()
            all_words.extend(words)
        
        # Maioria vence
        counter = Counter(all_words)
        voted_text = ' '.join([word for word, _ in counter.most_common()])
        
        return voted_text
```

---

## TROUBLESHOOTING

### Problema 1: Tesseract não encontrado

```python
# Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Linux
# sudo apt-get install tesseract-ocr

# Verificar
import subprocess
subprocess.run(['tesseract', '--version'])
```

### Problema 2: Accuracy baixa

```python
# 1. Pré-processamento inadequado
# Solução: Testar pipeline diferente

# 2. Idioma errado
# Solução: Verificar idioma
texto = pytesseract.image_to_string(img, lang='por')  # Não 'pt'!

# 3. PSM incorreto
# Solução: Testar PSM 6 ou 7
config = '--psm 7'
```

### Problema 3: EasyOCR/PaddleOCR lento

```python
# Usar GPU
reader = easyocr.Reader(['pt'], gpu=True)

# Reduzir tamanho da imagem
img_resized = cv2.resize(img, (1200, 1600))

# Batch processing
results = reader.readtext_batched([img1, img2, img3])
```

---

## TAGS DE BUSCA

`#ocr` `#tesseract` `#easyocr` `#paddleocr` `#text-extraction` `#document-processing` `#cnh` `#rg` `#nota-fiscal` `#preprocessing` `#deep-learning`

---

**Versão**: 1.0  
**Compatibilidade**: Python 3.7+, Tesseract 4.0+  
**Uso recomendado**: Extração de texto de documentos, automação
