# 📄 OCR - Optical Character Recognition

Curso completo de **OCR (Reconhecimento Óptico de Caracteres)** do básico ao avançado. Aprenda a extrair texto de imagens, documentos digitalizados e fotos usando Tesseract, EasyOCR e PaddleOCR.

---

## 🎯 Objetivo

Dominar técnicas de OCR para extração automatizada de texto:
- ✅ OCR básico com Tesseract
- ✅ Pré-processamento de imagens para OCR
- ✅ Extração de documentos (RG, CNH, notas fiscais)
- ✅ Comparação de engines (Tesseract, EasyOCR, PaddleOCR)
- ✅ Projeto prático com dados estruturados

**Por que OCR?**
- Digitalizar documentos físicos
- Automatizar entrada de dados
- Processar notas fiscais, recibos
- Extrair dados de CNH, RG, passaportes
- Tornar imagens pesquisáveis

---

## 📂 Estrutura dos Notebooks

```
OCR/
├── 01_OCR_Basico_Tesseract.ipynb                # Fundamentos
├── 02_Preprocessamento_Imagens.ipynb            # Melhorar qualidade
├── 02_Preprocessamento_Imagens_v2.ipynb         # Versão atualizada
├── 03_OCR_Completo_Extracao_Documentos.ipynb    # Docs reais
├── 04_EasyOCR_Comparacao.ipynb                  # EasyOCR
├── 05_PaddleOCR_Avancado.ipynb                  # PaddleOCR
└── 06_Projeto_Pratico_Dados_Estruturados.ipynb  # Projeto final
```

**Total**: 7 notebooks progressivos

---

## 🗺️ Jornada de Aprendizado

### Progressão Recomendada

```
Semana 1: Fundamentos
├── Dia 1-2: 01_OCR_Basico_Tesseract
│   └─ Tesseract, idiomas, qualidade
│
└── Dia 3-5: 02_Preprocessamento_Imagens
    └─ Threshold, resize, rotate, denoise

Semana 2: Aplicações
├── Dia 1-3: 03_OCR_Completo_Extracao_Documentos
│   └─ RG, CNH, notas fiscais
│
├── Dia 4: 04_EasyOCR_Comparacao
│   └─ Deep learning OCR
│
└── Dia 5: 05_PaddleOCR_Avancado
    └─ Estado-da-arte

Semana 3: Projeto
└── Dia 1-7: 06_Projeto_Pratico
    └─ Sistema completo de extração
```

**Tempo estimado**: 3 semanas

---

## 📚 Conteúdo Detalhado

### 1️⃣ 01_OCR_Basico_Tesseract.ipynb

**Objetivo**: Fundamentos de OCR com Tesseract

#### Tesseract - Engine Open Source

**O que é?**
- Engine de OCR desenvolvida pelo Google
- Suporta 100+ idiomas
- Open source e gratuito
- Accuracy: 85-95% (texto limpo)

#### Instalação

```bash
# Windows
# Baixar: https://github.com/UB-Mannheim/tesseract/wiki
# Instalar e adicionar ao PATH

# Linux
sudo apt-get install tesseract-ocr

# Mac
brew install tesseract

# Python
pip install pytesseract pillow opencv-python
```

#### OCR Básico

```python
import pytesseract
from PIL import Image

# Configurar caminho (Windows)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Ler imagem
img = Image.open('documento.png')

# Extrair texto
texto = pytesseract.image_to_string(img, lang='por')

print(texto)
```

**Output**:
```
Este é um documento de exemplo.
A qualidade do OCR depende da
imagem de entrada.
```

#### Múltiplos Idiomas

```python
# Português
texto_pt = pytesseract.image_to_string(img, lang='por')

# Inglês
texto_en = pytesseract.image_to_string(img, lang='eng')

# Múltiplos idiomas
texto_multi = pytesseract.image_to_string(img, lang='por+eng')

# Idiomas disponíveis
idiomas = pytesseract.get_languages()
print(idiomas)
# ['eng', 'por', 'spa', 'fra', 'deu', ...]
```

#### Informações Detalhadas

```python
# Obter dados estruturados
dados = pytesseract.image_to_data(img, output_type='dict', lang='por')

# Campos retornados:
# - level: nível hierárquico
# - page_num: número da página
# - block_num, par_num, line_num, word_num: hierarquia
# - left, top, width, height: coordenadas
# - conf: confiança (0-100)
# - text: texto extraído

# Filtrar por confiança
for i, texto in enumerate(dados['text']):
    conf = float(dados['conf'][i])
    if conf > 60:  # Confiança > 60%
        print(f"{texto} ({conf:.2f}%)")
```

#### Visualizar Bounding Boxes

```python
import cv2

# Ler imagem
img = cv2.imread('documento.png')
dados = pytesseract.image_to_data(img, output_type='dict', lang='por')

# Desenhar boxes
for i in range(len(dados['text'])):
    if float(dados['conf'][i]) > 60:
        x, y, w, h = dados['left'][i], dados['top'][i], dados['width'][i], dados['height'][i]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, dados['text'][i], (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

cv2.imshow('OCR Result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

#### Configurações de OCR (PSM e OEM)

```python
# PSM (Page Segmentation Mode)
# --psm 0: Orientação e script detection
# --psm 3: Automático (padrão)
# --psm 6: Bloco uniforme de texto
# --psm 7: Linha única de texto
# --psm 8: Palavra única
# --psm 11: Texto esparso

# OEM (OCR Engine Mode)
# --oem 0: Legacy engine
# --oem 1: Neural nets LSTM engine
# --oem 2: Legacy + LSTM
# --oem 3: Padrão (baseado no disponível)

# Exemplo: Linha única
custom_config = r'--oem 3 --psm 7'
texto = pytesseract.image_to_string(img, config=custom_config, lang='por')

# Exemplo: Apenas dígitos
custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'
numeros = pytesseract.image_to_string(img, config=custom_config)
```

---

### 2️⃣ 02_Preprocessamento_Imagens.ipynb

**Objetivo**: Melhorar qualidade da imagem antes do OCR

#### Por Que Pré-processar?

```
Imagem Ruim → OCR → Texto Ruim (60% accuracy)
     ↓
Imagem Ruim → Pré-processamento → Imagem Boa → OCR → Texto Bom (90% accuracy)
```

#### Técnicas Essenciais

##### 1. Grayscale (Tons de Cinza)

```python
import cv2

img = cv2.imread('documento.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Por que?
# - Reduz dados (3 canais → 1)
# - Simplifica processamento
# - Tesseract funciona melhor em grayscale
```

##### 2. Threshold (Binarização)

```python
# Threshold simples
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Threshold adaptativo (melhor para iluminação variável)
thresh_adaptive = cv2.adaptiveThreshold(
    gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11, 2
)

# Otsu's threshold (automático)
_, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
```

**Comparação**:
```
Original → Texto difícil de ler
Threshold Simples → Bom para iluminação uniforme
Threshold Adaptativo → Melhor para documentos reais ✓
Otsu → Automático, boa escolha geral
```

##### 3. Noise Removal (Remoção de Ruído)

```python
# Median Blur (remove "sal e pimenta")
denoised = cv2.medianBlur(gray, 3)

# Gaussian Blur (suaviza)
denoised = cv2.GaussianBlur(gray, (3, 3), 0)

# Bilateral Filter (preserva bordas)
denoised = cv2.bilateralFilter(gray, 9, 75, 75)

# Morphological operations
kernel = np.ones((1, 1), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
```

##### 4. Deskew (Correção de Inclinação)

```python
def deskew(image):
    """
    Corrige inclinação da imagem
    """
    # Detectar ângulo
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    
    # Rotacionar
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),
                            flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_REPLICATE)
    
    return rotated

# Usar
img_deskewed = deskew(gray)
```

##### 5. Border Removal (Remover Bordas)

```python
def remove_borders(image):
    """
    Remove bordas/margens
    """
    # Encontrar contornos
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Encontrar maior contorno (documento)
    largest = max(contours, key=cv2.contourArea)
    
    # Bounding box
    x, y, w, h = cv2.boundingRect(largest)
    
    # Crop
    cropped = image[y:y+h, x:x+w]
    
    return cropped
```

##### 6. Resize (Redimensionamento)

```python
# Tesseract funciona melhor com DPI 300
# Se imagem muito pequena, aumentar

def resize_for_ocr(image, target_height=1000):
    """
    Redimensiona mantendo proporção
    """
    h, w = image.shape[:2]
    if h < target_height:
        scale = target_height / h
        new_w = int(w * scale)
        resized = cv2.resize(image, (new_w, target_height), interpolation=cv2.INTER_CUBIC)
        return resized
    return image
```

#### Pipeline Completo de Pré-processamento

```python
def preprocess_for_ocr(image_path):
    """
    Pipeline completo
    """
    # 1. Ler
    img = cv2.imread(image_path)
    
    # 2. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 3. Resize
    gray = resize_for_ocr(gray)
    
    # 4. Denoise
    denoised = cv2.bilateralFilter(gray, 5, 50, 50)
    
    # 5. Threshold
    thresh = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    
    # 6. Deskew
    deskewed = deskew(thresh)
    
    # 7. Morphology
    kernel = np.ones((1, 1), np.uint8)
    final = cv2.morphologyEx(deskewed, cv2.MORPH_CLOSE, kernel)
    
    return final

# Usar
img_processada = preprocess_for_ocr('documento.png')

# OCR na imagem processada
texto = pytesseract.image_to_string(img_processada, lang='por')
```

---

### 3️⃣ 03_OCR_Completo_Extracao_Documentos.ipynb

**Objetivo**: Extrair informações de documentos reais (RG, CNH, notas)

#### Extração de CNH (Carteira de Motorista)

```python
def extrair_dados_cnh(image_path):
    """
    Extrai: Nome, CPF, RG, Data Nascimento, CNH
    """
    # Pré-processar
    img = preprocess_for_ocr(image_path)
    
    # OCR
    texto = pytesseract.image_to_string(img, lang='por')
    
    # Regex para extrair dados
    import re
    
    dados = {}
    
    # Nome (primeira linha após "NOME")
    match_nome = re.search(r'NOME[:\s]*([A-Z\s]+)', texto)
    if match_nome:
        dados['nome'] = match_nome.group(1).strip()
    
    # CPF (formato XXX.XXX.XXX-XX)
    match_cpf = re.search(r'(\d{3}\.\d{3}\.\d{3}-\d{2})', texto)
    if match_cpf:
        dados['cpf'] = match_cpf.group(1)
    
    # RG
    match_rg = re.search(r'RG[:\s]*(\d+[-\d]*)', texto)
    if match_rg:
        dados['rg'] = match_rg.group(1)
    
    # Data de Nascimento (DD/MM/AAAA)
    match_data = re.search(r'(\d{2}/\d{2}/\d{4})', texto)
    if match_data:
        dados['data_nascimento'] = match_data.group(1)
    
    # Número CNH
    match_cnh = re.search(r'CNH[:\s]*(\d+)', texto)
    if match_cnh:
        dados['numero_cnh'] = match_cnh.group(1)
    
    return dados

# Usar
dados = extrair_dados_cnh('cnh.jpg')
print(dados)
# {
#     'nome': 'JOÃO DA SILVA',
#     'cpf': '123.456.789-00',
#     'rg': '12.345.678-9',
#     'data_nascimento': '01/01/1990',
#     'numero_cnh': '12345678900'
# }
```

#### Extração de RG

```python
def extrair_dados_rg(image_path):
    """
    Extrai: Nome, RG, CPF, Data Nascimento, Filiação
    """
    img = preprocess_for_ocr(image_path)
    texto = pytesseract.image_to_string(img, lang='por')
    
    dados = {}
    
    # Nome
    match = re.search(r'NOME[:\s]*([A-Z\s]+)', texto)
    if match:
        dados['nome'] = match.group(1).strip()
    
    # Número RG
    match = re.search(r'N[°\s]+([0-9.-]+)', texto)
    if match:
        dados['rg'] = match.group(1)
    
    # CPF
    match = re.search(r'CPF[:\s]*(\d{3}\.\d{3}\.\d{3}-\d{2})', texto)
    if match:
        dados['cpf'] = match.group(1)
    
    # Data de Nascimento
    match = re.search(r'NASC(?:IMENTO)?[:\s]*(\d{2}/\d{2}/\d{4})', texto)
    if match:
        dados['data_nascimento'] = match.group(1)
    
    # Filiação
    match = re.search(r'FILIA[ÇC][AÃ]O[:\s]*([A-Z\s]+)', texto)
    if match:
        dados['filiacao'] = match.group(1).strip()
    
    return dados
```

#### Extração de Nota Fiscal

```python
def extrair_dados_nota_fiscal(image_path):
    """
    Extrai: CNPJ, Número NF, Data, Valor Total
    """
    img = preprocess_for_ocr(image_path)
    texto = pytesseract.image_to_string(img, lang='por')
    
    dados = {}
    
    # CNPJ (XX.XXX.XXX/XXXX-XX)
    match = re.search(r'(\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2})', texto)
    if match:
        dados['cnpj'] = match.group(1)
    
    # Número NF
    match = re.search(r'N[°\s]*NF[:\s]*(\d+)', texto, re.IGNORECASE)
    if match:
        dados['numero_nf'] = match.group(1)
    
    # Data (DD/MM/AAAA)
    match = re.search(r'DATA[:\s]*(\d{2}/\d{2}/\d{4})', texto, re.IGNORECASE)
    if match:
        dados['data'] = match.group(1)
    
    # Valor Total (R$ X.XXX,XX)
    match = re.search(r'TOTAL[:\s]*R\$\s*([\d.,]+)', texto, re.IGNORECASE)
    if match:
        dados['valor_total'] = match.group(1)
    
    return dados
```

---

### 4️⃣ 04_EasyOCR_Comparacao.ipynb

**Objetivo**: OCR com Deep Learning (melhor accuracy)

#### EasyOCR - Engine Baseada em Deep Learning

**Vantagens**:
- Baseado em CRAFT + CRNN (deep learning)
- Accuracy superior ao Tesseract (90-95%)
- Suporta 80+ idiomas
- Detecta rotação automaticamente
- GPU acceleration

**Instalação**:
```bash
pip install easyocr
```

#### Uso Básico

```python
import easyocr

# Criar leitor (primeira vez baixa modelos ~80MB)
reader = easyocr.Reader(['pt', 'en'])  # Português e Inglês

# Ler imagem
result = reader.readtext('documento.png')

# Resultado é lista de tuplas: (bbox, texto, confiança)
for (bbox, text, prob) in result:
    print(f"Texto: {text} (Confiança: {prob:.2f})")

# Output:
# Texto: Este é um documento (Confiança: 0.95)
# Texto: de exemplo para OCR (Confiança: 0.92)
```

#### Visualizar Resultados

```python
import cv2

img = cv2.imread('documento.png')

for (bbox, text, prob) in result:
    # bbox é lista de 4 pontos: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    pts = np.array(bbox, dtype=np.int32)
    cv2.polylines(img, [pts], True, (0, 255, 0), 2)
    
    # Texto
    x, y = pts[0]
    cv2.putText(img, text, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

cv2.imshow('EasyOCR Result', img)
cv2.waitKey(0)
```

#### Comparação: Tesseract vs EasyOCR

```python
import time

# Mesma imagem
img_path = 'documento.png'

# Tesseract
start = time.time()
texto_tesseract = pytesseract.image_to_string(Image.open(img_path), lang='por')
tempo_tesseract = time.time() - start

# EasyOCR
reader = easyocr.Reader(['pt'])
start = time.time()
result_easy = reader.readtext(img_path)
texto_easyocr = ' '.join([text for (_, text, _) in result_easy])
tempo_easyocr = time.time() - start

print(f"""
COMPARAÇÃO:
-----------
Tesseract:
- Tempo: {tempo_tesseract:.2f}s
- Texto: {texto_tesseract[:100]}...

EasyOCR:
- Tempo: {tempo_easyocr:.2f}s
- Texto: {texto_easyocr[:100]}...
""")
```

**Resultados típicos**:
```
COMPARAÇÃO:
-----------
Tesseract:
- Tempo: 0.5s
- Accuracy: 85%
- Melhor para: Documentos limpos, texto alinhado

EasyOCR:
- Tempo: 2.0s (CPU) / 0.3s (GPU)
- Accuracy: 92%
- Melhor para: Imagens naturais, texto rotacionado, múltiplas fontes
```

---

### 5️⃣ 05_PaddleOCR_Avancado.ipynb

**Objetivo**: Estado-da-arte em OCR (PaddlePaddle)

#### PaddleOCR - Melhor Performance

**Características**:
- Desenvolvido pela Baidu
- State-of-the-art accuracy (95-98%)
- Muito rápido (otimizado)
- Suporta 80+ idiomas
- Detecção de layout
- Table recognition

**Instalação**:
```bash
pip install paddlepaddle paddleocr
```

#### Uso Básico

```python
from paddleocr import PaddleOCR

# Criar OCR (primeira vez baixa modelos)
ocr = PaddleOCR(lang='pt')  # Português

# Processar imagem
result = ocr.ocr('documento.png')

# Resultado é lista de [bbox, (texto, confiança)]
for line in result[0]:
    bbox, (text, prob) = line
    print(f"Texto: {text} (Confiança: {prob:.2f})")
```

#### Detecção de Layout

```python
# Usar com detecção de layout
ocr = PaddleOCR(lang='pt', use_angle_cls=True, use_gpu=False)

# Processar
result = ocr.ocr('documento_complexo.png', cls=True)

# Resultado agrupa por blocos de texto
for idx in range(len(result)):
    res = result[idx]
    for line in res:
        print(line)
```

#### Table Recognition

```python
from paddleocr import PPStructure

# Criar estrutura de tabela
table_engine = PPStructure(show_log=False)

# Processar
result = table_engine('tabela.png')

# result contém estrutura da tabela
for line in result:
    if line['type'] == 'table':
        print("Tabela detectada!")
        print(line['res'])  # HTML da tabela
```

---

### 6️⃣ 06_Projeto_Pratico_Dados_Estruturados.ipynb

**Objetivo**: Sistema completo de extração de dados

#### Projeto: Extrator de Documentos

```python
class DocumentExtractor:
    """
    Sistema completo de extração de documentos
    """
    
    def __init__(self, ocr_engine='paddleocr'):
        """
        Engines: 'tesseract', 'easyocr', 'paddleocr'
        """
        self.engine = ocr_engine
        
        if engine == 'tesseract':
            # Configurar Tesseract
            pass
        elif engine == 'easyocr':
            self.reader = easyocr.Reader(['pt', 'en'])
        elif engine == 'paddleocr':
            self.ocr = PaddleOCR(lang='pt')
    
    def preprocess(self, image_path):
        """Pré-processamento"""
        return preprocess_for_ocr(image_path)
    
    def extract_text(self, image):
        """Extrair texto"""
        if self.engine == 'tesseract':
            return pytesseract.image_to_string(image, lang='por')
        elif self.engine == 'easyocr':
            result = self.reader.readtext(image)
            return ' '.join([text for (_, text, _) in result])
        elif self.engine == 'paddleocr':
            result = self.ocr.ocr(image)
            return ' '.join([line[1][0] for line in result[0]])
    
    def extract_document(self, image_path, doc_type='auto'):
        """
        Extrai dados estruturados
        
        doc_type: 'auto', 'cnh', 'rg', 'nota_fiscal'
        """
        # Pré-processar
        img = self.preprocess(image_path)
        
        # Extrair texto
        texto = self.extract_text(img)
        
        # Detectar tipo se auto
        if doc_type == 'auto':
            doc_type = self.detect_document_type(texto)
        
        # Extrair dados estruturados
        if doc_type == 'cnh':
            return extrair_dados_cnh_from_text(texto)
        elif doc_type == 'rg':
            return extrair_dados_rg_from_text(texto)
        elif doc_type == 'nota_fiscal':
            return extrair_dados_nota_from_text(texto)
        else:
            return {'texto': texto}
    
    def detect_document_type(self, text):
        """Detecta tipo de documento"""
        text_lower = text.lower()
        
        if 'cnh' in text_lower or 'carteira nacional de habilitação' in text_lower:
            return 'cnh'
        elif 'rg' in text_lower or 'identidade' in text_lower:
            return 'rg'
        elif 'nota fiscal' in text_lower or 'nf-e' in text_lower:
            return 'nota_fiscal'
        else:
            return 'desconhecido'
    
    def batch_process(self, image_paths):
        """Processar múltiplos documentos"""
        resultados = []
        
        for path in image_paths:
            try:
                resultado = self.extract_document(path)
                resultados.append({
                    'arquivo': path,
                    'sucesso': True,
                    'dados': resultado
                })
            except Exception as e:
                resultados.append({
                    'arquivo': path,
                    'sucesso': False,
                    'erro': str(e)
                })
        
        return resultados

# Usar
extractor = DocumentExtractor(ocr_engine='paddleocr')

# Processar um documento
dados = extractor.extract_document('cnh.jpg', doc_type='cnh')
print(dados)

# Processar múltiplos
documentos = ['cnh1.jpg', 'rg1.jpg', 'nota1.jpg']
resultados = extractor.batch_process(documentos)

# Salvar em JSON
import json
with open('resultados_ocr.json', 'w') as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)
```

---

## 📊 Comparação de Engines

| Aspecto | Tesseract | EasyOCR | PaddleOCR |
|---------|-----------|---------|-----------|
| **Accuracy** | 85-90% | 90-93% | 95-98% |
| **Velocidade (CPU)** | ⚡⚡⚡ | ⚡ | ⚡⚡ |
| **Velocidade (GPU)** | N/A | ⚡⚡⚡ | ⚡⚡⚡ |
| **Tamanho** | ~5 MB | ~80 MB | ~50 MB |
| **Idiomas** | 100+ | 80+ | 80+ |
| **Rotação** | ❌ | ✅ | ✅ |
| **Layout** | Básico | Não | ✅ |
| **Tabelas** | Não | Não | ✅ |
| **Instalação** | Difícil | Fácil | Fácil |
| **Quando usar** | Baseline, CPU only | Imagens naturais | Produção, best accuracy |

---

## 💻 Instalação Completa

### Requisitos

```
Python 3.7+
8GB RAM (mínimo)
GPU (opcional, acelera EasyOCR/PaddleOCR)
```

### Instalação

```bash
# Criar ambiente
conda create -n ocr_env python=3.9
conda activate ocr_env

# Tesseract (sistema)
# Windows: https://github.com/UB-Mannheim/tesseract/wiki
# Linux: sudo apt-get install tesseract-ocr tesseract-ocr-por
# Mac: brew install tesseract tesseract-lang

# Python
pip install pytesseract pillow opencv-python numpy

# EasyOCR
pip install easyocr

# PaddleOCR
pip install paddlepaddle paddleocr

# Extras
pip install regex pandas matplotlib
```

### Verificar Instalação

```python
# Tesseract
import pytesseract
print(pytesseract.get_tesseract_version())

# EasyOCR
import easyocr
reader = easyocr.Reader(['pt'])
print("EasyOCR OK")

# PaddleOCR
from paddleocr import PaddleOCR
ocr = PaddleOCR(lang='pt')
print("PaddleOCR OK")
```

---

## 🎯 Checklist de Conclusão

### Fundamentos
- [ ] Extrair texto com Tesseract
- [ ] Usar múltiplos idiomas
- [ ] Configurar PSM e OEM
- [ ] Visualizar bounding boxes

### Pré-processamento
- [ ] Aplicar grayscale e threshold
- [ ] Remover ruído
- [ ] Corrigir inclinação (deskew)
- [ ] Pipeline completo

### Documentos Reais
- [ ] Extrair dados de CNH
- [ ] Extrair dados de RG
- [ ] Extrair dados de nota fiscal
- [ ] Usar regex para estruturar

### Engines Avançadas
- [ ] Usar EasyOCR
- [ ] Usar PaddleOCR
- [ ] Comparar engines
- [ ] Escolher melhor para caso de uso

### Projeto
- [ ] Criar sistema completo
- [ ] Processar em lote
- [ ] Salvar resultados estruturados

---

## 📖 Recursos Complementares

### Documentação
- [Tesseract Docs](https://tesseract-ocr.github.io/)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)

### Datasets
- [ICDAR](https://rrc.cvc.uab.es/)
- [SROIE](https://rrc.cvc.uab.es/?ch=13)
- [Text-OCR](https://textvqa.org/textocr/)

### Ferramentas
- [Tesseract GUI](https://github.com/A9T9/Free-Ocr-Windows-Desktop)
- [Online OCR](https://www.onlineocr.net/)

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

**💡 Dica**: Pré-processamento é 80% do sucesso do OCR!

*Desenvolvido como parte do curso "Especialista em IA" - Módulo EAI_06*
