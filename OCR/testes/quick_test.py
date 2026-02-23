# OCR/quick_test.py
print("🔍 TESTE RÁPIDO DO AMBIENTE OCR")
print("=" * 40)

# 1. Testar imports básicos
try:
    import numpy as np
    print(f"✅ NumPy: {np.__version__}")
except:
    print("❌ NumPy falhou")

try:
    import cv2
    print(f"✅ OpenCV importado")
    # Testar funcionalidade básica
    img = np.zeros((10, 10, 3), np.uint8)
    cv2.rectangle(img, (2, 2), (8, 8), (255, 0, 0), 1)
    print(f"✅ OpenCV desenha retângulos")
except Exception as e:
    print(f"❌ OpenCV: {e}")

try:
    from PIL import Image
    print(f"✅ PIL/Pillow: {Image.__version__}")
except:
    print("❌ PIL falhou")

try:
    import pytesseract
    print(f"✅ PyTesseract importado")
except:
    print("❌ PyTesseract falhou")

print("\n" + "=" * 40)
print("🧪 TESTE PRÁTICO SIMPLES")
print("=" * 40)

# Criar imagem de teste MUITO simples
import numpy as np
import cv2

# Criar imagem branca com texto preto
img = np.ones((50, 200, 3), dtype=np.uint8) * 255
cv2.putText(img, 'TEST', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

print("Imagem criada (50x200 pixels com texto 'TEST')")

# Mostrar se possível
try:
    from PIL import Image
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # Tentar OCR
    import pytesseract
    
    # Configurar Tesseract no Windows
    import os
    if os.name == 'nt':
        paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        ]
        for path in paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                print(f"✅ Tesseract encontrado em: {path}")
                break
        else:
            print("❌ Tesseract não encontrado")
            print("💡 Instale com: winget install UB-Mannheim.TesseractOCR")
    
    # Tentar OCR
    try:
        text = pytesseract.image_to_string(img_pil, lang='eng')
        print(f"\n📝 TEXTO RECONHECIDO: '{text.strip()}'")
        
        if 'TEST' in text.upper():
            print("🎉 OCR FUNCIONANDO!")
        elif text.strip():
            print(f"⚠️  OCR reconheceu algo diferente: {text.strip()}")
        else:
            print("❌ OCR não reconheceu nada")
            
    except Exception as e:
        print(f"❌ Erro no OCR: {e}")
        
except Exception as e:
    print(f"❌ Erro no teste: {e}")

print("\n" + "=" * 40)
print("PRÓXIMO PASSO:")
print("Execute: winget install UB-Mannheim.TesseractOCR")
print("(No PowerShell como Administrador)")
print("=" * 40)