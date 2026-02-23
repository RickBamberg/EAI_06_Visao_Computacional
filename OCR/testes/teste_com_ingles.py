# teste_com_ingles.py
import os
import sys
import pytesseract

print("🧪 Testando OCR com inglês (que já está instalado)")
print("=" * 60)

# Configurar Tesseract
tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
if os.path.exists(tesseract_path):
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
    print(f"✅ Tesseract configurado: {tesseract_path}")
    
    # Verificar arquivo de idioma inglês
    eng_file = r'C:\Program Files\Tesseract-OCR\tessdata\eng.traineddata'
    if os.path.exists(eng_file):
        print(f"✅ Idioma inglês encontrado: {eng_file}")
        
        # Criar imagem de teste em INGLÊS
        import numpy as np
        import cv2
        from PIL import Image
        
        img = np.ones((150, 400, 3), dtype=np.uint8) * 255
        cv2.putText(img, 'HELLO WORLD', (50, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
        cv2.putText(img, 'OCR TEST 2024', (50, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
        
        # Converter
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # Testar OCR em INGLÊS
        try:
            texto = pytesseract.image_to_string(img_pil, lang='eng')
            print(f"\n📝 TEXTO RECONHECIDO (inglês):")
            print("-" * 40)
            print(texto)
            print("-" * 40)
            
            if texto.strip():
                print("🎉 OCR EM INGLÊS FUNCIONANDO!")
                print("\n✅ Tesseract está funcionando corretamente!")
                print("💡 Agora só falta instalar o idioma português")
            else:
                print("⚠️  OCR não reconheceu texto")
                
        except Exception as e:
            print(f"❌ Erro no OCR: {e}")
    else:
        print("❌ Arquivo de idioma inglês não encontrado")
else:
    print("❌ Tesseract não encontrado")
