# OCR/debug_opencv.py
import sys
print(f"Python: {sys.version}")
print(f"Python path: {sys.executable}")

print("\n" + "="*60)
print("DIAGNÓSTICO COMPLETO DO OPENCV")
print("="*60)

# Listar todos os módulos cv2
import pkgutil
import cv2

print("\n📦 Módulos cv2 disponíveis:")
for importer, modname, ispkg in pkgutil.iter_modules(cv2.__path__):
    print(f"  • {modname}")

print("\n🔍 Estrutura do pacote cv2:")
print(f"cv2.__file__ = {cv2.__file__}")
print(f"cv2.__path__ = {cv2.__path__}")

print("\n🧪 Testando imports específicos:")
try:
    # Tentar importar de submodules
    from cv2 import cv2 as cv2_full
    print("✅ from cv2 import cv2 funciona")
    
    # Testar funções
    import numpy as np
    img = np.zeros((10, 10, 3), np.uint8)
    cv2_full.rectangle(img, (2, 2), (8, 8), (255, 0, 0), 1)
    print("✅ cv2.rectangle funciona")
    
    # Verificar FONT
    if hasattr(cv2_full, 'FONT_HERSHEY_SIMPLEX'):
        print(f"✅ FONT_HERSHEY_SIMPLEX = {cv2_full.FONT_HERSHEY_SIMPLEX}")
    
except Exception as e:
    print(f"❌ Erro: {e}")
    import traceback
    traceback.print_exc()

print("\n🔄 Tentando import alternativo:")
try:
    # Em algumas instalações, as funções estão em cv2.cv2
    if hasattr(cv2, 'cv2'):
        cv = cv2.cv2
        print("✅ Usando cv2.cv2")
        
        # Testar
        import numpy as np
        img = np.zeros((10, 10, 3), np.uint8)
        cv.rectangle(img, (2, 2), (8, 8), (255, 0, 0), 1)
        print("✅ cv.rectangle funciona")
        
except Exception as e:
    print(f"❌ cv2.cv2 também falhou: {e}")

print("\n" + "="*60)
print("SOLUÇÕES POSSÍVEIS:")
print("1. pip uninstall opencv-python opencv-contrib-python -y")
print("2. pip install opencv-python==4.9.0.80")
print("3. pip install opencv-python-headless==4.9.0.80")
print("="*60)
