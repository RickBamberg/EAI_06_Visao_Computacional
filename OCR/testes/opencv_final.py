# test_opencv_final.py
print("🧪 TESTE FINAL DO OPENCV")

try:
    import cv2
    print(f"✅ cv2 importado")
    print(f"cv2.__file__ = {cv2.__file__}")
    
    # Testar funções básicas
    import numpy as np
    
    # 1. Criar imagem
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    print(f"✅ Imagem NumPy criada: {img.shape}")
    
    # 2. Testar rectangle
    cv2.rectangle(img, (20, 20), (80, 80), (255, 0, 0), 2)
    print(f"✅ cv2.rectangle funciona")
    
    # 3. Testar putText
    if hasattr(cv2, 'FONT_HERSHEY_SIMPLEX'):
        cv2.putText(img, 'TEST', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        print(f"✅ cv2.putText funciona")
    
    # 4. Testar cvtColor
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"✅ cv2.cvtColor funciona")
    
    # 5. Salvar imagem
    cv2.imwrite('test_opencv.png', img)
    print(f"✅ cv2.imwrite funciona - imagem salva: test_opencv.png")
    
    print(f"\\n🎉 OPENCV FUNCIONANDO PERFEITAMENTE!")
    
except Exception as e:
    print(f"❌ ERRO: {e}")
    import traceback
    traceback.print_exc()