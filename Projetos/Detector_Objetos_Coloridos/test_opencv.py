"""
Script de teste para verificar a instalação e funcionamento do OpenCV
"""
import cv2
import numpy as np

def test_opencv():
    """Testa a instalação do OpenCV"""
    print("=" * 50)
    print("🔍 TESTE DE INSTALAÇÃO DO OPENCV")
    print("=" * 50)
    
    # Versão do OpenCV
    print(f"\n✅ OpenCV versão: {cv2.__version__}")
    
    # Teste de criação de imagem
    print("\n🎨 Testando criação de imagem...")
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    img[:] = (100, 150, 200)
    print("✅ Imagem criada com sucesso")
    
    # Teste de conversão de cores
    print("\n🌈 Testando conversão BGR para HSV...")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    print("✅ Conversão realizada com sucesso")
    
    # Teste de detecção de contornos
    print("\n📐 Testando detecção de contornos...")
    mask = np.zeros((300, 400), dtype=np.uint8)
    cv2.circle(mask, (200, 150), 50, 255, -1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"✅ {len(contours)} contorno(s) detectado(s)")
    
    # Teste de câmera
    print("\n📹 Testando acesso à câmera...")
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print("✅ Câmera acessada com sucesso")
        ret, frame = cap.read()
        if ret:
            print(f"✅ Frame capturado: {frame.shape}")
        else:
            print("⚠️ Não foi possível capturar frame")
        cap.release()
    else:
        print("❌ Câmera não disponível")
        print("   Isso pode ser normal se você não tiver webcam")
        print("   O sistema ainda funcionará para testes")
    
    print("\n" + "=" * 50)
    print("✅ TODOS OS TESTES CONCLUÍDOS!")
    print("=" * 50)
    print("\n💡 Você pode executar a aplicação com: python app.py")

def test_color_detection():
    """Testa detecção de cores em imagem sintética"""
    print("\n" + "=" * 50)
    print("🎨 TESTE DE DETECÇÃO DE CORES")
    print("=" * 50)
    
    # Cria imagem de teste com círculos coloridos
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255
    
    # Círculo vermelho
    cv2.circle(img, (100, 200), 50, (0, 0, 255), -1)
    # Círculo verde
    cv2.circle(img, (250, 200), 50, (0, 255, 0), -1)
    # Círculo azul
    cv2.circle(img, (400, 200), 50, (255, 0, 0), -1)
    # Círculo amarelo
    cv2.circle(img, (550, 200), 50, (0, 255, 255), -1)
    
    # Converte para HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define ranges de cores
    colors = {
        'Vermelho': [(np.array([0, 120, 70]), np.array([10, 255, 255]))],
        'Verde': [(np.array([40, 40, 40]), np.array([80, 255, 255]))],
        'Azul': [(np.array([100, 150, 0]), np.array([140, 255, 255]))],
        'Amarelo': [(np.array([20, 100, 100]), np.array([30, 255, 255]))]
    }
    
    print("\n🔍 Testando detecção para cada cor:")
    for color_name, ranges in colors.items():
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for (lower, upper) in ranges:
            mask_temp = cv2.inRange(hsv, lower, upper)
            mask = cv2.bitwise_or(mask, mask_temp)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            area = cv2.contourArea(contours[0])
            print(f"  ✅ {color_name}: {len(contours)} objeto(s) detectado(s) - Área: {int(area)}px")
        else:
            print(f"  ❌ {color_name}: Nenhum objeto detectado")
    
    print("\n✅ Teste de detecção de cores concluído!")

if __name__ == "__main__":
    try:
        test_opencv()
        test_color_detection()
        
        print("\n" + "=" * 50)
        print("🎉 SISTEMA PRONTO PARA USO!")
        print("=" * 50)
        print("\n📝 Próximos passos:")
        print("   1. Execute: python app.py")
        print("   2. Abra o navegador em: http://localhost:5000")
        print("   3. Permita o acesso à câmera")
        print("   4. Selecione uma cor e teste a detecção!")
        
    except Exception as e:
        print(f"\n❌ Erro durante os testes: {e}")
        print("\n💡 Certifique-se de que as dependências estão instaladas:")
        print("   pip install -r requirements.txt")
