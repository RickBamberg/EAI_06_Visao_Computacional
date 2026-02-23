# instalar_portugues.py
import urllib.request
import os
import shutil

print("📥 INSTALANDO IDIOMA PORTUGUÊS PARA TESSERACT")
print("=" * 60)

# URLs dos arquivos de idioma (tessdata_fast - mais rápido)
urls = {
    'por': 'https://github.com/tesseract-ocr/tessdata_fast/raw/main/por.traineddata',
    'por_vert': 'https://github.com/tesseract-ocr/tessdata_fast/raw/main/por_vert.traineddata'
}

# Diretório destino
tessdata_dir = r'C:\Program Files\Tesseract-OCR\tessdata'
backup_dir = r'C:\Program Files\Tesseract-OCR\tessdata_backup'

# Criar backup primeiro
print(f"\n1. Criando backup em: {backup_dir}")
try:
    if os.path.exists(tessdata_dir):
        if os.path.exists(backup_dir):
            shutil.rmtree(backup_dir)
        shutil.copytree(tessdata_dir, backup_dir)
        print("   ✅ Backup criado")
except:
    print("   ⚠️  Não foi possível criar backup (continuando...)")

# Baixar arquivos
print(f"\n2. Baixando idiomas para: {tessdata_dir}")
os.makedirs(tessdata_dir, exist_ok=True)

for nome, url in urls.items():
    destino = os.path.join(tessdata_dir, f'{nome}.traineddata')
    
    print(f"\n   📥 {nome}.traineddata")
    print(f"      URL: {url}")
    print(f"      Destino: {destino}")
    
    try:
        # Baixar
        urllib.request.urlretrieve(url, destino)
        
        # Verificar
        if os.path.exists(destino):
            tamanho = os.path.getsize(destino) / (1024*1024)
            print(f"      ✅ Baixado: {tamanho:.2f} MB")
        else:
            print(f"      ❌ Falha ao baixar")
            
    except Exception as e:
        print(f"      ❌ Erro: {e}")

# Verificar o que foi instalado
print(f"\n3. VERIFICANDO IDIOMAS INSTALADOS:")
if os.path.exists(tessdata_dir):
    arquivos = [f for f in os.listdir(tessdata_dir) if f.endswith('.traineddata')]
    for arquivo in arquivos:
        caminho = os.path.join(tessdata_dir, arquivo)
        tamanho = os.path.getsize(caminho) / (1024*1024)
        print(f"   • {arquivo}: {tamanho:.1f} MB")

# Testar
print(f"\n4. TESTANDO CONFIGURAÇÃO...")
if os.path.exists(os.path.join(tessdata_dir, 'por.traineddata')):
    print("   ✅ Português instalado com sucesso!")
    
    import pytesseract
    tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    if os.path.exists(tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        # Teste rápido
        try:
            import numpy as np
            import cv2
            from PIL import Image
            
            img = np.ones((100, 300, 3), dtype=np.uint8) * 255
            cv2.putText(img, 'TESTE', (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
            
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            texto = pytesseract.image_to_string(img_pil, lang='por')
            
            print(f"   ✅ Teste OCR: '{texto.strip()}'")
            
        except Exception as e:
            print(f"   ⚠️  Erro no teste: {e}")
else:
    print("   ❌ Português não instalado")

print("\n" + "=" * 60)
print("📌 SE FALHAR, BAIXE MANUALMENTE:")
print("1. Acesse: https://github.com/tesseract-ocr/tessdata_fast")
print("2. Clique em 'por.traineddata'")
print("3. Clique em 'Download' (botão Raw)")
print(f"4. Salve em: {tessdata_dir}\\por.traineddata")
print("=" * 60)