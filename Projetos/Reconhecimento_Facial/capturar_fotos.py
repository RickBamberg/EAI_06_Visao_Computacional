import cv2
import os

# --- Configurações ---
NOME_PESSOA = "Rick"  # Mude aqui para cada pessoa que for cadastrar
NUM_FOTOS = 5        # Quantidade de fotos a serem tiradas

# --- Criação de Pastas ---
caminho_pessoa = os.path.join('data', NOME_PESSOA)
if not os.path.exists(caminho_pessoa):
    os.makedirs(caminho_pessoa)

# --- Captura ---
cap = cv2.VideoCapture(0) # 0 para a webcam padrão
count = 0

print("\nPressione 's' para salvar uma foto. Pressione 'q' para sair.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mostra a contagem na tela
    cv2.putText(frame, f"Fotos salvas: {count}/{NUM_FOTOS}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Captura de Fotos", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        if count < NUM_FOTOS:
            nome_arquivo = os.path.join(caminho_pessoa, f"{count+1}.jpg")
            cv2.imwrite(nome_arquivo, frame)
            print(f"Foto salva: {nome_arquivo}")
            count += 1
        else:
            print("Número máximo de fotos já foi atingido.")

    elif key == ord('q'):
        break

print(f"\n{count} fotos salvas para '{NOME_PESSOA}'.")
cap.release()
cv2.destroyAllWindows()