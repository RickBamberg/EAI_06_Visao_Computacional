from flask import Flask, render_template, request, url_for, jsonify, Response
import cv2
import os
import json
from datetime import datetime

app = Flask(__name__)

# Configurações globais para captura
captura_config = {
    'nome_pessoa': '',
    'num_fotos': 5,
    'fotos_capturadas': 0,
    'capturando': False
}

# Variável global para a câmera
camera = None

def inicializar_camera():
    """Inicializa a câmera se não estiver ativa"""
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
    return camera is not None

def liberar_camera():
    """Libera a câmera"""
    global camera
    if camera is not None:
        camera.release()
        camera = None
    cv2.destroyAllWindows()

def gerar_frames():
    """Gera frames da câmera para streaming"""
    global camera, captura_config
    
    if not inicializar_camera():
        return
    
    while captura_config['capturando']:
        ret, frame = camera.read()
        if not ret:
            break
        
        # Adiciona texto informativo no frame
        texto = f"Fotos: {captura_config['fotos_capturadas']}/{captura_config['num_fotos']}"
        cv2.putText(frame, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Converte para JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/capturar')
def capturar():
    return render_template('capturar.html')

@app.route('/tratar')
def tratar():
    return "Dados tratados com sucesso!"

@app.route('/reconhecer')
def reconhecer():
    return render_template('reconhecer.html')

@app.route('/iniciar_captura', methods=['POST'])
def iniciar_captura():
    """Inicia o processo de captura de fotos"""
    global captura_config
    
    dados = request.get_json()
    nome_pessoa = dados.get('nome_pessoa', 'Pessoa')
    num_fotos = int(dados.get('num_fotos', 5))
    
    # Cria a pasta para a pessoa se não existir
    caminho_pessoa = os.path.join('data', nome_pessoa)
    if not os.path.exists(caminho_pessoa):
        os.makedirs(caminho_pessoa)
    
    # Configura a captura
    captura_config.update({
        'nome_pessoa': nome_pessoa,
        'num_fotos': num_fotos,
        'fotos_capturadas': 0,
        'capturando': True
    })
    
    if inicializar_camera():
        return jsonify({'status': 'success', 'message': 'Captura iniciada'})
    else:
        return jsonify({'status': 'error', 'message': 'Erro ao acessar a câmera'})

@app.route('/parar_captura', methods=['POST'])
def parar_captura():
    """Para o processo de captura"""
    global captura_config
    captura_config['capturando'] = False
    liberar_camera()
    return jsonify({'status': 'success', 'message': 'Captura parada'})

@app.route('/capturar_foto', methods=['POST'])
def capturar_foto():
    """Captura uma foto"""
    global camera, captura_config
    
    if not captura_config['capturando'] or camera is None:
        return jsonify({'status': 'error', 'message': 'Captura não está ativa'})
    
    if captura_config['fotos_capturadas'] >= captura_config['num_fotos']:
        return jsonify({'status': 'error', 'message': 'Número máximo de fotos atingido'})
    
    ret, frame = camera.read()
    if ret:
        # Salva a foto
        nome_pessoa = captura_config['nome_pessoa']
        caminho_pessoa = os.path.join('data', nome_pessoa)
        nome_arquivo = os.path.join(caminho_pessoa, f"{captura_config['fotos_capturadas'] + 1}.jpg")
        
        cv2.imwrite(nome_arquivo, frame)
        captura_config['fotos_capturadas'] += 1
        
        # Se atingiu o número máximo, para a captura
        if captura_config['fotos_capturadas'] >= captura_config['num_fotos']:
            captura_config['capturando'] = False
            liberar_camera()
            return jsonify({
                'status': 'complete',
                'message': f'Todas as {captura_config["num_fotos"]} fotos foram capturadas!',
                'fotos_capturadas': captura_config['fotos_capturadas']
            })
        
        return jsonify({
            'status': 'success',
            'message': f'Foto {captura_config["fotos_capturadas"]} salva',
            'fotos_capturadas': captura_config['fotos_capturadas']
        })
    else:
        return jsonify({'status': 'error', 'message': 'Erro ao capturar foto'})

@app.route('/video_feed')
def video_feed():
    """Stream de vídeo da câmera"""
    return Response(gerar_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status_captura')
def status_captura():
    """Retorna o status atual da captura"""
    return jsonify(captura_config)

@app.route('/processar_captura', methods=['POST'])
def processar_captura():
    dados = request.form['dados']
    return f"Dados capturados: {dados}"

@app.route('/preview')
def preview():
    return render_template('preview.html')

if __name__ == '__main__':
    # Cria a pasta data se não existir
    if not os.path.exists('data'):
        os.makedirs('data')
    
    app.run(debug=True)