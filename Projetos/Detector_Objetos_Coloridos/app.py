from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
from threading import Lock

app = Flask(__name__)

# Configurações de cores em HSV (Hue, Saturation, Value)
COLOR_RANGES = {
    'vermelho': [
        (np.array([0, 120, 70]), np.array([10, 255, 255])),
        (np.array([170, 120, 70]), np.array([180, 255, 255]))
    ],
    'verde': [
        (np.array([40, 40, 40]), np.array([80, 255, 255]))
    ],
    'azul': [
        (np.array([100, 150, 0]), np.array([140, 255, 255]))
    ],
    'amarelo': [
        (np.array([20, 100, 100]), np.array([30, 255, 255]))
    ],
    'laranja': [
        (np.array([10, 100, 100]), np.array([20, 255, 255]))
    ],
    'roxo': [
        (np.array([140, 50, 50]), np.array([170, 255, 255]))
    ]
}

# Variáveis globais
camera = None
camera_lock = Lock()
current_color = 'vermelho'
min_area = 500
show_mask = False

def get_camera():
    """Inicializa ou retorna a câmera existente"""
    global camera
    with camera_lock:
        if camera is None or not camera.isOpened():
            camera = cv2.VideoCapture(0)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return camera

def detect_colored_objects(frame, color_name):
    """
    Detecta objetos de uma cor específica no frame
    """
    # Converte BGR para HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Cria máscara para a cor selecionada
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    
    if color_name in COLOR_RANGES:
        for (lower, upper) in COLOR_RANGES[color_name]:
            mask_temp = cv2.inRange(hsv, lower, upper)
            mask = cv2.bitwise_or(mask, mask_temp)
    
    # Aplica operações morfológicas para remover ruído
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Encontra contornos
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Cria cópia do frame para desenhar
    output = frame.copy()
    object_count = 0
    
    # Processa cada contorno
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Filtra por área mínima
        if area > min_area:
            object_count += 1
            
            # Obtém retângulo delimitador
            x, y, w, h = cv2.boundingRect(contour)
            
            # Desenha retângulo
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Calcula centro
            cx = x + w // 2
            cy = y + h // 2
            cv2.circle(output, (cx, cy), 5, (0, 0, 255), -1)
            
            # Adiciona informações
            label = f"{color_name.upper()} #{object_count}"
            cv2.putText(output, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Adiciona área
            area_text = f"Area: {int(area)}"
            cv2.putText(output, area_text, (x, y + h + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Adiciona contador total
    info_text = f"Objetos {color_name}: {object_count}"
    cv2.putText(output, info_text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Se show_mask estiver ativo, mostra a máscara ao lado
    if show_mask:
        mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        output = np.hstack([output, mask_colored])
    
    return output, object_count

def generate_frames():
    """Gera frames para streaming de vídeo"""
    cam = get_camera()
    
    while True:
        success, frame = cam.read()
        if not success:
            break
        
        # Detecta objetos coloridos
        processed_frame, count = detect_colored_objects(frame, current_color)
        
        # Codifica frame como JPEG
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()
        
        # Retorna frame no formato de streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    """Página principal"""
    return render_template('index.html', colors=list(COLOR_RANGES.keys()))

@app.route('/video_feed')
def video_feed():
    """Stream de vídeo"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_color', methods=['POST'])
def set_color():
    """Define a cor a ser detectada"""
    global current_color
    data = request.json
    color = data.get('color', 'vermelho')
    
    if color in COLOR_RANGES:
        current_color = color
        return jsonify({'success': True, 'color': current_color})
    return jsonify({'success': False, 'message': 'Cor inválida'})

@app.route('/set_min_area', methods=['POST'])
def set_min_area():
    """Define a área mínima para detecção"""
    global min_area
    data = request.json
    area = data.get('area', 500)
    
    try:
        min_area = int(area)
        return jsonify({'success': True, 'min_area': min_area})
    except ValueError:
        return jsonify({'success': False, 'message': 'Área inválida'})

@app.route('/toggle_mask', methods=['POST'])
def toggle_mask():
    """Alterna exibição da máscara"""
    global show_mask
    show_mask = not show_mask
    return jsonify({'success': True, 'show_mask': show_mask})

@app.route('/get_status')
def get_status():
    """Retorna status atual da detecção"""
    return jsonify({
        'current_color': current_color,
        'min_area': min_area,
        'show_mask': show_mask,
        'available_colors': list(COLOR_RANGES.keys())
    })

if __name__ == '__main__':
    try:
        app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
    finally:
        if camera is not None:
            camera.release()
