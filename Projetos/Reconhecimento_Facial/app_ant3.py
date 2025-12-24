from flask import Flask, render_template, request, url_for, jsonify, Response, send_file
import cv2
import os
import json
import numpy as np
from datetime import datetime
import glob
import pickle
from tqdm import tqdm

# Importações para o tratamento de imagens
try:
    from deepface import DeepFace
    deepface_disponivel = True
except ImportError:
    deepface_disponivel = False
    print("DeepFace não encontrado. Instale com: pip install deepface")

app = Flask(__name__)

# Configurações globais para captura
captura_config = {
    'nome_pessoa': '',
    'num_fotos': 5,
    'fotos_capturadas': 0,
    'capturando': False
}

# Configurações globais para tratamento
tratamento_config = {
    'processando': False,
    'progresso': 0,
    'total_imagens': 0,
    'imagens_processadas': 0,
    'status_message': 'Pronto para processar',
    'detector_carregado': False
}

# Variáveis globais
camera = None
detector = None
model_facenet = None

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

def carregar_detector_facial():
    """Carrega o detector facial do OpenCV"""
    global detector, tratamento_config
    
    if detector is not None:
        return True
    
    try:
        caminho_prototxt = 'deploy.prototxt'
        caminho_modelo = 'res10_300x300_ssd_iter_140000.caffemodel'
        
        if not os.path.exists(caminho_prototxt) or not os.path.exists(caminho_modelo):
            tratamento_config['status_message'] = 'Arquivos do detector facial não encontrados'
            return False
        
        detector = cv2.dnn.readNetFromCaffe(caminho_prototxt, caminho_modelo)
        tratamento_config['detector_carregado'] = True
        return True
    except Exception as e:
        tratamento_config['status_message'] = f'Erro ao carregar detector: {str(e)}'
        return False

def carregar_modelo_facenet():
    """Carrega o modelo FaceNet"""
    global model_facenet
    
    if not deepface_disponivel:
        return False
    
    try:
        if model_facenet is None:
            model_facenet = DeepFace.build_model('Facenet')
        return True
    except Exception as e:
        print(f"Erro ao carregar FaceNet: {e}")
        return False

def processar_deteccao_facial(pessoa_especifica=None):
    """Processa imagens para detectar e recortar faces
    
    Args:
        pessoa_especifica (str, optional): Nome da pessoa específica para processar.
                                         Se None, processa todas as pessoas.
    """
    global detector, tratamento_config
    
    caminho_dados_originais = 'data/'
    caminho_faces_recortadas = 'data/faces_recortadas/'
    
    # Criar pasta de saída se não existir
    if not os.path.exists(caminho_faces_recortadas):
        os.makedirs(caminho_faces_recortadas)
    
    total_processadas = 0
    total_salvas = 0
    
    # Determinar quais pessoas processar
    if pessoa_especifica:
        pessoas_para_processar = [pessoa_especifica] if pessoa_especifica in os.listdir(caminho_dados_originais) else []
        if not pessoas_para_processar:
            tratamento_config['status_message'] = f'Pessoa "{pessoa_especifica}" não encontrada'
            return 0, 0
    else:
        pessoas_para_processar = [nome for nome in os.listdir(caminho_dados_originais) 
                                if os.path.isdir(os.path.join(caminho_dados_originais, nome)) 
                                and nome != 'faces_recortadas']
    
    # Contar total de imagens
    total_imagens = 0
    for nome_pessoa in pessoas_para_processar:
        caminho_pessoa = os.path.join(caminho_dados_originais, nome_pessoa)
        for nome_arquivo in os.listdir(caminho_pessoa):
            if nome_arquivo.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                total_imagens += 1
    
    if total_imagens == 0:
        tratamento_config['status_message'] = 'Nenhuma imagem encontrada para processar'
        return 0, 0
    
    tratamento_config['total_imagens'] = total_imagens
    tratamento_config['imagens_processadas'] = 0
    
    # Processar cada pessoa
    for nome_pessoa in pessoas_para_processar:
        caminho_pessoa = os.path.join(caminho_dados_originais, nome_pessoa)
        
        tratamento_config['status_message'] = f'Processando imagens de: {nome_pessoa}'
        
        # Processar cada imagem da pessoa
        for nome_arquivo in os.listdir(caminho_pessoa):
            if not nome_arquivo.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue
                
            caminho_imagem = os.path.join(caminho_pessoa, nome_arquivo)
            
            # Carregar a imagem
            imagem = cv2.imread(caminho_imagem)
            if imagem is None:
                print(f"Erro ao ler: {nome_arquivo}")
                total_processadas += 1
                tratamento_config['imagens_processadas'] = total_processadas
                continue
            
            (h, w) = imagem.shape[:2]
            
            # Pré-processamento para a rede neural
            blob = cv2.dnn.blobFromImage(cv2.resize(imagem, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            
            # Detectar faces
            detector.setInput(blob)
            deteccoes = detector.forward()
            
            # Encontrar a melhor detecção
            if len(deteccoes[0, 0]) > 0:
                best_detection_index = np.argmax(deteccoes[0, 0, :, 2])
                confianca = deteccoes[0, 0, best_detection_index, 2]
                
                if confianca > 0.8:  # 80% de confiança
                    # Calcular coordenadas da caixa delimitadora
                    box = deteccoes[0, 0, best_detection_index, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    
                    # Garantir que está dentro dos limites
                    startX = max(0, startX)
                    startY = max(0, startY)
                    endX = min(w, endX)
                    endY = min(h, endY)
                    
                    # Recortar o rosto
                    rosto = imagem[startY:endY, startX:endX]
                    
                    if rosto.size != 0:
                        # Salvar o rosto recortado
                        nome_saida = f"{nome_pessoa}_{nome_arquivo}"
                        caminho_saida = os.path.join(caminho_faces_recortadas, nome_saida)
                        cv2.imwrite(caminho_saida, rosto)
                        total_salvas += 1
            
            total_processadas += 1
            tratamento_config['imagens_processadas'] = total_processadas
            tratamento_config['progresso'] = int((total_processadas / total_imagens) * 50)  # 50% para detecção
    
    return total_processadas, total_salvas

def gerar_embeddings(pessoa_especifica=None):
    """Gera embeddings das faces recortadas
    
    Args:
        pessoa_especifica (str, optional): Nome da pessoa específica para processar.
                                         Se None, processa todas as faces.
    """
    global tratamento_config
    
    if not deepface_disponivel:
        tratamento_config['status_message'] = 'DeepFace não disponível'
        return False
    
    if not carregar_modelo_facenet():
        tratamento_config['status_message'] = 'Erro ao carregar modelo FaceNet'
        return False
    
    caminho_faces_recortadas = 'data/faces_recortadas/'
    
    if not os.path.exists(caminho_faces_recortadas):
        tratamento_config['status_message'] = 'Pasta de faces recortadas não encontrada'
        return False
    
    # Carregar embeddings existentes se houver
    caminho_embeddings = 'data/embeddings.pickle'
    known_embeddings = []
    known_names = []
    
    if os.path.exists(caminho_embeddings):
        try:
            with open(caminho_embeddings, 'rb') as f:
                dados_existentes = pickle.load(f)
                known_embeddings = dados_existentes['embeddings'].tolist()
                known_names = dados_existentes['names']
        except Exception as e:
            print(f"Erro ao carregar embeddings existentes: {e}")
    
    # Filtrar arquivos de faces para processar
    if pessoa_especifica:
        arquivos_faces = [f for f in os.listdir(caminho_faces_recortadas) 
                         if f.startswith(f"{pessoa_especifica}_") and f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        # Remover embeddings existentes desta pessoa
        indices_para_remover = [i for i, nome in enumerate(known_names) if nome == pessoa_especifica]
        for i in reversed(indices_para_remover):  # Remove de trás para frente
            del known_embeddings[i]
            del known_names[i]
    else:
        arquivos_faces = [f for f in os.listdir(caminho_faces_recortadas) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        # Se processando tudo, limpa embeddings existentes
        known_embeddings = []
        known_names = []
    
    total_faces = len(arquivos_faces)
    if total_faces == 0:
        tratamento_config['status_message'] = f'Nenhuma face encontrada para processar{" para " + pessoa_especifica if pessoa_especifica else ""}'
        return False
    
    tratamento_config['status_message'] = f'Gerando embeddings{" para " + pessoa_especifica if pessoa_especifica else ""}...'
    
    for i, nome_arquivo in enumerate(arquivos_faces):
        nome_pessoa = nome_arquivo.split('_')[0]
        caminho_rosto = os.path.join(caminho_faces_recortadas, nome_arquivo)
        
        try:
            embedding_obj = DeepFace.represent(
                img_path=caminho_rosto,
                model_name='Facenet',
                enforce_detection=False
            )
            embedding = embedding_obj[0]['embedding']
            
            known_embeddings.append(embedding)
            known_names.append(nome_pessoa)
            
        except Exception as e:
            print(f"Erro ao processar {nome_arquivo}: {e}")
        
        # Atualizar progresso (50% para detecção + 50% para embeddings)
        progresso_embeddings = int(((i + 1) / total_faces) * 50)
        tratamento_config['progresso'] = 50 + progresso_embeddings
    
    # Salvar banco de dados de embeddings atualizado
    if known_embeddings:
        dados_embeddings = {
            "embeddings": np.array(known_embeddings), 
            "names": known_names
        }
        with open(caminho_embeddings, 'wb') as f:
            pickle.dump(dados_embeddings, f)
        
        tratamento_config['status_message'] = f'Processamento concluído! {len(known_embeddings)} embeddings salvos'
        return True
    else:
        tratamento_config['status_message'] = 'Nenhum embedding foi gerado'
        return False

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
    return render_template('tratar.html')

@app.route('/reconhecer')
def reconhecer():
    return render_template('reconhecer.html')

@app.route('/preview')
def preview():
    return render_template('preview.html')

# ================= ROTAS DE CAPTURA =================
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

# ================= ROTAS DE TRATAMENTO =================
@app.route('/iniciar_tratamento', methods=['POST'])
def iniciar_tratamento():
    """Inicia o processamento de tratamento de imagens"""
    global tratamento_config
    
    if tratamento_config['processando']:
        return jsonify({'status': 'error', 'message': 'Tratamento já em andamento'})
    
    # Pegar dados da requisição
    dados = request.get_json() if request.is_json else {}
    pessoa_especifica = dados.get('pessoa_especifica', None)  # None = processar todas
    
    # Verificar se existem imagens para processar
    caminho_dados = 'data/'
    if not os.path.exists(caminho_dados):
        return jsonify({'status': 'error', 'message': 'Pasta de dados não encontrada'})
    
    # Verificar se a pessoa específica existe (se fornecida)
    if pessoa_especifica:
        caminho_pessoa = os.path.join(caminho_dados, pessoa_especifica)
        if not os.path.exists(caminho_pessoa):
            return jsonify({'status': 'error', 'message': f'Pessoa "{pessoa_especifica}" não encontrada'})
    
    # Contar imagens disponíveis
    total_imagens = 0
    pessoas_para_processar = [pessoa_especifica] if pessoa_especifica else [
        nome for nome in os.listdir(caminho_dados) 
        if os.path.isdir(os.path.join(caminho_dados, nome)) and nome != 'faces_recortadas'
    ]
    
    for nome_pessoa in pessoas_para_processar:
        caminho_pessoa = os.path.join(caminho_dados, nome_pessoa)
        if os.path.isdir(caminho_pessoa):
            for arquivo in os.listdir(caminho_pessoa):
                if arquivo.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    total_imagens += 1
    
    if total_imagens == 0:
        return jsonify({'status': 'error', 'message': f'Nenhuma imagem encontrada para processar{" para " + pessoa_especifica if pessoa_especifica else ""}'})
    
    # Carregar detector facial
    if not carregar_detector_facial():
        return jsonify({'status': 'error', 'message': tratamento_config['status_message']})
    
    # Verificar se DeepFace está disponível
    if not deepface_disponivel:
        return jsonify({'status': 'error', 'message': 'DeepFace não está instalado'})
    
    # Iniciar processamento
    tratamento_config.update({
        'processando': True,
        'progresso': 0,
        'total_imagens': total_imagens,
        'imagens_processadas': 0,
        'status_message': f'Iniciando processamento{" para " + pessoa_especifica if pessoa_especifica else ""}...'
    })
    
    try:
        # Fase 1: Detecção e recorte de faces
        tratamento_config['status_message'] = f'Detectando e recortando faces{" de " + pessoa_especifica if pessoa_especifica else ""}...'
        total_processadas, total_salvas = processar_deteccao_facial(pessoa_especifica)
        
        # Fase 2: Geração de embeddings
        tratamento_config['status_message'] = f'Gerando embeddings{" para " + pessoa_especifica if pessoa_especifica else ""}...'
        success = gerar_embeddings(pessoa_especifica)
        
        if success:
            tratamento_config.update({
                'processando': False,
                'progresso': 100,
                'status_message': f'Processamento concluído! {total_salvas} faces detectadas e processadas{" para " + pessoa_especifica if pessoa_especifica else ""}'
            })
            return jsonify({
                'status': 'success',
                'message': f'Tratamento concluído! {total_salvas} faces processadas{" para " + pessoa_especifica if pessoa_especifica else ""}',
                'total_processadas': total_processadas,
                'total_salvas': total_salvas,
                'pessoa_processada': pessoa_especifica
            })
        else:
            tratamento_config.update({
                'processando': False,
                'status_message': 'Erro na geração de embeddings'
            })
            return jsonify({'status': 'error', 'message': 'Erro na geração de embeddings'})
            
    except Exception as e:
        tratamento_config.update({
            'processando': False,
            'status_message': f'Erro durante processamento: {str(e)}'
        })
        return jsonify({'status': 'error', 'message': f'Erro durante processamento: {str(e)}'})

@app.route('/status_tratamento')
def status_tratamento():
    """Retorna o status atual do tratamento"""
    # Adicionar informações sobre faces detectadas e embeddings
    status_completo = tratamento_config.copy()
    
    # Verificar se existe arquivo de embeddings para contar
    caminho_embeddings = 'data/embeddings.pickle'
    if os.path.exists(caminho_embeddings):
        try:
            with open(caminho_embeddings, 'rb') as f:
                dados_embeddings = pickle.load(f)
                status_completo['total_embeddings'] = len(dados_embeddings['names'])
        except:
            status_completo['total_embeddings'] = 0
    else:
        status_completo['total_embeddings'] = 0
    
    return jsonify(status_completo)

@app.route('/parar_tratamento', methods=['POST'])
def parar_tratamento():
    """Para o processamento de tratamento"""
    global tratamento_config
    tratamento_config.update({
        'processando': False,
        'status_message': 'Processamento interrompido pelo usuário'
    })
    return jsonify({'status': 'success', 'message': 'Tratamento parado'})

# ================= ROTAS DE PREVIEW =================
@app.route('/buscar_fotos')
def buscar_fotos():
    """Busca as fotos de uma pessoa específica"""
    nome_pessoa = request.args.get('nome', '')
    
    if not nome_pessoa:
        return jsonify({'status': 'error', 'message': 'Nome não fornecido'})
    
    caminho_pessoa = os.path.join('data', nome_pessoa)
    
    if not os.path.exists(caminho_pessoa):
        return jsonify({'status': 'error', 'message': 'Pasta não encontrada', 'fotos': []})
    
    # Busca todas as imagens na pasta
    extensoes = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    fotos = []
    
    for extensao in extensoes:
        padrao = os.path.join(caminho_pessoa, extensao)
        for caminho_foto in glob.glob(padrao):
            nome_arquivo = os.path.basename(caminho_foto)
            data_criacao = os.path.getctime(caminho_foto)
            data_formatada = datetime.fromtimestamp(data_criacao).strftime('%Y-%m-%d %H:%M:%S')
            
            fotos.append({
                'nome': nome_arquivo,
                'caminho': caminho_foto,
                'url': f'/foto/{nome_pessoa}/{nome_arquivo}',
                'data': data_formatada
            })
    
    # Ordena as fotos pelo nome
    try:
        fotos.sort(key=lambda x: int(os.path.splitext(x['nome'])[0]))
    except ValueError:
        fotos.sort(key=lambda x: x['nome'])
    
    return jsonify({
        'status': 'success',
        'nome_pessoa': nome_pessoa,
        'quantidade': len(fotos),
        'fotos': fotos
    })

@app.route('/foto/<nome_pessoa>/<nome_arquivo>')
def servir_foto(nome_pessoa, nome_arquivo):
    """Serve uma foto específica"""
    caminho_foto = os.path.join('data', nome_pessoa, nome_arquivo)
    
    if not os.path.exists(caminho_foto):
        return jsonify({'status': 'error', 'message': 'Foto não encontrada'}), 404
    
    return send_file(caminho_foto, mimetype='image/jpeg')

@app.route('/listar_pessoas')
def listar_pessoas():
    """Lista todas as pessoas com fotos no sistema"""
    caminho_dados = 'data/'
    pessoas = []
    
    if os.path.exists(caminho_dados):
        for nome_pasta in os.listdir(caminho_dados):
            caminho_pasta = os.path.join(caminho_dados, nome_pasta)
            if os.path.isdir(caminho_pasta) and nome_pasta != 'faces_recortadas':
                # Contar fotos na pasta
                extensoes = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
                total_fotos = 0
                for extensao in extensoes:
                    total_fotos += len(glob.glob(os.path.join(caminho_pasta, extensao)))
                
                if total_fotos > 0:
                    pessoas.append({
                        'nome': nome_pasta,
                        'total_fotos': total_fotos
                    })
    
    return jsonify({
        'status': 'success',
        'pessoas': pessoas
    })

@app.route('/verificar_embeddings')
def verificar_embeddings():
    """Verifica e retorna informações sobre os embeddings"""
    caminho_embeddings = 'data/embeddings.pickle'
    caminho_faces = 'data/faces_recortadas/'
    
    total_embeddings = 0
    total_faces = 0
    
    # Contar embeddings
    if os.path.exists(caminho_embeddings):
        try:
            with open(caminho_embeddings, 'rb') as f:
                dados_embeddings = pickle.load(f)
                total_embeddings = len(dados_embeddings['names'])
        except:
            total_embeddings = 0
    
    # Contar faces recortadas
    if os.path.exists(caminho_faces):
        extensoes = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        for extensao in extensoes:
            total_faces += len(glob.glob(os.path.join(caminho_faces, extensao)))
    
    return jsonify({
        'status': 'success',
        'total_embeddings': total_embeddings,
        'total_faces': total_faces
    })

if __name__ == '__main__':
    # Cria a pasta data se não existir
    if not os.path.exists('data'):
        os.makedirs('data')
    
    app.run(debug=True)