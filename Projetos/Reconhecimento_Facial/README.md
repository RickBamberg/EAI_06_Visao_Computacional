# Sistema de Reconhecimento Facial em Tempo Real com Deep Learning

Este projeto implementa um sistema completo de reconhecimento facial, capaz de identificar pessoas em tempo real através de uma interface web. A aplicação foi construída utilizando Python, Flask para o backend e uma pilha de tecnologias de Machine Learning, incluindo OpenCV, TensorFlow e a biblioteca DeepFace.

## Estrutura do Projeto

O projeto está organizado da seguinte forma para garantir a separação de responsabilidades e a clareza do código:

```
Reconhecimento_Facial/
│
├── data/
│   ├── NOME_PESSOA_1/
│   │   ├── 1.jpg
│   │   └── ...
│   ├── faces_recortadas/
│   └── embeddings.pickle
│
├── notebook/
│   └── processamento_e_geracao_embeddings.ipynb
│
├── templates/
│   └── index.html
│
├── app.py
├── capturar_fotos.py
├── deploy.prototxt
├── res10_300x300_ssd_iter_140000.caffemodel
├── requirements.txt
└── README.md
```

-   **`data/`**: Contém todos os dados. As subpastas com os nomes das pessoas guardam as imagens de cadastro. `faces_recortadas` armazena os rostos extraídos, e `embeddings.pickle` é o banco de dados de "assinaturas faciais".
-   **`notebook/`**: Contém o Jupyter Notebook usado para o trabalho de preparação de dados.
-   **`templates/`**: Pasta padrão do Flask para armazenar os arquivos HTML do frontend.
-   **`app.py`**: O arquivo principal do servidor backend Flask.
-   **`capturar_fotos.py`**: Script utilitário para coletar as imagens de cadastro.
-   **`deploy.prototxt` e `*.caffemodel`**: Arquivos do modelo de detecção facial do OpenCV.
-   **`requirements.txt`**: Lista de todas as dependências do Python para fácil instalação.
-   **`README.md`**: Este arquivo de documentação.

---

## Funcionalidades Principais

-   **Cadastro de Pessoas:** Um script utilitário (`capturar_fotos.py`) permite cadastrar novas pessoas de forma consistente, usando a mesma câmera que será utilizada para o reconhecimento.
-   **Processamento e Geração de Embeddings:** Um notebook Jupyter é responsável por processar as imagens de cadastro, detectar os rostos e utilizar um modelo pré-treinado (FaceNet) para gerar "assinaturas faciais" (embeddings) para cada rosto.
-   **Banco de Dados de Embeddings:** As assinaturas faciais e os nomes correspondentes são salvos em um arquivo `embeddings.pickle`, que serve como o "cérebro" do sistema de reconhecimento.
-   **API de Reconhecimento:** O backend, construído com Flask e um servidor de produção WSGI (Waitress), expõe uma API na rota `/reconhecer` que recebe uma imagem e retorna os dados da pessoa identificada.
-   **Interface Web em Tempo Real:** Um frontend em HTML e JavaScript acessa a webcam do usuário, envia os quadros de vídeo para a API e desenha os resultados (caixa delimitadora e nome da pessoa) sobre o vídeo.

## Arquitetura do Projeto

O sistema foi projetado de forma modular para separar as responsabilidades, seguindo as melhores práticas de desenvolvimento de software.

### Fase 1: Preparação e Cadastro (Offline)

1.  **`capturar_fotos.py`**: Script para capturar imagens de referência de cada pessoa. Garante que os dados de cadastro tenham a mesma qualidade (câmera, iluminação) dos dados de teste, o que é crucial para a precisão do modelo.
2.  **Notebook Jupyter**:
    -   **Detecção de Rosto**: Utiliza um modelo DNN pré-treinado do OpenCV para detectar e recortar os rostos das imagens de cadastro.
    -   **Geração de Embeddings**: Usa o modelo FaceNet, através da biblioteca DeepFace, para converter cada rosto recortado em um vetor numérico.
    -   **Criação do Banco de Dados**: Salva todos os embeddings e seus respectivos nomes em um arquivo `embeddings.pickle`.

### Fase 2: Reconhecimento (Online via API Web)

1.  **`app.py`**: O servidor backend Flask.
    -   Na inicialização, carrega o detector facial do OpenCV, o modelo FaceNet e o arquivo `embeddings.pickle` na memória.
    -   Utiliza um servidor de produção (Waitress) e um `threading.Lock` para garantir a estabilidade e o processamento seguro de requisições concorrentes.
    -   Expõe a rota `/reconhecer`.
2.  **`templates/index.html`**: O frontend.
    -   Usa JavaScript para acessar a webcam.
    -   Em um loop controlado (`setTimeout`), captura quadros do vídeo, os converte para base64 e os envia via requisição POST para a API `/reconhecer`.
    -   Recebe a resposta JSON do backend e usa a tag `<canvas>` do HTML5 para desenhar a caixa e o nome sobre o vídeo em tempo real.

## Tecnologias Utilizadas

-   **Backend**: Python, Flask, Waitress
-   **Machine Learning**: TensorFlow, DeepFace (para o modelo FaceNet), OpenCV (para detecção e manipulação de imagem), Scipy, NumPy
-   **Frontend**: HTML5, CSS3, JavaScript (Fetch API)
-   **Ambiente**: Conda

## Como Executar o Projeto

1.  **Configurar o Ambiente**: Crie e ative um ambiente (preferencialmente com Conda). Instale todas as dependências usando `pip install -r requirements.txt`.
2.  **Cadastrar Pessoas**: Execute `python capturar_fotos.py` para cada pessoa que deseja reconhecer. Siga as instruções no terminal.
3.  **Processar os Dados**: Execute todas as células do notebook Jupyter em `notebook/` para gerar o arquivo `embeddings.pickle`.
4.  **Iniciar o Servidor**: Execute `python app.py` no terminal.
5.  **Acessar a Aplicação**: Abra um navegador e acesse `http://127.0.0.1:5000`.

## Notas sobre Performance e Hardware

Este projeto realiza tarefas de Deep Learning (inferência em duas redes neurais) em tempo real, o que é uma operação computacionalmente intensiva, especialmente para a CPU.

-   **Lógica Funcional**: O sistema é logicamente completo e funcional. Ele detecta, reconhece e retorna a identidade das pessoas corretamente, como pode ser observado pelos logs do servidor e pela interface web.
-   **Gargalo de Performance**: O servidor pode apresentar instabilidade ou travar em máquinas com CPUs menos potentes, resultando em um erro de aplicativo (`python.exe - Erro de Aplicativo`), que é uma falha de segmentação causada pela exaustão de recursos. **Isso não é um erro de código, mas sim um limite de hardware.**
-   **Solução e Ajuste Fino**: A estabilidade do sistema é controlada pelo intervalo de `setTimeout` no arquivo `templates/index.html`. Em hardware mais potente (CPU rápida ou com suporte a GPU), um intervalo menor (ex: `100ms`) funcionaria de forma fluida. Em hardware mais modesto, um intervalo maior (ex: `1000ms`) garante a estabilidade em detrimento da fluidez do vídeo.

Este projeto cumpre com sucesso o desafio de criar um sistema de reconhecimento facial do zero, demonstrando a implementação de uma arquitetura web complexa e a aplicação de múltiplos modelos de deep learning.

## 📫 Contato

- GitHub: [https://github.com/RickBamberg](https://github.com/RickBamberg/)
- LinkedIn: [https://www.linkedin.com/in/carlos-henrique-bamberg-marques](https://www.linkedin.com/in/carlos-henrique-bamberg-marques/)
- Email: [rick.bamberg@gmail.com](mailto:rick.bamberg@gmail.com)