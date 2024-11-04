import cv2
import mediapipe as mp

# Inicializa a captura de vídeo (webcam)
video = cv2.VideoCapture(0)

# Inicializa a solução de detecção de mãos do MediaPipe
hands = mp.solutions.hands
Hands = hands.Hands(max_num_hands=1)  # Limita a detecção a uma só mão
mpDesenho = mp.solutions.drawing_utils  # Utilitário para desenhar as conexões dos pontos da mão

while True:
    # Lê a imagem da webcam (frame por frame)
    sucess, img = video.read()

    # Converte a imagem de BGR (padrão do OpenCV) para RGB (necessário para o MediaPipe)
    fRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Processa a imagem RGB para detectar as mãos e seus pontos
    resultado = Hands.process(fRGB)
    handPoints = resultado.multi_hand_landmarks  # Pega os pontos (landmarks) da mão detectada
    
    # Pega as dimensões da imagem para usar no mapeamento dos pontos
    h, w, _ = img.shape
    coordenadas = []  # Lista para armazenar as coordenadas dos pontos da mão

    if handPoints:  # Se algum ponto de mão foi detectado
        for pontosMao in handPoints:
            # Desenha os pontos e as conexões entre eles na imagem
            mpDesenho.draw_landmarks(img, pontosMao, hands.HAND_CONNECTIONS)

            # Para cada ponto da mão, pegar o índice (id) e as coordenadas (x, y) normalizadas
            for id, cord in enumerate(pontosMao.landmark):
                # Converte as coordenadas normalizadas para valores reais (baseados no tamanho da imagem)
                cx, cy = int(cord.x * w), int(cord.y * h)
                # Adiciona as coordenadas convertidas à lista de pontos
                coordenadas.append((cx, cy))

            # Lista dos índices que representam as pontas dos dedos (indicador, médio, anelar, mínimo)
            finger = [8, 12, 16, 20]
            cont = 0  # Contador para quantos dedos estão levantados

            if coordenadas:  # Se a lista de coordenadas não estiver vazia
                # Verifica se o dedão está levantado (comparando a posição do ponto 4 com o ponto 3)
                if coordenadas[4][0] < coordenadas[3][0]:
                    cont += 1

                # Verifica os outros dedos (se a ponta do dedo está acima da articulação do meio)
                for x in finger:
                    if coordenadas[x][1] < coordenadas[x-2][1]:
                        cont += 1

            # Desenha um retângulo para exibir o número de dedos levantados
            cv2.rectangle(img, (80, 10), (200, 110), (255, 0, 0), -1)
            # Coloca o número de dedos levantados dentro do retângulo
            cv2.putText(img, str(cont), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 5)

    # Mostra a imagem com os desenhos e o contador
    cv2.imshow('Imagem', img)
    cv2.waitKey(1)

    # Se a tecla 'r' for pressionada, encerra o loop e fecha o programa
    if cv2.waitKey(1) & 0xFF == ord('r'):
        break

# Libera a câmera e fecha as janelas
video.release()
cv2.destroyAllWindows()
