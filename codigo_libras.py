import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model

# Inicializando captura de vídeo e componentes
video_capture = cv2.VideoCapture(0)
hand_recognition = mp.solutions.hands.Hands(max_num_hands=1)
gesture_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
recognition_model = load_model('keras_model.h5')
model_input = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

while video_capture.isOpened():
    is_frame_read, video_frame = video_capture.read()
    if not is_frame_read:
        break

    # Conversão de cor e detecção de mãos
    frame_rgb = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
    hand_detections = hand_recognition.process(frame_rgb)
    frame_h, frame_w, _ = video_frame.shape

    if hand_detections.multi_hand_landmarks:
        for landmarks in hand_detections.multi_hand_landmarks:
            x_right, y_bottom = 0, 0
            x_left, y_top = frame_w, frame_h
            
            # Calculando limites para o retângulo
            for point in landmarks.landmark:
                x_coord, y_coord = int(point.x * frame_w), int(point.y * frame_h)
                x_right, y_bottom = max(x_right, x_coord), max(y_bottom, y_coord)
                x_left, y_top = min(x_left, x_coord), min(y_top, y_coord)
            
            # Desenhando um retângulo em torno da mão detectada
            cv2.rectangle(video_frame, (x_left - 50, y_top - 50), (x_right + 50, y_bottom + 50), (0, 255, 0), 2)

            try:
                # Preparando a imagem para o modelo de reconhecimento
                hand_crop = video_frame[y_top - 50:y_bottom + 50, x_left - 50:x_right + 50]
                resized_crop = cv2.resize(hand_crop, (224, 224))
                normalized_crop = (resized_crop.astype(np.float32) / 127.0) - 1
                model_input[0] = normalized_crop
                
                # Realizando predição
                model_predictions = recognition_model.predict(model_input)
                predicted_gesture = np.argmax(model_predictions)
                gesture_label = gesture_labels[predicted_gesture]

                # Mostrando a classe detectada na tela
                cv2.putText(video_frame, gesture_label, (x_left - 50, y_top - 65), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 0, 255), 5)

            except Exception as err:
                print(f"Erro ao processar imagem: {err}")
                continue

    # Exibindo o quadro final com a detecção
    cv2.imshow('Reconhecimento de Gestos', video_frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Pressionar Esc para sair
        break

video_capture.release()
cv2.destroyAllWindows()
