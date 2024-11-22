import cv2
import numpy as np
import os


video_path = "video_arara.mp4"

# Pasta com os frames das araras detectadas
output_dir = "frames_araras_azuis"
os.makedirs(output_dir, exist_ok=True)

# Função para detectar o azul 
def detect_blue_arara(frame):
   
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([90, 50, 50])   
    upper_blue = np.array([130, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    return cv2.countNonZero(mask) > 500

# Carrega o vídeo
cap = cv2.VideoCapture(video_path)
frame_count = 0
saved_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Processar um frame a cada 60 quadros
    if frame_count % 60 == 0:
        if detect_blue_arara(frame):  # Se azul for detectado
            frame_name = f"arara_azul_{frame_count}.jpg"
            cv2.imwrite(os.path.join(output_dir, frame_name), frame)
            saved_count += 1

cap.release()
print(f"Processamento concluído! {saved_count} frames com araras azuis salvos em '{output_dir}'.")
