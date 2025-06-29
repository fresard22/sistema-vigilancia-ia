import cv2
import pika
import numpy as np
from ultralytics import YOLO
import time
import os
from collections import defaultdict
import traceback # Usado para imprimir errores detallados

# --- 1. Cargar el modelo de IA (YOLOv8) ---
# Esta parte no cambia.
print("Cargando modelo de IA (YOLOv8)...")
model = YOLO('yolov8n.pt')
print("Modelo cargado exitosamente.")

# --- 2. Configuración de Heurísticas de Agresión ---
# Esta parte no cambia.
SPEED_THRESHOLD = 50
PROXIMITY_THRESHOLD = 50

# --- 3. Almacenamiento de datos de seguimiento ---
# Esta parte no cambia.
tracked_people = defaultdict(lambda: {'last_pos': None, 'last_time': None, 'is_alert': False})


# =================================================================
# --- 4. Bloque de Conexión SIMPLE (SIN TLS) ---
# Esta es la sección que hemos simplificado radicalmente.
# =================================================================
BROKER_HOST = os.environ.get('RABBITMQ_HOST', 'rabbitmq')
BROKER_HOST="rabbitmq"
BROKER_PORT = 5672 # <-- El puerto normal, no el 5671 de TLS
QUEUE_NAME = 'video_frames'
RABBITMQ_USER = os.environ.get('RABBITMQ_USER','usuario')
RABBITMQ_PASS = os.environ.get('RABBITMQ_PASS','pass')# ¡Usa una contraseña segura!

# Simplemente creamos las credenciales de usuario y contraseña
credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)

try:
    # La conexión es directa, sin parámetros SSL
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(
            host=BROKER_HOST,
            port=BROKER_PORT,
            credentials=credentials
        )
    )
    channel = connection.channel()
    channel.queue_declare(queue=QUEUE_NAME)
    print("--- ÉXITO --- Conexión SIMPLE a RabbitMQ establecida.")
    print(f"Esperando frames en la cola '{QUEUE_NAME}'...")

except Exception as e:
    print("\n--- ERROR DETALLADO EN LA CONEXIÓN ---")
    traceback.print_exc()
    print("---------------------------------------\n")
    exit() # Detiene el script si la conexión falla
# =================================================================


# --- 5. Función Callback (Esta no cambia en absoluto) ---
# Toda tu lógica de IA está segura aquí.
def callback(ch, method, properties, body):
    global tracked_people
    
    nparr = np.frombuffer(body, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is not None:
        current_time = time.time()
        results = model.track(frame, persist=True, verbose=False)

        current_frame_detections = {}
        alert_ids = set()

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            classes = results[0].boxes.cls.cpu().numpy().astype(int)

            for box, track_id, cls in zip(boxes, ids, classes):
                if model.names[cls].lower() == 'person':
                    x1, y1, x2, y2 = box
                    center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    current_pos = (center_x, center_y)
                    current_frame_detections[track_id] = {'pos': current_pos, 'box': box}
                    
                    if tracked_people[track_id]['last_pos'] is not None:
                        last_pos = tracked_people[track_id]['last_pos']
                        distance = np.linalg.norm(np.array(current_pos) - np.array(last_pos))
                        if distance > SPEED_THRESHOLD:
                            alert_ids.add(track_id)
                    
                    tracked_people[track_id]['last_pos'] = current_pos
                    tracked_people[track_id]['last_time'] = current_time

            detected_ids = list(current_frame_detections.keys())
            if len(detected_ids) > 1:
                for i in range(len(detected_ids)):
                    for j in range(i + 1, len(detected_ids)):
                        id1, id2 = detected_ids[i], detected_ids[j]
                        pos1 = np.array(current_frame_detections[id1]['pos'])
                        pos2 = np.array(current_frame_detections[id2]['pos'])
                        distance = np.linalg.norm(pos1 - pos2)
                        if distance < PROXIMITY_THRESHOLD:
                            alert_ids.add(id1)
                            alert_ids.add(id2)
        
        for track_id, data in current_frame_detections.items():
            x1, y1, x2, y2 = data['box']
            label = f'Persona {track_id}'
            
            if track_id in alert_ids:
                color = (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(frame, label + " [ALERTA]", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            else:
                color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        stale_ids = [tid for tid, data in tracked_people.items() if current_time - data['last_time'] > 5]
        for tid in stale_ids:
            del tracked_people[tid]

        cv2.imshow('Processing Node - Deteccion de Agresion V1', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            channel.stop_consuming()
    
    ch.basic_ack(delivery_tag=method.delivery_tag)

# --- 6. Consumir mensajes de la cola (Esta parte no cambia) ---
channel.basic_consume(queue=QUEUE_NAME, on_message_callback=callback)
try:
    channel.start_consuming()
except KeyboardInterrupt:
    print("Consumo interrumpido.")
finally:
    if 'connection' in locals() and connection.is_open:
        connection.close()
    cv2.destroyAllWindows()