# --- IMPORTS ---
import cv2
import pika
import numpy as np
from ultralytics import YOLO
import time
import os
from collections import defaultdict
import traceback
import ssl

print("--- [PROCESAMIENTO] Script iniciado. ---")

# --- 1. Cargar el modelo de IA (YOLOv8) ---
print("--- [PROCESAMIENTO] Cargando modelo de IA... ---")
model = YOLO('yolov8n.pt')
print("--- [PROCESAMIENTO] Modelo cargado exitosamente. ---")

# ... (El resto de la configuración no cambia)
SPEED_THRESHOLD = 50
PROXIMITY_THRESHOLD = 50
tracked_people = defaultdict(lambda: {'last_pos': None, 'last_time': None, 'is_alert': False})

# --- CONFIGURACIÓN Y CONEXIÓN ---
print("--- [PROCESAMIENTO] Leyendo configuración de entorno... ---")
RABBITMQ_HOST = os.getenv('RABBITMQ_HOST', 'localhost')
RABBITMQ_PORT = int(os.getenv('RABBITMQ_PORT', 5671))
QUEUE_NAME = 'camera_frames'
CA_CERT_PATH = os.getenv('CA_CERT')
CLIENT_CERT_PATH = os.getenv('CLIENT_CERT')
CLIENT_KEY_PATH = os.getenv('CLIENT_KEY')

print(f"--- [PROCESAMIENTO] Host: {RABBITMQ_HOST}, Puerto: {RABBITMQ_PORT} ---")

ssl_options = None
if RABBITMQ_PORT == 5671:
    print("--- [PROCESAMIENTO] Configurando conexión TLS... ---")
    try:
        context = ssl.create_default_context(cafile=CA_CERT_PATH)
        context.load_cert_chain(certfile=CLIENT_CERT_PATH, keyfile=CLIENT_KEY_PATH)
        ssl_options = pika.SSLOptions(context, server_hostname=RABBITMQ_HOST)
        print("--- [PROCESAMIENTO] Contexto SSL creado exitosamente. ---")
    except Exception as e:
        print(f"--- [PROCESAMIENTO] Error fatal creando el contexto SSL: {e} ---")
        exit(1)

connection = None
attempts = 0
print("--- [PROCESAMIENTO] Iniciando bucle de conexión... ---")

# ... (El bucle de conexión con reintentos se mantiene igual)
while attempts < 10 and not connection:
    try:
        params = pika.ConnectionParameters(
            host=RABBITMQ_HOST, port=RABBITMQ_PORT, ssl_options=ssl_options,
            credentials=pika.credentials.ExternalCredentials(), heartbeat=600,
            blocked_connection_timeout=300)
        connection = pika.BlockingConnection(params)
    except Exception as e:
        print(f"--- [PROCESAMIENTO] Falló el intento de conexión {attempts + 1} ---")
        attempts += 1
        time.sleep(5)

if not connection:
    print("--- [PROCESAMIENTO] No se pudo conectar. Saliendo. ---")
    exit(1)

channel = connection.channel()

print("--- [PROCESAMIENTO] ¡Conexión exitosa con RabbitMQ! ---")

# --- FUNCIÓN CALLBACK ---
def callback(ch, method, properties, body):
    global tracked_people
    
    # ... (toda tu lógica de IA se mantiene igual)
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

        # En un entorno de servidor real, esta línea debería ser removida o manejada de otra forma.
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

