import cv2
import pika
import traceback
import time
import os
import ssl # Importamos la biblioteca ssl

# --- 1. Configuración de RabbitMQ SEGURA ---
BROKER_HOST = os.getenv('RABBITMQ_HOST', 'localhost')
QUEUE_NAME = 'video_frames'
RABBITMQ_USER = os.getenv('RABBITMQ_USER','usuario')
RABBITMQ_PASS = os.getenv('RABBITMQ_PASS','pass')# ¡Usa una contraseña segura!
BROKER_PORT = 5672 # <-- El puerto normal, no el 5671

# Simplemente creamos las credenciales
credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)

try:
    # La conexión no tiene NADA de SSL. Es directa.
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

except Exception as e:
    print("\n--- ERROR DETALLADO ---")
    traceback.print_exc()
    print("-----------------------\n")
    exit()
# --- El resto del script (captura de cámara, envío de frames) es exactamente el mismo ---
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

print("Cámara iniciada. Enviando frames de forma segura... Presiona Ctrl+C para salir.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        result, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not result:
            continue

        channel.basic_publish(exchange='', routing_key=QUEUE_NAME, body=buffer.tobytes())
        print(f"Frame enviado ({len(buffer.tobytes())} bytes)")
        time.sleep(0.05)

except KeyboardInterrupt:
    print("Programa interrumpido.")
finally:
    print("Cerrando programa y conexión segura...")
    cap.release()
    if 'connection' in locals() and connection.is_open:
        connection.close()
