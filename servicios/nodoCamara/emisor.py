import cv2
import pika
import time

# --- 1. Configuración de RabbitMQ ---
BROKER_HOST = 'localhost'
QUEUE_NAME = 'video_frames'

# Conexión al broker
connection = pika.BlockingConnection(pika.ConnectionParameters(host=BROKER_HOST))
channel = connection.channel()

# Se asegura de que la cola exista. Si no, la crea.
channel.queue_declare(queue=QUEUE_NAME)
print(f"Conectado a RabbitMQ y la cola '{QUEUE_NAME}' está lista.")

# --- 2. Inicializar la captura de video ---
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

print("Cámara iniciada. Enviando frames... Presiona Ctrl+C para salir.")

try:
    # --- 3. Bucle principal para capturar y enviar ---
    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo recibir el frame. Saliendo...")
            break

        # --- 4. Codificar el frame para el envío ---
        # No podemos enviar el objeto 'frame' directamente. Lo codificamos a formato JPEG.
        # Esto reduce el tamaño y lo convierte en una secuencia de bytes.
        result, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not result:
            continue

        # --- 5. Publicar el frame en la cola de RabbitMQ ---
        channel.basic_publish(
            exchange='',          # Usamos el exchange por defecto
            routing_key=QUEUE_NAME, # El nombre de nuestra cola
            body=buffer.tobytes() # El cuerpo del mensaje son los bytes de la imagen
        )
        
        print(f"Frame enviado ({len(buffer.tobytes())} bytes)")
        
        # Opcional: Controlar la tasa de frames para no saturar
        time.sleep(0.05) # ~20 FPS

except KeyboardInterrupt:
    print("Programa interrumpido por el usuario.")

finally:
    # --- 6. Liberar recursos ---
    print("Cerrando programa y conexiones...")
    cap.release()
    connection.close()