import cv2
import pika
import numpy as np

# --- 1. Configuración de RabbitMQ ---
BROKER_HOST = 'localhost'
QUEUE_NAME = 'video_frames'

# Conexión al broker
connection = pika.BlockingConnection(pika.ConnectionParameters(host=BROKER_HOST))
channel = connection.channel()
channel.queue_declare(queue=QUEUE_NAME)
print(f"Conectado a RabbitMQ. Esperando frames en la cola '{QUEUE_NAME}'...")

# --- 2. Función Callback (se ejecuta por cada mensaje recibido) ---
def callback(ch, method, properties, body):
    """
    Esta función procesa el mensaje recibido.
    """
    # Convertir los bytes recibidos de vuelta a un array de numpy
    nparr = np.frombuffer(body, np.uint8)
    
    # Decodificar el array de numpy a una imagen de OpenCV
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Mostrar la imagen recibida
    if frame is not None:
        cv2.imshow('Video Recibido', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # Esta lógica de salida es simple, en un caso real sería más robusta
            channel.stop_consuming()
    
    # Confirmar que el mensaje fue procesado
    ch.basic_ack(delivery_tag=method.delivery_tag)

# --- 3. Consumir mensajes de la cola ---
# Le decimos al canal que use nuestra función 'callback' para procesar mensajes de la cola
channel.basic_consume(queue=QUEUE_NAME, on_message_callback=callback)

try:
    # Iniciar el consumo de mensajes. Este es un bucle bloqueante.
    channel.start_consuming()
except KeyboardInterrupt:
    print("Consumo interrumpido.")

finally:
    # Cerrar todo limpiamente
    connection.close()
    cv2.destroyAllWindows()