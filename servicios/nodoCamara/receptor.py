import cv2
import pika
import numpy as np
from ultralytics import YOLO

# --- 1. Cargar el modelo de IA (YOLOv8) ---
# 'yolov8n.pt' es un modelo pequeño y rápido, ideal para empezar.
# La primera vez que se ejecute, la biblioteca lo descargará automáticamente.
print("Cargando modelo de IA (YOLOv8)...")
model = YOLO('yolov8n.pt')
print("Modelo cargado exitosamente.")

# --- 2. Configuración de RabbitMQ ---
BROKER_HOST = 'localhost'
QUEUE_NAME = 'video_frames'

# Conexión al broker
try:
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=BROKER_HOST))
    channel = connection.channel()
    channel.queue_declare(queue=QUEUE_NAME)
    print(f"Conectado a RabbitMQ. Esperando frames en la cola '{QUEUE_NAME}'...")
except pika.exceptions.AMQPConnectionError as e:
    print(f"Error: No se pudo conectar a RabbitMQ en '{BROKER_HOST}'. Asegúrate de que el contenedor Docker esté corriendo.")
    print(f"Detalle del error: {e}")
    exit()

# --- 3. Función Callback para procesar cada frame ---
def callback(ch, method, properties, body):
    """
    Esta función se ejecuta por cada frame recibido de RabbitMQ.
    """
    # a. Decodificar la imagen
    nparr = np.frombuffer(body, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is not None:
        # b. PROCESAMIENTO CON IA: Realizar la detección de objetos
        results = model(frame, stream=True, verbose=False) # verbose=False para no imprimir tanto detalle

        # c. Procesar los resultados
        for result in results:
            # Obtener las cajas delimitadoras y las clases de los objetos detectados
            boxes = result.boxes
            for box in boxes:
                # Obtener las coordenadas de la caja
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Obtener la clase del objeto y el nombre
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                
                # d. DIBUJAR SOLO SI ES UNA PERSONA
                if class_name.lower() == 'person':
                    # Dibujar un rectángulo verde alrededor de la persona
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Poner el texto "Persona" encima de la caja
                    label = f'Persona {box.conf[0]:.2f}' # Confianza de la detección
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # e. Mostrar el frame procesado
        cv2.imshow('Processing Node - Deteccion de Personas', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            channel.stop_consuming()
    
    # Confirmar que el mensaje fue procesado
    ch.basic_ack(delivery_tag=method.delivery_tag)

# --- 4. Consumir mensajes de la cola ---
channel.basic_consume(queue=QUEUE_NAME, on_message_callback=callback)

try:
    channel.start_consuming()
except KeyboardInterrupt:
    print("Consumo interrumpido por el usuario.")
finally:
    print("Cerrando conexiones y ventanas...")
    if 'connection' in locals() and connection.is_open:
        connection.close()
    cv2.destroyAllWindows()