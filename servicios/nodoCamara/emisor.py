import pika
import time
import os
import cv2
import base64
import ssl
import traceback
import pika.credentials

## ----------------------------------------------------------------
## LEER CONFIGURACIÓN DESDE VARIABLES DE ENTORNO
## ----------------------------------------------------------------
RABBITMQ_HOST = os.getenv('RABBITMQ_HOST', 'localhost')
RABBITMQ_PORT = int(os.getenv('RABBITMQ_PORT', 5671)) 
CAMERA_INDEX = int(os.getenv('CAMERA_INDEX', 0))
CAMERA_ID = os.getenv('CAMERA_ID','Camara Notebook')

CA_CERT_PATH = os.getenv('CA_CERT')
CLIENT_CERT_PATH = os.getenv('CLIENT_CERT')
CLIENT_KEY_PATH = os.getenv('CLIENT_KEY')

## ----------------------------------------------------------------
## CONFIGURACIÓN DE CONEXIÓN SEGURA (TLS)
## ----------------------------------------------------------------
ssl_options = None
if RABBITMQ_PORT == 5671:
    print("Configurando conexión TLS...")
    try:
        context = ssl.create_default_context(cafile=CA_CERT_PATH)
        context.load_cert_chain(certfile=CLIENT_CERT_PATH, keyfile=CLIENT_KEY_PATH)
        ssl_options = pika.SSLOptions(context, server_hostname=RABBITMQ_HOST)
        print("Contexto SSL creado exitosamente.")
    except Exception as e:
        print(f"Error fatal creando el contexto SSL: {e}")
        exit(1)

## ----------------------------------------------------------------
## LÓGICA DE CONEXIÓN CON REINTENTOS
## ----------------------------------------------------------------
connection = None
attempts = 0
print(f"Intentando conectar a RabbitMQ en '{RABBITMQ_HOST}:{RABBITMQ_PORT}'...")

while attempts < 10 and not connection:
    try:
        params = pika.ConnectionParameters(
            host=RABBITMQ_HOST,
            port=RABBITMQ_PORT,
            ssl_options=ssl_options,
            credentials=pika.credentials.ExternalCredentials(),
            heartbeat=600,
            blocked_connection_timeout=300
        )
        connection = pika.BlockingConnection(params)
        print("¡Conexión exitosa con RabbitMQ!")
        
     #Capturar errores específicos de verificación de certificados
    except (ssl.SSLCertVerificationError, ssl.SSLError) as e:
        print(f"--- ERROR DE VERIFICACIÓN SSL ---")
        traceback.print_exc()
        attempts += 1
        print(f"Reintentando... (Intento {attempts}/10)")
        time.sleep(5)
        
    #Capturar otros errores de conexión de Pika
    except pika.exceptions.AMQPConnectionError as e:
        print(f"--- ERROR DE CONEXIÓN AMQP ---")
        traceback.print_exc()
        attempts += 1
        print(f"Reintentando... (Intento {attempts}/10)")
        time.sleep(5)
        
    #Capturar cualquier otro error
    except Exception as e:
        print(f"--- ERROR INESPERADO ---")
        traceback.print_exc()
        attempts += 1
        print(f"Reintentando... (Intento {attempts}/10)")
        time.sleep(5)

if not connection:
    print("No se pudo establecer conexión con RabbitMQ después de varios intentos. Saliendo.")
    exit(1)

## ----------------------------------------------------------------
## LÓGICA PRINCIPAL DEL EMISOR
## ----------------------------------------------------------------
try:

    #Conexión a la cola de rabbitmq
    channel = connection.channel()
    channel.queue_declare(queue='camera_frames', durable=True)
    print("Canal y cola 'camera_frames' declarados.")

    #Abrir cámara
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise IOError(f"No se puede abrir la cámara con índice {CAMERA_INDEX}")

    print("Cámara abierta. Empezando a enviar frames...")
    while True:

        #Lectura de frames
        ret, frame = cap.read()
        if not ret:
            print("No se pudo leer el frame de la cámara. Terminando bucle.")
            break
        
        #Escribir y codificar el frame a
        result, buffer = cv2.imencode('.jpg', frame,[int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not result:
            continue

        props = pika.BasicProperties(
            app_id=CAMERA_ID,
            delivery_mode=2,
            content_type='image/jpeg'
        )

        #Publicar el mensaje
        channel.basic_publish(
            exchange='',
            routing_key='camera_frames',
            body=buffer.tobytes(),
            properties=props)
        


        #Controlar la tasa de envío (ej. 10 frames por segundo)
        time.sleep(0.1)

#Para terminar la conexión
except KeyboardInterrupt:
    print("Interrupción por teclado detectada. Cerrando conexión.")
except Exception as e:
    print(f"Ocurrió un error en el bucle principal: {e}")
finally:
    if connection and not connection.is_closed:
        print("Cerrando la conexión con RabbitMQ.")
        connection.close()
    if 'cap' in locals() and cap.isOpened():
        cap.release()