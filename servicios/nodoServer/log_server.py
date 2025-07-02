# --- IMPORTS ---
import pika
import time
import os
import ssl
import traceback
import json

print("--- [LOG SERVER] Iniciando servicio de logs. ---")

## ----------------------------------------------------------------
## LEER VARIABLES DE ENTORNO
## ----------------------------------------------------------------
RABBITMQ_HOST = os.getenv('RABBITMQ_CONNECT_HOST', 'rabbitmq-1')
RABBITMQ_TLS_HOST = os.getenv('RABBITMQ_TLS_HOST', 'rabbitmq-1')
RABBITMQ_PORT = int(os.getenv('RABBITMQ_PORT', 5671))
CA_CERT_PATH = os.getenv('CA_CERT')
CLIENT_CERT_PATH = os.getenv('CLIENT_CERT')
CLIENT_KEY_PATH = os.getenv('CLIENT_KEY')

LOG_FILE_PATH = 'logs/system_alerts.log'

#Asegurarse de que la carpeta de logs exista
os.makedirs('logs', exist_ok=True)

## ----------------------------------------------------------------
## CONFIGURACIÓN TLS
## ----------------------------------------------------------------
ssl_options = None
if RABBITMQ_PORT == 5671:
    print("--- [LOG SERVER] Configurando conexión TLS... ---")
    try:
        context = ssl.create_default_context(cafile=CA_CERT_PATH)
        context.load_cert_chain(certfile=CLIENT_CERT_PATH, keyfile=CLIENT_KEY_PATH)
        ssl_options = pika.SSLOptions(context, server_hostname=RABBITMQ_TLS_HOST)
        print("--- [LOG SERVER] Contexto SSL creado exitosamente. ---")
    except Exception as e:
        print(f"--- [LOG SERVER] Error fatal creando el contexto SSL: {e}")
        exit(1)

## ----------------------------------------------------------------
## LÓGICA DE CONEXIÓN CON REINTENTOS
## ----------------------------------------------------------------
connection = None
attempts = 0
print(f"--- [LOG SERVER] Intentando conectar a RabbitMQ en '{RABBITMQ_HOST}:{RABBITMQ_PORT}'...")

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
        print("--- [LOG SERVER] ¡Conexión exitosa con RabbitMQ! ---")
    except Exception as e:
        print(f"--- [LOG SERVER] Falló el intento de conexión {attempts + 1}. Error: {e}")
        attempts += 1
        time.sleep(5)

if not connection:
    print("--- [LOG SERVER] No se pudo conectar a RabbitMQ después de varios intentos. Saliendo.")
    exit(1)


## ----------------------------------------------------------------
## LÓGICA DE ALERTAS 
## ----------------------------------------------------------------
def alert_callback(ch, method, properties, body):
    camera_id = properties.app_id if properties and properties.app_id else "Cámara desconocida"
    try:
        #Decodificar el mensaje JSON
        alert_data = json.loads(body)
        alert_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(alert_data.get('timestamp')))
        
        # Formatear y escribir el log
        log_message = f"ALERTA: Agresión detectada a las {alert_time} desde {camera_id}\n"
        
        with open(LOG_FILE_PATH, 'a') as log_file:
            log_file.write(log_message)
            
        
        ch.basic_ack(delivery_tag=method.delivery_tag)
        
    except Exception as e:
        print("Error procesando mensaje de alerta:")
        traceback.print_exc()

#Consumir mensajes de la cola
try:
    channel = connection.channel()
    queue_name = 'alerts_log'
    channel.queue_declare(queue=queue_name, durable=True)
    
    channel.basic_consume(queue=queue_name, on_message_callback=alert_callback)
    
    print(f"--- [LOG SERVER] Esperando alertas en la cola '{queue_name}'... ---")
    channel.start_consuming()

except KeyboardInterrupt:
    print("--- [LOG SERVER] Servicio detenido. ---")
finally:
    if connection and not connection.is_closed:
        connection.close()