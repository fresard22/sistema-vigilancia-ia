import cv2

# --- 1. Inicializar la captura de video ---
# El argumento 0 indica que queremos usar la cámara web por defecto.
# Si tuvieras varias cámaras, podrías probar con 1, 2, etc.
cap = cv2.VideoCapture(0)

# Verificar si la cámara se abrió correctamente
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

print("Cámara iniciada. Presiona la tecla 'q' para salir.")

# --- 2. Bucle principal para leer y mostrar frames ---
while True:
    # Leer un frame (una imagen) de la cámara.
    # 'ret' es un booleano (True si la lectura fue exitosa, False si no).
    # 'frame' es la imagen capturada.
    ret, frame = cap.read()

    # Si no se pudo leer el frame (ej. se desconectó la cámara), salimos del bucle.
    if not ret:
        print("No se pudo recibir el frame. Saliendo...")
        break

    # --- 3. Mostrar el frame en una ventana ---
    # 'cv2.imshow()' muestra la imagen en una ventana.
    # El primer argumento es el nombre de la ventana (puedes poner lo que quieras).
    # El segundo argumento es la imagen (el frame) a mostrar.
    cv2.imshow('Camara del Notebook', frame)

    # --- 4. Esperar por una tecla para salir ---
    # 'cv2.waitKey(1)' espera 1 milisegundo por una tecla.
    # Se comprueba si la tecla presionada fue 'q'.
    # El '0xFF == ord('q')' es la forma estándar de hacerlo compatible con todos los sistemas.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 5. Liberar recursos ---
# Cuando salimos del bucle, es MUY IMPORTANTE liberar la cámara y cerrar las ventanas.
print("Cerrando programa...")
cap.release()
cv2.destroyAllWindows()