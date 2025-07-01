# Sistema de Vigilancia IA con Detección de Agresión

Este proyecto es un sistema distribuido diseñado para monitorear flujos de video en tiempo real desde cámaras estratégicamente ubicadas. Utiliza un modelo de inteligencia artificial para detectar personas y analizar sus movimientos, con el fin de identificar comportamientos potencialmente agresivos y generar alertas tempranas.

## Características

* **Procesamiento de Video en Tiempo Real:** Captura de video desde múltiples fuentes.
* **Detección de Personas:** Utiliza el modelo YOLOv8 para identificar personas en el video.
* **Detección de Agresión (Heurística V1):** Analiza la velocidad y proximidad de las personas para detectar anomalías.
* **Arquitectura Distribuida:** Basado en microservicios que se comunican a través de un broker de mensajes (RabbitMQ).
* **Contenerizado con Docker:** Todo el sistema se despliega fácilmente con Docker y Docker Compose, garantizando la portabilidad.

## Arquitectura del Sistema

El sistema consta de tres servicios principales orquestados por Docker Compose:

1.  `rabbitmq`: El broker de mensajes que gestiona la comunicación.
2.  `camera-node`: Un servicio que captura el video de una cámara y lo publica en el broker.
3.  `processing-node`: Un servicio que consume el video del broker, ejecuta los modelos de IA y (actualmente) muestra el resultado en una ventana de depuración.

## Requisitos Previos

Antes de ejecutar el proyecto, asegúrate de tener instalado lo siguiente en tu máquina:

1.  **Git:** Para clonar el repositorio.
2.  **Docker y Docker Compose:** Es necesario tener Docker Engine y el plugin de Compose (V2, se usa con `docker compose`). [Instrucciones de instalación de Docker](https://docs.docker.com/engine/install/).
3.  **Sistema Operativo Linux (Recomendado):** La funcionalidad para mostrar la ventana de video (`cv2.imshow`) está configurada para funcionar en un entorno Linux con un servidor gráfico X11 (como Ubuntu, Fedora, etc.). En macOS o Windows, se requerirían configuraciones adicionales (ej. XQuartz en Mac, WSLg en Windows) que no están cubiertas en esta guía.
4.  **Una Cámara Web:** El sistema necesita acceso a una cámara conectada, típicamente en `/dev/video0`.

## Cómo Ejecutar el Proyecto

Sigue estos pasos para levantar todo el sistema:

1.  **Clonar el repositorio:**
    ```bash
    git clone <URL_DE_TU_REPOSITORIO_EN_GITHUB>
    cd <nombre-de-la-carpeta-del-proyecto>
    ```

2.  **Crear tu archivo de entorno:**
    Este proyecto usa un archivo `.env` para las credenciales. Copia el archivo de ejemplo:
    ```bash
    cp .env.example .env
    ```

3.  **(Solo para Linux) Dar permisos para la GUI:**
    Para que la ventana de video del `processing-node` pueda aparecer en tu escritorio, ejecuta este comando en tu terminal. Esto solo se necesita hacer una vez por sesión.
    ```bash
    xhost +local:docker
    ```

4.  **Construir y ejecutar los contenedores:**
    Este comando construirá las imágenes de Docker (la primera vez puede tardar varios minutos por la descarga e instalación de PyTorch) y levantará todos los servicios.
    ```bash
    docker compose up --build
    ```

¡Listo! Después de que los contenedores se inicien, deberías ver la ventana del `processing-node` con el video de tu cámara y las detecciones. Para detener todo el sistema, presiona `Ctrl + C` en la terminal.
