# servicios/nodoCamara/Dockerfile

# 1. Usar una imagen base oficial y ligera de Python.
FROM python:3.11

# 2. Establecer un directorio de trabajo dentro del contenedor.
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
 && rm -rf /var/lib/apt/lists/*

# 3. Copiar PRIMERO el archivo de requisitos. Esto aprovecha el caché de Docker.
# Si los requisitos no cambian, Docker no volverá a instalar todo cada vez.
COPY requirements.txt .

# 4. Instalar las dependencias del proyecto.
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copiar el resto del código de la aplicación al directorio de trabajo.
COPY . .

# 6. El comando que se ejecutará cuando el contenedor se inicie.
CMD ["python", "emisor.py"]