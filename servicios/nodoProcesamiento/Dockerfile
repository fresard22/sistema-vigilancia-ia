# servicios/processing_node/Dockerfile

# Usamos la misma estructura que el otro nodo.
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# NOTA: PyTorch (dependencia de ultralytics) puede ser grande.
# La primera vez que se construya esta imagen, puede tardar un poco.
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "procesamiento.py"]