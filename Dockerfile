FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libxrender1 \
    libxext6 \
    libsm6 \
    libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir tmap-viz && \
    git clone https://github.com/reymond-group/map4.git /tmp/map4 && \
    pip install --no-cache-dir /tmp/map4 && \
    rm -rf /tmp/map4

# Code only -- models/ and nn_data/ are mounted at run time (see README)
COPY ppb4_predict.py .
COPY templates/ ./templates/
COPY static/ ./static/

EXPOSE 5000

ENV FLASK_HOST=0.0.0.0

CMD ["python", "ppb4_predict.py"]
