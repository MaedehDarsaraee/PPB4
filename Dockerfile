FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libxrender1 \
    libxext6 \
    libsm6 \
    libfontconfig1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir tmap-viz && \
    git clone https://github.com/reymond-group/map4.git /tmp/map4 && \
    pip install --no-cache-dir /tmp/map4 && \
    rm -rf /tmp/map4

# Code only -- models/ and nn_data/ are mounted at run time (see README).
# models/ must contain the 8 separate models, the joint three-state model
# (ppb4_joint_ecfp4_full_model.h5) and the four label/metadata files:
#   PPB4_ACTIVE_DNNTARLABELS.txt   PPB4_INACTIVE_DNNTARLABELS.txt
#   PPB4_JOINT_DNNTARLABELS.txt    PPB4_TARGETSDETAILS.txt
#   PPB4_TARGETCLASSIFICATION.txt
COPY ppb4_predict.py .
COPY templates/ ./templates/
COPY static/ ./static/

EXPOSE 5000

ENV FLASK_HOST=0.0.0.0 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    TF_ENABLE_ONEDNN_OPTS=0

# The nine models load at import time, so the container is not ready
# immediately. start-period covers that.
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -fsS http://localhost:5000/ || exit 1

CMD ["python", "ppb4_predict.py"]
