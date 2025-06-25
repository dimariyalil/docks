FROM nvidia/cuda:12.9.0-runtime-ubuntu22.04

RUN apt-get update && \
    apt-get install -y python3 python3-pip git ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) зависимости FastAPI-обёртки
COPY lighthouse-api/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# 2) исходники Lighthouse, лежащие в корне контекста
COPY lighthouse-src /lighthouse
RUN pip3 install --no-cache-dir -e /lighthouse     # editable-режим

# 3) сам файл API
COPY lighthouse-api/app.py .

EXPOSE 5000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]
