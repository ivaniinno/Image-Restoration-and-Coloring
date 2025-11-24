FROM python:3.9-slim
WORKDIR /app
COPY . /app
RUN apt-get update && apt-get install -y build-essential libglib2.0-0 libsm6 libxrender1 libxext6 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8001
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
