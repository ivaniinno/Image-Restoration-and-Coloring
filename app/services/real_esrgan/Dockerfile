FROM python:3.9-slim
WORKDIR /app
COPY . /app
# Headless build: opencv-python-headless does not need X/GL libs; keep only essentials.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential libglib2.0-0 \
	&& rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8002
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8002"]
