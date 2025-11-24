import os
import time
import base64
import io
import psutil
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.middleware.cors import CORSMiddleware
import numpy as np
import torch
from PIL import Image
import cv2

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

def apply_degradation(img):
    img_array = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img_pil = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
    
    return img_pil

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])

import sys
sys.path.insert(0, '/app/real_esrgan')
from RealESRGAN.model import RealESRGAN
from PIL import Image

REAL_WEIGHTS_PATH = os.environ.get('REAL_WEIGHTS_PATH', '/app/weights')
INTERNAL_WEIGHTS_PATH = '/app/real_esrgan/weights'

# Prometheus metrics
REQUESTS = Counter('real_esrgan_requests_total', 'Total number of real-esrgan requests')
REQUEST_LATENCY = Histogram('real_esrgan_request_latency_seconds', 'Latency for real-esrgan requests')
CPU_GAUGE = Gauge('real_esrgan_cpu_percent', 'CPU percent at last inference')
MEM_GAUGE = Gauge('real_esrgan_proc_rss_bytes', 'RSS memory bytes at last inference')
LAST_LATENCY = Gauge('real_esrgan_last_duration_seconds', 'Last inference duration seconds')


def get_proc_metrics():
    p = psutil.Process()
    mem = p.memory_info().rss
    cpu = psutil.cpu_percent(interval=0.1)
    return {'proc_rss_bytes': mem, 'cpu_percent': cpu}

def resize_to(src, target_shape):
    # target_shape: (H, W, C) or (H, W)
    target_h, target_w = target_shape[0], target_shape[1]
    img = Image.fromarray(src)
    img = img.resize((target_w, target_h), Image.BICUBIC)
    return np.array(img)


@app.on_event('startup')
def load_model_on_startup():
    device = 'cpu'

    # build model instance and load weights once
    candidate = os.path.join(REAL_WEIGHTS_PATH, 'checkpoint_epoch24.pth')
    if not os.path.exists(candidate):
        for f in os.listdir(REAL_WEIGHTS_PATH):
            if f.endswith('.pth') or f.endswith('.pt'):
                candidate = os.path.join(REAL_WEIGHTS_PATH, f)
                break
    candidate = os.path.join(REAL_WEIGHTS_PATH, 'checkpoint_epoch24.pth')
    if not os.path.exists(candidate):
        # try internal repo weights
        internal = os.path.join(INTERNAL_WEIGHTS_PATH, 'checkpoint_epoch24.pth')
        if os.path.exists(internal):
            candidate = internal
        else:
            # fallback to any pth in mounted weights
            for f in os.listdir(REAL_WEIGHTS_PATH):
                if f.endswith('.pth') or f.endswith('.pt'):
                    candidate = os.path.join(REAL_WEIGHTS_PATH, f)
                    break

    if not os.path.exists(candidate):
        app.state.model = None
        app.state.device = device
        print('Warning: no real-esrgan weights found at', REAL_WEIGHTS_PATH)
        return

    model = RealESRGAN(device, scale=4)
    model.load_weights(candidate)
    app.state.model = model
    app.state.device = device
    app.state.weights = candidate
    print('Loaded real-esrgan weights:', candidate)


@app.post('/infer')
async def infer(image: UploadFile = File(...)):
    REQUESTS.inc()
    contents = await image.read()
    tmp_in = '/tmp/input_img'
    tmp_out = '/tmp/output.png'
    output_buffer = io.BytesIO()
    with open(tmp_in, 'wb') as f:
        f.write(contents)

    if not getattr(app.state, 'model', None):
        raise HTTPException(status_code=500, detail='Real-ESRGAN model not loaded on startup')

    start = time.time()
    try:
        img = Image.open(tmp_in).convert('RGB')
        img = apply_degradation(img)
        with torch.no_grad():
            sr = app.state.model.predict(img)
        sr = resize_to(np.array(sr), np.array(img).shape)
        sr = Image.fromarray(sr)
        sr.save(output_buffer, format='PNG')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    duration = time.time() - start

    metrics = get_proc_metrics()
    metrics.update({'duration_sec': duration})

    REQUEST_LATENCY.observe(duration)
    CPU_GAUGE.set(metrics['cpu_percent'])
    MEM_GAUGE.set(metrics['proc_rss_bytes'])
    LAST_LATENCY.set(duration)

    img_bytes = output_buffer.getvalue()
    img_b64 = base64.b64encode(img_bytes).decode('ascii')

    return JSONResponse({'image_b64': img_b64, 'metrics': metrics, 'model': 'real_esrgan'})


@app.get('/metrics')
def metrics_endpoint():
    data = generate_latest()
    return PlainTextResponse(data, media_type=CONTENT_TYPE_LATEST)
