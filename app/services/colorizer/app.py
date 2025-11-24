import os
import time
import base64
import io
import psutil
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.middleware.cors import CORSMiddleware

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])

# import building utils from the pix2pix folder
import sys
sys.path.insert(0, '/app/pix2pix')
from models import build_models
from utils import lab_to_rgb
from torchvision import transforms
from imageio import imwrite
from PIL import Image
import numpy as np

GEN_WEIGHTS_PATH = os.environ.get('GEN_WEIGHTS_PATH', '/app/weights')

# Prometheus metrics
REQUESTS = Counter('colorizer_requests_total', 'Total number of colorize requests')
REQUEST_LATENCY = Histogram('colorizer_request_latency_seconds', 'Latency for colorize requests')
CPU_GAUGE = Gauge('colorizer_cpu_percent', 'CPU percent at last inference')
MEM_GAUGE = Gauge('colorizer_proc_rss_bytes', 'RSS memory bytes at last inference')
LAST_LATENCY = Gauge('colorizer_last_duration_seconds', 'Last inference duration seconds')


def find_latest_weight(path, pattern='*.pt'):
    # reuse inference helper semantics: look for a .pt in path
    import glob
    if os.path.isdir(path):
        files = glob.glob(os.path.join(path, pattern))
        if not files:
            return None
        files = sorted(files, key=os.path.getmtime)
        return files[-1]
    elif os.path.isfile(path):
        return path
    return None


def get_proc_metrics():
    p = psutil.Process()
    mem = p.memory_info().rss
    cpu = psutil.cpu_percent(interval=0.1)
    return {'proc_rss_bytes': mem, 'cpu_percent': cpu}


@app.on_event('startup')
def load_model_on_startup():
    # load generator into app.state
    device = None
    try:
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    except Exception:
        device = 'cpu'

    w = find_latest_weight(GEN_WEIGHTS_PATH)
    if not w:
        app.state.generator = None
        app.state.device = device
        app.state.gen_weights = None
        print('Warning: no generator weights found at', GEN_WEIGHTS_PATH)
        return

    generator, _ = build_models(device=device, gen_weights=w)
    generator.eval()
    app.state.generator = generator
    app.state.device = device
    app.state.gen_weights = w
    print('Loaded colorizer weights:', w)


def process_with_generator(generator, input_path, output_path, device=None):
    # load and run inference similar to original inference.infer_image but using preloaded generator
    def load_image_as_L(path, size=(224, 224)):
        img = Image.open(path).convert('L').resize(size)
        arr = np.array(img).astype('float32') / 255.0
        arr = arr.reshape((1, 224, 224, 1))
        return arr

    arr = load_image_as_L(input_path)
    to_tensor = transforms.ToTensor()
    L = to_tensor(arr[0]).unsqueeze(0).to(device).float()
    with __import__('torch').no_grad():
        pred_ab = generator(L)
    rgb = lab_to_rgb(L, pred_ab)
    imwrite(output_path, (rgb[0] * 255).astype('uint8'))


@app.post('/infer')
async def infer(image: UploadFile = File(...)):
    REQUESTS.inc()
    tmp_in = '/tmp/input_image'
    tmp_out = '/tmp/output.png'
    contents = await image.read()
    with open(tmp_in, 'wb') as f:
        f.write(contents)

    if not getattr(app.state, 'generator', None):
        raise HTTPException(status_code=500, detail='Generator not loaded on startup')

    start = time.time()
    try:
        process_with_generator(app.state.generator, tmp_in, tmp_out, device=app.state.device)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    duration = time.time() - start

    metrics = get_proc_metrics()
    metrics.update({'duration_sec': duration})

    # record prometheus
    REQUEST_LATENCY.observe(duration)
    CPU_GAUGE.set(metrics['cpu_percent'])
    MEM_GAUGE.set(metrics['proc_rss_bytes'])
    LAST_LATENCY.set(duration)

    with open(tmp_out, 'rb') as f:
        img_bytes = f.read()
    img_b64 = base64.b64encode(img_bytes).decode('ascii')

    return JSONResponse({'image_b64': img_b64, 'metrics': metrics, 'model': 'colorizer'})


@app.get('/metrics')
def metrics_endpoint():
    data = generate_latest()
    return PlainTextResponse(data, media_type=CONTENT_TYPE_LATEST)
