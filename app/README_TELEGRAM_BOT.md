# Telegram bot + model services (colorizer + real-esrgan)

This workspace contains a minimal setup to run a Telegram bot that accepts images and forwards them to two model services:
- colorizer (pix2pix-based)
- real-esrgan (super-resolution)

Each model runs in its own container and exposes a small FastAPI `/infer` endpoint that accepts an image and returns the processed image (base64) together with simple runtime metrics (duration, cpu, memory).

The `telegram-bot` container receives user images, presents choices (colorize / enhance), forwards the image to the chosen model service, stores inference metrics in a local SQLite DB, collects user feedback, and can generate simple `stats` plots.

Files added:
- `docker-compose.yml` — orchestrates three services
- `services/colorizer` — FastAPI wrapper around `pix2pix colorizer` code
- `services/real_esrgan` — FastAPI wrapper around `real-esrgan` code
- `bot/` — Telegram bot code (aiogram), sqlite storage, simple plotting

Important notes before running
- Place your model weights in the repository `weights/` folder. Current expected filenames (examples):
  - `ResUnet_epoch_4.pt` (colorizer)
  - `checkpoint_epoch24` (real-esrgan)
- The colorizer service expects the `pix2pix colorizer` code to be present in the repository (it is in this workspace) and mounts it into the container.
- These Dockerfiles install `torch` via pip which will download a large package; adapt the Dockerfiles if you want to use GPU-optimized images (CUDA) or pre-built torch wheels.

Run (example)

1. Copy example env and set your bot token + admin chat id:

```bash
cp bot/.env.example .env
export BOT_TOKEN=your_token
export ADMIN_CHAT_ID=your_chat_id
```

2. Build and start services with docker-compose (from repository root):

```bash
docker-compose up --build
```

3. Use the bot in Telegram. Send a photo and choose an action.

Monitoring and time-series storage
- A new `monitoring` service scrapes `/metrics` from both model services and will send Telegram alerts to `ADMIN_CHAT_ID` when thresholds are exceeded (long duration, high CPU). Configure thresholds via `ALERT_DURATION` and `ALERT_CPU` env vars.

Environment variables available to control monitoring (set in your shell or CI):

```bash
export ALERT_DURATION=30
export ALERT_CPU=80
```
