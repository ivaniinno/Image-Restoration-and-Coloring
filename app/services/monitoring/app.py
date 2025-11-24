import os
import time
import requests
import threading
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.environ.get('BOT_TOKEN')
ADMIN_CHAT_ID = os.environ.get('ADMIN_CHAT_ID')
COLORIZER_METRICS = os.environ.get('COLORIZER_METRICS','http://colorizer:8001/metrics')
REAL_METRICS = os.environ.get('REAL_METRICS','http://real_esrgan:8002/metrics')
SCRAPE_INTERVAL = int(os.environ.get('SCRAPE_INTERVAL', '5'))
INFLUX_URL = os.environ.get('INFLUX_URL')


def parse_prom_metrics(text):
    # very small parser: return mapping of metric_name->value for simple gauges
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) >= 2:
            try:
                out[parts[0]] = float(parts[1])
            except Exception:
                continue
    return out


def send_telegram(text):
    if not BOT_TOKEN or not ADMIN_CHAT_ID:
        print('Bot token or admin chat id not set; skipping telegram send')
        return
    url = f'https://api.telegram.org/bot{BOT_TOKEN}/sendMessage'
    try:
        requests.post(url, json={'chat_id': int(ADMIN_CHAT_ID), 'text': text})
    except Exception as e:
        print('Failed to send telegram message', e)


def push_to_influx(lines):
    # INFLUX_URL expected to accept line protocol (v1 write endpoint)
    if not INFLUX_URL:
        return
    try:
        resp = requests.post(INFLUX_URL, data='\n'.join(lines), timeout=5)
        if resp.status_code not in (200,204):
            print('Influx push failed', resp.status_code, resp.text)
    except Exception as e:
        print('Influx push exception', e)


def run_monitor():
    while True:
        alerts = []
        for name, url in [('colorizer', COLORIZER_METRICS), ('real_esrgan', REAL_METRICS)]:
            try:
                r = requests.get(url, timeout=5)
                if r.status_code != 200:
                    alerts.append(f'{name} metrics scrape failed: {r.status_code}')
                    continue
                metrics = parse_prom_metrics(r.text)
                # look for last_duration, cpu, mem
                dur_key = f'{name}_last_duration_seconds'
                cpu_key = f'{name}_cpu_percent'
                mem_key = f'{name}_proc_rss_bytes'
                dur = metrics.get(dur_key)
                cpu = metrics.get(cpu_key)
                mem = metrics.get(mem_key)
                if dur and dur > float(os.environ.get('ALERT_DURATION', '10')):
                    alerts.append(f'Long inference on {name}: {dur:.1f}s')
                if cpu and cpu > float(os.environ.get('ALERT_CPU', '80')):
                    alerts.append(f'High CPU on {name}: {cpu:.1f}%')

                # optional: push to influx in line protocol
                lines = []
                if dur is not None:
                    lines.append(f'{name}_last_duration value={dur}')
                if cpu is not None:
                    lines.append(f'{name}_cpu value={cpu}')
                if mem is not None:
                    lines.append(f'{name}_mem value={mem}')
                if lines:
                    push_to_influx(lines)

            except Exception as e:
                alerts.append(f'{name} scrape exception: {e}')

        if alerts:
            text = '\n'.join(alerts)
            print('ALERTS:\n', text)
            send_telegram(text)

        time.sleep(SCRAPE_INTERVAL)


if __name__ == '__main__':
    t = threading.Thread(target=run_monitor, daemon=True)
    t.start()
    while True:
        time.sleep(60)
