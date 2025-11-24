import os
import io
import base64
import time
import sqlite3
import requests
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, executor, types
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import matplotlib.pyplot as plt
import numpy as np

load_dotenv()
BOT_TOKEN = os.environ.get('BOT_TOKEN')
ADMIN_CHAT_ID = int(os.environ.get('ADMIN_CHAT_ID','-3452168000'))
COLORIZER_URL = os.environ.get('COLORIZER_URL','http://localhost:8001/infer')
REAL_URL = os.environ.get('REAL_ESRGAN_URL','http://localhost:8002/infer')

if not BOT_TOKEN:
    print('BOT_TOKEN not set')
    exit(1)

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(bot)

DB_PATH = '/app/data/metrics.db'
os.makedirs('/app/data', exist_ok=True)


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS inferences (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        model TEXT,
        duration REAL,
        cpu REAL,
        mem INTEGER,
        created_at REAL,
        feedback TEXT,
        rating INTEGER
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS pending_files (
        token TEXT PRIMARY KEY,
        user_id INTEGER,
        path TEXT,
        created_at REAL
    )''')
    conn.commit()
    conn.close()

init_db()

def save_metrics(user_id, model, duration, cpu, mem):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('INSERT INTO inferences (user_id, model, duration, cpu, mem, created_at) VALUES (?,?,?,?,?,?)',
              (user_id, model, duration, cpu, mem, time.time()))
    conn.commit()
    conn.close()

@dp.message_handler(commands=['start','help'])
async def send_welcome(message: types.Message):
    await message.reply('Пришлите фото, я предложу варианты: раскрасить или улучшить качество.')

@dp.message_handler(content_types=['photo'])
async def handle_photo(message: types.Message):
    # get highest resolution photo
    photo = message.photo[-1]
    file = await bot.get_file(photo.file_id)
    data = await bot.download_file(file.file_path)
    # save to temp
    tmp = f'/tmp/{photo.file_id}.jpg'
    with open(tmp, 'wb') as f:
        f.write(data.getvalue())

    # store a short token in DB to avoid oversized callback_data
    import uuid, time as _time
    token = uuid.uuid4().hex[:16]
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('INSERT OR REPLACE INTO pending_files (token, user_id, path, created_at) VALUES (?,?,?,?)',
              (token, message.from_user.id, tmp, _time.time()))
    conn.commit(); conn.close()

    kb = InlineKeyboardMarkup().add(
        InlineKeyboardButton('Colorize', callback_data=f'c|{token}')
        # for fast enabling: uncomment the line below
        # , InlineKeyboardButton('Enhance', callback_data=f'e|{token}')
    )
    await message.reply('Выберите действие:', reply_markup=kb)

@dp.callback_query_handler(lambda c: c.data and (c.data.startswith('c|') or c.data.startswith('e|')))
async def process_callback(callback_query: types.CallbackQuery):
    data = callback_query.data.split('|', 1)
    action = data[0]
    token = data[1]

    # lookup file path by token
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    row = c.execute('SELECT path FROM pending_files WHERE token=?', (token,)).fetchone()
    conn.close()
    if not row:
        await bot.answer_callback_query(callback_query.id, text='Файл не найден, пришлите фото снова')
        return
    filepath = row[0]
    try:
        await bot.answer_callback_query(callback_query.id, text='Запущена обработка...')
    except Exception:
        pass

    url = COLORIZER_URL if action=='c' else REAL_URL
    files = {'image': open(filepath, 'rb')}
    start = time.time()
    try:
        r = requests.post(url, files=files, timeout=600)
        r.raise_for_status()
    except Exception as e:
        await bot.send_message(callback_query.from_user.id, f'Ошибка модели: {e}')
        return
    duration = time.time() - start
    j = r.json()
    img_b64 = j.get('image_b64')
    metrics = j.get('metrics', {})
    model_name = j.get('model')

    # save metrics
    save_metrics(callback_query.from_user.id, model_name, metrics.get('duration_sec', duration), metrics.get('cpu_percent',0), metrics.get('proc_rss_bytes',0))

    # send output image
    img_bytes = base64.b64decode(img_b64)
    await bot.send_photo(callback_query.from_user.id, types.InputFile(io.BytesIO(img_bytes), filename='result.png'))

    # ask for feedback
    fb_kb = InlineKeyboardMarkup().add(
        InlineKeyboardButton('👍', callback_data=f'fb_like|{model_name}'),
        InlineKeyboardButton('👎', callback_data=f'fb_dislike|{model_name}')
    )
    await bot.send_message(callback_query.from_user.id, 'Нравится результат?', reply_markup=fb_kb)

    # cleanup token record
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('DELETE FROM pending_files WHERE token=?', (token,))
        conn.commit(); conn.close()
    except Exception:
        pass

@dp.callback_query_handler(lambda c: c.data and c.data.startswith('fb_'))
async def handle_feedback(cb: types.CallbackQuery):
    parts = cb.data.split('|')
    fb = parts[0]
    model = parts[1] if len(parts)>1 else None
    # record simple feedback: update last inference for this user & model
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    if fb == 'fb_like':
        c.execute('UPDATE inferences SET feedback=? WHERE id=(SELECT id FROM inferences WHERE user_id=? AND model=? ORDER BY created_at DESC LIMIT 1)', ('like', cb.from_user.id, model))
    else:
        c.execute('UPDATE inferences SET feedback=? WHERE id=(SELECT id FROM inferences WHERE user_id=? AND model=? ORDER BY created_at DESC LIMIT 1)', ('dislike', cb.from_user.id, model))
    conn.commit(); conn.close()
    await bot.answer_callback_query(cb.id, text='Спасибо за отзыв!')

@dp.message_handler(commands=['stats'])
async def cmd_stats(message: types.Message):
    # generate extended stats: latency, resources, feedback
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    rows = c.execute('SELECT model, duration, cpu, mem, feedback, created_at FROM inferences').fetchall()
    conn.close()
    if not rows:
        await message.reply('Нет данных')
        return

    # Cumulative avg duration for colorizer only
    colorizer_rows = [r for r in rows if r[0] == 'colorizer']
    durations = [r[1] for r in colorizer_rows]
    plt.figure()
    if durations:
        plt.plot(np.cumsum(np.ones(len(durations))), np.cumsum(durations)/np.arange(1,len(durations)+1), label='colorizer')
    plt.xlabel('Requests')
    plt.ylabel('Cumulative avg duration (s)')
    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    await bot.send_photo(message.chat.id, types.InputFile(buf, filename='stats_latency_colorizer.png'))

    # Для быстрого возврата real_esrgan latency — раскомментируйте строки ниже
    # real_rows = [r for r in rows if r[0] == 'real_esrgan']
    # real_durations = [r[1] for r in real_rows]
    # plt.figure()
    # if real_durations:
    #     plt.plot(np.cumsum(np.ones(len(real_durations))), np.cumsum(real_durations)/np.arange(1,len(real_durations)+1), label='real_esrgan')
    # plt.xlabel('Requests')
    # plt.ylabel('Cumulative avg duration (s)')
    # plt.legend()
    # buf_real = io.BytesIO()
    # plt.savefig(buf_real, format='png')
    # buf_real.seek(0)
    # await bot.send_photo(message.chat.id, types.InputFile(buf_real, filename='stats_latency_real_esrgan.png'))

    # CPU usage plot (colorizer only)
    plt.figure()
    colorizer_cpu = [r[2] for r in colorizer_rows]
    colorizer_times = [r[5] for r in colorizer_rows]
    if colorizer_cpu:
        plt.plot(colorizer_times, colorizer_cpu, label='CPU % (colorizer)')
    plt.xlabel('Timestamp')
    plt.ylabel('CPU usage (%)')
    plt.legend()
    plt.tight_layout()
    buf_cpu = io.BytesIO()
    plt.savefig(buf_cpu, format='png')
    buf_cpu.seek(0)
    await bot.send_photo(message.chat.id, types.InputFile(buf_cpu, filename='stats_cpu_colorizer.png'))

    # RAM usage plot (colorizer only)
    plt.figure()
    colorizer_mem = [r[3]/(1024*1024) for r in colorizer_rows]
    if colorizer_mem:
        plt.plot(colorizer_times, colorizer_mem, label='RAM MB (colorizer)')
    plt.xlabel('Timestamp')
    plt.ylabel('RAM usage (MB)')
    plt.legend()
    plt.tight_layout()
    buf_mem = io.BytesIO()
    plt.savefig(buf_mem, format='png')
    buf_mem.seek(0)
    await bot.send_photo(message.chat.id, types.InputFile(buf_mem, filename='stats_ram_colorizer.png'))

    # Для быстрого возврата real_esrgan CPU/RAM — раскомментируйте строки ниже
    # real_cpu = [r[2] for r in real_rows]
    # real_mem = [r[3]/(1024*1024) for r in real_rows]
    # real_times = [r[5] for r in real_rows]
    # plt.figure()
    # if real_cpu:
    #     plt.plot(real_times, real_cpu, label='CPU % (real_esrgan)')
    # plt.xlabel('Timestamp')
    # plt.ylabel('CPU usage (%)')
    # plt.legend()
    # plt.tight_layout()
    # buf_real_cpu = io.BytesIO()
    # plt.savefig(buf_real_cpu, format='png')
    # buf_real_cpu.seek(0)
    # await bot.send_photo(message.chat.id, types.InputFile(buf_real_cpu, filename='stats_cpu_real_esrgan.png'))
    # plt.figure()
    # if real_mem:
    #     plt.plot(real_times, real_mem, label='RAM MB (real_esrgan)')
    # plt.xlabel('Timestamp')
    # plt.ylabel('RAM usage (MB)')
    # plt.legend()
    # plt.tight_layout()
    # buf_real_mem = io.BytesIO()
    # plt.savefig(buf_real_mem, format='png')
    # buf_real_mem.seek(0)
    # await bot.send_photo(message.chat.id, types.InputFile(buf_real_mem, filename='stats_ram_real_esrgan.png'))

    # Feedback plot
    plt.figure()
    feedbacks = [r[4] for r in rows if r[4] is not None]
    from collections import Counter
    fb_counts = Counter(feedbacks)
    labels = list(fb_counts.keys())
    values = [fb_counts[k] for k in labels]
    plt.bar(labels, values)
    plt.xlabel('Feedback')
    plt.ylabel('Count')
    plt.title('User feedback (like/dislike)')
    buf3 = io.BytesIO()
    plt.savefig(buf3, format='png')
    buf3.seek(0)
    await bot.send_photo(message.chat.id, types.InputFile(buf3, filename='stats_feedback.png'))

async def monitor_alerts():
    # check last 20 inferences for high duration or cpu
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    rows = c.execute('SELECT id, user_id, model, duration, cpu, mem, created_at FROM inferences ORDER BY created_at DESC LIMIT 20').fetchall()
    conn.close()
    alerts = []
    for r in rows:
        if r[3] and r[3] > float(os.environ.get('ALERT_DURATION', 0)):
            alerts.append(f'Long inference: id={r[0]} model={r[2]} dur={r[3]:.1f}s')
        if r[4] and r[4] > float(os.environ.get('ALERT_CPU', 0)):
            alerts.append(f'High CPU: id={r[0]} model={r[2]} cpu={r[4]}%')
    if alerts and ADMIN_CHAT_ID:
        text = '\n'.join(alerts)
        try:
            await bot.send_message(ADMIN_CHAT_ID, f'ALERTS:\n{text}')
        except Exception:
            pass

async def on_startup(dp_: Dispatcher):
    scheduler = AsyncIOScheduler()
    scheduler.add_job(monitor_alerts, 'interval', minutes=5)
    scheduler.start()

if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True, on_startup=on_startup)
