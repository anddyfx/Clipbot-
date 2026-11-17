import asyncio
import glob
import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from telegram import Update
from telegram.ext import (Application, CommandHandler, ContextTypes,
                          MessageHandler, filters)

import helpers  # your hashtag / posting time / viral score code

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("clipbot")

TOKEN = os.getenv("BOT_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN")
if not TOKEN:
    raise SystemExit("Missing BOT_TOKEN variable in Railway")

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "tiny")
CLIP_SECONDS = int(os.getenv("CLIP_SECONDS", "45"))
NUM_CLIPS = int(os.getenv("NUM_CLIPS", "3"))
VERTICAL = os.getenv("VERTICAL", "1") == "1"
SR = 16000
URL_RE = re.compile(r"https?://\S+")

_model = None
_busy = asyncio.Semaphore(1)  # one video at a time, keeps memory low


def get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        _model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    return _model


def run(cmd, cwd=None):
    return subprocess.run(cmd, check=True, capture_output=True, cwd=cwd)


def download_url(url, folder):
    import yt_dlp
    opts = {
        "outtmpl": str(Path(folder) / "source.%(ext)s"),
        "format": "bv*[height<=720]+ba/b[height<=720]/b",
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
    files = glob.glob(str(Path(folder) / "source.*"))
    if not files:
        raise RuntimeError("Download failed")
    return files[0]


def load_audio(path):
    out = run(["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1",
               "-ar", str(SR), "-f", "s16le", "-"]).stdout
    return np.frombuffer(out, dtype=np.int16).astype(np.float32) / 32768.0


def srt_time(t):
    t = max(0.0, t)
    h, rem = divmod(int(t), 3600)
    m, s = divmod(rem, 60)
    ms = int((t - int(t)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def build_srt(segments, start, end, path):
    """Short punchy caption chunks (about 5 words each)."""
    lines, idx = [], 1
    for seg in segments:
        if seg.end < start or seg.start > end:
            continue
        words = seg.text.strip().split()
        if not words:
            continue
        chunks = [words[i:i + 5] for i in range(0, len(words), 5)]
        span = max(0.1, seg.end - seg.start)
        per = span / len(chunks)
        for i, chunk in enumerate(chunks):
            a = seg.start + i * per - start
            b = seg.start + (i + 1) * per - start
            if b <= 0 or a >= end - start:
                continue
            lines.append(f"{idx}\n{srt_time(a)} --> {srt_time(b)}\n"
                         f"{' '.join(chunk).upper()}\n")
            idx += 1
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def analyze(video):
    audio = load_audio(video)
    secs = max(1, len(audio) // SR)
    rms = np.sqrt(np.mean(audio[:secs * SR].reshape(secs, SR) ** 2, axis=1))
    loud = np.clip(rms / (np.percentile(rms, 99) + 1e-9), 0, 1)

    segments = list(get_model().transcribe(audio, vad_filter=True)[0])
    speech = np.zeros(secs)
    for s in segments:
        speech[int(s.start):min(secs, int(s.end) + 1)] = 1.0

    # slide a window over the video and score each one
    length = min(CLIP_SECONDS, secs)
    windows = []
    for start in range(0, max(1, secs - length + 1), 10):
        end = start + length
        text = helpers.clip_text(segments, start, end)
        score = helpers.viral_score(loud, speech, start, end, text, secs)
        windows.append((score, start, end))
    windows.sort(reverse=True)

    # pick the best non-overlapping ones
    picked = []
    for score, start, end in windows:
        if all(end <= p[1] or start >= p[2] for p in picked):
            picked.append((score, start, end))
        if len(picked) >= NUM_CLIPS:
            break
    return segments, picked


def make_clip(video, folder, n, start, end, segments):
    srt = Path(folder) / f"cap{n}.srt"
    out = Path(folder) / f"clip{n}.mp4"
    build_srt(segments, start, end, srt)
    style = ("FontName=DejaVu Sans,Bold=1,FontSize=14,Alignment=2,"
             "MarginV=90,Outline=2,Shadow=1,PrimaryColour=&H00FFFFFF")
    vf = f"subtitles={srt.name}:force_style='{style}'"
    if VERTICAL:
        vf = "scale=-2:1920,crop=1080:1920," + vf
    run(["ffmpeg", "-y", "-v", "error", "-ss", str(start), "-i", video,
         "-t", str(end - start), "-vf", vf, "-c:v", "libx264",
         "-preset", "veryfast", "-crf", "24", "-c:a", "aac",
         "-movflags", "+faststart", str(out)], cwd=folder)
    return out


def process(video, folder):
    segments, picked = analyze(video)
    results = []
    for n, (score, start, end) in enumerate(picked, 1):
        clip = make_clip(video, folder, n, start, end, segments)
        text = helpers.clip_text(segments, start, end)
        tags = " ".join(helpers.suggest_hashtags(text))
        when, label = helpers.best_post_time()
        caption = (f"Clip {n} | score {score}/100 | {start}s-{end}s\n"
                   f"{tags}\n"
                   f"Best time to post: {when.strftime('%a %H:%M')} UTC "
                   f"({label})")
        results.append((clip, caption))
    return results


async def run_pipeline(update: Update, source):
    msg = update.message
    status = await msg.reply_text("Working on it, this can take a few minutes...")
    folder = tempfile.mkdtemp(prefix="clipbot_")
    try:
        async with _busy:
            if source["kind"] == "url":
                video = await asyncio.to_thread(download_url, source["url"], folder)
            else:
                tg_file = await source["file"].get_file()
                video = str(Path(folder) / "source.mp4")
                await tg_file.download_to_drive(video)
            results = await asyncio.to_thread(process, video, folder)
        if not results:
            await status.edit_text("Couldn't find a good moment in that video.")
            return
        for clip, caption in results:
            with open(clip, "rb") as f:
                await msg.reply_video(video=f, caption=caption,
                                      read_timeout=180, write_timeout=180)
        await status.delete()
    except Exception as e:
        log.exception("Processing failed")
        await status.edit_text(f"Something went wrong: {str(e)[:200]}")
    finally:
        shutil.rmtree(folder, ignore_errors=True)


async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hi! Send me a video file or a YouTube/Twitch link and I'll cut the "
        "best moments into captioned clips, with hashtags and a posting time.")


async def on_video(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    m = update.message
    f = m.video or m.document
    if f.file_size and f.file_size > 20 * 1024 * 1024:
        await m.reply_text("Telegram limits bot downloads to 20 MB. "
                           "Send a YouTube/Twitch link for bigger videos.")
        return
    await run_pipeline(update, {"kind": "file", "file": f})


async def on_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    found = URL_RE.search(update.message.text or "")
    if not found:
        await update.message.reply_text("Send a video file or a YouTube/Twitch link.")
        return
    await run_pipeline(update, {"kind": "url", "url": found.group(0)})


async def on_error(update, ctx: ContextTypes.DEFAULT_TYPE):
    log.error("Update error: %s", ctx.error)


def main():
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler(["start", "help"], start))
    app.add_handler(MessageHandler(
        filters.VIDEO | filters.Document.VIDEO, on_video))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.add_error_handler(on_error)
    log.info("Bot started, polling Telegram...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
