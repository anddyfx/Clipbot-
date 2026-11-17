import os, re, json, uuid, shutil, asyncio, subprocess, logging
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel
from telegram import Update
from telegram.ext import (ApplicationBuilder, CommandHandler,
                          MessageHandler, ContextTypes, filters)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("clipbot")

BOT_TOKEN = os.environ["BOT_TOKEN"]
CLIP_MIN = int(os.getenv("CLIP_MIN", "30"))
CLIP_MAX = int(os.getenv("CLIP_MAX", "60"))
NUM_CLIPS = int(os.getenv("NUM_CLIPS", "3"))
WORK = Path("/tmp/clipbot")
WORK.mkdir(exist_ok=True)

# small model keeps Railway memory usage low
whisper = WhisperModel("base", device="cpu", compute_type="int8")

URL_RE = re.compile(r"https?://\S+")


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, check=True)


def download(url, out_dir):
    out = str(out_dir / "source.%(ext)s")
    run(["yt-dlp", "-f", "bv*[height<=720]+ba/b[height<=720]/b",
         "--merge-output-format", "mp4", "--no-playlist",
         "-o", out, url])
    return next(out_dir.glob("source.*"))


def duration(path):
    r = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "json", str(path)])
    return float(json.loads(r.stdout)["format"]["duration"])


def extract_audio(video, wav):
    run(["ffmpeg", "-y", "-i", str(video), "-vn", "-ac", "1",
         "-ar", "16000", str(wav)])


def loudness_curve(wav, total):
    """RMS loudness per second, used to find hype moments."""
    raw = subprocess.run(
        ["ffmpeg", "-i", str(wav), "-f", "s16le", "-ac", "1",
         "-ar", "16000", "-"], capture_output=True, check=True).stdout
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    sr = 16000
    secs = int(total)
    curve = np.zeros(secs)
    for i in range(secs):
        chunk = audio[i * sr:(i + 1) * sr]
        if len(chunk):
            curve[i] = np.sqrt(np.mean(chunk ** 2))
    return curve


def pick_moments(curve, words_per_sec, length):
    """Score each window by loudness + speech density, pick top non-overlapping."""
    secs = len(curve)
    if secs <= CLIP_MIN:
        return [(0, secs)]
    loud = curve / (curve.max() + 1e-9)
    speech = words_per_sec / (words_per_sec.max() + 1e-9)
    score = 0.6 * loud + 0.4 * speech
    win = min(CLIP_MAX, max(CLIP_MIN, 45))
    csum = np.concatenate([[0], np.cumsum(score)])
    scores = [(csum[i + win] - csum[i], i) for i in range(0, secs - win + 1)]
    scores.sort(reverse=True)
    picked = []
    for _, start in scores:
        if all(abs(start - s) >= win for s, _ in picked):
            picked.append((start, start + win))
        if len(picked) == NUM_CLIPS:
            break
    return sorted(picked)


def ts(t):
    h, m, s = int(t // 3600), int(t % 3600 // 60), t % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def make_ass(segments, start, end, path):
    header = ("[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\n\n"
              "[V4+ Styles]\nFormat: Name,Fontname,Fontsize,PrimaryColour,"
              "OutlineColour,BackColour,Bold,Outline,Shadow,Alignment,MarginV\n"
              "Style: Default,Arial,64,&H00FFFFFF,&H00000000,&H64000000,"
              "1,4,1,2,300\n\n[Events]\nFormat: Layer,Start,End,Style,Text\n")
    lines = []
    for seg in segments:
        if seg.end < start or seg.start > end:
            continue
        s = max(seg.start, start) - start
        e = min(seg.end, end) - start
        text = seg.text.strip().replace("\n", " ").upper()
        lines.append(f"Dialogue: 0,{ts(s)},{ts(e)},Default,{text}")
    Path(path).write_text(header + "\n".join(lines), encoding="utf-8")


def render_clip(video, start, end, segments, out, work):
    ass = work / f"cap_{uuid.uuid4().hex[:6]}.ass"
    make_ass(segments, start, end, ass)
    # vertical 9:16 with blurred background, captions burned in
    vf = ("split[a][b];[a]scale=1080:1920,boxblur=20:5[bg];"
          "[b]scale=1080:-2[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2,"
          f"subtitles={ass}")
    run(["ffmpeg", "-y", "-ss", str(start), "-t", str(end - start),
         "-i", str(video), "-filter_complex", vf,
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
         "-c:a", "aac", "-movflags", "+faststart", str(out)])


def process(video, work):
    total = duration(video)
    wav = work / "audio.wav"
    extract_audio(video, wav)
    segs, _ = whisper.transcribe(str(wav), vad_filter=True)
    segments = list(segs)
    curve = loudness_curve(wav, total)
    wps = np.zeros(len(curve))
    for s in segments:
        n = len(s.text.split())
        span = max(1, int(s.end - s.start))
        for t in range(int(s.start), min(int(s.end) + 1, len(wps))):
            wps[t] += n / span
    moments = pick_moments(curve, wps, total)
    outs = []
    for i, (a, b) in enumerate(moments, 1):
        out = work / f"clip_{i}.mp4"
        render_clip(video, a, b, segments, out, work)
        outs.append(out)
    return outs


async def start_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Send me a video or a YouTube/Twitch link and I'll clip the best "
        "moments with captions.")


async def handle(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    work = WORK / uuid.uuid4().hex
    work.mkdir()
    try:
        await msg.reply_text("Got it, working on it. This can take a few minutes.")
        loop = asyncio.get_running_loop()
        if msg.video or msg.document:
            f = await (msg.video or msg.document).get_file()
            video = work / "source.mp4"
            await f.download_to_drive(str(video))
        else:
            m = URL_RE.search(msg.text or "")
            if not m:
                await msg.reply_text("Send a video file or a link please.")
                return
            video = await loop.run_in_executor(None, download, m.group(0), work)
        clips = await loop.run_in_executor(None, process, video, work)
        for c in clips:
            with open(c, "rb") as fh:
                await msg.reply_video(fh, supports_streaming=True)
    except Exception as e:
        log.exception("failed")
        await msg.reply_text(f"Something went wrong: {str(e)[:200]}")
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(MessageHandler(
        filters.VIDEO | filters.Document.VIDEO | (filters.TEXT & ~filters.COMMAND),
        handle))
    app.run_polling()


if __name__ == "__main__":
    main()
