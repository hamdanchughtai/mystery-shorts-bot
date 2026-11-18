"""Automated YouTube Shorts pipeline.

Gemini (script) -> edge-tts (voice + word timings) -> Pixabay (footage)
-> ffmpeg (9:16 render + captions) -> YouTube upload.
"""
import asyncio
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import edge_tts
import requests
from google import genai
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# ---------- settings ----------
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
VOICE = os.environ.get("TTS_VOICE", "en-US-AndrewNeural")
# "private" is safest until your API project passes YouTube's audit.
# Change to "public" later (set PRIVACY in the workflow file).
PRIVACY = os.environ.get("PRIVACY", "private")
SEGMENT_SECONDS = 5  # how long each stock clip is shown
WORK = Path("work")
TOPICS_FILE = Path("topics_done.json")
MUSIC_DIR = Path("music")

FORMATS = [
    "Case file: tell one well-documented unsolved mystery in a gripping way.",
    "Three theories: present the 3 main theories about one real mystery.",
    "What we know vs. what we don't: split one real mystery into facts and unknowns.",
    "Strange history: one odd, true, well-documented historical event most people never heard of.",
]

FALLBACK_KEYWORDS = ["dark forest fog", "ocean waves night", "old library", "abandoned building"]


def log(msg):
    print(f"[shorts-bot] {msg}", flush=True)


def run(cmd, **kw):
    subprocess.run(cmd, check=True, **kw)


# ---------- 1. script ----------
def generate_script(done_topics):
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    fmt = random.choice(FORMATS)
    prompt = f"""You write scripts for YouTube Shorts about unsolved mysteries and strange history.

Format for this video: {fmt}

Rules:
- Pick a REAL, well-documented case. Do not invent facts, dates, names or statistics.
- If unsure about a detail, leave it out. Never accuse real living people of crimes.
- Plain text only: NO asterisks, markdown, bullet points, parentheses or emojis. Write years and numbers as spoken words (e.g. \"nineteen eighty-two\", not 1982).
- Script: 110-130 words, spoken text only, short sentences, strong hook in the first sentence, ends with an open question.
- Do NOT reuse these topics: {json.dumps(done_topics[-150:])}

Return ONLY valid JSON with exactly these keys:
"topic": short topic name,
"script": the spoken text,
"title": under 70 characters, curiosity-driven, no hashtags,
"description": 2-3 sentences summarizing the video,
"tags": list of 6-10 short tags,
"visual_keywords": list of 6 simple English stock-footage search terms (e.g. "foggy forest", "old map", "stormy ocean")."""
    resp = None
    for model in (GEMINI_MODEL, "gemini-2.5-flash-lite", GEMINI_MODEL):
        for attempt in range(1, 4):
            try:
                resp = client.models.generate_content(model=model, contents=prompt)
                break
            except Exception as e:
                log(f"Gemini {model} attempt {attempt} failed: {type(e).__name__}")
                time.sleep(15 * attempt)
        if resp is not None:
            break
    if resp is None:
        raise RuntimeError("Gemini unavailable after several retries")
    raw = resp.text.strip()
    text = raw[raw.find("{"): raw.rfind("}") + 1]
    data = json.loads(text)
    for key in ("topic", "script", "title", "description", "tags", "visual_keywords"):
        if key not in data:
            raise ValueError(f"Gemini response missing '{key}'")
    data["title"] = data["title"][:95]
    return data


# ---------- 2. voice + word timings ----------
async def make_voice(text, path):
    comm = edge_tts.Communicate(text, VOICE, boundary="WordBoundary")
    words = []
    with open(path, "wb") as f:
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                start = chunk["offset"] / 1e7
                words.append({"text": chunk["text"], "start": start, "end": start + chunk["duration"] / 1e7})
    if not words:
        raise RuntimeError("No word timings returned by edge-tts")
    return words


PIPER_VOICE = "en_US-ryan-high"
PIPER_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/high/" + PIPER_VOICE


def download_file(url, dest):
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for part in r.iter_content(1 << 20):
                f.write(part)


def make_voice_piper(text, path):
    """Local fallback voice. Word timings are estimated from word length."""
    model = WORK / f"{PIPER_VOICE}.onnx"
    if not model.exists():
        log("Downloading Piper voice model...")
        download_file(PIPER_URL + ".onnx", model)
        download_file(PIPER_URL + ".onnx.json", WORK / f"{PIPER_VOICE}.onnx.json")
    subprocess.run(
        [sys.executable, "-m", "piper", "-m", str(model), "-f", str(path)],
        input=text.encode("utf-8"), check=True,
    )
    total = audio_duration(path)
    tokens = text.split()
    weights = []
    for t in tokens:
        w = len(t) + 1
        if t.endswith((".", "?", "!")):
            w += 6
        elif t.endswith((",", ";", ":")):
            w += 3
        weights.append(w)
    scale = (total - 0.2) / sum(weights)
    words, cur = [], 0.1
    for t, w in zip(tokens, weights):
        words.append({"text": t, "start": cur, "end": cur + w * scale})
        cur += w * scale
    return words


def make_voice_any(text):
    """Try edge-tts first, then fall back to Piper."""
    try:
        path = WORK / "voice.mp3"
        words = asyncio.run(make_voice(text, path))
        log("Voice: edge-tts")
        return path, words
    except Exception as e:
        log(f"edge-tts failed ({type(e).__name__}); using Piper instead")
    path = WORK / "voice.wav"
    words = make_voice_piper(text, path)
    log("Voice: Piper")
    return path, words


ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def two_digits(n):
    if n < 20:
        return ONES[n]
    return TENS[n // 10] + ("" if n % 10 == 0 else "-" + ONES[n % 10])


def year_to_words(y):
    if y == 2000:
        return "two thousand"
    if 2001 <= y <= 2009:
        return "two thousand " + ONES[y - 2000]
    hi, lo = divmod(y, 100)
    first = two_digits(hi)
    if lo == 0:
        return first + " hundred"
    if lo < 10:
        return first + " oh " + ONES[lo]
    return first + " " + two_digits(lo)


def decade_to_words(y):
    w = year_to_words(y)
    if w.endswith("y"):
        return w[:-1] + "ies"
    return w + "s"


def clean_script(text):
    """Remove markdown symbols and spell out years so the voice reads them naturally."""
    text = re.sub(r"[*_#`~]", "", text)
    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"\b(1[1-9]\d\d|20\d\d)s\b", lambda m: decade_to_words(int(m.group(1))), text)
    text = re.sub(r"\b(1[1-9]\d\d|20\d\d)\b", lambda m: year_to_words(int(m.group(1))), text)
    text = text.replace("&", " and ").replace("%", " percent")
    return re.sub(r"\s+", " ", text).strip()


def align_with_whisper(audio_path, fallback_words):
    """Let Whisper listen to the voiceover and time every spoken word."""
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel("base.en", device="cpu", compute_type="int8")
        segments, _ = model.transcribe(str(audio_path), word_timestamps=True)
        words = []
        for seg in segments:
            for w in seg.words or []:
                t = w.word.strip()
                if t:
                    words.append({"text": t, "start": w.start, "end": w.end})
        if len(words) >= 20:
            log(f"Captions timed by Whisper ({len(words)} words)")
            return words
        log("Whisper returned too few words; using original timings")
    except Exception as e:
        log(f"Whisper alignment failed ({type(e).__name__}: {e}); using original timings")
    return fallback_words


def audio_duration(path):
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)]
    )
    return float(out.decode().strip())


# ---------- 3. captions (ASS) ----------
def ass_time(t):
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def make_captions(words, path, group=3):
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
Style: Default,DejaVu Sans,86,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,7,2,5,60,60,0,1

[Events]
Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
"""
    lines = []
    for i in range(0, len(words), group):
        chunk = words[i:i + group]
        start = chunk[0]["start"]
        nxt = words[i + group]["start"] if i + group < len(words) else chunk[-1]["end"] + 0.3
        text = " ".join(w["text"] for w in chunk).upper()
        text = re.sub(r"[{}\\]", "", text)
        lines.append(f"Dialogue: 0,{ass_time(start)},{ass_time(nxt)},Default,,0,0,0,,{text}")
    Path(path).write_text(header + "\n".join(lines) + "\n", encoding="utf-8")


# ---------- 4. footage ----------
def search_clips(keyword, key):
    r = requests.get(
        "https://pixabay.com/api/videos/",
        params={"key": key, "q": keyword, "per_page": 10, "safesearch": "true"},
        timeout=30,
    )
    r.raise_for_status()
    hits = [h for h in r.json().get("hits", []) if h.get("duration", 0) >= 4]
    random.shuffle(hits)
    return hits


def download_clip(hit, dest):
    vids = hit["videos"]
    for size in ("large", "medium", "small"):
        url = vids.get(size, {}).get("url")
        if url:
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(dest, "wb") as f:
                    for part in r.iter_content(1 << 20):
                        f.write(part)
            return True
    return False


def get_footage(keywords, needed):
    key = os.environ["PIXABAY_API_KEY"]
    clips = []
    for kw in list(keywords) + FALLBACK_KEYWORDS:
        if len(clips) >= needed:
            break
        try:
            for hit in search_clips(kw, key):
                dest = WORK / f"raw_{len(clips)}.mp4"
                if download_clip(hit, dest):
                    clips.append(dest)
                    break
        except Exception as e:  # keep going with other keywords
            log(f"Footage search failed for '{kw}': {e}")
    if not clips:
        raise RuntimeError("Could not download any footage from Pixabay")
    return clips


# ---------- 5. render ----------
def render(clips, voice_path, captions_path, out_path):
    total = audio_duration(voice_path) + 0.6
    n_segments = max(1, math.ceil(total / SEGMENT_SECONDS))
    seg_len = total / n_segments

    segments = []
    for i in range(n_segments):
        src = clips[i % len(clips)]
        seg = WORK / f"seg_{i}.mp4"
        run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-stream_loop", "-1", "-i", str(src), "-t", f"{seg_len:.3f}",
            "-vf", "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,setsar=1,fps=30",
            "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
            str(seg),
        ])
        segments.append(seg)

    concat_list = WORK / "concat.txt"
    concat_list.write_text("".join(f"file '{s.name}'\n" for s in segments))
    video_only = WORK / "video_only.mp4"
    run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
         "-i", "concat.txt", "-c", "copy", "video_only.mp4"], cwd=WORK)

    music = sorted(MUSIC_DIR.glob("*.mp3")) if MUSIC_DIR.exists() else []
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", "video_only.mp4", "-i", voice_path.name]
    if music:
        cmd += ["-stream_loop", "-1", "-i", str(Path("..") / random.choice(music))]
        audio_filter = "[1:a]volume=1.0[v];[2:a]volume=0.12[m];[v][m]amix=inputs=2:duration=first[a]"
    else:
        audio_filter = "[1:a]anull[a]"
    cmd += [
        "-filter_complex", f"[0:v]ass={captions_path.name}[vout];{audio_filter}",
        "-map", "[vout]", "-map", "[a]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k", "-t", f"{total:.3f}", "-movflags", "+faststart",
        out_path.name,
    ]
    run(cmd, cwd=WORK)


# ---------- 6. upload ----------
def upload(video_path, data):
    creds = Credentials(
        token=None,
        refresh_token=os.environ["YT_REFRESH_TOKEN"],
        token_uri="https://oauth2.googleapis.com/token",
        client_id=os.environ["YT_CLIENT_ID"],
        client_secret=os.environ["YT_CLIENT_SECRET"],
        scopes=["https://www.googleapis.com/auth/youtube.upload"],
    )
    youtube = build("youtube", "v3", credentials=creds)
    description = (
        f"{data['description']}\n\n#Shorts #mystery #unsolved #history\n\n"
        "Narration: AI-generated voice. Footage: Pixabay (free license)."
    )
    body = {
        "snippet": {
            "title": data["title"],
            "description": description,
            "tags": [str(t) for t in data["tags"]][:15],
            "categoryId": "27",
        },
        "status": {"privacyStatus": PRIVACY, "selfDeclaredMadeForKids": False},
    }
    media = MediaFileUpload(str(video_path), mimetype="video/mp4", resumable=True)
    req = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
    response = None
    while response is None:
        _, response = req.next_chunk()
    return response["id"]


# ---------- main ----------
def main():
    WORK.mkdir(exist_ok=True)
    done = json.loads(TOPICS_FILE.read_text()) if TOPICS_FILE.exists() else []

    log("Writing script with Gemini...")
    data = generate_script(done)
    log(f"Topic: {data['topic']} | Title: {data['title']}")

    data["script"] = clean_script(data["script"])
    data["title"] = re.sub(r"[*_#`]", "", data["title"]).strip()
    data["description"] = re.sub(r"[*_#`]", "", data["description"]).strip()

    log("Generating voiceover...")
    voice_path, words = make_voice_any(data["script"])
    words = align_with_whisper(voice_path, words)

    captions_path = WORK / "captions.ass"
    make_captions(words, captions_path)

    needed = min(6, max(3, math.ceil(audio_duration(voice_path) / SEGMENT_SECONDS)))
    log("Downloading footage...")
    clips = get_footage(data["visual_keywords"], needed)

    log("Rendering video...")
    out_path = WORK / "short.mp4"
    render(clips, voice_path, captions_path, out_path)

    log(f"Uploading to YouTube ({PRIVACY})...")
    video_id = upload(out_path, data)
    log(f"Done: https://youtube.com/shorts/{video_id}")

    done.append(data["topic"])
    TOPICS_FILE.write_text(json.dumps(done, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"FAILED: {type(e).__name__}: {e}")
        sys.exit(1)
