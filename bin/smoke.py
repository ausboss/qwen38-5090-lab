#!/usr/bin/env python3
"""Quick check that the local Qwen3.8 server does the things you actually use:
a system prompt, back-and-forth chat, and reading an image.

Run via:  qwen test
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
import urllib.request

import labauth

_ap = argparse.ArgumentParser()
_ap.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
_ap.add_argument("--model", default="qwen38")
_args, _ = _ap.parse_known_args()

BASE = _args.base_url
MODEL = _args.model


def call(messages, max_tokens=800, **extra):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens,
            "temperature": 0.3, **extra}
    req = urllib.request.Request(
        f"{BASE}/chat/completions", data=json.dumps(body).encode(),
        headers=labauth.headers())
    t0 = time.perf_counter()
    d = json.loads(urllib.request.urlopen(req, timeout=600).read())
    msg = d["choices"][0]["message"]
    return (msg.get("content") or "").strip(), d.get("usage", {}), time.perf_counter() - t0


def make_test_image() -> str:
    """White image: a red circle plus the text '42', rendered with a real font
    so a wrong answer means the model misread it, not that we drew it badly."""
    import glob
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (420, 220), "white")
    d = ImageDraw.Draw(img)
    d.ellipse([30, 55, 160, 185], fill="red")

    font = None
    for pat in ("/usr/share/fonts/**/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/**/*Bold.ttf",
                "/usr/share/fonts/**/*.ttf"):
        hits = glob.glob(pat, recursive=True)
        if hits:
            try:
                font = ImageFont.truetype(hits[0], 130)
                break
            except Exception:  # noqa: BLE001
                continue
    if font is None:  # last resort: tiny bitmap font, scaled up
        tmp = Image.new("RGB", (60, 20), "white")
        ImageDraw.Draw(tmp).text((2, 2), "42", fill="black")
        img.paste(tmp.resize((190, 130), Image.LANCZOS), (210, 55))
    else:
        d.text((215, 40), "42", fill="black", font=font)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def main():
    ok = True

    print("1. system prompt + chat")
    sys_msg = {"role": "system",
               "content": "You are a terse assistant. Answer in under 10 words."}
    ans, usage, dt = call([sys_msg, {"role": "user", "content": "What is the capital of Japan?"}])
    print(f"   -> {ans!r}  ({dt:.1f}s, {usage.get('completion_tokens')} tok, "
          f"{usage.get('reasoning_tokens', 0)} thinking)")
    ok &= "tokyo" in ans.lower()

    print("2. multi-turn (does it keep context?)")
    convo = [sys_msg,
             {"role": "user", "content": "My favourite colour is teal. Remember it."},
             {"role": "assistant", "content": "Noted."},
             {"role": "user", "content": "What is my favourite colour?"}]
    ans, _, dt = call(convo)
    print(f"   -> {ans!r}  ({dt:.1f}s)")
    ok &= "teal" in ans.lower()

    print("3. image understanding")
    try:
        b64 = make_test_image()
        ans, _, dt = call([{"role": "user", "content": [
            {"type": "text",
             "text": "What number and what shape do you see? Answer briefly."},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]}], max_tokens=900)
        print(f"   -> {ans!r}  ({dt:.1f}s)")
        got = ("42" in ans) or ("circle" in ans.lower()) or ("red" in ans.lower())
        ok &= got
        if not got:
            print("   (image was received but not described as expected)")
    except Exception as e:  # noqa: BLE001
        print(f"   IMAGE TEST FAILED: {e!r}")
        ok = False

    print("\n" + ("ALL GOOD" if ok else "SOMETHING IS OFF — see above"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
