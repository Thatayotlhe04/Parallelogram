#!/usr/bin/env python3
"""Render an asciicast through xterm.js in Chrome, frame by frame, then encode MP4.

    npm install --prefix scripts/demo
    python scripts/demo/render.py SESSION.cast OUT.mp4 POSTER.webp

Layout is drawn at 2x in CSS pixels so the video stays sharp on high-density
screens. Needs Playwright with a Chromium/Chrome, and ffmpeg.
"""
import asyncio
import functools
import http.server
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading

from playwright.async_api import async_playwright

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FPS = 30


def serve():
    """Serve the render page, xterm.js and fonts from one origin."""
    site = Path(tempfile.mkdtemp(prefix='parallelogram-render-'))
    modules = HERE / 'node_modules'
    copies = {
        'render.html': HERE / 'render.html',
        'xterm.js': modules / '@xterm/xterm/lib/xterm.js',
        'xterm.css': modules / '@xterm/xterm/css/xterm.css',
        'addon-canvas.js': modules / '@xterm/addon-canvas/lib/addon-canvas.js',
        'Inter-Variable.woff2': ROOT / 'landing/fonts/Inter-Variable.woff2',
    }
    for weight in ('400', '700'):
        name = f'jetbrains-mono-latin-{weight}-normal.woff2'
        copies[name] = modules / '@fontsource/jetbrains-mono/files' / name
    copies['jetbrains-mono-latin-ext-400-normal.woff2'] = modules / '@fontsource/jetbrains-mono/files/jetbrains-mono-latin-ext-400-normal.woff2'
    for name, source in copies.items():
        if not source.is_file():
            sys.exit(f'missing {source}; run: npm install --prefix scripts/demo')
        shutil.copy(source, site / name)
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass
    handler = functools.partial(Quiet, directory=str(site))
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, site


async def frames(cast, out_dir, poster_at):
    lines = Path(cast).read_text().splitlines()
    header, events = json.loads(lines[0]), [json.loads(line) for line in lines[1:]]
    server, site = serve()
    executable = shutil.which('google-chrome') or shutil.which('chromium')
    async with async_playwright() as p:
        browser = await p.chromium.launch(executable_path=executable, args=['--no-sandbox'])
        page = await browser.new_page(viewport={'width': 2200, 'height': 1800}, device_scale_factor=1)
        await page.goto(f'http://127.0.0.1:{server.server_port}/render.html')
        await page.evaluate(f'setup({header["width"]}, {header["height"]})')
        window = await page.query_selector('#win')
        i, count = 0, int((events[-1][0] + 0.05) * FPS) + 1
        poster = None
        for frame in range(count):
            t = frame / FPS
            chunk = []
            while i < len(events) and events[i][0] <= t:
                chunk.append(events[i][2])
                i += 1
            name = out_dir / f'f{frame:05d}.png'
            if chunk or frame == 0:
                await page.evaluate('c => feed(c)', chunk)
                await window.screenshot(path=name)
            else:
                os.link(out_dir / f'f{frame - 1:05d}.png', name)   # nothing changed on screen
            if poster is None and t >= poster_at:
                poster = name
        await browser.close()
    server.shutdown()
    shutil.rmtree(site)
    return poster


def main():
    cast, mp4, poster_out = sys.argv[1:4]
    duration = json.loads(Path(cast).read_text().splitlines()[-1])[0]
    with tempfile.TemporaryDirectory(prefix='parallelogram-frames-') as tmp:
        out_dir = Path(tmp)
        # The poster is the finished run, just before the closing Ctrl-L.
        poster = asyncio.run(frames(cast, out_dir, duration - 1.2))
        subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-framerate', str(FPS), '-i', str(out_dir / 'f%05d.png'),
                        '-c:v', 'libx264', '-preset', 'slow', '-tune', 'stillimage', '-crf', '22',
                        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-an', mp4], check=True)
        subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-i', str(poster), '-c:v', 'libwebp',
                        '-quality', '88', poster_out], check=True)
    print(f'{mp4}: {os.path.getsize(mp4) // 1024} KB, {poster_out}: {os.path.getsize(poster_out) // 1024} KB')


if __name__ == '__main__':
    main()
