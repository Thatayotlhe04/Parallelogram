#!/usr/bin/env python3
"""Check (or rebuild) the landing-page terminal recording.

The CLI panel plays a screen recording of a real bash session on the demo
dataset (scripts/demo_dataset.py): scripts/demo/record.py records the session
as an asciicast, scripts/demo/render.py renders it through xterm.js into
landing/demo/*.mp4. This script verifies that every command's output in the
recorded sessions is exactly what the installed CLI prints today, and keeps
the page's transcript and markup in step.

    python scripts/capture_terminal_demo.py           # verify
    python scripts/capture_terminal_demo.py --write   # rewrite the page block
    python scripts/capture_terminal_demo.py --record  # re-record the sessions (then run render.py)
"""
import argparse
import html
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / 'scripts/demo'
sys.path.insert(0, str(DEMO))
sys.path.insert(0, str(ROOT / 'scripts'))
from record import SCRIPT, project  # noqa: E402

PAGE = ROOT / 'landing/index.html'
START = '<!-- terminal-recording:start -->'
END = '<!-- terminal-recording:end -->'
SIZES = {'wide': (108, 22), 'narrow': (60, 30)}
EDITOR_FIRST, EDITOR_LINE, EDITOR_ROWS = 2128, 2131, 22   # train.jsonl lines in the editor; 2131 holds the key
EDITOR_CLIP = 150                                         # characters rendered per line; the rest is off-screen
TOKEN = re.compile(r'"(?:[^"\\]|\\.)*"|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|true|false|null|[{}\[\],:]|\s+|.')
ANSI = re.compile(r'\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07]*\x07|[\x01\x02]')


def commands():
    """The command lines the session runs, as bash sees them after history recall."""
    lines, last = [], []
    for action, value in SCRIPT:
        if action == 'type':
            last.append(value)
        elif action == 'up':
            last = [lines[-value]]
        elif action == 'enter':
            lines.append(''.join(last))
            last = []
    return lines


def fresh_runs(cols):
    """Run each command in a fresh copy of the demo project, as the recording did."""
    runs = []
    with tempfile.TemporaryDirectory(prefix='parallelogram-verify-') as tmp:
        home = Path(tmp)
        project(home)
        cwd = home / 'code/support-sft'
        env = {**os.environ, 'COLUMNS': str(cols), 'FORCE_COLOR': '1', 'TERM': 'xterm-256color',
               'PATH': os.pathsep.join([str(Path(sys.executable).parent), os.environ.get('PATH', '')])}
        env.pop('NO_COLOR', None)
        for line in commands():
            result = subprocess.run(['bash', '-c', line], cwd=cwd, env=env, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
            runs.append({'command': line, 'exit_code': result.returncode, 'output': result.stdout})
    return runs


def cast_text(path):
    lines = path.read_text().splitlines()
    header = json.loads(lines[0])
    text = ''.join(json.loads(line)[2] for line in lines[1:])
    return header, ANSI.sub('', text).replace('\r\n', '\n')


def verify_cast(name, cols, rows):
    path = DEMO / f'session-{cols}.cast'
    header, text = cast_text(path)
    assert (header['width'], header['height']) == (cols, rows), f'{path.name} is not {cols}×{rows}'
    runs = fresh_runs(cols)
    position = 0
    for run in runs:
        expected = ANSI.sub('', run['output'])
        found = text.find(expected, position)
        if found < 0:
            sys.exit(f'{path.name}: the recorded output of `{run["command"]}` differs from the CLI today; '
                     'run --record, then scripts/demo/render.py')
        position = found + len(expected)
    for media in (f'landing/demo/terminal-{cols}.mp4', f'landing/demo/terminal-{cols}.webp'):
        if not (ROOT / media).is_file():
            sys.exit(f'missing {media}; run scripts/demo/render.py')
    return runs


def json_html(raw, mark=None):
    """Highlight one JSONL line like an editor; `mark` is a (start, end) selection."""
    kinds = [''] * len(raw)
    tokens = [(m.start(), m.end(), m.group()) for m in TOKEN.finditer(raw)]
    for i, (a, b, value) in enumerate(tokens):
        if value.startswith('"'):
            following = next((t[2] for t in tokens[i + 1:] if not t[2].isspace()), '')
            kind = 'j-k' if following == ':' else 'j-s'
        elif value in '{}[],:':
            kind = 'j-p'
        elif value.isspace():
            kind = ''
        else:
            kind = 'j-n'
        kinds[a:b] = [kind] * (b - a)
    out, run, key = [], '', None
    for index, ch in enumerate(raw):
        state = (kinds[index], bool(mark) and mark[0] <= index < mark[1])
        if state != key and run:
            out.append(_span(run, *key))
            run = ''
        key, run = state, run + ch
    if run:
        out.append(_span(run, *key))
    return ''.join(out)


def _span(text, kind, selected):
    value = html.escape(text, quote=False)
    value = f'<span class="{kind}">{value}</span>' if kind else value
    return f'<span class="sel">{value}</span>' if selected else value


def editor_lines():
    """The real train.jsonl lines the editor shows, and the selected key on line 2131."""
    from demo_dataset import build, dumps
    lines = dumps(build()[0]).splitlines()
    target = lines[EDITOR_LINE - 1]
    start = target.index('sk-proj-')
    end = start + re.match(r'sk-proj-[A-Za-z0-9_-]+', target[start:]).end()
    rows = []
    for number in range(EDITOR_FIRST, EDITOR_FIRST + EDITOR_ROWS):
        raw = lines[number - 1][:EDITOR_CLIP]
        current = number == EDITOR_LINE
        rows.append(f'<div class="ed-line{" is-current" if current else ""}"><span class="ln">{number}</span>'
                    f'<span class="code">{json_html(raw, (start, end) if current else None)}</span></div>')
    return ''.join(rows), len(lines), start + 1, end - start


def render(runs):
    version = importlib.metadata.version('parallelogram')
    transcript = ''.join(f'$ {html.escape(run["command"])}\n{html.escape(ANSI.sub("", run["output"]))}\n'
                         for run in runs)
    train = re.search(r'(\d+) records', ANSI.sub('', runs[0]['output'])).group(1)
    issues = len(re.findall(r'^  [✗!] ', ANSI.sub('', runs[0]['output']), re.MULTILINE))
    clean = re.search(r'Wrote (\d+) records', ANSI.sub('', runs[2]['output'])).group(1)
    rows, total, column, selected = editor_lines()
    scroll = round(100 * (EDITOR_FIRST - 1) / total, 1)
    tree = ''.join(f'<div class="tree-row {cls}">{label}</div>' for cls, label in (
        ('root', '⌄ SUPPORT-SFT'), ('d1', '› .github'), ('d1', '⌄ configs'),
        ('d2', '<i class="fi fi-yml">≡</i>sft-qwen2.5-7b.yaml'), ('d1', '⌄ data'),
        ('d2', '<i class="fi fi-json">{}</i>eval.jsonl'), ('d2 is-active', '<i class="fi fi-json">{}</i>train.jsonl'),
        ('d1', '› scripts'), ('d1', '<i class="fi fi-git">◆</i>.gitignore'),
        ('d1', '<i class="fi fi-md">ⓘ</i>README.md'), ('d1', '<i class="fi fi-py">◆</i>train.py')))
    sources = ' '.join(f'data-{name}="demo/terminal-{cols}.mp4" data-{name}-poster="demo/terminal-{cols}.webp"'
                       for name, (cols, _) in SIZES.items())
    return f'''{START}
        <div class="scene" id="demo-desktop" data-layout="wide" aria-hidden="true">
          <div class="scene-stage" id="scene-stage">
            <div class="win-editor">
              <div class="ed-bar"><span class="lights"><i></i><i></i><i></i></span>train.jsonl — support-sft</div>
              <div class="ed-body">
                <div class="ed-tree"><div class="tree-head">EXPLORER</div>{tree}</div>
                <div class="ed-main">
                  <div class="ed-tabs"><span class="ed-tab is-active"><i class="fi fi-json">{{}}</i>train.jsonl</span><span class="ed-tab"><i class="fi fi-yml">≡</i>sft-qwen2.5-7b.yaml</span></div>
                  <div class="ed-crumbs">data <span>›</span> train.jsonl</div>
                  <div class="ed-code">{rows}<span class="ed-thumb" style="--at:{scroll}%"></span></div>
                </div>
              </div>
              <div class="ed-status"><span>⑂ main</span><span class="ed-status-right"><span>Ln {EDITOR_LINE}, Col {column} ({selected} selected)</span><span>Spaces: 4</span><span>UTF-8</span><span>LF</span><span>JSON Lines</span></span></div>
            </div>
            <div class="win-term" id="demo-window">
              <video id="demo-video" {sources} muted playsinline loop preload="none" disablepictureinpicture></video>
            </div>
          </div>
        </div>
        <figcaption class="scene-foot"><span><span class="local-dot" aria-hidden="true"></span> Screen recording · real bash session · synthetic {int(train):,}-ticket support export</span><span>parallelogram {version}</span></figcaption>
        <div id="terminal-static" class="terminal-static"><h3>{int(train):,} support conversations. {issues} issues. One clean file.</h3><p>The first check finds a pasted API key, customer contact details, eval leakage, duplicates, empty and unfinished turns, an undeclared tool call and mojibake, and exits 2. jq shows the key in record 2131. The fix writes {int(clean):,} records to train.clean.jsonl with the key redacted, and leaves train.jsonl untouched; the re-check exits 0.</p><pre>{transcript}</pre></div>
        {END}'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--write', action='store_true', help='rewrite the page block from fresh runs')
    parser.add_argument('--record', action='store_true', help='re-record both sessions')
    arguments = parser.parse_args()
    if arguments.record:
        for cols, rows in SIZES.values():
            subprocess.run([sys.executable, str(DEMO / 'record.py'), str(DEMO / f'session-{cols}.cast'),
                            str(cols), str(rows)], check=True)
        print('Recorded. Now render: python scripts/demo/render.py for each session.')
        sys.exit()
    if not shutil.which('jq'):
        sys.exit('jq is required to verify the recording')
    wide = None
    for name, (cols, rows) in SIZES.items():
        runs = verify_cast(name, cols, rows)
        if name == 'wide':
            wide = runs
    if [run['exit_code'] for run in wide] != [2, 0, 1, 0, 0]:
        sys.exit('The recorded CLI behavior changed; review before publishing.')
    block = render(wide)
    source = PAGE.read_text()
    if arguments.write:
        first, last = source.index(START), source.index(END) + len(END)
        PAGE.write_text(source[:first] + block + source[last:])
        print('Wrote the recording block.')
    elif block not in source:
        sys.exit('The page block does not match the recording; run --write.')
    else:
        print('Recorded sessions match the current CLI.')
