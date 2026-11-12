#!/usr/bin/env python3
"""Record a real interactive bash session on the demo dataset (asciicast v2).

The keystrokes are scripted; everything else is real: readline echoes them,
bash runs the commands, and parallelogram and jq write their own output to a
real pty of the given size.

    python scripts/demo/record.py OUT.cast COLS ROWS

Needs bash, git, jq, and parallelogram installed next to this Python.
"""
import fcntl
import json
import os
from pathlib import Path
import pty
import random
import select
import shutil
import struct
import subprocess
import sys
import tempfile
import termios
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demo_dataset import build, dumps

# What the visitor sees typed, in order. ("up", n) recalls history n entries back.
SCRIPT = [
    ('pause', 0.9),
    ('type', 'parallelogram check data/train.jsonl --eval data/eval.jsonl'), ('enter', 3.0),
    ('type', "sed -n 2131p data/train.jsonl | jq '.messages[0]'"), ('enter', 2.6),
    ('up', 2), ('type', ' --fix -o data/train.clean.jsonl'), ('enter', 2.8),
    ('type', "sed -n 2125p data/train.clean.jsonl | jq '.messages[0]'"), ('enter', 2.4),
    ('type', 'parallelogram check data/train.clean.jsonl --eval data/eval.jsonl'), ('enter', 4.6),
    ('clear', 0.8),   # Ctrl-L returns to the opening frame, so the video loops seamlessly
]
BASHRC = r'''
export LANG=C.UTF-8 LC_ALL=C.UTF-8
HISTFILE=/dev/null
__st=0; __first=1
__prompt() { __st=$?; if [ "$__first" = 1 ]; then __first=0; else echo; fi; }
PROMPT_COMMAND=__prompt
__char() { if [ "$__st" -eq 0 ]; then printf '\001\033[38;2;110;208;140m\002❯\001\033[0m\002'; else printf '\001\033[38;2;244;124;124m\002❯\001\033[0m\002'; fi; }
PS1='\[\033[38;2;122;166;218m\]\w\[\033[0m\] \[\033[38;2;110;110;110m\]$(git branch --show-current 2>/dev/null)\[\033[0m\]\n$(__char) '
cd ~/code/support-sft
'''
FILES = {
    'configs/sft-qwen2.5-7b.yaml': 'base_model: Qwen/Qwen2.5-7B-Instruct\ndatasets:\n  - path: data/train.clean.jsonl\n'
                                   '    type: chat_template\nval_set: data/eval.jsonl\nsequence_len: 4096\n',
    '.gitignore': 'data/*.jsonl\n.venv/\n__pycache__/\n',
    'README.md': '# support-sft\n\nFine-tune Qwen2.5-7B on exported support tickets.\n',
    'train.py': '# training entry point; see configs/\n',
}


def project(home):
    root = home / 'code/support-sft'
    (root / 'data').mkdir(parents=True)
    for name, text in FILES.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
    train, eval_rows = build()
    (root / 'data/train.jsonl').write_text(dumps(train))
    (root / 'data/eval.jsonl').write_text(dumps(eval_rows))
    git = ['git', '-c', 'user.name=dev', '-c', 'user.email=dev@example.com']
    subprocess.run([*git, 'init', '-q', '-b', 'main'], cwd=root, check=True)
    subprocess.run([*git, 'add', '-A'], cwd=root, check=True)
    subprocess.run([*git, 'commit', '-qm', 'export and training config'], cwd=root, check=True)


def record(out, cols, rows):
    for tool in ('bash', 'git', 'jq'):
        if not shutil.which(tool):
            sys.exit(f'{tool} is required')
    with tempfile.TemporaryDirectory(prefix='parallelogram-session-') as tmp:
        home = Path(tmp)
        project(home)
        (home / '.bashrc').write_text(BASHRC)
        path = os.pathsep.join([str(Path(sys.executable).parent), '/usr/local/bin', '/usr/bin', '/bin'])
        env = {'HOME': str(home), 'TERM': 'xterm-256color', 'LANG': 'C.UTF-8', 'LC_ALL': 'C.UTF-8',
               'PATH': path, 'COLORTERM': 'truecolor'}
        pid, fd = pty.fork()
        if pid == 0:
            fcntl.ioctl(0, termios.TIOCSWINSZ, struct.pack('HHHH', rows, cols, 0, 0))
            os.execve(shutil.which('bash'), ['bash', '--noprofile', '--rcfile', str(home / '.bashrc'), '-i'], env)
        events, rng = [], random.Random(4)
        start = time.monotonic()

        def pump(seconds):
            end = time.monotonic() + seconds
            while (left := end - time.monotonic()) > 0:
                if select.select([fd], [], [], left)[0]:
                    try:
                        data = os.read(fd, 65536)
                    except OSError:
                        return
                    events.append([round(time.monotonic() - start, 4), data.decode('utf-8', 'replace')])

        def wait_prompt():
            mark, end = len(events), time.monotonic() + 60
            while time.monotonic() < end:
                pump(0.1)
                if '❯' in ''.join(e[1] for e in events[mark:]) and time.monotonic() - start - events[-1][0] > 0.35:
                    return
            raise TimeoutError('the shell did not return to a prompt')

        wait_prompt()
        events.clear()
        start = time.monotonic()
        os.write(fd, b'\x0c')
        wait_prompt()
        for action, value in SCRIPT:
            if action == 'pause':
                pump(value)
            elif action == 'type':
                for ch in value:
                    os.write(fd, ch.encode())
                    pump(rng.uniform(0.038, 0.085) + (0.06 if ch == ' ' and rng.random() < 0.3 else 0))
            elif action == 'up':
                for _ in range(value):
                    os.write(fd, b'\x1b[A')
                    pump(0.5)
            elif action == 'enter':
                pump(0.32)
                os.write(fd, b'\r')
                wait_prompt()
                pump(value)
            elif action == 'clear':
                os.write(fd, b'\x0c')
                pump(value)
        os.kill(pid, 9)
    header = {'version': 2, 'width': cols, 'height': rows, 'env': {'SHELL': '/bin/bash', 'TERM': 'xterm-256color'}}
    with open(out, 'w') as f:
        f.write(json.dumps(header) + '\n')
        for t, data in events:
            f.write(json.dumps([t, 'o', data], ensure_ascii=False) + '\n')
    return events[-1][0]


if __name__ == '__main__':
    out, cols, rows = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    print(f'{out}: {record(out, cols, rows):.1f}s')
