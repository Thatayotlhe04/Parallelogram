#!/usr/bin/env python3
"""Reproduce the landing terminal recording from the installed local CLI."""
import argparse
import importlib.metadata
import html
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / 'landing/terminal-recording.json'
PAGE = ROOT / 'landing/index.html'
START = '<!-- terminal-recording:start -->'
END = '<!-- terminal-recording:end -->'


def ansi_html(text):
    """Preserve terminal characters and SGR styling; never reinterpret output."""
    classes = set()
    output = []
    palette = {1: 'ansi-bold', 2: 'ansi-dim', 31: 'ansi-red', 32: 'ansi-green',
               33: 'ansi-yellow', 36: 'ansi-cyan', 95: 'ansi-magenta'}
    for part in re.split(r'(\x1b\[[0-9;]*m)', text):
        if part.startswith('\x1b['):
            for number in (int(value or '0') for value in part[2:-1].split(';')):
                if number == 0:
                    classes.clear()
                elif number in palette:
                    classes.add(palette[number])
                else:
                    raise ValueError(f'Unsupported terminal style: {number}')
        elif part:
            value = html.escape(part)
            output.append(f'<span class="{" ".join(sorted(classes))}">{value}</span>' if classes else value)
    return ''.join(output)


def render(recording):
    runs = []
    transcript = []
    for clip in recording['commands']:
        command = html.escape(clip['command'])
        output = ansi_html(clip['ansi'])
        runs.append(f'<div class="terminal-run" data-run><div class="terminal-command"><span class="shell-prompt">❯</span> <span data-command="{command}">{command}</span><span class="shell-cursor" aria-hidden="true"></span></div><pre class="terminal-output" data-output>{output}</pre></div>')
        transcript.append(f'<span class="shell-prompt">❯</span> {command}\n{output}')
    return f'''{START}
        <figcaption><span class="mac-lights" aria-hidden="true"><i></i><i></i><i></i></span><span class="terminal-path">datasets — zsh</span><span class="terminal-window-spacer" aria-hidden="true"></span></figcaption>
        <div class="terminal-scroll" id="terminal-screen" tabindex="0" role="region" aria-label="Automatic terminal demonstration. Focus to pause and read." hidden><div class="terminal-history" aria-hidden="true">{''.join(runs)}<div class="terminal-idle" id="terminal-idle"><span class="shell-prompt">❯</span> <span id="terminal-clear"></span><span class="shell-cursor"></span></div></div></div>
        <div id="terminal-static" class="terminal-static"><pre>{''.join(transcript)}</pre></div>
        {END}'''


def capture():
    def chat(question, answer):
        return {'messages': [{'role': 'user', 'content': question},
                             {'role': 'assistant', 'content': answer}]}

    clean = chat('What is 2 + 2?', '4')
    held = chat('Capital of France?', 'Paris.')
    # This synthetic key is the existing detector-test fixture, not a credential.
    rows = [clean,
            chat('Review this config: sk-proj-Ab3dEf9hIjK2mNoPqR5sTuVwXyZ01234',
                 'Keep credentials out of training data.'),
            {'prompt': 'Choose a greeting.', 'chosen': 'Hello.', 'rejected': 'Hello.'},
            clean, chat('Who should I contact?', 'Itâ€™s team@acme.com.'), held]
    commands = [
        ['check', 'train.jsonl', '--eval', 'eval.jsonl'],
        ['check', 'train.jsonl', '--eval', 'eval.jsonl', '--fix', '--output', 'clean.jsonl'],
        ['check', 'clean.jsonl', '--eval', 'eval.jsonl'],
    ]
    env = {**os.environ, 'FORCE_COLOR': '1', 'COLUMNS': '96', 'TERM': 'xterm-256color'}
    env.pop('NO_COLOR', None)
    clips = []
    with tempfile.TemporaryDirectory(prefix='parallelogram-demo-') as directory:
        work = Path(directory)
        (work / 'train.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
        (work / 'eval.jsonl').write_text(json.dumps(held) + '\n')
        for arguments in commands:
            result = subprocess.run(
                [sys.executable, '-c', 'from parallelogram.cli import app; app()', *arguments],
                cwd=work, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, check=False,
            )
            clips.append({'command': 'parallelogram ' + ' '.join(arguments),
                          'exit_code': result.returncode, 'ansi': result.stdout})
    if [clip['exit_code'] for clip in clips] != [2, 1, 0]:
        raise RuntimeError('The recorded CLI behavior changed; review before publishing.')
    return {'version': importlib.metadata.version('parallelogram'),
            'columns': 96, 'commands': clips}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true', help='Replace the saved recording')
    arguments = parser.parse_args()
    recording = capture()
    if arguments.write:
        DESTINATION.write_text(json.dumps(recording, ensure_ascii=False, indent=2) + '\n')
        source = PAGE.read_text()
        if START in source:
            first, last = source.index(START), source.index(END) + len(END)
            PAGE.write_text(source[:first] + render(recording) + source[last:])
        print('Captured actual CLI output and exit codes.')
    elif recording != json.loads(DESTINATION.read_text()):
        sys.exit('Recording does not match the current CLI.')
    elif render(recording) not in PAGE.read_text():
        sys.exit('The displayed transcript does not match the recording.')
    else:
        print('Recording matches the current CLI.')
