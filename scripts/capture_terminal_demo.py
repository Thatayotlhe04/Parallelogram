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
    # Encode literal line-end spaces so the exact terminal text survives clean diffs.
    return re.sub(r' +(?=\n)', lambda match: '&#32;' * len(match.group()), ''.join(output))


def record_html(records, clean=False):
    """Show the actual file as JSONL with visual wrapping, not invented records."""
    output = []
    for index, record in enumerate(records):
        messages = record['messages']
        lines = ['{<span class="json-key">"messages"</span>: [']
        for n, message in enumerate(messages):
            role = html.escape(json.dumps(message['role']))
            content = html.escape(json.dumps(message['content'], ensure_ascii=False))
            if 'maya@acme.com' in content:
                content = content.replace('maya@acme.com', '<mark class="private-value">maya@acme.com</mark>')
            if '[REDACTED_EMAIL]' in content:
                content = content.replace('[REDACTED_EMAIL]', '<mark class="redacted-value">[REDACTED_EMAIL]</mark>')
            comma = ',' if n < len(messages) - 1 else ''
            lines.append('  {<span class="json-key">"role"</span>: <span class="json-string">' + role + '</span>, <span class="json-key">"content"</span>: <span class="json-string">' + content + '</span>}' + comma)
        lines.append(']}')
        issue = '' if clean else (' duplicate-record' if index == 1 else ' private-record' if index == 2 else '')
        joined = '\n'.join(lines)
        output.append(f'<div class="source-record{issue}"><span class="record-number">{index + 1}</span><pre>{joined}</pre></div>')
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
        <div class="demo-desktop" id="demo-desktop" data-scene="intro" aria-hidden="true">
          <div class="editor-window">
            <div class="desktop-titlebar"><span class="mac-lights"><i></i><i></i><i></i></span><span>support-model</span><span class="window-symbol"></span></div>
            <div class="editor-layout">
              <aside class="editor-sidebar"><span class="explorer-title">EXPLORER</span><div class="project-folder">⌄ &nbsp; support-model</div><div class="explorer-file original-file"><span class="file-icon">&#123;&#125;</span> train.jsonl <span class="problem-count">2</span></div><div class="explorer-file clean-file"><span class="file-icon">&#123;&#125;</span> clean.jsonl <span class="file-new">U</span></div><div class="explorer-file"><span class="python-icon">◇</span> train.py</div><div class="explorer-file"><span class="readme-icon">≡</span> README.md</div></aside>
              <div class="editor-main">
                <div class="editor-tabs"><span class="editor-tab original-file"><span class="file-icon">&#123;&#125;</span> train.jsonl <span>×</span></span><span class="editor-tab clean-file"><span class="file-icon">&#123;&#125;</span> clean.jsonl <span>×</span></span></div>
                <div class="editor-breadcrumb">support-model <span>›</span> <span id="editor-filename">train.jsonl</span><span class="jsonl-label">JSON Lines</span></div>
                <div class="editor-code"><div id="source-before">{record_html(recording['before'])}</div><div id="source-after" hidden>{record_html(recording['after'], clean=True)}</div></div>
                <div class="editor-diagnostic" id="editor-diagnostic"><span class="diagnostic-dot"></span><span id="editor-note">3 training examples · customer support</span></div>
              </div>
            </div>
            <div class="editor-statusbar"><span>⑂ main</span><span id="editor-status">UTF-8 &nbsp; · &nbsp; JSONL</span></div>
          </div>
          <div class="walkthrough-caption"><span id="story-label">YOUR TRAINING DATA</span><h3 id="story-title">Small file. Hidden problems.</h3><p id="story-body">A repeated example. A private email. Both easy to miss in a training set.</p></div>
          <div class="demo-terminal-window">
            <div class="desktop-titlebar"><span class="mac-lights"><i></i><i></i><i></i></span><span>support-model — zsh</span><span class="terminal-window-spacer"></span></div>
            <div class="terminal-scroll" id="terminal-screen"><div class="terminal-history">{''.join(runs)}<div class="terminal-idle" id="terminal-idle"><span class="shell-prompt">❯</span> <span class="shell-cursor"></span></div></div></div>
          </div>
        </div>
        <figcaption class="walkthrough-footer"><span><span class="local-dot" aria-hidden="true"></span> On your machine. Before training.</span><span>Example dataset · actual CLI output</span></figcaption>
        <div id="terminal-static" class="terminal-static"><h3>Three examples. Two issues. One clean file.</h3><p>Parallelogram finds a duplicate record and a private email in train.jsonl. The fix writes two records to clean.jsonl: the duplicate is removed and the email becomes [REDACTED_EMAIL]. The original file stays untouched. A second check passes with no errors or warnings.</p><pre>{''.join(transcript)}</pre></div>
        {END}'''


def capture():
    def chat(question, answer):
        return {'messages': [{'role': 'user', 'content': question},
                             {'role': 'assistant', 'content': answer}]}

    clean = chat('Can I return an item?', 'Returns are accepted within 30 days.')
    rows = [clean, clean, chat('Where is my order?', 'Email maya@acme.com for an update.')]
    commands = [
        ['check', 'train.jsonl'],
        ['check', 'train.jsonl', '--fix', '--output', 'clean.jsonl'],
        ['check', 'clean.jsonl'],
    ]
    env = {**os.environ, 'FORCE_COLOR': '1', 'COLUMNS': '72', 'TERM': 'xterm-256color'}
    env.pop('NO_COLOR', None)
    clips = []
    with tempfile.TemporaryDirectory(prefix='parallelogram-demo-') as directory:
        work = Path(directory)
        (work / 'train.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
        for arguments in commands:
            result = subprocess.run(
                [sys.executable, '-c', 'from parallelogram.cli import app; app()', *arguments],
                cwd=work, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, check=False,
            )
            clips.append({'command': 'parallelogram ' + ' '.join(arguments),
                          'exit_code': result.returncode, 'ansi': result.stdout})
        fixed = [json.loads(line) for line in (work / 'clean.jsonl').read_text().splitlines()]
    if [clip['exit_code'] for clip in clips] != [2, 1, 0]:
        raise RuntimeError('The recorded CLI behavior changed; review before publishing.')
    return {'version': importlib.metadata.version('parallelogram'),
            'columns': 72, 'before': rows, 'after': fixed, 'commands': clips}


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
