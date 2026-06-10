# Parallelogram Mac setup + PyPI release checklist

This repo has two parts:

- `cli/` — the Python package that becomes `pip install parallelogram`
- `landing/` — the static website for `parallelogram.dev`

## 1. Install Mac dev tools

```bash
xcode-select --install
```

Install Homebrew if you do not have it yet, then:

```bash
brew install python git gh
```

## 2. Set up the CLI locally

From the folder that contains `cli/` and `landing/`:

```bash
cd cli
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,tokenizer]'
pytest -q
```

Run the CLI:

```bash
parallelogram check examples/broken.jsonl
parallelogram check examples/broken.jsonl --fix --output clean.jsonl
```

## 3. Build the package locally

```bash
cd cli
source .venv/bin/activate
python -m pip install --upgrade build twine
rm -rf dist
python -m build
python -m twine check dist/*
```

You should see a `.whl` and `.tar.gz` file in `cli/dist/`.

## 4. Create PyPI + TestPyPI accounts

Create accounts on:

- https://pypi.org
- https://test.pypi.org

Enable 2FA on both.

## 5. First safe release: TestPyPI

Optional manual test upload:

```bash
python -m pip install --upgrade twine
python -m twine upload --repository testpypi dist/*
```

Then test install in a fresh folder:

```bash
python3 -m venv /tmp/pg-test
source /tmp/pg-test/bin/activate
python -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple parallelogram
parallelogram --help
```

## 6. Real release with GitHub Actions + PyPI Trusted Publishing

The included workflow is:

```text
.github/workflows/release.yml
```

On PyPI, add a Trusted Publisher for the `parallelogram` project:

- Owner: your GitHub username or org
- Repository name: your repo name
- Workflow filename: `release.yml`
- Environment name: `pypi`

Then release by bumping the version in `cli/pyproject.toml`, committing, tagging, and pushing:

```bash
git add cli/pyproject.toml
git commit -m "Release CLI v0.2.0"
git tag cli-v0.2.0
git push origin main --tags
```

The tag `cli-v0.2.0` triggers the PyPI publish workflow.

## 7. Update landing page after PyPI works

Once PyPI has the package, update the landing page install commands so they show real commands, not GitHub URL placeholders:

```bash
pip install parallelogram
parallelogram check data.jsonl
parallelogram check data.jsonl --fix --output clean.jsonl
```

Then deploy only the `landing/` folder to Vercel, Netlify, Cloudflare Pages, or GitHub Pages.
