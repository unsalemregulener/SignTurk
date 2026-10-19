# Contributing

## Setup

Use Python 3.11.

```bash
pip install -r requirements-dev.txt
python tools/download_models.py
```

## Conventions

- Code, comments and docstrings are in English. Interface text and API
  messages are in Turkish.
- Never commit `.env`, `local_dev.db` or model weights. `.gitignore` covers
  them.
- Changes to the sentence engine come with cases in
  `text_processing/eval/dataset.jsonl`.

## Checks

Run these before opening a pull request; CI runs the same ones.

```bash
ruff check .
python -m unittest discover -s tests
python -m text_processing.eval --check --min-exact 0.98
python tests/backend_smoke.py
```

Interface changes also go through the JSX check, with the Babel version that
`frontend/app/index.html` loads:

```bash
npm install --no-save @babel/standalone@7.29.9
node tools/check_jsx.mjs
```

Report security problems privately, as described in [SECURITY.md](SECURITY.md).

## Publishing new model weights

1. Upload the file to a new release (for example `model-assets-v2`).
2. Update its entry in `model-assets.json`: `url`, `bytes` and `sha256`.
3. Run `python tools/download_models.py --check`, then
   `python tools/download_models.py --force`.
4. Measure the model with `tools/evaluate_226.py`, and update the models
   section of the README and the model card.
