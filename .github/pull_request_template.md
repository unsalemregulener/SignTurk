## What changes

<!-- What and why, in a few sentences. -->

## Checks

- [ ] `ruff check .`
- [ ] `python -m unittest discover -s tests`
- [ ] `python -m text_processing.eval --check --min-exact 0.98`
- [ ] `node tools/check_jsx.mjs`, for interface changes
- [ ] `python tests/backend_smoke.py`, for recognition or API changes (needs the weights)
- [ ] The README or the model card follows any change in behaviour or numbers
