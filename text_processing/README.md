# text_processing

Turns the stream of approved sign glosses into a grammatical Turkish
sentence (and optional speech). A deterministic **rule-based engine** is the
primary path; an optional **ML layer** (Hugging Face Inference API or local
transformers) proposes a candidate that is validated and arbitrated against
the rule-based output per request.

## Architecture

Dependencies point inward: pure domain at the centre, I/O at the edges.

```
glosses ─▶ lexicon ─▶ pipeline ─▶ grammar ─▶ tts
                                    │
                    web (FastAPI) ──┘
```

| module | responsibility |
| --- | --- |
| `lexicon` | Turkish spelling of the 226 AUTSL glosses (`cay` → `çay`, `afiyet_olsun` → `afiyet olsun`) |
| `buffer` | word buffer with silence detection |
| `pipeline` | `SignTextPipeline`: words in, sentence (+ optional audio) out; waits for at least two words |
| `grammar/linguistics` | Turkish phonology and morphology, no I/O |
| `grammar/rules` | `RuleBasedCorrector`: SOV order, case suffixes, present continuous, copula (`ben evli` → `Ben evliyim.`) |
| `grammar/registry` | `ModelSpec`, `MODEL_REGISTRY`, `GrammarConfig` |
| `grammar/prompts` | versioned zero-shot prompt templates |
| `grammar/validation` | sanity checks on ML candidates |
| `grammar/adapters` | model backends and ML error classification |
| `grammar/arbiter` | per-request ML-vs-rule comparison |
| `grammar/telemetry` | in-process decision log for the health endpoint |
| `grammar/corrector` | `GrammarCorrector`, the public hybrid facade |
| `tts` | gTTS with an optional offline Piper fallback |
| `web/` | Pydantic schemas, bounded pipeline cache, async job store and the `/api/text/*` routers |

The live application maps glosses through `lexicon` before calling the
engine. Free text sent to `/api/text/correct` is used as is, because a few
glosses are also ordinary words with another meaning (`gol`, `sinir`, `hali`).

## Conventions

- Code and docstrings are written in English.
- User-facing strings (interface, `ml_error` and `ping` messages) are Turkish.
- The rule-based engine is the floor and the fallback for every ML error; it
  is never removed.
- The decision log records counts, not the signed words.

## Validation

```bash
python -m unittest discover -s tests -v
python -m text_processing.eval --check --min-exact 0.98   # rule-based floor
```

To try the engine or the API without TensorFlow, ONNX Runtime or a camera:

```bash
python -m text_processing.demo --words "ben çay içmek"   # Ben çay içiyorum.
python -m text_processing.devserver                      # http://127.0.0.1:8000/docs
```

## Notes

- The rule-based engine covers common sentence patterns; some constructions,
  such as questions, a few case suffixes and proper nouns, are simplified.
- The optional local ML stack should use transformers ≥ 4.48, torch ≥ 2.6 and
  sentencepiece ≥ 0.2.1; older releases have published CVEs.
