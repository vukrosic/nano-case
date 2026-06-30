# nano-case

A **1,016,960-parameter** (~1M) byte-level transformer that converts a messy
identifier to a target case style — `snake`, `kebab`, `camel`, `pascal`, `const` —
and, crucially, segments **boundary-destroyed** inputs (no separators, one global
case: `sdkmodel`, `SENDREQUEST`) that a regular expression provably cannot split.

```
const  | sdkmodel          => SDK_MODEL
camel  | usertablehandler  => userTableHandler
kebab  | sqlqueryname      => sql-query-name
snake  | md5cache          => md5_cache
pascal | rendertokenerror  => RenderTokenError
```

It runs on a CPU in milliseconds and was trained **entirely on code-generated
data** — no scraping, no labelling, no distillation. Weights + a self-contained
inference file are here and on
[Hugging Face](https://huggingface.co/vukrosic/nano-case).

📄 [Technical report (PDF)](report/nano-case-report.pdf) · 📝 [Full writeup (gist)](https://gist.github.com/vukrosic/ee6bcffad64d90efa59e4874f78a4397)

## Why a model and not a regex

Converting a *clean* identifier between cases is a solved, free problem — a regex
splits on separators and camel-humps and re-renders. So that slice has no value.
The value is the **regex-killer slice**: inputs where the separators and casing are
gone (`userprofilecache`, `HTTPREQUESTHANDLER`), leaving nothing to split on. The
only way back to the intended words is a **learned vocabulary**. That is what
nano-case has, and what a script cannot have.

## Benchmark (held-out, seed 987654321, N=4000)

Exact-match accuracy, model vs a standard regex case-converter. Mean ± std over 3
training seeds (0/1/2).

|                | model | regex script |
|----------------|------:|-------------:|
| **overall**    | **99.8% ± 0.0%** | 61.8% |
| **smushed slice** (N=1410) | **99.7% ± 0.0%** | **8.2%** |

The **smushed slice** is the regex-killer: boundary-destroyed, single-case,
multi-word inputs. **8.2%** for the script vs **99.7%** for the model is the
"you genuinely need a model here" result.

Reproduce: `python eval_nano_case.py --n 4000`.

## Where it breaks (out-of-distribution)

nano-case's segmentation prior **is** its ~120-token training vocabulary. Honest
limits, measured:

| input type | accuracy |
|---|---:|
| in-vocab smushed (the trained slice) | 100% |
| **out-of-vocabulary words smushed** | 2% |
| chains longer than trained (5–6 words) | 33% |

So it nails smushing of *known* words and degrades on unknown tokens / very long
chains — the expected ceiling of a 1M model on a vocabulary task, reported rather
than hidden.

## Use it

```bash
pip install -r requirements.txt
python modeling_nano_case.py            # demo
```
```python
from modeling_nano_case import load, to_case
m = load("model.safetensors", "config.json")
to_case(m, "const", "sdkmodel")          # -> "SDK_MODEL"
to_case(m, "camel", "user_table_handler")# -> "userTableHandler"
```

## How it was trained

Code-generated data (sample words from a fixed vocabulary → render the gold target
canonically → corrupt a copy into a messy input, ~45% boundary-destroyed), SFT with
the prompt masked so only the target + newline EOS is supervised. ~1M-param
byte-level transformer (RMSNorm, RoPE, GQA, SwiGLU), 12k steps, AdamW, cosine LR.
Full recipe and exact config in [TRAINING.md](TRAINING.md).

## Files

- `modeling_nano_case.py` — self-contained model + `load()` / `to_case()` (torch + safetensors only).
- `data_cases.py` — the code data generator (shared by train and eval).
- `eval_nano_case.py` — the model-vs-regex benchmark.
- `test_nano_case.py` — labels-correct / no-leakage / determinism / published-weights regression.
- `model.safetensors`, `config.json` — weights + architecture.
- `report/nano-case-report.pdf` — the technical report.
- `TRAINING.md` — reproduction recipe.

## License

MIT. Built by **Vuk Rosić**.
