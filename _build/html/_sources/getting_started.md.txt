# Getting Started

This page contains short instructions to build the docs locally and where to find examples.

## Build locally

Install the documentation dependencies and build:

```bash
python -m pip install -r docs/requirements.txt
python -m pip install -e .
cd docs
sphinx-build -b html . _build/html
```

The built HTML will be in `docs/_build/html`.

## Examples

See the `soliton_solver/examples` module for runnable examples you can adapt into tutorials.
