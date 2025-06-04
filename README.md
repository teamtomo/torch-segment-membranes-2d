# torch-segment-membranes-2d

[![License](https://img.shields.io/pypi/l/torch-segment-membranes-2d.svg?color=green)](https://github.com/tlambert03/torch-segment-membranes-2d/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/torch-segment-membranes-2d.svg?color=green)](https://pypi.org/project/torch-segment-membranes-2d)
[![Python Version](https://img.shields.io/pypi/pyversions/torch-segment-membranes-2d.svg?color=green)](https://python.org)
[![CI](https://github.com/tlambert03/torch-segment-membranes-2d/actions/workflows/ci.yml/badge.svg)](https://github.com/tlambert03/torch-segment-membranes-2d/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/tlambert03/torch-segment-membranes-2d/branch/main/graph/badge.svg)](https://codecov.io/gh/tlambert03/torch-segment-membranes-2d)

Segment membranes in cryo-EM images

## Development

The easiest way to get started is to use the [github cli](https://cli.github.com)
and [uv](https://docs.astral.sh/uv/getting-started/installation/):

```sh
gh repo fork tlambert03/torch-segment-membranes-2d --clone
# or just
# gh repo clone tlambert03/torch-segment-membranes-2d
cd torch-segment-membranes-2d
uv sync
```

Run tests:

```sh
uv run pytest
```

Lint files:

```sh
uv run pre-commit run --all-files
```
