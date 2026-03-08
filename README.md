# Silver Fund Quant
Python package for Silver Fund quant team research and trading tools. 

## Installation

To install run

```bash
pip install sf-quant
```

## Environment Configuration

`sf-quant` requires two environment variables to locate data: `ROOT` and `DATABASE`.

### Option 1: `.env` file or environment variables

Create a `.env` file in your project root (automatically loaded via `python-dotenv`):

```
ROOT=/path/to/root
DATABASE=your_database
```

Or export them in your shell:

```bash
export ROOT=/path/to/root
export DATABASE=your_database
```

### Option 2: Programmatic configuration

Set values at runtime using `sfd.env()`:

```python
import sf_quant.data as sfd

sfd.env(root="/path/to/root", database="your_database")
```

This will override any values set via environment variables.

## Documentation Development

To run a local server of the sphinx documentation run

```bash
uv run sphinx-autobuild docs docs/_build/html
```

## Release Process
1. Create PR
2. Merge PR(s)
3. Increment version in pyproject.toml
4. git tag v*.*.*
5. git push origin main --tags
6. Create a release and publish release notes (github)
7. uv build
8. uv publish
