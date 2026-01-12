# Silver Fund Quant
Python package for Silver Fund quant team research and trading tools. 

## Installation

To install run

```bash
pip install sf-quant
```

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
