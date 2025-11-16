# Installing CIVIC as a Python Package

## Development Installation (Editable)

For development, install in editable mode so changes are immediately reflected:

```bash
# From the civic directory
pip install -e .
```

After installation, you can use the `civic` command anywhere:

```bash
# Instead of: python scripts/civic.py run config/my_project.yaml
civic run config/my_project.yaml

# All commands work the same:
civic run config/my_project.yaml --dry-run
civic run config/my_project.yaml --auto
civic run config/my_project.yaml --skip-existing

# Individual steps:
civic tile config/my_project.yaml
civic annotate config/my_project.yaml
civic review config/my_project.yaml
civic reassemble config/my_project.yaml
```

## Production Installation

For production deployment or distribution:

```bash
# Build the package
pip install build
python -m build

# This creates:
# - dist/river-segmentation-0.1.0.tar.gz
# - dist/river_segmentation-0.1.0-whl

# Install from wheel
pip install dist/river_segmentation-0.1.0-py3-none-any.whl
```

## Available CLI Commands

After installation, you'll have these commands available:

| Command | Description |
|---------|-------------|
| `civic` | Main pipeline command (tile → annotate → review → reassemble) |
| `civic-download-models` | Download required model weights |
| `civic-check-hardware` | Check GPU/CPU capabilities |

## Verify Installation

```bash
# Check that civic is installed
civic --help

# Check version
python -c "import civic; print(civic.__version__)"

# Test with dry-run
civic run config/quinhagak_test_subset.yaml --dry-run
```

## Uninstall

```bash
pip uninstall river-segmentation
```

## Publishing to PyPI (Future)

When ready to publish to PyPI:

```bash
# Install twine
pip install twine

# Upload to Test PyPI first
python -m twine upload --repository testpypi dist/*

# Test installation from Test PyPI
pip install --index-url https://test.pypi.org/simple/ river-segmentation

# Upload to PyPI
python -m twine upload dist/*

# Users can then install with:
# pip install river-segmentation
```

## Entry Points Configured

The following entry points are configured in `pyproject.toml`:

```toml
[project.scripts]
civic = "civic.cli:main"
civic-download-models = "river_segmentation.utils.model_downloader:main"
civic-check-hardware = "river_segmentation.utils.hardware_detect:main"
```

This means after installation:
- `civic` → runs the main pipeline CLI
- `civic-download-models` → downloads model weights
- `civic-check-hardware` → checks hardware capabilities
