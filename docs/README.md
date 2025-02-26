# Advanced AI Assistant Documentation

This directory contains the documentation for the Advanced AI Assistant.

## Building the Documentation

The documentation is built using [MkDocs](https://www.mkdocs.org/) with the [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) theme.

To build the documentation:

1. Install the required dependencies:
   ```bash
   pip install mkdocs mkdocs-material
   ```

2. Build the documentation:
   ```bash
   mkdocs build
   ```

3. Serve the documentation locally:
   ```bash
   mkdocs serve
   ```

4. View the documentation in your browser at `http://localhost:8000`.

## Documentation Structure

- `index.md`: Main documentation page
- `configuration.md`: Configuration guide
- `api-reference.md`: API reference (to be created)
- `contributing.md`: Contributing guide (to be created)

## Adding New Documentation

To add new documentation:

1. Create a new Markdown file in the `docs` directory.
2. Add the file to the `nav` section in `mkdocs.yml`.
3. Build and serve the documentation to preview your changes.

## Generating API Reference

The API reference can be generated using [pdoc](https://pdoc.dev/):

```bash
pip install pdoc
pdoc --html --output-dir docs/api src
``` 