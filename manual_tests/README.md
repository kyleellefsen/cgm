# Manual Tests

This directory contains tests that require manual/visual inspection and interaction.
It also serves to document how to code interactively with the visualization.

## Running Tests

### Visualization Tests

To run the visual inspection test for the conditioning interface:
```bash
python -m manual_tests.viz.visual_inspection
```

## Test Descriptions

### `viz/visual_inspection.py`
Tests the visual aspects of node conditioning in the graph visualization:
- Verifies that conditioning circles appear correctly
- Checks CPD table display and highlighting
- Tests interactive conditioning through the UI 