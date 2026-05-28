# PTODSL Tests

`ptodsl/tests/` is the canonical home for PTODSL-specific regression checks.

This directory intentionally keeps three PTODSL testing layers close together:

- `test_*.py`: the canonical entrypoint pattern for PTODSL regressions, including compile-only, diagnostics, frontend-handoff, docs-as-test, and focused unit tests
- `support/`: helper modules shared by the test entrypoints; these are intentionally not auto-discovered as tests

Related PTODSL validation still lives nearby, but with different roles:

- `ptodsl/examples/`: launchable example programs; these stay user-facing and are validated by regressions here
- `test/dsl-st/`: simulator / ST cases for PTODSL kernels that need runtime execution rather than compile-only checks

Typical local runs:

```bash
cd $PTOAS_REPO_ROOT
python3 ptodsl/tests/test_jit_compile.py
python3 ptodsl/tests/test_docs_as_test.py
python3 -m unittest discover -s ptodsl/tests -p 'test_*.py'
```
