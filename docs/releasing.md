# Releasing

This is the minimum engineering gate for a release. The v4 packaging and CI
issues will expand it into a complete release playbook.

## Preconditions

- The release version and migration notes are current.
- Required CI checks pass from a clean checkout.
- Default tests use no provider credentials or external network.
- The source distribution and wheel build successfully.
- The wheel installs and `import raglight` succeeds in an isolated environment.
- Each documented optional extra is validated by its CI matrix.
- Public API and deprecation documentation match the release.

Run locally:

```bash
make check
```

`make package-check` builds fresh artifacts and validates the wheel in a
temporary virtual environment. It does not publish anything.

Publishing, signing, tagging, and changelog automation remain maintainer-owned
until a dedicated release decision documents them.

