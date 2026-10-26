# Dependency security follow-up (October 2026)

The October 25 scan supplied in `dt_nspect_vuln_details_dd_1.csv` reports
44 findings across seven Python dependencies. The workspace lockfile and root
constraints enforce the compatible fixes below.
Run `make sync` to install the updated lockfile.

Following [DataDesigner's transitive dependency guidance](https://github.com/NVIDIA-NeMo/DataDesigner/blob/main/DEVELOPMENT.md#transitive-dependencies),
transitive security floors stay out of published plugin requirements. Downstream
applications should apply the recommended versions below in their own lockfiles.
Only retrieval-sdg directly uses filelock, NLTK, and Pillow, so it retains those
requirements. All plugins require DataDesigner 0.9.7 or later, which enforces the
exceptional `certifi>=2026.10.19` CA trust floor through data-designer-config.

| Dependency | Recommended version | Report findings |
| --- | --- | --- |
| MCP | `>=1.30.0,<2` | CVE-2026-59951 |
| NLTK | `>=3.10.4,<3.11` | CVE-2026-81726 |
| fsspec | `>=2026.6.0` | CVE-2026-104851 |
| certifi | `>=2026.10.19` | BDSA-2026-37951 (eMudhra trust-store incident) |
| filelock, Python 3.11+ | `>=4.2.1,<5` | CVE-2026-112940 |
| filelock, Python 3.10 | `>=3.20,<3.20.4` | Excludes the affected ReadWriteLock implementation introduced in 3.20.4 |
| Pillow, Python 3.11+ | `>=13.0.0` | All 38 Pillow findings in the supplied report |

## Applying the recommendations in an application

Update the application's lockfile with targeted upgrades:

```bash
uv lock --upgrade-package certifi --upgrade-package filelock --upgrade-package fsspec \
    --upgrade-package mcp --upgrade-package nltk --upgrade-package pillow
```

If a minimum does not survive re-resolution, add it to the application's root
`[tool.uv].constraint-dependencies`. Preserve the Python markers from this
repository for filelock and Pillow when supporting Python 3.10. Workspace
constraints are not included in published plugin metadata.

## Unresolved findings

- **OpenTelemetry: CVE-2026-114603.** The fixed API release is 1.45.1.
  DataDesigner 0.9.7 requires `opentelemetry-api>=1.43,<1.44` and
  `opentelemetry-sdk>=1.43,<1.44`, so adding the fixed API floor makes the
  dependency graph unsatisfiable. The lockfile retains API 1.43.0. An upstream
  DataDesigner release must permit the patched telemetry dependency family
  before this finding can be resolved without overriding its compatibility
  contract.
- **Pillow on Python 3.10: all 38 reported Pillow findings remain open.**
  Pillow 13.0.0 requires Python 3.11 or newer, and PyPI has no 12.x security
  backport as of October 26, 2026. Python 3.10 retains the existing 12.3.0
  floor to preserve this repository's supported Python baseline. Use Python
  3.11+ to obtain the patched Pillow dependency. Full remediation on Python
  3.10 requires an upstream backport or a separately approved change to the
  supported Python baseline.

This change does not claim a clean vulnerability scan. Re-run the scanner
against the installed environment; a scanner that ignores Python markers may
still flag the Python 3.10 Pillow entry in the universal lockfile.

## Upstream references

- [Pillow 13.0.0 security fixes and Python support](https://pillow.readthedocs.io/en/stable/releasenotes/13.0.0.html)
- [NLTK 3.10.4 release](https://github.com/nltk/nltk/releases/tag/v3.10.4)
- [certifi 2026.10.19 release](https://github.com/certifi/python-certifi/releases/tag/2026.10.19)
- [filelock security advisories](https://github.com/tox-dev/filelock/security/advisories)
- [DataDesigner 0.9.7 package metadata](https://pypi.org/pypi/data-designer/0.9.7/json)
