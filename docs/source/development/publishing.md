# Publishing CPax to PyPI

CPax uses two separate manual GitHub Actions workflows. The
[Manual CalVer Release](https://github.com/JaLuka98/CPax/blob/main/.github/workflows/manual-release.yaml)
creates the version bump commit and annotated Git tag; the
[Publish to PyPI](https://github.com/JaLuka98/CPax/blob/main/.github/workflows/publish-pypi.yaml)
workflow validates and publishes that exact tagged source.

See also the [versioning policy](versioning) and
[GitHub Releases](https://github.com/JaLuka98/CPax/releases).

## One-time setup (repository owner)

1. Create or verify the PyPI project **CPax**. If not yet registered,
   configure a *pending trusted publisher* from your PyPI account; if
   registered and owned by you, add a trusted publisher to that project.
2. On PyPI, set owner **JaLuka98**, repository **CPax**, workflow filename
   **publish-pypi.yaml**, and environment **pypi**. These values must
   match exactly. No PyPI token is necessary.
3. In GitHub **Settings → Environments**, create **pypi**. Configure
   required reviewer approval and prevent self-review where available.
   Publication is deliberately dispatched from **main** even though the
   build checks out a chosen release tag. Restrict deployment to main
   (not merely tag patterns) and enforce permissions on who can manually
   run release workflows.

No credentials should be committed to the repository.

## Normal release procedure

1. Confirm CI is green on main. In **Actions → Manual CalVer Release**,
   choose **main** and `dry_run=true`. Inspect its proposed version.
2. Repeat with **main** and `dry_run=false`; it atomically pushes
   the version commit and **annotated** `vYYYY.M.N` tag.
3. In **Actions → Publish to PyPI**, choose **main**, enter exactly
   that existing tag, and initially select `dry_run=true`. The workflow
   checks annotation, ancestry from main, tagged `pyproject.toml` and
   `uv.lock`, builds the wheel and sdist, validates metadata, installs
   the wheel in a clean environment and solves a harmonic oscillator.
4. Repeat the PyPI workflow with `dry_run=false`; approve the **pypi**
   environment when prompted. Only the publish job has OIDC permission.
5. Inspect the [PyPI project](https://pypi.org/project/CPax/) and
   install the published version from PyPI in a fresh environment.
   Then create the GitHub Release and describe breaking API changes.
6. In the Read the Docs **Versions** dashboard, confirm the release tag
   is active and built; see the [versioning guide](versioning).

The version is **not** hard-coded to October 2026: the CalVer workflow
determines the release version from UTC. The publication workflow never
creates, moves or repairs tags and will not overwrite an existing PyPI
release. A tag must exist before starting a real publish. If publication
fails after the tag is made, correct the issue and rerun publication for
the same tag rather than creating another release blindly.

## Security boundaries

The release tag must be annotated, have a valid CalVer name, be reachable
from current main, and match both package and lockfile version metadata.
The publishing job does **not** build code or install dependencies, and
does **not** receive repository write permission. It downloads only the
distributions built in the preceding validated job.

Do not assume a `push: tags` workflow will run for a tag created with
the `GITHUB_TOKEN`; use the explicit Publish to PyPI dispatch. Trusted
Publishing relies on the GitHub environment's approval and PyPI's
registered publisher identity; any authorised person who can start an
approved real publication needs to be treated as a release operator.
