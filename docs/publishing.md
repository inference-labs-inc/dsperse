# Publishing jstprove to PyPI

This guide documents the release process using GitHub Actions (tag-based publish), plus a manual fallback.

## Release via GitHub Actions (recommended)

<details>
<summary><strong>PyPI Setup for GitHub Actions (Trusted Publishing)</strong></summary>

Needs to be done only once. To enable automated publishing from GitHub Actions,
the repository owner must configure Trusted Publishing on the PyPI side:

1. **Log in to PyPI** as an Owner or Maintainer of the project at https://pypi.org/manage/account/publishing/
2. **Scroll to the "Add a new pending publisher" section.**
3. **Click "GutHub" tab**.
4. **Fill in the following:**
   - **Owner**: `inference-labs-inc` (your GitHub organization or username)
   - **Repository name**: `dsperse` (your repo name)
   - **Workflow name**: (optional, leave blank to allow any workflow)
5. **Save the publisher.**
6. **Verify the publisher is listed and active.**

No PyPI API token is needed when using Trusted Publishing. The GitHub Actions workflow must use the `pypa/gh-action-pypi-publish` action, which will authenticate automatically.

For more details, see: https://docs.pypi.org/trusted-publishers/adding-a-publisher/

</details>

Steps to cut a release:

1. Update version in `pyproject.toml` under `[project].version` (semver recommended).
2. Commit the change; ensure CI is green.
3. Create and push a tag that matches the version, prefixed with `v`.

```bash
git add pyproject.toml
git commit -m "chore(release): v0.1.1"
git tag v0.1.1
git push origin v0.1.1
```

The workflow will build and publish automatically when the tag lands.
Ensure PyPI Trusted Publishing is configured for the repo/org.

## Manual build/publish (fallback)

<details>
<summary><strong>If you need to publish manually (e.g., testing or local validation)</strong></summary>

### Prerequisites

- Python 3.10+
- A virtual environment (recommended)
- PyPI account with Maintainer/Owner permission for `jstprove`
- API token created at https://pypi.org/manage/account/token/
- Required tools:

```bash
python -m pip install --upgrade pip build twine
```

Optional: configure `~/.pypirc` for convenience:

```ini
# ~/.pypirc
[distutils]
index-servers = pypi testpypi

[pypi]
username = __token__
password = pypi-<your-token>

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-<your-testpypi-token>
```

You can also export credentials per session:

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-<your-token>
```

### Build the package

```bash
python -m pip install --upgrade pip build twine
python -m build
python -m twine check dist/*
```

Optional: inspect file contents (sdist):

```bash
tar -tf dist/kubz-<version>.tar.gz | head -n 50
```

### Test the package locally

Install the wheel into a clean environment and smoke-test the CLI.

```bash
python -m venv .venv-test
source .venv-test/bin/activate
pip install dist/kubz-*.whl
kubz --help
```

### Publish to TestPyPI (recommended)

Upload to TestPyPI first to validate metadata and installability.

```bash
python -m twine upload -r testpypi dist/*
```

Test install from TestPyPI (note: dependencies resolve from PyPI):

```bash
pip install --index-url https://test.pypi.org/simple --extra-index-url https://pypi.org/simple jstprove==<version>
```

### Publish to PyPI

When satisfied with testing, upload to PyPI:

```bash
python -m twine upload dist/*
```

</details>
