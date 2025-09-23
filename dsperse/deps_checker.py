import sys
from dsperse.install import install_deps, check_ezkl, check_srs_files


def ensure_dependencies():
    """Check and install dependencies"""
    ezkl_path, ezkl_version = check_ezkl()
    present_srs, missing_srs = check_srs_files()

    if ezkl_path and not missing_srs:
        return True

    if not ezkl_path:
        print("[INFO] EZKL not found. Installing dependencies...", file=sys.stderr)

    if missing_srs:
        print(
            f"[INFO] SRS files missing for sizes: {missing_srs}",
            file=sys.stderr,
        )

    print("[INFO] Running dependency installer...", file=sys.stderr)

    if install_deps(skip_pip=True, interactive=False):
        print("[INFO] Dependencies installed successfully!", file=sys.stderr)
        return True
    else:
        print("[ERROR] Failed to install dependencies automatically.", file=sys.stderr)
        print("[INFO] Please install EZKL manually from:", file=sys.stderr)
        print("https://github.com/zkonduit/ezkl#installation", file=sys.stderr)
        return False
