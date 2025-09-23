import sys
from dsperse.install import install_deps, check_ezkl, check_srs_files


def ensure_dependencies():
    """Check and install dependencies"""
    ezkl_path, ezkl_version = check_ezkl()
    present_srs, missing_srs = check_srs_files()

    critical_sizes = [16, 18, 20, 22]
    critical_missing = [s for s in critical_sizes if s in missing_srs]

    if ezkl_path and not critical_missing:
        return True

    if not ezkl_path:
        print("[INFO] EZKL not found. Installing dependencies...", file=sys.stderr)

    if critical_missing:
        print(
            f"[INFO] Critical SRS files missing for sizes: {critical_missing}",
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
