#!/usr/bin/env bash
# install.sh - Installer for Dsperse CLI and EZKL (with lookup tables)
# This script installs:
#   - Dsperse CLI (python package, console script: dsperse)
#   - EZKL CLI (if missing)
#   - EZKL lookup tables (if missing)
# It detects existing installations and will skip or prompt accordingly.

set -euo pipefail

INTERACTIVE=true
FORCE_REINSTALL=false
PYTHON_BIN="python3"
PIP_BIN=""

print_usage() {
  cat <<EOF
Usage: $0 [OPTIONS]

Install Dsperse CLI and EZKL dependencies.

Options:
  -h, --help              Show this help and exit
  -n, --non-interactive   Run without prompts (best-effort automatic install)
  -f, --force             Force reinstall Dsperse package with pip

Examples:
  $0                      Run interactively with prompts
  $0 -n                   Non-interactive install (CI-friendly)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      print_usage
      exit 0
      ;;
    -n|--non-interactive)
      INTERACTIVE=false
      shift
      ;;
    -f|--force)
      FORCE_REINSTALL=true
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      print_usage
      exit 1
      ;;
  esac
done

say() { echo -e "$1"; }
info() { say "[INFO] $1"; }
warn() { say "[WARN] $1"; }
err()  { say "[ERROR] $1"; }

confirm() {
  local prompt="$1"
  if [[ "$INTERACTIVE" == true ]]; then
    read -r -p "$prompt [y/N]: " resp || true
    [[ "$resp" =~ ^[Yy]$ ]]
  else
    return 0
  fi
}

# Resolve pip for the chosen python
resolve_pip() {
  if command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    PIP_BIN="$PYTHON_BIN -m pip"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
    PIP_BIN="python -m pip"
  else
    err "Python not found. Please install Python 3 and re-run."
    exit 1
  fi
}

# Install Dsperse CLI via pip
install_dsperse_cli() {
  info "Installing Dsperse CLI (pip editable install) ..."
  if [[ "$FORCE_REINSTALL" == true ]]; then
    eval $PIP_BIN install -e .
  else
    # Try to detect if dsperse is already available
    if command -v dsperse >/dev/null 2>&1; then
      info "Dsperse CLI already installed: $(command -v dsperse)"
      if [[ "$INTERACTIVE" == true ]] && confirm "Reinstall/upgrade Dsperse CLI?"; then
        eval $PIP_BIN install -e .
      else
        info "Skipping Dsperse CLI install."
      fi
    else
      eval $PIP_BIN install -e .
    fi
  fi
  if command -v dsperse >/dev/null 2>&1; then
    info "✓ Dsperse CLI installed: $(command -v dsperse)"
  else
    warn "Dsperse CLI not found on PATH yet. It may appear after you restart your shell or activate your venv."
  fi
}

# Try installing ezkl via cargo (preferred if available)
install_ezkl_via_cargo() {
  if ! command -v cargo >/dev/null 2>&1; then
    return 1
  fi
  info "Installing EZKL via cargo (this may take several minutes) ..."
  if cargo install --list | grep -q "^ezkl "; then
    info "EZKL already installed via cargo. Updating to latest compatible version ..."
  fi
  if cargo install --locked ezkl; then
    # Ensure cargo bin in PATH for current session
    export PATH="$HOME/.cargo/bin:$PATH"
    return 0
  fi
  return 1
}

# Try installing ezkl via pip (provides python package; CLI availability may vary)
install_ezkl_via_pip() {
  info "Attempting to install ezkl via pip ..."
  if eval $PIP_BIN install -U ezkl; then
    return 0
  fi
  return 1
}

# Attempt to download lookup tables with available subcommands
install_lookup_tables() {
  if ! command -v ezkl >/dev/null 2>&1; then
    warn "Skipping EZKL lookup tables: ezkl not found."
    return 0
  fi
  info "Installing EZKL lookup tables (if required) ..."
  set +e
  if ezkl --help 2>/dev/null | grep -qi "lookup"; then
    # Try common subcommands (names may vary by version)
    ezkl get-lookup-tables 2>/dev/null && { set -e; info "✓ Lookup tables installed (get-lookup-tables)."; return 0; }
    ezkl download-lookup-tables 2>/dev/null && { set -e; info "✓ Lookup tables installed (download-lookup-tables)."; return 0; }
    ezkl lookup --download 2>/dev/null && { set -e; info "✓ Lookup tables installed (lookup --download)."; return 0; }
  fi
  set -e
  warn "Could not detect a lookup tables subcommand on this EZKL version. If witness generation asks for lookup tables, please consult EZKL docs to download them."
}

# Ensure EZKL installed
ensure_ezkl() {
  if command -v ezkl >/dev/null 2>&1; then
    info "EZKL already installed: $(command -v ezkl)"
    ezkl --version || true
    return 0
  fi

  if [[ "$INTERACTIVE" == true ]]; then
    say ""
    say "EZKL CLI is not installed. Choose an installation method:"
    say "  1) cargo install (recommended if Rust is installed)"
    say "  2) pip install ezkl (Python package; CLI availability may vary)"
    say "  3) Skip (I'll install manually)"
    read -r -p "Select [1/2/3]: " choice || true
    case "$choice" in
      1)
        install_ezkl_via_cargo || warn "cargo install failed."
        ;;
      2)
        install_ezkl_via_pip || warn "pip install ezkl failed."
        ;;
      *)
        warn "Skipping EZKL installation as per user choice."
        ;;
    esac
  else
    # Non-interactive: best-effort cargo then pip
    install_ezkl_via_cargo || install_ezkl_via_pip || true
  fi

  # Fallback: try ezkl install script if Linux or previous methods failed
  if ! command -v ezkl >/dev/null 2>&1; then
    if [[ "$OSTYPE" == "linux-gnu"* ]] || [[ -f /etc/os-release ]]; then
      info "Trying EZKL installation via official install script..."
      if curl -s https://raw.githubusercontent.com/zkonduit/ezkl/main/install_ezkl_cli.sh | bash; then
        info "✓ EZKL installed via official script"
        # Ensure cargo bin in PATH for current session
        export PATH="$HOME/.cargo/bin:$PATH"
      else
        warn "EZKL install script failed"
      fi
    fi
  fi

  if command -v ezkl >/dev/null 2>&1; then
    info "✓ EZKL installed: $(command -v ezkl)"
    ezkl --version || true
  else
    warn "EZKL CLI not found after installation attempts. You may need to add its install location to PATH or install manually."
    warn "Common options:"
    warn "  - Cargo: cargo install --locked ezkl (ensure \"$HOME/.cargo/bin\" on PATH)"
    warn "  - Prebuilt binaries: see EZKL GitHub releases"
  fi
}

# Ensure SRS files (kzg commitment) exist under ~/.ezkl/srs
ensure_srs() {
  if ! command -v ezkl >/dev/null 2>&1; then
    warn "Skipping SRS setup: ezkl not found."
    return 0
  fi
  local SRS_DIR="$HOME/.ezkl/srs"
  local MIN_LOGROWS=${MIN_LOGROWS:-2}
  local MAX_LOGROWS=${MAX_LOGROWS:-24}
  mkdir -p "$SRS_DIR"

  info "Checking EZKL SRS files in $SRS_DIR (kzg, logrows ${MIN_LOGROWS}-${MAX_LOGROWS}) ..."
  local missing=()
  local present=()
  local n
  for (( n=$MIN_LOGROWS; n<=$MAX_LOGROWS; n++ )); do
    if [[ -f "$SRS_DIR/kzg${n}.srs" ]]; then
      present+=("kzg${n}.srs")
    else
      missing+=("$n")
    fi
  done

  if (( ${#missing[@]} == 0 )); then
    info "✓ All required SRS files already present (${#present[@]} files)."
    return 0
  fi

  say "SRS files missing for logrows: ${missing[*]}"
  if [[ "$INTERACTIVE" == true ]]; then
    say "These downloads can take a while (several minutes per file), but having them locally will speed up EZKL circuitization and proof steps."
    if ! confirm "Download missing SRS files now using 'ezkl get-srs --commitment kzg'?"; then
      warn "Skipping SRS download at user request. You can run it later manually, e.g.: ezkl get-srs --logrows 20 --commitment kzg"
      return 0
    fi
  else
    warn "Non-interactive mode: skipping automatic SRS downloads to avoid long install times."
    warn "You can download later with: ezkl get-srs --logrows <N> --commitment kzg (files are stored under $HOME/.ezkl/srs)."
    return 0
  fi

  local ok_count=0
  local fail_count=0
  for n in "${missing[@]}"; do
    info "Downloading SRS for logrows=$n (kzg) ..."
    if ezkl get-srs --logrows "$n" --commitment kzg >/dev/null 2>&1; then
      info "✓ Downloaded $SRS_DIR/kzg${n}.srs"
      ok_count=$((ok_count+1))
    else
      warn "Failed to download SRS for logrows=$n. You can retry: ezkl get-srs --logrows $n --commitment kzg"
      fail_count=$((fail_count+1))
    fi
  done
  info "SRS download summary: success=$ok_count, failed=$fail_count, total_missing=${#missing[@]}"
}

main() {
  info "Installing dependencies for Dsperse ..."
  resolve_pip

  # Display python and pip info
  eval $PIP_BIN --version || true

  # Install Dsperse CLI
  install_dsperse_cli

  # Ensure EZKL
  ensure_ezkl

  # Ensure SRS data (kzg) exists
  ensure_srs

  # Lookup tables (for some ezkl versions)
  install_lookup_tables

  say ""
  if command -v dsperse >/dev/null 2>&1; then
    info "You can now run the Dsperse CLI: dsperse --help"
  else
    warn "dsperse command not yet available on PATH. Try restarting your shell or activating your virtualenv."
  fi

  if command -v ezkl >/dev/null 2>&1; then
    info "EZKL is ready: ezkl --help"
  else
    warn "EZKL not detected; some features will fall back to ONNX or fail."
  fi

  say "\nInstallation complete!"
}

main "$@"
