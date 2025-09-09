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

# Deprecated installers (cargo/pip) removed: we now install EZKL only via the official source.

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
  if [[ "$INTERACTIVE" == true ]]; then
    say "Manual instructions:"
    say "  - Visit EZKL docs: https://github.com/zkonduit/ezkl"
    say "  - Look for 'lookup tables' instructions for your version"
    read -r -p "Press Enter to continue after manual lookup-table setup (or Ctrl+C to abort)..." _ || true
  fi
}

# Ensure EZKL installed
ensure_ezkl() {
  if command -v ezkl >/dev/null 2>&1; then
    info "EZKL already installed: $(command -v ezkl)"
    ezkl --version || true
    return 0
  fi

  # Install only from the official source
  info "Installing EZKL from the official source ..."
  if curl -fsSL https://raw.githubusercontent.com/zkonduit/ezkl/main/install_ezkl_cli.sh | bash; then
    info "✓ EZKL installed via official script"
  else
    err "Failed to install EZKL via the official script. Please see https://github.com/zkonduit/ezkl for manual installation instructions."
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
    err "EZKL CLI not found after installation attempt."
    warn "- Ensure your PATH includes the installation directory."
    warn "- Manual install instructions: https://github.com/zkonduit/ezkl#installation"
    if [[ "$INTERACTIVE" == true ]]; then
      read -r -p "Install EZKL manually now, then press Enter to retry detection (or type 's' to skip): " resp || true
      if [[ ! "$resp" =~ ^[Ss]$ ]]; then
        if command -v ezkl >/dev/null 2>&1; then
          info "Detected EZKL after manual install."
        else
          warn "EZKL still not detected. You can proceed, but ezkl-dependent features will not work."
        fi
      else
        warn "Skipping EZKL after user choice."
      fi
    fi
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
  if (( fail_count > 0 )); then
    warn "Some SRS downloads failed."
    warn "You can manually fetch missing files with: ezkl get-srs --logrows <N> --commitment kzg"
    warn "Files are stored under: $SRS_DIR"
    if [[ "$INTERACTIVE" == true ]]; then
      read -r -p "Try to download SRS manually now and press Enter to continue (or type 's' to skip): " resp || true
      if [[ ! "$resp" =~ ^[Ss]$ ]]; then
        info "Continuing after manual SRS step."
      else
        warn "Skipping remaining SRS downloads as per user choice."
      fi
    fi
  fi
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
