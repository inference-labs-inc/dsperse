import sys


def main():
    try:
        from dsperse._native import cli_main
    except ImportError:
        print("dsperse native extension not found; install with: pip install dsperse", file=sys.stderr)
        return 1

    try:
        cli_main()
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001 - top-level CLI wrapper to convert any error to exit code 1
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
