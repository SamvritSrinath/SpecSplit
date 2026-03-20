"""Patch generated gRPC stub imports for the installed Python package layout.

`grpc_tools.protoc` generates Python gRPC stubs that import the sibling module
directly (e.g. `import spec_decoding_pb2`).

In this repository, the protobuf package is exposed as `specsplit.proto`, so
the generated import must be rewritten to:

`from specsplit.proto import spec_decoding_pb2`

This avoids platform-specific `sed -i` behavior differences (notably on Windows).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


IMPORT_RE = re.compile(r"(?m)^import spec_decoding_pb2\s*$")


def patch_file(path: Path, *, dry_run: bool) -> None:
    original = path.read_text(encoding="utf-8")

    already_patched = "from specsplit.proto import spec_decoding_pb2" in original
    if already_patched:
        return

    updated, count = IMPORT_RE.subn(
        "from specsplit.proto import spec_decoding_pb2",
        original,
    )
    if count == 0:
        raise SystemExit(
            f"No matching import found in {path}. "
            "Expected a line exactly matching `import spec_decoding_pb2`."
        )

    if dry_run:
        return

    path.write_text(updated, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch protobuf gRPC stub imports.")
    parser.add_argument(
        "--file",
        type=Path,
        default=Path("specsplit/proto/spec_decoding_pb2_grpc.py"),
        help="Path to the generated *_pb2_grpc.py file to patch.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write changes; exit successfully if patch is applicable.",
    )
    args = parser.parse_args()

    patch_file(args.file, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

