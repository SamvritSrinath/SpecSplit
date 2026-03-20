# Platforms & Builds

SpecSplit supports local development and CI unit testing on Linux/macOS natively,
and Windows via WSL2 (or Docker).

## Linux / Unix

1. Generate protobuf stubs:
   - `make proto`
2. Run unit tests:
   - `make test`

The `Makefile` drives `grpc_tools.protoc` using `specsplit/proto/spec_decoding.proto`.

## macOS

`make proto` is supported on macOS as well (the `Makefile` includes a Darwin
`sed -i` variant).

For heavier runs, prefer the Docker workflow below.

## Windows

Recommendation: use WSL2.

In WSL2 you get a consistent `make proto` toolchain (matching Linux), and you
avoid Windows `sed` differences.

## Docker (recommended for consistent local runs)

`docker-compose.yml` provides a CPU-only local simulation of the distributed
Draft/Target/Orchestrator services.

Usage:

```bash
docker compose up --build
```

The compose file maps:

- Draft Worker: `50051:50051`
- Target Worker: `50052:50052`

GPU passthrough is optional (commented out in `docker-compose.yml`) and should be
enabled only when Docker + NVIDIA runtime are configured.

## Proto import patching note

`make proto` generates stubs and then patches the import line for the Python
package layout.

On some native Windows environments, `sed -i` behavior differs; if you hit
proto-import patch failures, follow the “proto import robustness” guidance in
the project `scripts/` directory (and/or use WSL2).

