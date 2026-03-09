#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  scripts/manage_remote_worker.sh <start|stop|restart|status|update|logs|url> <env-file>

The env file should define ROLE=draft or ROLE=target and the usual
SPECSPLIT_* variables for that worker.
EOF
}

note() {
    printf '%s\n' "$*"
}

die() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

require_executable() {
    local target="$1"
    if [[ "$target" == */* ]]; then
        [[ -x "$target" ]] || die "Executable not found: $target"
        return
    fi

    command -v "$target" >/dev/null 2>&1 || die "Command not found: $target"
}

read_pid() {
    local pid_file="$1"
    [[ -f "$pid_file" ]] || return 1
    tr -d '[:space:]' <"$pid_file"
}

pid_is_running() {
    local pid="$1"
    [[ "$pid" =~ ^[0-9]+$ ]] || return 1
    kill -0 "$pid" 2>/dev/null
}

is_running() {
    local pid_file="$1"
    local pid
    pid="$(read_pid "$pid_file")" || return 1
    pid_is_running "$pid"
}

clear_stale_pidfile() {
    local pid_file="$1"
    if [[ -f "$pid_file" ]] && ! is_running "$pid_file"; then
        rm -f "$pid_file"
    fi
}

stop_process() {
    local label="$1"
    local pid_file="$2"
    local timeout="${3:-20}"
    local pid
    local waited=0

    clear_stale_pidfile "$pid_file"
    if ! is_running "$pid_file"; then
        note "$label is not running."
        return 0
    fi

    pid="$(read_pid "$pid_file")"
    note "Stopping $label (pid $pid)..."
    kill "$pid" 2>/dev/null || true

    while pid_is_running "$pid"; do
        if (( waited >= timeout )); then
            note "$label did not exit after ${timeout}s; sending SIGKILL."
            kill -KILL "$pid" 2>/dev/null || true
            break
        fi
        sleep 1
        waited=$((waited + 1))
    done

    rm -f "$pid_file"
}

tail_logs() {
    note "==> $WORKER_LOG <=="
    if [[ -f "$WORKER_LOG" ]]; then
        tail -n "${TAIL_LINES:-40}" "$WORKER_LOG"
    else
        note "(missing)"
    fi

    if [[ "$NGROK_ENABLE" == "1" ]]; then
        note
        note "==> $NGROK_LOG <=="
        if [[ -f "$NGROK_LOG" ]]; then
            tail -n "${TAIL_LINES:-40}" "$NGROK_LOG"
        else
            note "(missing)"
        fi
    fi
}

json_python() {
    local candidate
    for candidate in "$PYTHON_BIN" python3 python; do
        [[ -n "$candidate" ]] || continue
        if [[ "$candidate" == */* ]]; then
            [[ -x "$candidate" ]] && {
                printf '%s\n' "$candidate"
                return 0
            }
        elif command -v "$candidate" >/dev/null 2>&1; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

url_from_ngrok_api() {
    command -v curl >/dev/null 2>&1 || return 1

    local parser api_response
    parser="$(json_python)" || return 1
    api_response="$(
        curl --silent --show-error --max-time "${NGROK_API_TIMEOUT_SECONDS:-2}" \
            "${NGROK_API_URL%/}/tunnels" 2>/dev/null
    )" || return 1

    LOCAL_PORT="$LOCAL_PORT" "$parser" -c '
import json
import os
import sys

port = os.environ["LOCAL_PORT"]
candidates = {
    port,
    f"localhost:{port}",
    f"127.0.0.1:{port}",
    f"[::1]:{port}",
    f"http://localhost:{port}",
    f"http://127.0.0.1:{port}",
}

try:
    payload = json.load(sys.stdin)
except Exception:
    raise SystemExit(1)

for tunnel in payload.get("tunnels", []):
    if str(tunnel.get("proto", "")).startswith("tcp"):
        addr = str((tunnel.get("config") or {}).get("addr", ""))
        if addr in candidates or not addr:
            public_url = str(tunnel.get("public_url", "")).strip()
            if public_url:
                print(public_url)
                raise SystemExit(0)

raise SystemExit(1)
' <<<"$api_response"
}

url_from_ngrok_log() {
    [[ -f "$NGROK_LOG" ]] || return 1
    grep -Eo 'tcp://[^[:space:]]+' "$NGROK_LOG" | tail -n 1
}

cache_public_url() {
    local url="$1"
    [[ -n "$url" ]] || return 1
    printf '%s\n' "$url" >"$PUBLIC_URL_FILE"
}

read_cached_public_url() {
    [[ -f "$PUBLIC_URL_FILE" ]] || return 1
    head -n 1 "$PUBLIC_URL_FILE"
}

resolve_public_url() {
    [[ "$NGROK_ENABLE" == "1" ]] || return 1

    if [[ -n "${NGROK_TCP_URL:-}" ]]; then
        printf '%s\n' "$NGROK_TCP_URL"
        return 0
    fi

    local url
    url="$(url_from_ngrok_api || true)"
    if [[ -z "$url" ]]; then
        url="$(url_from_ngrok_log || true)"
    fi
    if [[ -n "$url" ]]; then
        cache_public_url "$url" || true
        printf '%s\n' "$url"
        return 0
    fi

    read_cached_public_url
}

wait_for_public_url() {
    local attempts="${NGROK_URL_LOOKUP_ATTEMPTS:-10}"
    local sleep_seconds="${NGROK_URL_LOOKUP_INTERVAL_SECONDS:-1}"
    local i url

    for ((i = 0; i < attempts; i++)); do
        url="$(resolve_public_url || true)"
        if [[ -n "$url" ]]; then
            printf '%s\n' "$url"
            return 0
        fi
        sleep "$sleep_seconds"
    done

    return 1
}

print_status() {
    clear_stale_pidfile "$WORKER_PID_FILE"
    clear_stale_pidfile "$NGROK_PID_FILE"

    if is_running "$WORKER_PID_FILE"; then
        note "$ROLE worker: running (pid $(read_pid "$WORKER_PID_FILE"))"
    else
        note "$ROLE worker: stopped"
    fi
    note "worker log: $WORKER_LOG"

    if [[ "$NGROK_ENABLE" == "1" ]]; then
        if is_running "$NGROK_PID_FILE"; then
            note "ngrok: running (pid $(read_pid "$NGROK_PID_FILE"))"
        else
            note "ngrok: stopped"
        fi
        note "ngrok log: $NGROK_LOG"
        local public_url
        public_url="$(resolve_public_url || true)"
        if [[ -n "$public_url" ]]; then
            note "public tcp address: $public_url"
        else
            note "public tcp address: unknown (try the url action or inspect $NGROK_LOG)"
        fi
    else
        note "ngrok: disabled"
    fi
}

start_worker() {
    clear_stale_pidfile "$WORKER_PID_FILE"
    if is_running "$WORKER_PID_FILE"; then
        note "$ROLE worker already running (pid $(read_pid "$WORKER_PID_FILE"))."
        return 0
    fi

    note "Starting $ROLE worker on localhost:$LOCAL_PORT..."
    nohup "$PYTHON_BIN" -m "$MODULE" >"$WORKER_LOG" 2>&1 </dev/null &
    echo "$!" >"$WORKER_PID_FILE"
    sleep "${WORKER_STARTUP_WAIT_SECONDS:-3}"

    if ! is_running "$WORKER_PID_FILE"; then
        rm -f "$WORKER_PID_FILE"
        note "$ROLE worker exited during startup. Recent log output:"
        tail -n 40 "$WORKER_LOG" || true
        return 1
    fi

    note "$ROLE worker started (pid $(read_pid "$WORKER_PID_FILE"))."
}

start_ngrok() {
    clear_stale_pidfile "$NGROK_PID_FILE"
    if [[ "$NGROK_ENABLE" != "1" ]]; then
        note "ngrok disabled for this environment."
        return 0
    fi

    if is_running "$NGROK_PID_FILE"; then
        note "ngrok already running (pid $(read_pid "$NGROK_PID_FILE"))."
        return 0
    fi

    note "Starting ngrok TCP tunnel for localhost:$LOCAL_PORT..."
    if [[ -n "${NGROK_TCP_URL:-}" ]]; then
        nohup "$NGROK_BIN" tcp --url "$NGROK_TCP_URL" "$LOCAL_PORT" >"$NGROK_LOG" 2>&1 </dev/null &
    else
        nohup "$NGROK_BIN" tcp "$LOCAL_PORT" >"$NGROK_LOG" 2>&1 </dev/null &
    fi
    echo "$!" >"$NGROK_PID_FILE"
    sleep "${NGROK_STARTUP_WAIT_SECONDS:-3}"

    if ! is_running "$NGROK_PID_FILE"; then
        rm -f "$NGROK_PID_FILE"
        note "ngrok exited during startup. Recent log output:"
        tail -n 40 "$NGROK_LOG" || true
        return 1
    fi

    note "ngrok started (pid $(read_pid "$NGROK_PID_FILE"))."
    local public_url
    public_url="$(wait_for_public_url || true)"
    if [[ -n "$public_url" ]]; then
        note "Public TCP address: $public_url"
    else
        note "Public TCP address could not be auto-detected yet; try:"
        note "  $0 url $ENV_FILE_ABS"
    fi
}

stop_all() {
    if [[ "$NGROK_ENABLE" == "1" ]]; then
        stop_process "ngrok" "$NGROK_PID_FILE"
    fi
    stop_process "$ROLE worker" "$WORKER_PID_FILE"
}

start_all() {
    require_executable "$PYTHON_BIN"
    if [[ "$NGROK_ENABLE" == "1" ]]; then
        require_executable "$NGROK_BIN"
    fi
    start_worker
    start_ngrok
}

ACTION="${1:-}"
ENV_FILE="${2:-}"

if [[ -z "$ACTION" || -z "$ENV_FILE" ]]; then
    usage
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE_ABS="$(cd "$(dirname "$ENV_FILE")" && pwd)/$(basename "$ENV_FILE")"

[[ -f "$ENV_FILE_ABS" ]] || die "Env file not found: $ENV_FILE"

set -a
# shellcheck disable=SC1090
source "$ENV_FILE_ABS"
set +a

ROLE="${ROLE:?ROLE must be set in the env file}"
WORKDIR="${WORKDIR:-$REPO_ROOT}"
STATE_DIR="${STATE_DIR:-$WORKDIR/.run/$ROLE}"
PYTHON_BIN="${PYTHON_BIN:-$WORKDIR/.venv/bin/python}"
NGROK_BIN="${NGROK_BIN:-ngrok}"
NGROK_ENABLE="${NGROK_ENABLE:-1}"
NGROK_API_URL="${NGROK_API_URL:-http://127.0.0.1:4040/api}"
REINSTALL_AFTER_PULL="${REINSTALL_AFTER_PULL:-0}"
GIT_REMOTE="${GIT_REMOTE:-origin}"

case "$ROLE" in
    draft)
        MODULE="specsplit.workers.draft.service"
        LOCAL_PORT="${LOCAL_PORT:-${SPECSPLIT_DRAFT_GRPC_PORT:-50051}}"
        ;;
    target)
        MODULE="specsplit.workers.target.service"
        LOCAL_PORT="${LOCAL_PORT:-${SPECSPLIT_TARGET_GRPC_PORT:-50052}}"
        ;;
    *)
        die "ROLE must be draft or target, got: $ROLE"
        ;;
esac

mkdir -p "$STATE_DIR"
cd "$WORKDIR"

WORKER_PID_FILE="$STATE_DIR/worker.pid"
NGROK_PID_FILE="$STATE_DIR/ngrok.pid"
WORKER_LOG="$STATE_DIR/worker.log"
NGROK_LOG="$STATE_DIR/ngrok.log"
PUBLIC_URL_FILE="$STATE_DIR/ngrok_public_url.txt"

case "$ACTION" in
    start)
        start_all
        print_status
        ;;
    stop)
        stop_all
        print_status
        ;;
    restart)
        stop_all
        start_all
        print_status
        ;;
    status)
        print_status
        ;;
    logs)
        tail_logs
        ;;
    url)
        if [[ "$NGROK_ENABLE" != "1" ]]; then
            die "ngrok is disabled in this environment"
        fi
        resolve_public_url || die "Could not determine the public TCP address. Ensure ngrok is running and its local API is reachable at ${NGROK_API_URL%/}."
        ;;
    update)
        require_executable git
        stop_all
        CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
        GIT_BRANCH="${GIT_BRANCH:-$CURRENT_BRANCH}"
        note "Pulling $GIT_REMOTE/$GIT_BRANCH..."
        if ! git pull --ff-only "$GIT_REMOTE" "$GIT_BRANCH"; then
            note "git pull failed; restarting the previous code."
            start_all
            exit 1
        fi

        if [[ "$REINSTALL_AFTER_PULL" == "1" ]]; then
            note "Refreshing editable install..."
            "$PYTHON_BIN" -m pip install -e .
        fi

        start_all
        print_status
        ;;
    *)
        usage
        exit 1
        ;;
esac
