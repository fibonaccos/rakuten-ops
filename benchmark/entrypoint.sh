#!/usr/bin/env sh
set -eu

SIMULATION_CONFIG="${SIMULATION_CONFIG:-/benchmark/config/simulation.yaml}"
REPORT_DIR="${REPORT_DIR:-/benchmark/reports}"
mkdir -p "$REPORT_DIR"

read_yaml() {
  python3 -c "
import yaml
with open('${SIMULATION_CONFIG}', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
print(cfg['run']['$1'])
"
}

USERS=$(read_yaml users)
SPAWN_RATE=$(read_yaml spawn_rate)
RUN_TIME=$(read_yaml run_time)

if [ -z "${TARGET_BASE_URL:-}" ]; then
  echo "[entrypoint] TARGET_BASE_URL n'est pas defini, abandon." >&2
  exit 1
fi

echo "[entrypoint] host=${TARGET_BASE_URL} users=${USERS} spawn_rate=${SPAWN_RATE} run_time=${RUN_TIME}"

exec locust \
  -f /benchmark/locustfile.py \
  --headless \
  --host "${TARGET_BASE_URL}" \
  --users "${USERS}" \
  --spawn-rate "${SPAWN_RATE}" \
  --run-time "${RUN_TIME}" \
  --csv "${REPORT_DIR}/results" \
  --html "${REPORT_DIR}/report.html"
