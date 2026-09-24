#!/usr/bin/env bash
# 여러 sweep 설정을 순서대로 돈다.  sweep 마다 로그 하나, 끝에 요약.
#   ./scripts/run_queue.sh sweeps/B_euler_steps_1300.sh sweeps/C1_dtau_ladder_pilot400.sh
# 하나가 실패해도 다음으로 간다 — 드레인 감지는 run_sweep.sh 안에 있다.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."
[[ $# -gt 0 ]] || { echo "사용법: $0 sweeps/A.sh [sweeps/B.sh ...]" >&2; exit 2; }
mkdir -p logs
# 큐 전체가 잠금을 한 번 쥔다 — sweep 사이의 빈틈에 다른 run 이 끼어들지 못하게.
# 안쪽의 run_sweep.sh·run.sh 는 ALPAMAYO_LOCK_HELD 를 보고 다시 잠그지 않는다.
LOCK_FILE="${LOCK_FILE:-/tmp/alpamayo-inference.lock}"
exec 9>"$LOCK_FILE"
flock -n 9 || { echo "다른 추론·sweep 이 돌고 있다 ($LOCK_FILE). 끝나길 기다린다." >&2; exit 1; }
export ALPAMAYO_LOCK_HELD=1
STARTED=$(date +%s); RESULTS=()
for cfg in "$@"; do
  name="$(basename "$cfg" .sh)"; log="logs/queue_${name}_$(date +%m%d_%H%M).log"
  echo "═══ $(date '+%m-%d %H:%M')  $cfg  →  $log"
  t=$(date +%s)
  if SWEEP_CONFIG="$cfg" bash "$HERE/run_sweep.sh" >"$log" 2>&1; then r="완료"; else r="실패"; fi
  el=$(( ( $(date +%s) - t ) / 60 ))
  RESULTS+=("$r|$cfg|${el}분")
  echo "    $r  ${el}분   $(grep -cE '^\[[0-9]+/[0-9]+\]' "$log" 2>/dev/null || echo 0) 클립 줄"
done
echo
echo "═══ queue 끝  $(( ( $(date +%s) - STARTED ) / 3600 ))시간 $(( ( ( $(date +%s) - STARTED ) % 3600 ) / 60 ))분"
for r in "${RESULTS[@]}"; do IFS='|' read -r st c el <<<"$r"; printf "  %-4s %-44s %s\n" "$st" "$c" "$el"; done
