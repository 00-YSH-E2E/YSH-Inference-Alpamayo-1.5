#!/usr/bin/env bash
# =============================================================================
#  여러 설정을 한 번에 돌린다.  아래 [축] 만 고치고 실행:  ./scripts/sweep.sh
# =============================================================================
#
#  hydra 의 --multirun 과 같은 생각이다: 축을 여러 개 적으면 **조합이 전부** 돈다.
#
#  다만 결과를 담는 방식이 다르다. 기록 규약은 **한 run 에 여러 평가를 몰아넣는 걸
#  금지**한다 — 앞 결과가 덮이기 때문이다. 그래서 sweep 은 "결과 N개를 가진 run
#  하나" 가 아니라 **run N개 + 그것들을 묶는 `sweep` 태그** 다. 표에서 한 줄씩 서고,
#  태그로 한 묶음만 골라낼 수 있다.
#
#  기본값은 run.sh 에서 가져온다. 설정이 사는 곳은 거기 하나다 — 두 곳에 적어 두면
#  반드시 갈라진다. 여기서는 **바꿀 축만** 적는다.
#
# =============================================================================
#  [축] ────────────────────────────────────────────────────────── 여기만 고친다
# =============================================================================

# 이 묶음의 이름. 모든 run 에 sweep 태그로 붙어서 나중에 이것만 골라볼 수 있다
SWEEP_NAME="pruning-comparison"

# 축마다 값을 여러 개 적으면 조합이 전부 돈다.
# 값이 하나면 그 축은 안 도는 것과 같다 (run.sh 기본값을 덮어쓴다).
# 배열을 통째로 비우면 run.sh 값을 그대로 쓴다.
SWEEP_VARIANT=("Vanilla" "Pruned-24L" "INT8")
SWEEP_MODEL=()                    # 변형마다 체크포인트가 다르면 VARIANT 와 같은 길이로
SWEEP_NUM_TRAJ_SAMPLES=()         # 예: (1 6 16)
SWEEP_TEMPERATURE=()              # 예: (0.6 0.9)
SWEEP_INFERENCE_STEP=()           # 예: (5 10 20)

# 하나가 죽어도 나머지를 계속 돌릴까.  0 이면 첫 실패에서 멈춘다
CONTINUE_ON_FAILURE=1

# 진짜로 돌리기 전에 무엇이 돌지만 본다
DRY_RUN=0

# =============================================================================
#  아래는 안 고쳐도 된다
# =============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."

# run.sh 를 source 하면 설정만 들어오고 실행은 안 된다 (거기 sourced 가드가 있다)
# shellcheck disable=SC1091
source "$HERE/run.sh"

RED=$'\033[31m'; YEL=$'\033[33m'; GRN=$'\033[32m'; DIM=$'\033[2m'; BLD=$'\033[1m'; OFF=$'\033[0m'

# 비어 있는 축은 run.sh 의 값 하나짜리로 채운다 — 그래야 조합 계산이 균일해진다
[[ ${#SWEEP_VARIANT[@]}           -eq 0 ]] && SWEEP_VARIANT=("$VARIANT")
[[ ${#SWEEP_MODEL[@]}             -eq 0 ]] && SWEEP_MODEL=("$MODEL")
[[ ${#SWEEP_NUM_TRAJ_SAMPLES[@]}  -eq 0 ]] && SWEEP_NUM_TRAJ_SAMPLES=("$NUM_TRAJ_SAMPLES")
[[ ${#SWEEP_TEMPERATURE[@]}       -eq 0 ]] && SWEEP_TEMPERATURE=("$TEMPERATURE")
[[ ${#SWEEP_INFERENCE_STEP[@]}    -eq 0 ]] && SWEEP_INFERENCE_STEP=("$INFERENCE_STEP")

# 변형마다 체크포인트가 다른 경우는 조합이 아니라 짝이다. 길이가 같으면 짝으로 본다.
PAIRED_MODEL=0
if [[ ${#SWEEP_MODEL[@]} -gt 1 && ${#SWEEP_MODEL[@]} -eq ${#SWEEP_VARIANT[@]} ]]; then
  PAIRED_MODEL=1
fi

# ── 조합 만들기 ─────────────────────────────────────────────────────────────
JOBS=()
for vi in "${!SWEEP_VARIANT[@]}"; do
  v="${SWEEP_VARIANT[$vi]}"
  if [[ "$PAIRED_MODEL" == "1" ]]; then MODELS=("${SWEEP_MODEL[$vi]}"); else MODELS=("${SWEEP_MODEL[@]}"); fi
  for m in "${MODELS[@]}"; do
    for k in "${SWEEP_NUM_TRAJ_SAMPLES[@]}"; do
      for t in "${SWEEP_TEMPERATURE[@]}"; do
        for s in "${SWEEP_INFERENCE_STEP[@]}"; do
          JOBS+=("${v}|${m}|${k}|${t}|${s}")
        done
      done
    done
  done
done

TOTAL=${#JOBS[@]}
echo
echo "${BLD}── sweep: ${SWEEP_NAME} ─ ${TOTAL}개 조합 ──────────────────────${OFF}"
printf "  %-3s %-14s %-6s %-6s %-6s %s\n" "#" "VARIANT" "K" "temp" "step" "MODEL"
i=0
for job in "${JOBS[@]}"; do
  IFS='|' read -r v m k t s <<<"$job"; i=$((i+1))
  printf "  %-3s %-14s %-6s %-6s %-6s %s\n" "$i" "$v" "$k" "$t" "${s:-기본}" "$m"
done
echo
echo "  ${DIM}각각이 별개의 MLflow run 이 된다. sweep 태그로 묶인다.${OFF}"
[[ "$LIMIT" != "0" ]] && echo "  ${DIM}클립 ${LIMIT}개씩 · 총 $((TOTAL * LIMIT)) 클립분${OFF}"
echo

if [[ "$DRY_RUN" == "1" ]]; then
  echo "${YEL}DRY_RUN=1 — 여기까지.${OFF}"
  exit 0
fi

# ── 돌리기 ──────────────────────────────────────────────────────────────────
declare -a RESULTS=()
STARTED=$(date +%s)
i=0
for job in "${JOBS[@]}"; do
  IFS='|' read -r v m k t s <<<"$job"; i=$((i+1))
  echo
  echo "${BLD}[$i/$TOTAL] ${v}  K=${k}  temp=${t}${s:+  step=$s}${OFF}"
  echo "────────────────────────────────────────────────────────────"
  t0=$(date +%s)
  # run.sh 를 그대로 다시 부른다 — 점검이 조합마다 돈다. 설정은 환경변수로 덮는다.
  if OVERRIDE_VARIANT="$v" OVERRIDE_MODEL="$m" OVERRIDE_NUM_TRAJ_SAMPLES="$k" \
     OVERRIDE_TEMPERATURE="$t" OVERRIDE_INFERENCE_STEP="$s" SWEEP="$SWEEP_NAME" \
     bash "$HERE/run.sh"; then
    RESULTS+=("${GRN}완료${OFF}|$v|$k|$t|$(( $(date +%s) - t0 ))초")
  else
    RESULTS+=("${RED}실패${OFF}|$v|$k|$t|$(( $(date +%s) - t0 ))초")
    if [[ "$CONTINUE_ON_FAILURE" != "1" ]]; then
      echo "${RED}CONTINUE_ON_FAILURE=0 — 여기서 멈춘다.${OFF}" >&2
      break
    fi
    echo "${YEL}계속한다. 나머지 조합은 이것과 무관하다.${OFF}" >&2
  fi
done

# ── 요약 ────────────────────────────────────────────────────────────────────
echo
echo "${BLD}── sweep 끝: ${SWEEP_NAME} ─ $(( $(date +%s) - STARTED ))초 ────────────${OFF}"
printf "  %-6s %-14s %-6s %-6s %s\n" "결과" "VARIANT" "K" "temp" "걸린 시간"
ok_n=0
for r in "${RESULTS[@]}"; do
  IFS='|' read -r st v k t el <<<"$r"
  printf "  %-6s %-14s %-6s %-6s %s\n" "$st" "$v" "$k" "$t" "$el"
  [[ "$st" == *"완료"* ]] && ok_n=$((ok_n+1))
done
echo
echo "  ${ok_n}/${#RESULTS[@]} 완료"
echo "  ${DIM}허브에서 sweep=${SWEEP_NAME} 로 이 묶음만 골라볼 수 있다${OFF}"
[[ "$ok_n" -eq "${#RESULTS[@]}" ]] || exit 1
