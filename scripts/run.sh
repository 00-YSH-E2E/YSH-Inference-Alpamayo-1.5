#!/usr/bin/env bash
# =============================================================================
#  추론 한 번 돌리기.  아래 [설정] 만 고치고 실행한다:  ./scripts/run.sh
# =============================================================================
#
#  왜 이 파일이 있나: 플래그를 외우지 않기 위해서, 그리고 **GPU 시간을 쓰기 전에**
#  틀린 걸 잡기 위해서다. 기록은 소급이 안 된다 — run 이 자기 GPU 를 안 봤으면
#  나중에 어느 기계였는지 적어 넣을 방법이 없고, 세 시간 뒤에 알아차리면 세 시간이
#  날아간다. 그래서 아래 검사가 전부 **추론 시작 전에** 돈다.
#
# =============================================================================
#  [설정] ─────────────────────────────────────────────────────────── 여기만 고친다
# =============================================================================

# 무엇이 다른 run 인가.  비교 축이다 (SKILL §14-1)
#   좋은 예: Vanilla · Pruned-24L · INT8 · KD-student
#   나쁜 예: exp4 · v2 · Vanilla-26.09.01   ← 날짜·번호는 run_id 가 이미 안다
VARIANT="Vanilla"

# 어느 기계인가.  비우면 호스트명을 쓴다.
#   폴더 이름과 run 이름에 들어가고, config/experiments.yaml 이 이 값으로 기기를 가른다.
#   호스트명이 그 기계를 부르는 이름이 아닐 때만 적는다 (대여 박스, 컨테이너)
MACHINE=""

# 몇 개나 돌릴 것인가
CLIP_LIST="notebooks/clip_ids.parquet"   # clip_id 열이 있는 parquet
LIMIT=1                                  # 0 이면 목록 전부
NUM_TRAJ_SAMPLES=6                       # 클립당 뽑을 궤적 수 (K)

# 특정 클립만 돌릴 때.  적으면 CLIP_LIST 와 LIMIT 을 무시한다
#   CLIP_IDS=("030c760c-ae38-49aa-9ad8-f5650a545d26" "0347d9f9-...")
CLIP_IDS=()

# 클립 안 어느 시점부터 예측하나 (마이크로초).  바꾸면 다른 상황을 보는 셈이라
# 이전 run 과 비교가 안 된다
T0_US=5100000

# 샘플링
TEMPERATURE=0.6
TOP_P=0.98
SEED=42
MAX_GENERATION_LENGTH=256
INFERENCE_STEP=""                        # 비우면 체크포인트 기본값

# 모델과 데이터
MODEL="nvidia/Alpamayo-1.5-10B"
ATTN="sdpa"                              # Jetson 은 sdpa. flash_attention_2 는 aarch64 휠이 없다
DATA_SPEC="Cam-4"
DATA_CACHE="/home/thor/Documents/Alpamayo/Data/Alpamayo-1.5_Cam-4_Vanilla"
OUT_ROOT="out"                           # run 폴더가 생길 곳

# 캐시에 없는 클립을 허브에서 받아올까.  기본은 아니오 —
# 켜면 조용히 10배 느려지고, 캐시본과 스트림본이 섞여 무엇을 읽었는지 흐려진다
ALLOW_STREAM=0

# 이 run 이 왜 존재하는지 한두 문장.  6개월 뒤의 네가 읽는다
NOTES="baseline 재측정"

# 어디에 기록하나
EXPERIMENT="alpamayo-1.5"                # 프로젝트 이름. 실행마다 바꾸지 않는다
EVALS_REPO="YSHRobotics/Alpamayo-Evals"
ML_PLATFORM_HOST="ysh-jetson-orin-nano.tail4570ef.ts.net"

# 껐다 켜는 것들 (1 = 켬)
UPLOAD=1                                 # HF 에 산출물 올리기
TRACK=1                                  # MLflow 에 기록하기.  테일넷 밖이면 0
SAMPLES=1                                # 시각화 PNG 만들기
INCLUDE_GT=0                             # 정답 궤적을 로컬에 남기기 (업로드는 안 된다)

# =============================================================================
#  아래는 안 고쳐도 된다
# =============================================================================
# sweep.sh 가 이 파일을 source 해서 위 설정을 기본값으로 가져간다. 설정을 두 곳에
# 적어 두면 반드시 갈라지므로, 설정이 사는 곳은 여기 하나다.
(return 0 2>/dev/null) && return 0

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# sweep.sh 가 조합마다 덮어쓰는 자리. 위 설정 블록은 손대지 않는다 —
# 거기에 ${VAR:-기본값} 을 쓰면 읽기 어려워지고, 무엇이 기본값인지 흐려진다.
VARIANT="${OVERRIDE_VARIANT:-$VARIANT}"
MODEL="${OVERRIDE_MODEL:-$MODEL}"
NUM_TRAJ_SAMPLES="${OVERRIDE_NUM_TRAJ_SAMPLES:-$NUM_TRAJ_SAMPLES}"
TEMPERATURE="${OVERRIDE_TEMPERATURE:-$TEMPERATURE}"
INFERENCE_STEP="${OVERRIDE_INFERENCE_STEP-$INFERENCE_STEP}"
SEED="${OVERRIDE_SEED:-$SEED}"
LIMIT="${OVERRIDE_LIMIT:-$LIMIT}"
SWEEP="${SWEEP:-}"

RED=$'\033[31m'; YEL=$'\033[33m'; GRN=$'\033[32m'; DIM=$'\033[2m'; OFF=$'\033[0m'
# 처음 걸린 것에서 멈추지 않는다. 하나 고치고 다시 돌려서 다음 걸 발견하는 건
# 두더지잡기고, 그 사이에 고칠 마음이 식는다. 전부 모아서 한 번에 보여준다.
PROBLEMS=0
fail() { echo "${RED}막힘${OFF}  $1" >&2; PROBLEMS=$((PROBLEMS + 1)); }
warn() { echo "${YEL}주의${OFF}  $1" >&2; }
ok()   { echo "${GRN}OK${OFF}    $1"; }

echo
echo "── 돌리기 전 점검 ──────────────────────────────────────────"

# ── 이름 규칙 (SKILL §14-1) ─────────────────────────────────────────────────
[[ -n "$VARIANT" ]] || fail "VARIANT 가 비었다. 무엇이 다른 run 인지가 비교 축이다"
if [[ "$VARIANT" =~ [0-9]{2}[.\_-][0-9]{2}[.\_-][0-9]{2} ]]; then
  fail "VARIANT 에 날짜가 있다 ($VARIANT). 언제 돌렸는지는 run_id 와 시작 시각이 이미 안다.
       변형 이름은 비교 축이라 같은 변형이면 언제 돌리든 같은 문자열이어야 한다"
fi
if [[ "$VARIANT" =~ ^(exp|run|test|tmp)[-_]?[0-9]+$ ]] || [[ "$VARIANT" =~ ^(v|ver)?[0-9]+$ ]]; then
  fail "VARIANT 가 '$VARIANT' 이면 무엇이 다른지 안 보인다.
       Vanilla · Pruned-24L · INT8 처럼 바뀐 내용을 적는다"
fi
MACHINE_SHOWN="${MACHINE:-$(hostname -s)}"
ok "이름:  Alpamayo-1.5_${DATA_SPEC}_${VARIANT}_${MACHINE_SHOWN}_$(date +%y.%m.%d)_<run_id>"

# ── 소급 불가한 것들 ────────────────────────────────────────────────────────
[[ -n "$NOTES" ]] || warn "NOTES 가 비었다. 6개월 뒤에 이 run 이 왜 있는지 알 수 없다"

# git 이 실패한 것과 워킹트리가 깨끗한 것은 둘 다 빈 출력을 낸다. 가르지 않으면
# 레포 밖에서 돌린 run 이 "clean" 으로 보이고, 정작 git_commit 은 비어서 나간다.
if ! git rev-parse --git-dir >/dev/null 2>&1; then
  fail "여기는 git 레포가 아니다: $(pwd)
       코드 좌표(git_commit)가 통째로 비고, 그건 나중에 채워 넣을 수 없다"
elif [[ -n "$(git status --porcelain)" ]]; then
  warn "워킹트리가 dirty 다. 이 run 은 정확히 재현되지 않는다 — 먼저 커밋하는 게 낫다"
  git status --porcelain | head -5 | sed 's/^/      /'
else
  ok "git:   $(git rev-parse --short HEAD) (clean)"
fi

# ── 데이터 ─────────────────────────────────────────────────────────────────
DATA_OK=1
if [[ ! -d "$DATA_CACHE" ]]; then
  if [[ "$ALLOW_STREAM" == "1" ]]; then
    warn "DATA_CACHE 가 없는데 ALLOW_STREAM=1 이다: $DATA_CACHE
       전부 허브에서 받는다 — 분 단위로 끝날 일이 시간 단위가 된다"
  else
    DATA_OK=0
    fail "DATA_CACHE 가 없다: $DATA_CACHE
       캐시가 없으면 스트리밍으로 떨어져 10배 느려지고, 출력에는 아무 표시도 안 남는다
       일부러 그러는 거면 ALLOW_STREAM=1"
  fi
fi
# 클립을 직접 지정했으면 목록 파일은 안 읽는다
if [[ ${#CLIP_IDS[@]} -eq 0 && -n "$CLIP_LIST" && ! -f "$CLIP_LIST" ]]; then
  DATA_OK=0; fail "CLIP_LIST 가 없다: $CLIP_LIST"
fi
if [[ "$DATA_OK" == "1" ]]; then
  if [[ ${#CLIP_IDS[@]} -gt 0 ]]; then
    ok "데이터: $DATA_CACHE  ·  클립 ${#CLIP_IDS[@]}개 직접 지정"
  else
    ok "데이터: $DATA_CACHE${CLIP_LIST:+  ·  $CLIP_LIST}$([[ "$LIMIT" != "0" ]] && echo "  ·  앞 $LIMIT 개")"
  fi
fi

# ── 자격 ───────────────────────────────────────────────────────────────────
if [[ -z "${HF_TOKEN:-}" && ! -f "$HOME/.cache/huggingface/token" ]]; then
  warn "HF_TOKEN 이 없다. @main 이 40자리 sha 로 안 풀려서 좌표가 움직이는 이름으로 남는다"
  [[ "$UPLOAD" == "1" ]] && fail "업로드도 못 한다. export HF_TOKEN=... 하거나 UPLOAD=0"
fi

# ── MLflow 도달 ────────────────────────────────────────────────────────────
# 여기서 3초 만에 확인한다. 안 그러면 mlflow 클라이언트가 재시도 7회 × 타임아웃
# 120초로 몇 분을 매달린 뒤에야 죽는다 — 그것도 추론이 시작되기도 전에.
if [[ "$TRACK" == "1" ]]; then
  URI="${MLFLOW_TRACKING_URI:-http://${ML_PLATFORM_HOST}:5000}"
  if curl -fsS -m 3 "${URI}/health" >/dev/null 2>&1; then
    ok "MLflow: $URI"
  else
    fail "MLflow 에 3초 안에 못 닿았다: $URI
       테일넷 밖이면 MLflow 는 안 열린다 (100.81.70.49 에만 묶여 있다).
       고르는 법:
         · 테일넷에 붙는다:        tailscale up
         · 터널을 판다:            ssh -N -L 5000:100.81.70.49:5000 <중계>
                                  export MLFLOW_TRACKING_URI=http://localhost:5000
         · 기록 없이 돌린다:        이 파일에서 TRACK=0
                                  (추론과 HF 업로드는 그대로 된다)"
  fi
  export MLFLOW_TRACKING_URI="$URI"
else
  warn "TRACK=0 — MLflow 기록 없이 돈다. 산출물은 HF 로 가고 숫자는 로컬 run.json 에만 남는다"
fi

export ML_PLATFORM_HOST

if [[ "$PROBLEMS" -gt 0 ]]; then
  echo
  echo "${RED}${PROBLEMS}건이 막고 있다. 위를 고치고 다시 돌린다.${OFF}" >&2
  echo "${DIM}아직 아무것도 안 돌았다 — GPU 시간은 안 썼다.${OFF}" >&2
  exit 1
fi

# ── 명령 조립 ──────────────────────────────────────────────────────────────
ARGS=(
  --variant "$VARIANT"
  --num-traj-samples "$NUM_TRAJ_SAMPLES"
  --t0-us "$T0_US"
  --temperature "$TEMPERATURE" --top-p "$TOP_P" --seed "$SEED"
  --max-generation-length "$MAX_GENERATION_LENGTH"
  --model "$MODEL" --attn "$ATTN"
  --data-spec "$DATA_SPEC" --data-cache "$DATA_CACHE"
  --out-root "$OUT_ROOT"
  --experiment "$EXPERIMENT" --evals-repo "$EVALS_REPO"
)
[[ -n "$MACHINE" ]]        && ARGS+=(--machine "$MACHINE")
[[ -n "$NOTES" ]]          && ARGS+=(--notes "$NOTES")
[[ -n "$SWEEP" ]]          && ARGS+=(--sweep "$SWEEP")
[[ -n "$INFERENCE_STEP" ]] && ARGS+=(--inference-step "$INFERENCE_STEP")
# 클립을 직접 지정했으면 목록과 개수 제한은 뜻이 없다
if [[ ${#CLIP_IDS[@]} -gt 0 ]]; then
  for c in "${CLIP_IDS[@]}"; do ARGS+=(--clip-id "$c"); done
else
  [[ -n "$CLIP_LIST" ]] && ARGS+=(--clip-list "$CLIP_LIST")
  [[ "$LIMIT" != "0" ]] && ARGS+=(--limit "$LIMIT")
fi
[[ "$ALLOW_STREAM" == "1" ]] && ARGS+=(--allow-stream)
[[ "$UPLOAD"       == "0" ]] && ARGS+=(--no-upload)
[[ "$TRACK"        == "0" ]] && ARGS+=(--no-track)
[[ "$SAMPLES"      == "0" ]] && ARGS+=(--no-samples)
[[ "$INCLUDE_GT"   == "1" ]] && ARGS+=(--include-gt)

echo
echo "${DIM}python scripts/run_inference_tracked.py ${ARGS[*]}${OFF}"
echo "────────────────────────────────────────────────────────────"
echo
exec python scripts/run_inference_tracked.py "${ARGS[@]}"
