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
MACHINE="Pro6000 x 1 - Lent"

# 어느 클립을 돌릴 것인가.  clip_id 열이 있는 parquet
#
#   clip_ids_gold644.parquet  NVIDIA 공식 평가 세트.  recipes 의 eval.py 가
#                             --parquet 기본값으로 쓰는 그 644개 (열 이름만 맞춤).
#                             세트 선택이 우리 판단이 아니라는 게 이 파일의 값어치다 —
#                             "왜 그 클립을 골랐냐" 는 질문이 안 나온다.
#   clip_ids.parquet          데모용 1181개.  gold 644 를 부분집합으로 포함한다.
#                             더 넓게 보고 싶을 때만.
# CLIP_LIST="notebooks/clip_ids_gold644.parquet"    # 공식 644
CLIP_LIST="notebooks/clip_ids.parquet"           # 전체 1181 (gold 를 포함)

# 앞에서 몇 개만 볼까.  0 이면 목록 전부 — **실험에서는 항상 0 이다.**
#   LIMIT 은 스모크용 손잡이지 부분집합을 만드는 방법이 아니다. 250 으로 두면
#   "gold 644 의 앞 250개" 인데, 그 250개가 왜 그 250개인지는 parquet 에 저장된
#   순서 말고 근거가 없다. 진짜로 작은 세트가 필요하면 CLIP_LIST 를 새로 만든다 —
#   그래야 "이 기준으로 골랐다" 가 파일로 남는다 (gold644 를 그렇게 만들었다).
LIMIT=0
NUM_TRAJ_SAMPLES=6                       # 클립당 뽑을 궤적 수 (K).
                                         # 공식 지표가 minADE_6 이라 6 이 아니면 그 수치와 비교가 안 된다

# 특정 클립만 돌릴 때.  적으면 CLIP_LIST 와 LIMIT 을 무시한다
#   CLIP_IDS=("030c760c-ae38-49aa-9ad8-f5650a545d26" "0347d9f9-...")
CLIP_IDS=()

# ── 아래 여섯 개는 NVIDIA 의 값이다 ────────────────────────────────────────
# alpamayo-recipes/recipes/alpamayo1_5_quant/eval.py 의 argparse 기본값과 동일하다.
# 모델 카드가 보고하는 minADE_6 @6.4s = 0.916m 이 이 설정에서 나온 숫자이므로,
# 하나라도 바꾸면 그 수치와 비교할 근거가 사라진다.  바꿀 거면 왜 바꾸는지 적을 것.
#
# 클립 안 어느 시점에서 예측하나 (마이크로초).  t0 이전 1.5초를 보고 이후 6.4초를 예측한다.
# gold parquet 에 event_t0s 열이 있지만 eval.py 는 쓰지 않는다 — 고정 5.1초다.
T0_US=5100000
TEMPERATURE=0.6                          # CoT 텍스트 생성용.  확산 노이즈와 무관하다
TOP_P=0.98
SEED=42                                  # 클립마다 추론 직전에 건다.  같은 시드 → 같은 CoT
                                         # → 같은 초기 노이즈.  paired 비교의 전제다
MAX_GENERATION_LENGTH=256                # 상한일 뿐. 실측 CoT 는 6~14 토큰이다
MODEL="nvidia/Alpamayo-1.5-10B"
# ──────────────────────────────────────────────────────────────────────────

INFERENCE_STEP=10                        # Euler 적분 스텝.  ⚠️ 비우면 열이 null 로 남아
                                         # 다른 run 과 그룹핑이 안 된다.  기본값도 10 이지만 명시한다

# 어느 파이썬으로 돌리나.  "python" 이면 PATH 에 있는 것 (venv 를 미리 활성화한 경우)
#   venv 가 이 레포 밖에 있으면 절대경로로 적는다
PYTHON="/workspace/YSH-KD-Alpamayo-1.5/a1_5_venv/bin/python"

# 이 레포의 src 를 sys.path 앞에 놓을까.  venv 에 alpamayo1_5 가 이미 설치돼 있으면
# 그쪽이 먼저 잡혀서 trace/ 가 없는 코드가 돈다 — 에러가 아니라 조용히 다른 코드다.
#   0 이면 안 건드린다 (이 레포를 uv sync 로 직접 설치한 경우)
FORCE_LOCAL_SRC=1

# 모델과 데이터 (MODEL 은 위 NVIDIA 블록에 있다)
ATTN="sdpa"                              # Jetson 은 sdpa. flash_attention_2 는 aarch64 휠이 없다
DATA_SPEC="Cam-4"
DATA_CACHE="/workspace/.hf_home/hub"     # HF 허브 캐시 루트 (cache_dir 로 그대로 넘어간다)
OUT_ROOT="/workspace/runs"               # run 폴더가 생길 곳. 레포 밖에 둬서
                                         # git add -A 로 결과가 딸려 들어가지 않게 한다

# 캐시에 없는 클립을 허브에서 받아올까.  기본은 아니오 —
# 켜면 조용히 10배 느려지고, 캐시본과 스트림본이 섞여 무엇을 읽었는지 흐려진다
ALLOW_STREAM=0

# 이 run 이 왜 존재하는지 한두 문장.  6개월 뒤의 네가 읽는다
NOTES="Euler 스텝 스윕. minADE 와 meanADE 가 반대로 움직이는지, 그 괴리가 기동에 따라 커지는지"

# 어디에 기록하나
EXPERIMENT="alpamayo-1.5"                # 프로젝트 이름. 실행마다 바꾸지 않는다
EVALS_REPO="YSHRobotics/Alpamayo-Evals"
ML_PLATFORM_HOST="ysh-jetson-orin-nano.tail4570ef.ts.net"

# MLflow 는 테일넷 안에만 열려 있다.  이 기계가 테일넷에 커널 수준으로 붙지 못하면
# (비특권 컨테이너는 /dev/net/tun 이 없어 userspace 모드로만 돈다) SOCKS 프록시를 거쳐야 한다.
#   비우면 직접 연결한다.  MLflow 는 http 라 http_proxy 만 걸면 되고,
#   HF 는 https 라 프록시를 안 타므로 업로드는 그대로 직결이다.
NETWORK_PROXY="socks5h://127.0.0.1:1055" # 비우면 직접 연결

# 껐다 켜는 것들 (1 = 켬)
UPLOAD=1                                 # HF 에 산출물 올리기
TRACK=1                                  # MLflow 에 기록하기.  테일넷 밖이면 0
SAMPLES=1                                # 시각화 PNG 만들기
INCLUDE_GT=1                             # 정답 궤적을 로컬에 남기기 (업로드는 안 된다).
                                         # 이게 있어야 나중에 지표를 GPU 없이 다시 계산할 수 있다 —
                                         # 없으면 정의가 바뀔 때마다 모델을 다시 돌려야 한다

# =============================================================================
#  아래는 안 고쳐도 된다
# =============================================================================
# run_sweep.sh 가 이 파일을 source 해서 위 설정을 기본값으로 가져간다. 설정을 두 곳에
# 적어 두면 반드시 갈라지므로, 설정이 사는 곳은 여기 하나다.
(return 0 2>/dev/null) && return 0

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# run_sweep.sh 가 조합마다 덮어쓰는 자리. 위 설정 블록은 손대지 않는다 —
# 거기에 ${VAR:-기본값} 을 쓰면 읽기 어려워지고, 무엇이 기본값인지 흐려진다.
VARIANT="${OVERRIDE_VARIANT:-$VARIANT}"
MODEL="${OVERRIDE_MODEL:-$MODEL}"
# 클립 목록은 **비교 축이 아니라 실험의 경계**다. 한 sweep 안에서 바뀌면 arm 마다
# 다른 클립을 보게 되어 paired 비교가 성립하지 않으므로 SWEEP_ 축은 두지 않는다.
# 대신 오버라이드는 둔다 — 같은 sweep 을 다른 세트로 한 번 더 돌리는 건 별개의 실험이고,
# 그때 이 파일을 고치면 워킹트리가 더러워져 run 들이 git_dirty 로 기록된다.
CLIP_LIST="${OVERRIDE_CLIP_LIST:-$CLIP_LIST}"
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
# 러너가 만드는 것과 같은 규칙으로 조립한다. 여기 미리보기와 실제 폴더 이름이
# 다르면 미리보기가 없느니만 못하다.
if [[ ${#CLIP_IDS[@]} -gt 0 ]]; then CLIPS_SHOWN="${#CLIP_IDS[@]}"
elif [[ "$LIMIT" != "0" ]];    then CLIPS_SHOWN="$LIMIT"
else                                CLIPS_SHOWN="N"; fi
NAME_TAIL="${CLIPS_SHOWN}clip_k${NUM_TRAJ_SAMPLES}-temp${TEMPERATURE}"
[[ -n "${LABEL:-}" ]] && NAME_TAIL="${NAME_TAIL}-${LABEL}"
ok "이름:  Alpamayo-1.5_${DATA_SPEC}_${VARIANT}_${NAME_TAIL}_${MACHINE_SHOWN}_$(date +%y.%m.%d)_<run_id>"

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
  ok "git:   $(git rev-parse --short HEAD) ($(git rev-parse --abbrev-ref HEAD), clean)"
fi

# 커밋이 어느 원격 브랜치에도 없으면 git_commit 은 이 디스크에만 있는 좌표다.
# 기록 자체는 정상으로 남는다 — 그래서 조용하다. 나중에 rebase 나 amend 를 하면
# 그 좌표는 아무것도 가리키지 않게 된다.
if git rev-parse --git-dir >/dev/null 2>&1 \
   && [[ -z "$(git branch -r --contains HEAD 2>/dev/null)" ]]; then
  warn "이 커밋은 아직 push 안 됐다. 파라미터와 지표는 정상 기록되지만,
       git_commit 이 이 기계에만 있는 커밋을 가리킨다 — 다른 사람은 못 찾고,
       나중에 rebase/amend 하면 그 좌표는 죽는다"
fi

# ── 파이썬 ─────────────────────────────────────────────────────────────────
# 인터프리터가 있는지가 아니라 **어느 alpamayo1_5 를 import 하는지**를 본다.
# venv 에 다른 체크아웃이 설치돼 있으면 그쪽이 먼저 잡히고, trace/ 가 없어서
# 계측이 통째로 빠진 채 돈다 — 에러 없이, 결과 파일만 조용히 비어서.
if ! command -v "$PYTHON" >/dev/null 2>&1 && [[ ! -x "$PYTHON" ]]; then
  fail "PYTHON 을 못 찾는다: $PYTHON
       venv 를 활성화했거나, 이 파일의 PYTHON 에 절대경로를 적어야 한다"
else
  PY_SRC=$( [[ "$FORCE_LOCAL_SRC" == "1" ]] && echo "PYTHONPATH=$PWD/src" || echo "" )
  RESOLVED=$(env $PY_SRC "$PYTHON" -c \
    'import alpamayo1_5.trace, os; print(os.path.dirname(os.path.dirname(alpamayo1_5.trace.__file__)))' 2>&1)
  if [[ "$RESOLVED" != "$PWD/src/alpamayo1_5" ]]; then
    fail "alpamayo1_5 가 다른 체크아웃에서 잡힌다:
       기대: $PWD/src/alpamayo1_5
       실제: $RESOLVED
       FORCE_LOCAL_SRC=1 로 두거나, 이 레포를 venv 에 설치한다"
  else
    ok "python: $("$PYTHON" -c 'import sys;print(sys.version.split()[0])') · alpamayo1_5 ← 이 레포"
  fi
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
# hf auth login 은 $HF_HOME 아래에 토큰을 쓴다. HF_HOME 을 옮겨 둔 기계에서
# ~/.cache/huggingface 만 보면 토큰이 있는데도 없다고 막는다 — 실제로 그랬다.
HF_TOKEN_FILE="${HF_HOME:-$HOME/.cache/huggingface}/token"
if [[ -z "${HF_TOKEN:-}" && ! -f "$HF_TOKEN_FILE" ]]; then
  warn "HF_TOKEN 이 없다 ($HF_TOKEN_FILE 도 없음).
       @main 이 40자리 sha 로 안 풀려서 좌표가 움직이는 이름으로 남는다"
  [[ "$UPLOAD" == "1" ]] && fail "업로드도 못 한다. export HF_TOKEN=... 하거나 UPLOAD=0"
else
  ok "HF:    ${HF_TOKEN:+환경변수}${HF_TOKEN:-$HF_TOKEN_FILE}"
fi

# ── 프록시 ─────────────────────────────────────────────────────────────────
# http_proxy 만 건다. MLflow 는 http 라 이걸 타고, HF 는 https 라 안 타므로
# 업로드는 직결로 남는다 — 87GB 를 릴레이로 보내는 사고를 구조적으로 막는다.
if [[ -n "$NETWORK_PROXY" ]]; then
  export http_proxy="$NETWORK_PROXY" HTTP_PROXY="$NETWORK_PROXY"
  export no_proxy="localhost,127.0.0.1" NO_PROXY="localhost,127.0.0.1"
  ok "프록시: $NETWORK_PROXY (http 만 — https/HF 는 직결)"
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
         · 프록시를 거친다:         이 파일에서 NETWORK_PROXY=\"socks5h://127.0.0.1:1055\"
                                  (userspace 모드 tailscaled 가 여는 SOCKS 포트)
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
[[ -n "${LABEL:-}" ]]      && ARGS+=(--label "$LABEL")
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

[[ "$FORCE_LOCAL_SRC" == "1" ]] && export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

echo
echo "${DIM}${PYTHON} scripts/run_inference_tracked.py ${ARGS[*]}${OFF}"
echo "────────────────────────────────────────────────────────────"
echo
exec "$PYTHON" scripts/run_inference_tracked.py "${ARGS[@]}"
