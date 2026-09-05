# A1 — K 사다리, 지연시간 목적.  30 클립이면 충분하다 (정확도 주장은 안 한다).
# 묻는 것: head 지연이 K 에 얼마나 비례하나.  A0 프로파일이 "스텝당 런치 ~4900, copy/cat 지배"
# 라 했으므로 K=1 이 K=6 의 1/6 이 아니라 거의 같을 것으로 예상 — 그러면 K 는 공짜고
# 스텝 수만이 지연 레버다.  12 arm × 30 클립 ≈ 26 분.
SWEEP_NAME="K-Ladder-latency30"
SWEEP_NUM_TRAJ_SAMPLES=(1 2 4 6 12 16)
SWEEP_INFERENCE_STEP=(10 1)
SWEEP_CLIP_LIST="notebooks/clip_ids_latency30.parquet"
