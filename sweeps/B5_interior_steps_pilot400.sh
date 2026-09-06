# B5 — 스윕의 안쪽.  10/4/2/1 네 점으로는 무릎을 못 찾는다.
# 묻는 것 둘: (a) minADE·다양성 곡선의 모양 (어디서 꺾이나), (b) 각 스텝 arm 을 C1 τ 곡선 위에
# 놓았을 때 "다양성으로 설명 안 되는 잔여" 가 몇 스텝부터 생기나 — 2 에서 0, 1 에서 0.1 이면
# 3 은 어디인가.  10 을 같은 400 클립에서 다시 돌리는 이유: 게이트의 clip_sets 검사가 arm 간
# 동일 클립을 요구한다 (1300 run 은 부분집합이라 못 쓴다).
SWEEP_NAME="Euler-Interior-pilot400"
SWEEP_INFERENCE_STEP=(10 8 6 5 3)
SWEEP_CLIP_LIST="notebooks/clip_ids_pilot400.parquet"
