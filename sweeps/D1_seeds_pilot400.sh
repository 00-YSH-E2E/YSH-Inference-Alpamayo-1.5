# D1 — run-to-run 밴드.  지금까지 모든 정확도 숫자는 seed 42 하나다.
# clip-hash 시드 뒤에는 시드를 바꾸는 것이 진짜 복제다 (옛 프로토콜에서는 노이즈 6개를 1300번
# 복사한 것이라 시드 바꿈 = 노이즈 6개 바꿈이었다).  seed 42 arm 은 이미 있다 (s10 = C1 dt1.0,
# s2 = B 블록 1300 의 부분집합).  묻는 것: 부트스트랩 CI 가 시드 간 분산을 덮는가.
SWEEP_NAME="Seeds-pilot400"
SWEEP_SEED=(7 1234)
SWEEP_INFERENCE_STEP=(10 2)
SWEEP_CLIP_LIST="notebooks/clip_ids_pilot400.parquet"
