# D2 — minADE_K 곡선의 K>6 쪽.  K≤6 은 새 run 이 필요 없다: 샘플이 교환 가능하므로 (clip-hash)
# K=6 run 의 부분집합 min 이 곧 minADE_k (k=1..6) 이다.  K=12 만 실제로 돈다.
# 묻는 것: 잃은 커버리지를 샘플 수로 되살 수 있나 — s2 의 K=12 가 s10 의 K=6 에 닿는가.
# A1: K=12 는 expert 75 ms/step, vision·prefill 도 K 배 → 클립당 ≈ 5 s.
SWEEP_NAME="K12-pilot400"
SWEEP_NUM_TRAJ_SAMPLES=(12)
SWEEP_INFERENCE_STEP=(10 2)
SWEEP_CLIP_LIST="notebooks/clip_ids_pilot400.parquet"
