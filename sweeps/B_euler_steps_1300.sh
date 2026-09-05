# B 블록 — 논문 Table 1.
# 시드 수정(clip-hash) 뒤의 네 arm.  기존 4개 run 은 per-clip 재시드라 샘플이 비교환이어서
# 절대 minADE_6 에 못 쓴다 — 짝지은 비교만 유효했다.  1300 클립 전수 (split 열이 실리므로
# 보고는 val+test 700 으로 자른다; train 600 은 학생의 학습 세트 기준선).
SWEEP_NAME="Euler-Step-1300"
SWEEP_INFERENCE_STEP=(10 4 2 1)
SWEEP_CLIP_LIST="notebooks/clip_ids_cached1300.parquet"
