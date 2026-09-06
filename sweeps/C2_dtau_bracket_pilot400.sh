# C2 — s1 을 온도 사다리 안에 넣는다.
# C1 판정: s4·s2 의 페널티는 같은 다양성의 τ 곡선과 일치 (gap +0.03, +0.00), s1 만 +0.10 [+0.05, +0.15].
# 단 s1 의 diversity_final 1.06 이 사다리 바닥 (τ=0.2 → 1.15) 아래라 그 0.10 은 외삽이다.
# 스텝 수와 다양성의 선형 관계로 s1 은 τ≈0.17 에 해당한다.  0.17 과 0.12 로 아래를 막으면
# "1스텝의 진짜 적분 오차 ≈ 0.08–0.10 m" 가 보간이 된다.  1.0 을 다시 넣는 이유: 큐 뒤 커밋
# (4c02466: path: 좌표·--x0-from·distill/ 복사·게이트 수정) 이 추론 경로에 영향이 없음을
# C1 의 dt1.0 (95ed8fa8) 과 비트 비교로 증명하고, 이 sweep 만으로도 게이트가 닫히게 한다.
SWEEP_NAME="DTau-Bracket-pilot400"
SWEEP_INFERENCE_STEP=(10)
SWEEP_DIFFUSION_TEMPERATURE=(1.0 0.17 0.12)
SWEEP_CLIP_LIST="notebooks/clip_ids_pilot400.parquet"
