# C1 — 다양성 맞춘 대조군.  make-or-break #1.
# 스텝을 10 으로 고정하고 확산 초기 노이즈 온도만 내린다.  expert 호출 수는 그대로다.
# s1 의 diversity_final (1.058) 을 재현하는 온도에서 minADE 가 s1 의 1.28 근처면
# 1스텝 페널티 전체가 다양성 붕괴이고 (논문 = 증류가 분포를 복원), 0.82 근처에 머물면
# 진짜 적분 오차다 (논문 = 속도장 보정).  둘은 다른 논문이다.
# 1.0 을 포함하는 이유: 같은 400 클립 위의 기준선이 있어야 게이트가 통과한다.
SWEEP_NAME="DTau-Ladder-pilot400"
SWEEP_INFERENCE_STEP=(10)
SWEEP_DIFFUSION_TEMPERATURE=(1.0 0.85 0.7 0.5 0.35 0.2)
SWEEP_CLIP_LIST="notebooks/clip_ids_pilot400.parquet"
