#!/bin/bash

# 시작 시간 기록
start_time=$(date +%s)

python scripts/1_down_cut.py
python scripts/2_asr.py
python -m dac encode ./data/cut/ --output ./data/codes 

# 종료 시간 기록 및 총 소요 시간 계산
end_time=$(date +%s)
duration=$((end_time - start_time))

# 시간을 시:분:초 형식으로 변환
hours=$((duration / 3600))
minutes=$(( (duration % 3600) / 60 ))
seconds=$((duration % 60))

echo "파이프라인 총 실행 시간: ${hours}시간 ${minutes}분 ${seconds}초" 