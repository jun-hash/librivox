#!/bin/bash

# 시작 시간 기록
start_time=$(date +%s)

python scripts/1_down_cut.py --start_idx 0 --end_idx 500

# ASR 스크립트 실행 - 성공할 때까지 재시도
max_attempts=3
attempt=1
asr_success=false

while [ $attempt -le $max_attempts ] && [ "$asr_success" = false ]; do
    echo "ASR 실행 시도 #$attempt"
    if python scripts/2_asr.py; then
        asr_success=true
        echo "ASR 성공적으로 완료"
    else
        echo "ASR 실행 실패 (시도 $attempt/$max_attempts)"
        if [ $attempt -lt $max_attempts ]; then
            echo "30초 후 재시도..."
            sleep 30
        fi
        attempt=$((attempt + 1))
    fi
done

if [ "$asr_success" = false ]; then
    echo "최대 재시도 횟수 초과. 파이프라인을 중단합니다."
    exit 1
fi

python scripts/3_dac.py

# 종료 시간 기록 및 총 소요 시간 계산
end_time=$(date +%s)
duration=$((end_time - start_time))

# 시간을 시:분:초 형식으로 변환
hours=$((duration / 3600))
minutes=$(( (duration % 3600) / 60 ))
seconds=$((duration % 60))

echo "파이프라인 총 실행 시간: ${hours}시간 ${minutes}분 ${seconds}초" 