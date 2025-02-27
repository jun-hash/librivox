import pandas as pd
import numpy as np

# CSV 파일 읽기
df = pd.read_csv('data/raw_audio_metadata.csv')

# 총 화자 수 (unique speaker_id)
total_speakers = df['speaker_id'].nunique()

# 화자당 playtime 통계 (초 -> 시간 변환)
speaker_playtime = df.groupby('speaker_id')['playtime'].sum() / 3600  # 시간으로 변환

# 통계 계산
total_time = df['playtime'].sum() / 3600  # 시간으로 변환
total_segments = len(df)
median_per_speaker = speaker_playtime.median()
std_per_speaker = speaker_playtime.std()
mean_per_speaker = speaker_playtime.mean()

print(f"=== balanced_final.csv 분석 결과 ===")
print(f"총 화자 수: {total_speakers}명")
print(f"\n화자당 playtime 통계 (시간):")
print(f"- 평균: {mean_per_speaker:.2f}시간")
print(f"- 중앙값: {median_per_speaker:.2f}시간")
print(f"- 표준편차: {std_per_speaker:.2f}시간")
print(f"\n전체 데이터:")
print(f"- 총 시간: {total_time:.2f}시간")
print(f"- 총 세그먼트 수: {total_segments}개") 