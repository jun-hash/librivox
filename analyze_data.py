import pandas as pd
import numpy as np

# 데이터 읽기
df = pd.read_csv('final.csv')

def analyze_and_suggest_balancing(df, target_hours=31000):
    speaker_times = df.groupby('speaker_id')['playtime'].sum()
    
    # 현재 총 시간
    current_total_hours = speaker_times.sum() / 3600
    
    # 화자당 최대 발화 시간 설정 (예: 99퍼센타일 값으로)
    max_time_per_speaker = np.percentile(speaker_times, 90)
    
    # 화자당 최소 발화 시간 설정 (예: 25퍼센타일 값으로)
    min_time_per_speaker = np.percentile(speaker_times, 25)
    
    # 조정된 발화 시간 계산
    balanced_times = speaker_times.clip(lower=min_time_per_speaker, upper=max_time_per_speaker)
    balanced_total_hours = balanced_times.sum() / 3600
    
    print("현재 상태:")
    print(f"- 총 시간: {current_total_hours:.2f} 시간")
    print(f"- 화자당 평균: {speaker_times.mean()/3600:.2f} 시간")
    print(f"- 화자당 중앙값: {speaker_times.median()/3600:.2f} 시간")
    
    print("\n균형 조정 후:")
    print(f"- 총 시간: {balanced_total_hours:.2f} 시간")
    print(f"- 화자당 평균: {balanced_times.mean()/3600:.2f} 시간")
    print(f"- 화자당 중앙값: {balanced_times.median()/3600:.2f} 시간")
    
    # 31000시간에 맞추기 위한 스케일링 팩터 계산
    scaling_factor = target_hours / balanced_total_hours
    
    print(f"\n제안된 처리 방법:")
    print(f"1. 화자당 최소 발화 시간: {min_time_per_speaker/3600:.2f} 시간")
    print(f"2. 화자당 최대 발화 시간: {max_time_per_speaker/3600:.2f} 시간")
    print(f"3. 전체 데이터 스케일링 팩터: {scaling_factor:.2f}")
    
    return balanced_times, min_time_per_speaker, max_time_per_speaker

balanced_times, min_time, max_time = analyze_and_suggest_balancing(df)

# 실제 데이터 처리를 위한 함수
def balance_dataset(df, min_time, max_time, target_hours=31000):
    """
    데이터셋을 실제로 처리하는 함수
    """
    # 화자별 총 발화시간 계산
    speaker_times = df.groupby('speaker_id')['playtime'].sum()
    
    # 처리가 필요한 화자 식별
    speakers_to_reduce = speaker_times[speaker_times > max_time].index
    speakers_to_increase = speaker_times[speaker_times < min_time].index
    
    # 결과 데이터프레임 준비
    balanced_df = df.copy()
    
    # 발화시간이 너무 긴 화자의 데이터 처리
    for speaker in speakers_to_reduce:
        speaker_data = balanced_df[balanced_df['speaker_id'] == speaker]
        reduction_factor = max_time / speaker_times[speaker]
        # 각 발화 구간의 시간을 비례적으로 줄임
        balanced_df.loc[speaker_data.index, 'playtime'] *= reduction_factor
    
    # 발화시간이 너무 짧은 화자의 데이터 처리
    for speaker in speakers_to_increase:
        speaker_data = balanced_df[balanced_df['speaker_id'] == speaker]
        increase_factor = min_time / speaker_times[speaker]
        # 각 발화 구간의 시간을 비례적으로 늘림
        balanced_df.loc[speaker_data.index, 'playtime'] *= increase_factor
    
    # 전체 시간을 target_hours에 맞추기 위한 스케일링
    current_total = balanced_df['playtime'].sum() / 3600
    final_scaling = target_hours / current_total
    balanced_df['playtime'] *= final_scaling
    
    return balanced_df

# balanced_df = balance_dataset(df, min_time, max_time)
# balanced_df.to_csv('balanced_final.csv', index=False)

def sample_speakers(df, random_seed=42):
    """
    각 화자당 하나의 발화를 무작위로 샘플링
    """
    # 재현성을 위한 시드 설정
    np.random.seed(random_seed)
    
    # 각 speaker_id 그룹에서 하나의 행을 무작위로 선택
    sampled_df = df.groupby('speaker_id').apply(
        lambda x: x.iloc[np.random.randint(len(x))]
    ).reset_index(drop=True)
    
    # 필요한 컬럼만 선택 (예시, 실제 컬럼명에 맞게 조정 필요)
    columns_to_keep = ['speaker_id', 'playtime', 'audio_link']
    sampled_df = sampled_df[columns_to_keep]
    
    # 발화 시간 순으로 정렬
    sampled_df = sampled_df.sort_values('playtime', ascending=False)
    
    print(f"총 {len(sampled_df)} 개의 샘플이 추출되었습니다.")
    print(f"총 재생 시간: {sampled_df['playtime'].sum() / 3600:.2f} 시간")
    
    return sampled_df

# 샘플링 실행
sampled_df = sample_speakers(df)

# CSV 파일로 저장
output_path = 'speaker_samples.csv'
sampled_df.to_csv(output_path, index=False)
print(f"\n결과가 {output_path}에 저장되었습니다.") 