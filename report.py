# --------------------------------------------------------------
# 전체 코드: 최종 보고서 생성 + 매수/매도 추천 + 지표 설명
# --------------------------------------------------------------

# Google Drive 마운트
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

######################
# (1) 평가 함수
######################
def evaluate_predictions(data, target_columns, forecast_horizon):
    """
    이 함수는 실제 값과 예측 값을 비교하여 (7일 후 예측 기준),
    MAE, MSE, RMSE, MAPE, Accuracy 등의 다양한 평가 지표를 계산합니다.

    - MAE (평균 절대 오차): 실제 값과 예측 값 간의 평균 절대 오차 (값이 낮을수록 좋음, 원래 데이터 단위와 동일)
    - MSE (평균 제곱 오차): 오차의 제곱 평균 (값이 낮을수록 좋음)
    - RMSE (평균 제곱근 오차): MSE의 제곱근 (값이 낮을수록 좋음, MAE와 함께 사용)
    - MAPE (평균 절대 백분율 오차): 실제 값 대비 오차 비율을 백분율로 표시 (값이 낮을수록 좋음)
    - Accuracy (%): 100 - MAPE로 계산된 단순한 정확도 측정 값
    """

    metrics = []

    for col in target_columns:
        predicted_col = f'{col}_Predicted'
        actual_col = f'{col}_Actual'

        # 데이터에 해당 컬럼이 존재하는지 확인
        if predicted_col not in data.columns or actual_col not in data.columns:
            print(f"{col} 스킵됨: 데이터에 해당 컬럼이 없음")
            continue

        # 예측 값과 실제 값 가져오기
        predicted = data[predicted_col]
        # 실제 값을 예측 기간(7일) 만큼 이동하여 정렬
        actual = data[actual_col].shift(-forecast_horizon)

        # 유효한 (NaN이 아닌) 데이터만 선택
        valid_idx = ~predicted.isna() & ~actual.isna()
        predicted = predicted[valid_idx]
        actual = actual[valid_idx]

        if len(predicted) == 0:
            print(f"{col} 스킵됨: 유효한 예측/실제 값이 없음")
            continue

        # 평가 지표 계산
        mae = mean_absolute_error(actual, predicted)
        mse = mean_squared_error(actual, predicted)
        rmse = mse ** 0.5
        mape = (abs((actual - predicted) / actual).mean()) * 100
        accuracy = 100 - mape

        metrics.append({
            '종목': col,
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE (%)': mape,
            'Accuracy (%)': accuracy
        })

    return pd.DataFrame(metrics)

###############################
# (2) 미래 상승 분석
###############################
def analyze_rise_predictions(data, target_columns):
    """
    이 함수는 데이터프레임의 마지막 행(가장 최근 날짜)을 기준으로
    실제 값과 예측 값을 비교하여 상승/하락 여부와 상승 확률(%)을 계산합니다.
    """

    last_row = data.iloc[-1]
    results = []

    for col in target_columns:
        last_actual_price = last_row.get(f'{col}_Actual', np.nan)
        predicted_future_price = last_row.get(f'{col}_Predicted', np.nan)

        # 상승/하락 여부 및 상승 확률 계산
        if pd.notna(last_actual_price) and pd.notna(predicted_future_price):
            predicted_rise = predicted_future_price > last_actual_price
            rise_probability = ((predicted_future_price - last_actual_price) / last_actual_price) * 100
        else:
            predicted_rise = np.nan
            rise_probability = np.nan

        results.append({
            '종목': col,
            '최근 실제 가격': last_actual_price,
            '예측된 미래 가격': predicted_future_price,
            '예측 상승 여부': predicted_rise,
            '상승 확률 (%)': rise_probability
        })

    return pd.DataFrame(results)

#######################################
# (3) 매수/매도 추천 및 분석
#######################################
def generate_recommendation(row):
    """
    추천 로직:
    - (예측 상승 == True) 그리고 (상승 확률 > 0) => 매수
    - (상승 확률 > 2) => 강력 매수
    - 그 외에는 매도
    """
    rise_prob = row.get('상승 확률 (%)', 0)
    predicted_rise = row.get('예측 상승 여부', False)

    if pd.isna(rise_prob) or pd.isna(predicted_rise):
        return "데이터 없음"

    if predicted_rise and rise_prob > 0:
        if rise_prob > 2:
            return "강력 매수"
        else:
            return "매수"
    else:
        return "매도"

def generate_analysis(row):
    """
    각 종목에 대한 간단한 분석 코멘트를 생성
    """
    stock_name = row['종목']
    rise_prob = row.get('상승 확률 (%)', 0)
    predicted_rise = row.get('예측 상승 여부', False)

    if pd.isna(rise_prob) or pd.isna(predicted_rise):
        return f"{stock_name}: 데이터 부족"

    if predicted_rise:
        return f"{stock_name} 종목이 약 {rise_prob:.2f}% 상승할 것으로 예상됩니다. 매수 또는 보유 고려."
    else:
        return f"{stock_name} 종목이 약 {-rise_prob:.2f}% 하락할 것으로 예상됩니다. 신중한 접근 필요."

#######################
# (4) 메인 코드 실행
#######################
# 파일 경로 설정
predicted_file_path = '/content/drive/My Drive/predicted_stock.csv'

# 1) 데이터 불러오기
data = pd.read_csv(predicted_file_path, parse_dates=['날짜'])

# 2) 타겟 종목 설정
target_columns = ['QQQ ETF']
forecast_horizon = 7  # 7일 후 예측

# 3) 예측 평가
evaluation_results = evaluate_predictions(data, target_columns, forecast_horizon)
print("============ 예측 평가 결과 ============")
print(evaluation_results)

# 4) 미래 상승 분석
rise_results = analyze_rise_predictions(data, target_columns)
print("============ 상승 예측 결과 ============")
print(rise_results)

# 5) 결과 병합 및 정렬
final_results = pd.merge(evaluation_results, rise_results, on='종목', how='outer')
final_results = final_results.sort_values(by='상승 확률 (%)', ascending=False)

# 6) 매수/매도 추천 및 분석 생성
final_results['추천'] = final_results.apply(generate_recommendation, axis=1)
final_results['분석'] = final_results.apply(generate_analysis, axis=1)

# 7) 결과 저장 및 출력
final_output_path = '/content/drive/My Drive/final_stock_analysis.csv'
final_results.to_csv(final_output_path, index=False)
print(f"\n최종 분석 결과 저장됨: {final_output_path}\n")
print("=============== 최종 보고서 ===============")
print(final_results.to_string(index=False))


# 8) Save final results to CSV
final_output_path = '/content/drive/My Drive/final_stock_analysis.csv'
final_results.to_csv(final_output_path, index=False)
print(f"\nFinal combined results saved to {final_output_path}\n")

# 9) Print final report
print("=============== Final Report ===============")
print(final_results.to_string(index=False))




# 타겟 컬럼들
# target_columns = [
#     '애플', '마이크로소프트', '아마존', '구글 A', '구글 C',
#     '메타', '테슬라', '엔비디아', '페이팔', '어도비',
#     '넷플릭스', '컴캐스트', '펩시코', '인텔', '시스코',
#     '브로드컴', '텍사스 인스트루먼트', '퀄컴', '코스트코', '암젠'
# ]
