from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler # 주가 및 독립적인 데이터를 0에서 1범위로 정규화 / 사용하는 이유 : 딥러닝 모델에서 각 범위안에서 학습할 때 훨씬 안정적이기 때문
from tensorflow.keras.models import Model # Model, Input, Dense는 신경망 구성 요소를 만들 때 사용
from tensorflow.keras.layers import ( # 멀티 헤드 어텐션은 Transformer 모델의 핵심적인 계층
    Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, Add, GlobalAveragePooling1D)
from tensorflow.keras.optimizers import Adam

plt.rc('font', family = 'nanumgothic')

# Transformer Encoder 정의
# 입력 데이터에 대해서 Self-Attention을 적용해 시계열 데이터의 패턴 파악
def transformer_encoder(inputs, num_heads, ff_dim, dropout=0.1):
    # Self-Attention 레이어
    # 입력 데이터 간의 관계를 학습하여 중요한 정보에 가중치 부여
    attention_output = MultiHeadAttention(num_heads = num_heads, key_dim = inputs.shape[-1])(inputs, inputs) # 데이터의 관계를 학습하는 역할
    attention_output = Dropout(dropout)(attention_output) # 과적합 방지
    attention_output = Add()([inputs, attention_output])
    attention_output = LayerNormalization(epsilon=1e-6)(attention_output) # 레이어 정규화로 학습의 안정성 향상

    # 피드포워드 네트워크
    # Attention에서 학습된 특징을 복합적으로 변환하는 과정
    ffn = Dense(ff_dim, activation='relu')(attention_output) # 비선형 변환으로 특징을 추출출
    ffn = Dense(inputs.shape[-1])(ffn)
    ffn_output = Dropout(dropout)(ffn) # 과적합 방지
    ffn_output = Add()([attention_output, ffn_output]) # attension_output과 ffn_output에 대해서 텐서를 요소별로 더함
    ffn_output = LayerNormalization(epsilon=1e-6)(ffn_output) # 레이어 정규화로 학습의 안정성 향상

    # 즉 Attention Output인 원본 특징이 보존되면서, 변환된 특징인 ffn_output을 추가
    return ffn_output

# Transformer Model 정의
# 이중 입력 모델 (주가 데이터, 경제 데이터, 멀티헤드어텐션의 헤드 개수, 피드포워드 차원, 타겟 사이즈)

def build_transformer_with_two_inputs(stock_shape, econ_shape, num_heads, ff_dim, target_size):
    stock_inputs = Input(shape=stock_shape) # Input으로 데이터를 입력받음음
    stock_encoded = stock_inputs
    for _ in range(4): # 4개의 Transforemr Layers
        stock_encoded = transformer_encoder(stock_encoded, num_heads=num_heads, ff_dim=ff_dim)
    stock_encoded = Dense(64, activation='relu')(stock_encoded) # 64차원으로 특징을 압축 및 변환

    econ_inputs = Input(shape=econ_shape)
    econ_encoded = econ_inputs
    for _ in range(4):
        econ_encoded = transformer_encoder(econ_encoded, num_heads=num_heads, ff_dim=ff_dim)
    econ_encoded = Dense(64, activation='relu')(econ_encoded)

    merged = Add()([stock_encoded, econ_encoded]) # 특징을 결합
    merged = Dense(128, activation='relu')(merged) # 128노드의 Dense층으로 특징을 추출
    merged = Dropout(0.2)(merged) # 과적합 방지
    merged = GlobalAveragePooling1D()(merged) # 데이터 전체를 시계열 차원으로 압축
    outputs = Dense(target_size)(merged) # 타겟사이즈 만큼 압축

    return Model(inputs=[stock_inputs, econ_inputs], outputs=outputs)

print("로딩 중..")
file_path = '/content/drive/My Drive/total.csv'
data = pd.read_csv(file_path, parse_dates=['날짜'])
data.sort_values(by='날짜', inplace=True)

data.fillna(method='ffill', inplace=True) # 결측값을 앞의 값으로 채움
data.fillna(method='bfill', inplace=True) # ffill에서 채우지 못한 값이 있으면, 다음 값으로 빈자리를 채움
data = data.apply(pd.to_numeric, errors='coerce') # 데이터프레임의 모든 값을 숫자로 변경, 변환할 수 없는 값은 NaN값
data.dropna(inplace=True) # NaN값 제거

forecast_horizon = 7 # 예측기간

# target_columns = [
#     '애플', '마이크로소프트', '아마존', '구글 A', '구글 C',
#     '메타', '테슬라', '엔비디아', '페이팔', '어도비',
#     '넷플릭스', '컴캐스트', '펩시코', '인텔', '시스코',
#     '브로드컴', '텍사스 인스트루먼트', '퀄컴', '코스트코', '암젠'
# ]

target_columns = [
    'QQQ ETF'
]

economic_features = [
    '10년 기대 인플레이션율',
    '장단기 금리차',
    '기준금리',
    '미시간대 소비자 심리지수',
    '실업률',
    '2년 만기 미국 국채 수익률',
    '10년 만기 미국 국채 수익률',
    '금융스트레스지수',
    '소비자 물가지수',
    '5년 변동금리 모기지',
    '미국 달러 환율',
    '가계 부채 비율',
    'GDP 성장률',
    '나스닥 종합지수',
    'S&P 500 지수',
    '금 가격',
    '달러 인덱스',
    '나스닥 100',
    'S&P 500 ETF',
    # 'QQQ ETF',
    '러셀 2000 ETF',
    '다우 존스 ETF',
    'VIX 지수'
]

train_size = int(len(data) * 0.8)
train_data = data.iloc[:train_size]
test_data = data.iloc[train_size:]

data_scaled = data.copy()
stock_scaler = MinMaxScaler() # 주식 데이터를 0과 1사이로 변환
econ_scaler = MinMaxScaler() # 경제 지표를 0과 1사이로 변환

data_scaled[target_columns] = stock_scaler.fit_transform(data[target_columns]) # 주식 관련 데이터를 0~1 범위로 정규화
data_scaled[economic_features] = econ_scaler.fit_transform(data[economic_features]) # 경제 지표 데이터를 0~1 범위로 정규화

lookback = 90 # 주가를 예측할 때 과거 90일을 기준

X_stock_train = []
X_econ_train = []
y_train = []

# 학습 데이터셋 생성
for i in range(lookback, len(data_scaled) - forecast_horizon): # i번째 샘플을 만들기 위해 90일 만큼 과거 데이터를 들고옴
    X_stock_seq = data_scaled[target_columns].iloc[i - lookback:i].to_numpy() # 주가 데이터에서 90일 만큼 시퀀스를 추출
    X_econ_seq = data_scaled[economic_features].iloc[i - lookback:i].to_numpy() # 경제 데이터에서 90일 만큼 시퀀스를 추출
    y_val = data_scaled[target_columns].iloc[i + forecast_horizon - 1].to_numpy() # 예측할 타겟 값 (7일 만큼의 미래 주가 데이터)
    X_stock_train.append(X_stock_seq)
    X_econ_train.append(X_econ_seq)
    y_train.append(y_val)


# numpy 배열로 변환
X_stock_train = np.array(X_stock_train)
X_econ_train = np.array(X_econ_train)
y_train = np.array(y_train)

X_stock_full = []
X_econ_full = []

# 전체 예측 데이터 생성
for i in range(lookback, len(data_scaled)):
    X_stock_seq = data_scaled[target_columns].iloc[i - lookback:i].to_numpy()
    X_econ_seq = data_scaled[economic_features].iloc[i - lookback:i].to_numpy()
    X_stock_full.append(X_stock_seq)
    X_econ_full.append(X_econ_seq)

X_stock_full = np.array(X_stock_full)
X_econ_full = np.array(X_econ_full)

# 입력 데이터 형태
stock_shape = (lookback, len(target_columns))
econ_shape = (lookback, len(economic_features))

# 모델 생성
model = build_transformer_with_two_inputs(stock_shape, econ_shape, num_heads=8, ff_dim=256, target_size=len(target_columns))
model.compile(optimizer = Adam(learning_rate = 0.0001), loss='mse', metrics = ['mae'])
model.summary()

# 모델 학습
history = model.fit([X_stock_train, X_econ_train], y_train, epochs=50, batch_size=32, verbose = 1)

# 모델 예측
predicted_prices = model.predict([X_stock_full, X_econ_full], verbose=1)
predicted_prices_actual = stock_scaler.inverse_transform(predicted_prices)
pred_len = len(predicted_prices_actual)

# 날짜 데이터 추출
today_dates = data['날짜'].iloc[lookback : lookback + pred_len].values

# 실제 데이터 준비 (오늘 날짜에 해당하는 실제값), 데이터 범위 넘어가면 NaN 처리
actual_data_end = min(lookback + pred_len, len(data))
actual_full = data[target_columns].iloc[lookback:actual_data_end].values

# 만약 actual_full 길이가 pred_len보다 짧다면 부족한 부분을 NaN으로 채움
if actual_full.shape[0] < pred_len:
    nan_padding = np.full((pred_len - actual_full.shape[0], len(target_columns)), np.nan)
    actual_full = np.vstack([actual_full, nan_padding])

# 결과 데이터프레임
result_data = pd.DataFrame({'날짜' : today_dates})
for idx, col in enumerate(target_columns):
    result_data[f'{col}_Predicted'] = predicted_prices_actual[:, idx]
    result_data[f'{col}_Actual'] = actual_full[:, idx]

# 날짜 형식 변환
result_data['날짜'] = pd.to_datetime(result_data['날짜'], errors='coerce')
result_data['날짜'] = result_data['날짜'].dt.strftime('%Y-%m-%d')

# 결과 저장
output_file_path = '/content/drive/My Drive/predicted_stock.csv'
result_data.to_csv(output_file_path, index=False)
print(f'{output_file_path}에 저장 완료')

# 학습 손실 시각화
plt.figure(figsize=(12,6))
plt.plot(history.history['loss'], label = 'Train Loss')
plt.title('학습 손실')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 실제 값과 예측 값 비교
for col in target_columns:
    plt.figure(figsize=(12, 6))
    plt.plot(pd.to_datetime(result_data['날짜']), result_data[f'{col}_Actual'], label='오늘 데이터', alpha=0.7)
    plt.plot(pd.to_datetime(result_data['날짜']), result_data[f'{col}_Predicted'], label='예측된 7일 뒤 데이터', alpha=0.7)
    plt.title(f'{col} - 실제와 예측 데이터 관측')
    plt.xlabel('날짜')
    plt.ylabel('가격')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()
    plt.show()
