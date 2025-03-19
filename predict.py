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
    merged = Dropout(0.2) # 과적합 방지
    merged = GlobalAveragePooling1D()(merged) # 데이터 전체를 시계열 차원으로 압축
    outputs = Dense(target_size)(merged) # 타겟사이즈 만큼 압축
    
    return Model(inputs=[stock_inputs, econ_inputs], outputs=outputs)

print("로딩 중..")
file_path = '/content/drive/My drive/total.csv'
data = pd.read_csv(file_path, parse_dates=['날짜']) # 날짜열을 데이터 타입으로 파싱
data.sort_values(by='날짜', inplace=True) # 날짜열을 오름차순

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

x_stock_train = []
x_econ_train = []
y_train = []

# 학습 데이터셋 생성성
for i in range(lookback, len(data_scaled) - forecast_horizon): # i번째 샘플을 만들기 위해 90일 만큼 과거 데이터를 들고옴
    x_stock_seq = data_scaled[target_columns].iloc[i - lookback:i].to_numpy() # 주가 데이터에서 90일 만큼 시퀀스를 추출
    x_econ_seq = data_scaled[economic_features].iloc[i - lookback:i].to_numpy() # 경제 데이터에서 90일 만큼 시퀀ㅅ느를 추출 
    y_val = data_scaled[target_columns].iloc[i + forecast_horizon - 1].to_numpy() # 예측할 타겟 값 (7일 만큼의 미래 주가 데이터)
    x_stock_train.append(x_stock_seq)
    x_econ_train.append(x_econ_seq)
    y_train.append(y_val)

x_stock_train = np.array(x_stock_train) # 시퀀스들을 numpy로 변환하여 모델에서 인자에 넣기 쉽도록 하기 위함
x_econ_train = np.array(x_econ_train)
y_train = np.array(y_train)

# 전체 예측 데이터 생성

x_stock_full = []
x_econ_full = []

for i in range(lookback, len([data_scaled])):
    x_stock_seq = data_scaled[target_columns].iloc[i - lookback:i].to_numpy()
    x_econ_seq = data_scaled[economic_features].iloc[i - lookback:i].to_numpy()
    x_stock_full.append(x_stock_seq)
    x_econ_full.append(x_econ_seq)
    
x_stock_full = np.array(x_stock_full)
x_econ_full = np.array(x_econ_full)

# 모델 학습

stock_shape = (lookback, len(target_columns))
econ_shape = (lookback, len(economic_features))

model = build_transformer_with_two_inputs(stock_shape, econ_shape, num_heads=8, ff_dim=256, target_size=len(target_columns))
model.compile(optimizer = Adam(learning_rate = 0.0001), loss='mse', metrics = ['mae'])
model.summary()
