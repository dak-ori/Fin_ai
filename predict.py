from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.preprocessing import MinMaxScaler # 주가 및 독립적인 데이터를 0에서 1범위로 정규화
from tensorflow.keras.models import Model # Model, Input, Dense는 신경망 구성 요소를 만들 때 사용
from tensorflow.keras.layers import ( # 멀티 헤드 어텐션은 Transformer 모델의 핵심적인 계층
    Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, Add, GlobalAveragePooling1D)

import tensorflow as tf
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