# 모델 설명

## 모델 요약 : Transformer를 활용하여, 시계열 데이터를 동시에 처리 하는 구조

    - 인풋이 2개이며, 각 입력에 대해 4개의 Transformer Encoder를 사용
    - GlobalAveragePooling1D를 사용하여 최종적으로 1개 크기의 출력을 내는 형태

## 1. 입력 레이어

```
input_layer (None, 90, 1)
```

    - 90은 lookback(시계열 길이), 1은 특성 개수

## 2. Multi-Head Attention 레이어

```
multi_head_attention (None, 90, 1)
```

    - 모델의 구조가 주가, 경제 지표 각각에 대해 사용
    - 각 블록마다 MultiHeadAttention 이 1개씩 등장하는 구조

## 3. Dropout -> add -> LayerNormalization

    1. Dropout : 과적합 방지를 위해 일부 뉴럴을 0으로 만드는 기법
    2. add : Self-Attention 결과와 원본 입력을 더함
    3. LayerNormalization : 각 시점 내 채녈 방향으로 평균을 정규화

## 4. Feed-Forward Network : Dense -> Dense

```
dense (None, 90, 256)
```

    - dense : ff_dim이 256이라서, 채널 수 x 256 만큼 생성
    - 파라미터가 512인 이유는, 채널 수(1) x 256 + bias

## 5. 동일한 패턴이 4번 반복

    - Multi-Head Attention + FFN 구조를 4층 구조로 설계했기 때문

## 6. 최종 병합 후 Dense

    1. add_16 (None, 90, 64) : 마지막 출력을 단순히 더하는 방식
    2. Dense_18 (None, 90, 128) : 각각의 차원은 64차원이므로 더하면 128차원
    3. dropout (None, 90, 128) : 과적합 방지
    4. global_average_polling1d (None, 128) : 시계열 길이를 압축하는 단계
    6. Dense_19 (None, 1) : 마지막 출력, 예측 해야하는 타깃이 1개

# 정리

    - 모델 입력 : (주가, 경제지표 90일) -> 각각 4번씩 Transformer 인코딩
    - 중간 단계 :  Multi-Head Attention, Dropout, add, LayerNorm, FFN이 반복
    - 병합 : 인코딩 결과를 합쳐 Dense로 변환 후 Globalaveragepolling1d로 시계열 길이를 평균
    - 출력 : Dense(1) 로 일주일 주가 1개 예측
