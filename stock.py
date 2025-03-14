import requests
import pandas as pd
import yfinance as yf
from datetime import datetime

# FRED API KEY 
api_key = '017f90d099b9597068064fd94a386378'

# FRED에서 제공하는 지표 코드와 명칭
fred_indicators = {
    'T10YIE': '10년 기대 인플레이션율',  # 10년 만기 기대 인플레이션율 (일간): 10년 만기 국채의 물가연동국채(TIPS)와 명목 국채 간 수익률 차이를 통해 시장이 기대하는 평균 인플레이션율을 나타냅니다.
    'T10Y2Y': '장단기 금리차',  # 10년-2년 국채 수익률 스프레드 (일간): 장기 금리와 단기 금리 간의 차이로, 경제 성장 가능성과 경기 침체 신호를 제공합니다.
    'FEDFUNDS': '기준금리',  # 연방기금 금리 (월간): 연방준비제도가 설정한 단기 금리로, 소비와 투자, 경제 활동 전반에 영향을 미칩니다.
    'UMCSENT': '미시간대 소비자 심리지수',  # 소비자 신뢰 지수 (월간): 소비자들이 경제를 얼마나 낙관적으로 바라보는지를 측정하는 지표로, 소비 지출의 선행 지표 역할을 합니다.
    'UNRATE': '실업률',  # 실업률 (월간): 노동 시장의 상태를 나타내는 핵심 지표로, 경제 성장 및 경기 둔화의 주요 신호로 활용됩니다.
    'USREC': '경기침체',  # 경기침체 지수 (월간): 미국 경제가 경기 침체 상태에 있는지를 나타냅니다. 일반적으로 장단기 금리차(T10Y2Y)를 통해도 간접적으로 파악 가능합니다.
    'DGS2': '2년 만기 미국 국채 수익률',  # 2년 만기 국채 수익률 (일간): 단기 국채 수익률로, 시장이 예측하는 단기 금리 동향과 통화 정책 방향을 반영합니다.
    'DGS10': '10년 만기 미국 국채 수익률',  # 10년 만기 국채 수익률 (일간): 장기 국채 수익률로, 경제의 장기 성장성과 인플레이션 기대를 반영합니다.

    # 추가 지표
    'STLFSI4': '금융스트레스지수',  # 금융 스트레스 지수 (주간): 금융 시스템에서 발생하는 스트레스(위험 수준)를 측정하며, 금융 위기의 가능성을 조기에 감지하는 데 사용됩니다.
    'PCE': '개인 소비 지출',  # 개인 소비 지출 (월간): 가계가 상품과 서비스에 지출한 금액을 측정하며, 경제 활동과 물가 상승에 대한 주요 지표로 활용됩니다.
    # 'INDPRO': '산업생산',  # 산업 생산 지수 (월간): 제조업, 광업, 유틸리티에서의 생산 수준을 측정하여 경제 활동을 파악합니다.
    # 'HOUST': '주택 착공',  # 신규 주택 착공 건수 (월간): 주택 시장 활동의 선행 지표로, 경제 성장과 소비 심리를 나타냅니다.
    # 'UNEMPLOY': '실업자수',  # 실업자의 총 수 (월간): 노동 시장 내 실업 상태에 있는 총 인구 수를 나타내며, 경제 활동의 강도를 평가합니다.
    # 'RSAFS': '소매판매',  # 소매판매 지수 (월간): 소비자들이 상품과 서비스에 지출한 금액을 측정하여 소비 동향을 나타냅니다.
    'CPIENGSL': '에너지 가격 지수',  # 소비자 물가지수 중 에너지 부문 (월간): 에너지 가격의 변동을 측정하며, 소비자 물가지수(CPI) 중 특정 부문을 나타냅니다.
    'AHETPI': '임금 성장률',  # 시간당 평균 임금 성장률 (월간): 노동 시장 내 임금 상승 속도를 측정하며, 소비자 지출 및 물가 상승에 영향을 미칩니다.
    # 'PPIACO': '농산물 가격 지수',  # 생산자 물가지수 중 농산물 부문 (월간): 농산물 부문의 가격 변동을 나타내며, 생산자 물가지수(PPI)의 일부로 인플레이션을 평가하는 데 사용됩니다.
    'CPIAUCSL': '소비자 물가지수',  # 전체 소비자 물가지수 (월간): 일반 소비자가 구매하는 상품과 서비스의 평균 가격 변동을 나타내며, 인플레이션의 대표적인 지표입니다.
    'CSUSHPINSA': '주택가격지수',  # 케이스-실러 주택 가격 지수 (월간): 미국 주택 시장에서 평균 주택 가격 변동을 측정하여 부동산 시장 상태를 평가합니다.
    # 'MORTGAGE30US': '30년 고정금리 모기지',  # 30년 만기 고정금리 모기지 금리 (주간): 30년 고정금리 주택 대출 금리를 나타내며, 주택 시장과 가계 부채 상태를 반영합니다.
    # 'MORTGAGE15US': '15년 고정금리 모기지',  # 15년 만기 고정금리 모기지 금리 (주간): 15년 만기 주택 대출 금리를 나타내며, 주택 구매력을 평가하는 데 사용됩니다.
    'MORTGAGE5US': '5년 변동금리 모기지',  # 5년 변동금리 모기지 금리 (주간): 5년 변동금리 대출 금리를 나타내며, 주택 시장의 단기 대출 동향을 반영합니다.
    'DTWEXM': '미국 달러 환율',  # 미국 무역가중 환율 (월간): 미국 달러의 국제 시장에서 상대적 가치를 나타내며, 무역과 수출입 활동에 영향을 미칩니다.
    'M2': '통화 공급량 M2',  # M2 통화 공급량 (주간): 유통되는 통화량(현금, 요구불 예금 등)을 측정하여 경제 내 유동성을 나타냅니다.
    # 'TEDRATE': 'TED 스프레드',  # 3개월 만기 미국 국채와 유로달러 금리 스프레드 (일간): 금융 시장의 신용 위험을 반영하며, 유동성과 신뢰도를 평가하는 데 사용됩니다.
    # 'BAMLH0A0HYM2': '미국 하이일드 채권 스프레드',  # 미국 하이일드 채권과 국채 스프레드 (일간): 투자자들이 고위험 자산에 대해 요구하는 추가 수익률을 나타냅니다.
    # 'BAMLC0A0CM': '미국 회사채 스프레드',  # 미국 회사채와 국채 스프레드 (일간): 회사채와 국채 간 금리 차이를 측정하여 신용 위험을 평가합니다.
    # 'BAMLCC0A0CMTRIV': '미국 회사채 수익률',  # 미국 회사채 수익률 (일간): 회사채에서 발생하는 수익률을 나타내며, 기업의 신용 상태를 평가하는 데 유용합니다.
		# 'BAMLCC0A1AAATRIV': '미국 회사채 AAA등급 수익률',  # AAA등급 회사채 수익률 (일간)
    # 'BAMLCC0A4BBBTRIV': '미국 회사채 BBB등급 수익률',  # BBB등급 회사채 수익률 (일간)
    # 'BAMLHYH0A0HYM2TRIV': '미국 하이일드 채권 수익률',  # 하이일드 채권 수익률 (일간)
    # 'BAMLHYH0A3CMTRIV': '미국 하이일드 채권 CCC등급 수익률',  # CCC등급 하이일드 채권 수익률 (일간)
    # 'BAMLHE00EHYIEY': '미국 하이일드 채권 기대수익률',  # 하이일드 채권 기대수익률 (일간)
    
    'TDSP': '가계 부채 비율',  # 가계의 부채 상환 비율을 나타냄 (분기): 가계의 소득 대비 부채 상환 부담을 측정하여 가계의 재무 건전성을 평가합니다.
    # 'A939RX0Q048SBEA': '실질 GDP 성장률',  # 계절 조정된 연간 실질 GDP 성장률 (분기): 물가 변동을 반영하지 않은 국내총생산(GDP) 증가율로, 경제 성장 상태를 나타냅니다.
    'GDPC1': 'GDP 성장률',  # 실질 국내총생산 성장률, 물가 조정을 반영 (분기)
    # 'W019RCQ027SBEA': '정부 지출',  # 정부의 총 지출 금액 (분기)
    # 'DRBLACBS': '대출 연체율',  # 기업 대출의 연체율 (분기): 기업 대출의 연체 비율을 나타내며, 금융 시장의 신용 위험을 평가합니다.

    # 주식시장 관련 추가 지표
    'DJIA': '다우존스 산업평균지수',  # 미국 대형 30개 기업의 주가 평균 (일간): 미국 주식 시장의 전반적인 흐름을 반영하며, 대형주의 안정성을 평가하는 지표입니다.
    'NASDAQCOM': '나스닥 종합지수'  # 나스닥 시장 전체 종합 주가 지수 (일간): 기술주 중심의 주식 시장 동향을 나타냅니다.
}

yfinance_indicators = {
    'S&P 500 지수': '^GSPC',    # S&P 500 지수: 미국 주식 시장의 대형주 500개로 구성된 지수로, 전체 시장의 대표 지표입니다.
    '금 가격': 'GC=F',           # 금 가격 (선물): 시장의 안전자산 선호도를 나타내며, 인플레이션 및 경제 불확실성에 민감합니다.
    '달러 인덱스': 'DX-Y.NYB',    # 달러 인덱스: 미국 달러의 글로벌 통화 대비 상대적 가치를 나타냅니다.

    # 추가 지표
    '나스닥 100': '^NDX',           # 나스닥 100 지수: 나스닥 상장 대형 기술주 100개를 포함하며, 기술주 중심의 시장 흐름을 반영합니다.
    'S&P 500 ETF': 'SPY',           # S&P 500 추종 ETF: S&P 500 지수를 추종하는 상장지수펀드로, 시장 전체의 동향을 추적합니다.
    'QQQ ETF': 'QQQ',               # 나스닥 100 추종 ETF: 기술주 중심의 나스닥 100 지수를 추종하는 ETF입니다.
    '러셀 2000 ETF': 'IWM',         # 러셀 2000 추종 ETF: 중소형주로 구성된 러셀 2000 지수를 추종하며, 경제 성장성을 반영합니다.
    '다우 존스 ETF': 'DIA',         # 다우 존스 추종 ETF: 다우존스 산업평균지수를 추종하며, 안정적인 대형주의 흐름을 나타냅니다.
    'VIX 지수': '^VIX'              # ^VIX (변동성 지수, 공포 지수): 시장의 변동성 기대치를 반영하며, 투자 심리를 측정하는 지표입니다.
}

# 나스닥 100 상위 종목 티커 리스트와 한글 이름
nasdaq_top_100 = [
    ("AAPL", "애플"), ("MSFT", "마이크로소프트"), ("AMZN", "아마존"), ("GOOGL", "구글 A"),
    ("GOOG", "구글 C"), ("META", "메타"), ("TSLA", "테슬라"), ("NVDA", "엔비디아"), ("PYPL", "페이팔"),
    ("ADBE", "어도비"), ("NFLX", "넷플릭스")

    , ("CMCSA", "컴캐스트"), ("PEP", "펩시코"),
    ("INTC", "인텔"), ("CSCO", "시스코"), ("AVGO", "브로드컴"), ("TXN", "텍사스 인스트루먼트"),
    ("QCOM", "퀄컴"), ("COST", "코스트코"), ("AMGN", "암젠")

    # , ("CHTR", "차터 커뮤니케이션"),
    # ("SBUX", "스타벅스"), ("AMD", "AMD")
    # , ("MDLZ", "몬델리즈"), ("INTU", "인트윗"),
    # ("ISRG", "인튜이티브 서지컬"), ("BKNG", "부킹홀딩스"), ("ADP", "ADP"),
    # ("VRTX", "버텍스"), ("MU", "마이크론"), ("AMAT", "어플라이드 머티리얼즈"), ("REGN", "리제네론"),
    # ("LRCX", "램 리서치"), ("KDP", "케우리그 닥터페퍼"), ("FISV", "피서브"), ("CSX", "CSX"),
    # ("GILD", "길리어드 사이언스"), ("MELI", "메르카도 리브레"), ("SNPS", "시놉시스"),
    # ("EA", "일렉트로닉 아츠")
    #
    # , ("KLAC", "KLA"), ("ADSK", "오토데스크"), ("CTAS", "신타스"),
    # ("XEL", "엑셀 에너지"), ("PANW", "팔로알토 네트웍스"), ("ANSS", "앤시스"), ("TEAM", "아틀라시안"),
    # ("WDAY", "워크데이"), ("ILMN", "일루미나"), ("DOCU", "도큐사인"),
    # ("MRNA", "모더나"), ("IDXX", "아이덱스"), ("ZM", "줌 비디오"), ("DXCM", "덱스컴"),
    # ("ROST", "로스 스토어스"), ("CRWD", "크라우드스트라이크"), ("MAR", "메리어트"),
    # ("EXC", "엑셀론"), ("MNST", "몬스터 비버리지"), ("PCAR", "PACCAR"), ("LCID", "루시드 모터스"),
    # ("ALGN", "얼라인 테크놀로지"), ("BIIB", "바이오젠"),
    # ("MTCH", "매치 그룹"), ("OKTA", "옥타"), ("BKR", "베이커 휴즈"), ("ZS", "지스케일러"),
    # ("CDNS", "케이던스"), ("CPRT", "코파트"), ("FAST", "패스트널"), ("AEP", "아메리칸 일렉트릭"),
    # ("ORLY", "오라일리"), ("VRSK", "버리스크"), ("CTSH", "코그니전트"), ("PDD", "핀둬둬"),
    # ("CHKP", "체크포인트"), ("JD", "징둥"), ("NTES", "넷이즈"), ("KHC", "크래프트 하인즈"),
    # ("DLTR", "달러 트리"), ("EPAM", "EPAM 시스템즈"), ("SWKS", "스카이웍스"),
    # ("NXPI", "NXP 반도체"), ("TTD", "트레이드 데스크"),
    # ("PAYX", "페이첵스"), ("BIDU", "바이두"), ("WDC", "웨스턴 디지털"), ("TRMB", "트림블"),
    # ("FTNT", "포티넷"), ("VRSN", "베리사인"), ("ASML", "ASML 홀딩"), ("BMRN", "바이오마린"),
    # ("LULU", "룰루레몬"), ("EBAY", "이베이"), ("CEG", "컨스텔레이션 에너지"), ("RIVN", "리비안")
]

start_date = '2016-01-01'
end_date = datetime.today().strftime('%Y-%m-%d')

fred_data_frames = []
for code, name in fred_indicators.items():
    # 지표별 제공 주기에 따른 요청 주기를 설정
    if code in ['FEDFUNDS', 'UMCSENT', 'USREC', 'PCE', 'INDPRO', 'HOUST', 'UNNEMPLOY',
    'RSAFS', 'CPIENGSL', 'AHETPI', 'PPIACO', 'CPIAUCSL', 'CSUSHPINSA', 'DTWEXM', 'UNRATE']:
        frequency = 'm' # 월간
    elif code in ['STLFSI4', 'M2', 'MORTGAGE15US', 'MORTGAGE5US']:
        frequency = 'w' # 주간
    elif code in ['TDSP', 'A939RX0Q04BSBEA', 'GDPC1', 'W019RCQ027SBEA', 'DRBLACBS']:
        frequency = 'q' # 분기
    else:
        frequency = 'd' # 일간

    url = f'https://api.stlouisfed.org/fred/series/observations'
    params = {
        'series_id' : code,
        'api_key' : api_key,
        'file_type' : 'json',
        'observation_start' : start_date,
        'observation_end' : end_date,
        'frequency' : frequency
    }

    response = requests.get(url, params=params)

    # 데이터를 DateFrame 으로 변환 후 날짜를 인덱스로 설정하여 리스트에 추가하는 코드
    if response.status_code == 200:
        data = response.json().get('observations', [])
        if data:
            df = pd.DataFrame(data)[['date', 'value']]
            df.columns = ['date', name]
            df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None) # 타임존이 포함된 날짜 데이터는 문제가 발생
            fred_data_frames.append(df.set_index('date')) # date를 인덱스 데이터로 추가
        else:
            print(f"No data {name} {(code)}")
    else:
            print(f"fail {name} {(code)} : {response.status_code}")

# 데이터 빈도에 따른 리샘플링
for i, df in enumerate(fred_data_frames):
    if df.empty:
        print(f"DataFrame {i} is Empty")
        continue

    try:
        inferred_freq = df.index.inferred_freq
        if inferred_freq in ['M', 'MS']: # 월간
            fred_data_frames[i] = df.resample('D').ffill()
        if inferred_freq in ['W', 'W-FRI']: # 주간
            fred_data_frames[i] = df.resample('D').ffill()
        if inferred_freq in ['Q', 'Q5-QCT']: # 분기
            fred_data_frames[i] = df.resample('D').ffill()
        if inferred_freq in ['B']: # 일간
            fred_data_frames[i] = df.resample('D').ffill()
        else:
            fred_data_frames[i] = df.resample('D').ffill()
    except Exception as e:
        print(f"Error {i} : {e}")

# CSV 파일 형식으로 저장
if fred_data_frames:
    merged_df = pd.concat(fred_data_frames, axis=1) # 데이터 배열들을 concat으로 통합
    merged_df.to_csv('fred_data.csv', index=True, encoding='utf-8-sig')
    print("저장 완료")
else:
    print("저장 할 데이터 없음")