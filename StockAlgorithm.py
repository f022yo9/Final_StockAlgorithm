#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 19:58:20 2023

@author: f___yo_
"""

# -*- coding: utf-8 -*-
##############################################################################################################################################################

# 주가 시계열 분석

# 팀이름: GBT

# 작성자: 최규진, 심태율, 김홍표, 유진권

##############################################################################################################################################################

#라이브러리 임포트
import os
import random
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import FinanceDataReader as fdr
import pmdarima as pm
from pmdarima.arima import ndiffs

warnings.filterwarnings("ignore")

##############################################################################################################################################################

#시드 세팅

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(42) # Seed 고정

##############################################################################################################################################################

# 트레인셋 불러오기

file_path = '/Users/f___yo_/github/StockAlgorithm/data/'
train = pd.read_csv(file_path + 'train.csv')

# 추론 결과를 저장하기 위한 dataframe 생성
results_df = pd.DataFrame(columns=['종목코드', 'final_return', 'accuracy'])

# train 데이터에 존재하는 독립적인 종목코드 추출
unique_codes = train['종목코드'].unique()

##############################################################################################################################################################

# 날짜 설정
start_date = datetime.strptime(str(train['일자'].min()), '%Y%m%d')
end_date = datetime.strptime(str(train['일자'].max()), '%Y%m%d')

# KOSPI 지수, USD/KRW 환율, 비트코인 데이터 가져오기
df_kospi = fdr.DataReader('KS11', start_date, end_date)
df_kosdaq = fdr.DataReader('KQ11', start_date, end_date)
df_USD = fdr.DataReader('USD/KRW', start_date, end_date)

df_3years = pd.read_csv(file_path + '3years.csv')
df_3years['날짜'] = pd.to_datetime(df_3years['날짜'])
df_3years_sorted = df_3years[['날짜', '값']].sort_values('날짜')
df_3years_sorted.set_index('날짜', inplace=True)

###############################################################################################################################

# 각 종목코드에 대해서 모델 학습 및 추론 반복

for code in tqdm(unique_codes):
    # 학습 데이터 생성
    train_close = train[train['종목코드'] == code][['일자', '종가', '시가', '거래량']]  # Include '시가' column in the DataFrame
    train_close['일자'] = pd.to_datetime(train_close['일자'], format='%Y%m%d')
    train_close.set_index('일자', inplace=True)
   
    # Create the new column '변화량' ('change') using subtraction
    train_close['변화량'] = train_close['종가'] - train_close['시가']  
    tc = train_close['종가']
    change = train_close['변화량']

    # KOSPI 지수 데이터 가져오기
    kospi_data = df_kospi[['Adj Close']].copy()
    kospi_data.rename(columns={'Adj Close': 'KOSPI'}, inplace=True)

    #KOSDAQ 지수 데이터 가져오기
    kosdaq_data = df_kosdaq[['Adj Close']].copy()
    kosdaq_data.rename(columns={'Adj Close': 'KOSDAQ'}, inplace=True)
   
    # USD/KRW 환율 데이터 가져오기
    usd_data = df_USD[['Adj Close']].copy()
    usd_data.rename(columns={'Adj Close': 'USD'}, inplace=True)

    # 국고채금리(3년) 데이터 가져오기    
    years_data = df_3years_sorted[['값']].copy()
    years_data.rename(columns={'값': '3YEARS'}, inplace=True)

    # 종목 데이터와 KOSPI, USD, 비트코인, 거래량, OIL, GOLD 데이터 병합
    merged_data = pd.merge(train_close, kospi_data, left_index=True, right_index=True, how='outer')
    merged_data = pd.merge(merged_data, kosdaq_data, left_index=True, right_index=True, how='outer')
    merged_data = pd.merge(merged_data, usd_data, left_index=True, right_index=True, how='outer')
    merged_data = pd.merge(merged_data, years_data, left_index=True, right_index=True, how='outer')
     
    # 트레인 데이터의 행을 인덱스로 설정하고 나머지 잘라주기
    merged_data = merged_data.loc[train_close.index]

    # 선형 보간
    merged_data['KOSPI'] = merged_data['KOSPI'].interpolate(method='linear')
    merged_data['KOSDAQ'] = merged_data['KOSDAQ'].interpolate(method='linear')
    merged_data['USD'] = merged_data['USD'].interpolate(method='linear')
    merged_data['3YEARS'] = merged_data['3YEARS'].interpolate(method='linear')
     
    # 트레인과 테스트셋 분리 (80%와 20%)
    train_data, test_data = train_test_split(merged_data, test_size=0.2, shuffle=False)

    # 학습 데이터 다시 분리
    filtered_tc = train_data['종가']
    filtered_change = train_data['변화량']
    filtered_kospi = train_data['KOSPI']
    filtered_kosdaq = train_data['KOSDAQ']
    filtered_usd = train_data['USD']
    filtered_3years = train_data['3YEARS']
 
    # 모델 선언, 학습시키기
    model = ARIMA(filtered_tc, exog=pd.concat([filtered_change, filtered_kospi, filtered_kosdaq, filtered_usd, filtered_3years], axis=1), order=(1, 1, 1))
    model_fit = model.fit()

    # 테스트 데이터의 길이만큼 예측값을 저장
    steps = len(test_data)  # Number of steps to predict (same as the length of the test data)
    predictions = model_fit.forecast(steps=steps, exog=pd.concat([filtered_change[-steps:], filtered_kospi[-steps:], filtered_kosdaq[-steps:], filtered_usd[-steps:], filtered_3years[-steps:]], axis=1))

    # 예측값과 실제값의 길이를 맞춰서 예측 정확도 계산
    actual_values = test_data['종가'].values
    predicted_values = predictions.values
    accuracy = np.mean(np.abs((actual_values - predicted_values) / actual_values)) * 100

    # 결과 저장
    final_return = (predictions.iloc[-1] - predictions.iloc[0]) / predictions.iloc[0]
    results_df = results_df.append({'종목코드': code, 'final_return': final_return, 'accuracy': accuracy}, ignore_index=True)
   
results_df['순위'] = results_df['final_return'].rank(method='first').astype('int') # 각 순위를 중복없이 생성

results_df

##############################################################################################################################################################

# 결과 저장하기
len(results_df) #2000 인지 확인할것
print('정확도 : ',np.mean(results_df['accuracy']),'%')

results_df.to_csv(file_path + 'results_df.csv', index=False)
results_submission =results_df[['종목코드']].merge(results_df[['종목코드', '순위']], on='종목코드', how='left')
results_submission.to_csv(file_path + 'results_submission.csv', index=False)

##############################################################################################################################################################

# 차분 결정

data = merged_data['종가']
n_diffs = ndiffs(data, alpha=0.05, test='adf', max_d=6)
print(f"추정된 차수 d = {n_diffs}") # 결과 : 추정된 차수 d = 1

# 모형 차수 결정
model = pm.auto_arima(
            y=data,
            d=1,
            start_p=0, max_p=3,
            start_q=0, max_q=3,
            m=1, seasonal=False,
            stepwise=True,
            trace=True
)
print(model_fit.summary()) # Best model:  ARIMA(1,1,1)(0,0,0)[0]  

# 잔차 검정

model_fit.plot_diagnostics()
plt.show()

y = merged_data['종가']
y_1diff = merged_data['종가'].diff().dropna()
result = adfuller(y)
print(f'원 데이터 ADF Statistic: {result[0]:.3f}')
print(f'원 데이터 p-value: {result[1]:.3f}')

result = adfuller(y_1diff)
print(f'1차 차분 ADF Statistic: {result[0]:.3f}')
print(f'1차 차분 p-value: {result[1]:.3f}')

##############################################################################################################################################################

correlation_matrix = merged_data[['거래량','변화량', 'KOSPI', 'KOSDAQ', 'USD', '3YEARS']].corr()

# VIF 계산
vif = pd.DataFrame()
vif["Variable"] = correlation_matrix.index
vif["VIF"] = [variance_inflation_factor(correlation_matrix.values, i) for i in range(len(correlation_matrix))]

print(vif)

merged_data.corr()


