#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 15:38:55 2024

@author: guyuchen
"""

#Station 1 ETL-Data import



# Data capture -01

# import requests
# from datetime import datetime
# import pandas as pd
# import numpy as np
# def Station1_loadData(crypto_pairs, api_key, limit, filepath,map):
#     def fetch_crypto_data(symbol, api_key, limit=limit):
#         if api_key.strip():  
#             headers = {'Apikey': api_key}  
#         else:
#             headers = {}  

#         url = f'https://min-api.cryptocompare.com/data/v2/histoday?fsym={symbol}&tsym=USD&limit={limit}'
#         response = requests.get(url, headers=headers)
#         data = response.json()
#         return data['Data']['Data']
#     df = pd.DataFrame()
#     for symbol in crypto_pairs:
#         print(f"Fetching data for {symbol}...")
#         data = pd.DataFrame(fetch_crypto_data(symbol, api_key=api_key,
#                                           limit=limit))  
#         data['date'] = pd.to_datetime(data['time'], unit='s') 
#         data.drop(['time'], inplace=True, axis=1)
#         data['ticker'] = symbol
#         data['Crypto']=map[symbol]
#         df = pd.concat([df, data], axis=0)

#     dfStation1 = df.set_index(['ticker', 'date'])
#     dfStation1 = dfStation1.sort_index(level=['ticker', 'date'])
#     dfStation1 = dfStation1.replace([np.inf, -np.inf], np.nan)
#     dfStation1 = dfStation1.reset_index()  
#     dfStation1 = dfStation1[~(
#                 (dfStation1['close'] == 0) | (dfStation1['close'].isna()) | (dfStation1['volumeto'] == 0) | (
#             dfStation1['volumeto'].isna()))]

#     dfStation1 = dfStation1.set_index(['ticker', 'date'])
#     dfStation1.to_csv(filepath)

#     # print("____")
#     # print(f"Data successfully saved to {filepath}")
#     return df

'''ETL'''
'''1.1# Data capture -02
Get cryptocurrency information from the Crypto compare website'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from datetime import datetime

def fetch_crypto_data(symbol, api_key, limit = 1000):
    if api_key.strip():
        headers = {'Apikey':api_key}
    else:
        headers = {}
        
    url = f'https://min-api.cryptocompare.com/data/v2/histoday?fsym={symbol}&tsym=USD&limit={limit}'    
    response = requests.get(url, headers=headers)
    data = response.json()
    return data['Data']['Data']


'''1.1.2 Get a list of cryptocurrencies:
#Get a list of all cryptocurrencies from Cryptocompare and map them'''

def Station1_allcoins(api_key):
    headers = {'Apikey': api_key}  
    url=f'https://min-api.cryptocompare.com/data/all/coinlist'
    response = requests.get(url, headers=headers)
    data = response.json()
    symbol_list=data['Data'].keys()
    symbol_coin_map={}
    for name in symbol_list:
        coin_name=data['Data'][name]['CoinName']
        symbol_coin_map[name]=coin_name
    return symbol_coin_map

#Get the top five cryptocurrency names for further research

def get_top_20_coins(api_key):
    headers = {'Apikey': api_key}
    url = 'https://min-api.cryptocompare.com/data/top/mktcapfull?limit=20&tsym=USD'
    response = requests.get(url, headers=headers)
    data = response.json()
    top_20_coins = [coin['CoinInfo']['Name'] for coin in data['Data']]
    return top_20_coins
if __name__ == "__main__":
    api_key = 'your_api_key_here'
    top_20_coins = get_top_20_coins(api_key)
    # print("Top 20 cryptocurrencies by market cap:")
    # print(top_20_coins)
    
    
#Obtain the names of the top five cryptocurrencies for further research 
#, of which USDT, due to too stable research value,
# abandoned the research and chose other digital currencies such as  XRP 
 #ETL- transform   
 
 
 
def main(limit= 1000,api_key = ''):
    crypto_pairs = ['BTC', 'ETH', 'BNB', 'XRP', 'TONCOIN', 'ADA', 'DOGE', 'TRX', 'ONDO', 'SHIB', 'OP']
    df= pd.DataFrame()
    for symbol in crypto_pairs:
        data = pd.DataFrame(fetch_crypto_data(symbol, api_key= api_key,limit=limit))
        data['date']=pd.to_datetime(data['time'],unit = 's')
        data.drop(['time'],inplace=True,axis=1)
        data['sym'] = symbol
        df = pd.concat([df,data], axis=0)
    df = df[~((df['close'] == 0) | (df['close'].isna()) | (df['volumeto'] == 0) | (df['volumeto'].isna()))]
    return df

#ETL - load

df = main(api_key="")
df.to_csv('crypto.csv')
    


'''1.3 Research on specific functions (BTC)'''
df = pd.read_csv('crypto.csv',index_col=0) 
btc = df[df['sym']=='BTC']
# 1.information，Message checking, checking for data errors or null values for specific currencies
# print(btc.info())
# check for nan
# print(df.isnull().sum())

# #select data,close,voluemfrom,volumeto
# ETL- extract -> feature selection
btc = btc[['date','close', 'volumefrom','volumeto']]

## set index
btc = btc.set_index(['date'])


'''##1.3.1.Construct a combination of price and volume'''
btc.rename(columns= {'close': "prc"},inplace = True)

# btc['prc'].plot(label ='price')
# btc['volumefrom'].plot(label= 'vf')
# (btc['volumeto']/1000000).plot(label='v-mil')
# plt.title('vol-time')
# plt.show()


'''#2Feature Engineering
2.1 Building portfolios and statistical models - Feature Engineering'''


btc['ret'] =btc['prc'].pct_change()

'''##2.1.1 Cumulative return#'''

((1+btc['ret']).cumprod() - 1).plot()
# plt.title('Cumulative return-BTC')
# plt.show()

#2.1.2 Holding period return 
hpr =np.round((btc['prc'].iloc[-1]/btc['prc'].iloc[0] - 1) * 100 , 3)
# print(f"BTC/USD holding-period return is {hpr}%")

#2.1.3 Annual Return
annual_ret = np.round(btc['ret'].mean() * 100 * 365, 3) 
# print(f"BTC/USD annual return is {annual_ret}%")

#2.1.4 Annual volatility
annual_vol = np.round(btc['ret'].std() * np.sqrt(365) * 100, 3) 
# print(f"BTC/USD annual volatility is {annual_vol}%")

#2.1.5 Annual sharpe ratio
sharpe_ratio = np.round(annual_ret / annual_vol, 3) 
# print(f"BTC/USD annual sharpe ratio is {sharpe_ratio}%")

#2.1.6 Annual Sortino ratio
sortino_ratio = np.round((btc['ret'].mean() /  
                          btc['ret'][btc['ret'] < 0].std()) * np.sqrt(365), 3)
# print(f"BTC/USD annual Sortino ratio is {sortino_ratio}")


'''# Apply the function to all data sets
# ETL - feature selection'''

df.rename(columns={'close': 'prc'}, inplace=True)
subset = df[['prc','date','sym','volumefrom','volumeto']]
# print(subset)


'''##2.2 Cumulative return#'''
subset['return'] = subset['prc'].pct_change()
# print(subset)
ret_by_sym = subset.groupby('sym')['return'].mean()
# print(ret_by_sym)
# print(ret_by_sym*7)


'''# 2.2.1 Holding Period Return'''
def calculate_holding_period_return(group):
    return np.round((group['prc'].iloc[-1] / group['prc'].iloc[0] - 1) * 100, 3)

hpr_by_sym = subset.groupby('sym').apply(calculate_holding_period_return)
# print("Holding Period Return by Symbol:")
# print(hpr_by_sym)

'''# 2.2.2 Annual Return'''
def calculate_annual_return(group):
    return np.round(group['return'].mean() * 100 * 365, 3)

annual_ret_by_sym = subset.groupby('sym').apply(calculate_annual_return)
# print("Annual Return by Symbol:")
# print(annual_ret_by_sym)

'''# 3.2.3 Annual Volatility'''
def calculate_annual_volatility(group):
    return np.round(group['return'].std() * np.sqrt(365) * 100, 3)

annual_vol_by_sym = subset.groupby('sym').apply(calculate_annual_volatility)
# print("Annual Volatility by Symbol:")
# print(annual_vol_by_sym)

'''# 2.2.4 Annual Sharpe Ratio'''
def calculate_sharpe_ratio(group):
    annual_return = group['return'].mean() * 365
    annual_volatility = group['return'].std() * np.sqrt(365)
    return np.round(annual_return / annual_volatility, 3)

sharpe_ratio_by_sym = subset.groupby('sym').apply(calculate_sharpe_ratio)
# print("Sharpe Ratio by Symbol:")
# print(sharpe_ratio_by_sym)

'''# 2.2.5 Annual Sortino Ratio'''
def calculate_sortino_ratio(group):
    annual_return = group['return'].mean() * 365
    downside_volatility = group['return'][group['return'] < 0].std() * np.sqrt(365)
    return np.round(annual_return / downside_volatility, 3)

sortino_ratio_by_sym = subset.groupby('sym').apply(calculate_sortino_ratio)
# print("Sortino Ratio by Symbol:")
# print(sortino_ratio_by_sym)

'''#2.2.6 Calculate the standard deviation of the return rate and measure the risk value'''
std_by_sym = subset.groupby('sym')['return'].std()
std_by_sym = std_by_sym.sort_values(ascending=False)
print(std_by_sym)


# Feature Engineering
'''2.3 Feature engineering visualization'''

'''2.3.1 Returns by Cryptocurrency，Check the normal distribution'''
# fig, ax = plt.subplots(figsize=(10,8))
# for i, j in subset.groupby('sym'):
#     plt.hist(j['return'],label =i, density = True , bins =50)   
# plt.legend()
# plt.xlabel('Return')
# plt.ylabel('Density')
# plt.title('Returns by Cryptocurrency')
# plt.show()


'''2.3.2 Price curve over time for cryptocurrencies'''
fig, ax =plt.subplots(figsize=(10,8))
# for i, j in subset.groupby('sym'):
#     plt.plot(j['date'],j['prc'], label = i)
# plt.legend()
# plt.xlabel('Date')
# plt.ylabel('Price (USD)')
# plt.title('Cryptocurrency Prices Over Time')
# plt.show()


'''
 2.3.3 A line chart of returns over time to distinguish between
  trends and fluctuations in returns for different cryptocurrencies'''

# fig, ax =plt.subplots(figsize=(10,8))
# for i, j in subset.groupby('sym'):
#     plt.plot(j['date'],j['return'], label = i)
# plt.legend()
# plt.xlabel('Date')
# plt.ylabel('Return')
# plt.title('Cryptocurrency Returns Over Time')
# plt.show()


'''
2.3.4Return - A line chart of the rate of return over time, used to distinguish
 the trend and volatility of the rate of return of different cryptocurrencies,
 and the monthly volatility can also reflect the risk profile'''
# fig, ax =plt.subplots(figsize=(10,8))
# for i, j in subset.groupby('sym'):
#     j = j.set_index('date')
#     j.index = pd.to_datetime(j.index)
#     j = j.resample('M').last()
#     plt.plot(j.index,j['return'],label=i)
# plt.legend()
# plt.xlabel('Date')
# plt.ylabel('Monthly Return')
# plt.title('Monthly Cryptocurrency Returns Over Time')
# plt.show()


'''2.3.5 Return on holding-Line chart of monthly holding returns'''

# fig, ax = plt.subplots(figsize=(10, 8))
# for sym, group in subset.groupby('sym'):
#     group = group.set_index('date')
#     group.index = pd.to_datetime(group.index)
#     group = group.resample('M').apply(calculate_holding_period_return)
#     plt.plot(group.index, group, label=sym)

# plt.legend()
# plt.xlabel('Date')
# plt.ylabel('Holding Period Return (%)')
# plt.title('Monthly Holding Period Return of Cryptocurrencies')
# plt.show()

'''2.3.6 Return on holding - Box plot - A box plot is built based on the characteristic
  distribution of each cryptocurrency calculated by the holding period return rate, 
  so as to visually show the distribution characteristics of different currency data。'''
 
# hpr_data = []
# labels = []
# for sym in hpr_by_sym.index:
#     returns = subset[subset['sym'] == sym]['return'].dropna().values
#     if len(returns) > 0:
#         hpr_data.append(returns)
#         labels.append(sym)

# fig, ax = plt.subplots(figsize=(10, 8))
# ax.boxplot(hpr_data, labels=labels)
# ax.set_title('Box Plot of Holding Period Return by Cryptocurrency')
# ax.set_xlabel('Cryptocurrency')
# ax.set_ylabel('Holding Period Return (%)')
# plt.show()

'''# 2.3.7 Annual Volatility - Bar chart'''
# fig, ax = plt.subplots(figsize=(10, 8))
# ax.bar(annual_vol_by_sym.index, annual_vol_by_sym.values, color='b')
# ax.set_title('Annual Volatility by Cryptocurrency')
# ax.set_xlabel('Cryptocurrency')
# ax.set_ylabel('Annual Volatility (%)')
# plt.show()

'''2.3.8 Annual Volatility'''
# annual_vol_df = annual_vol_by_sym.reset_index()
# annual_vol_df.columns = ['Cryptocurrency', 'Annual Volatility']
# plt.figure(figsize=(12, 8))
# sns.barplot(data=annual_vol_df, x='Cryptocurrency', y='Annual Volatility', palette='viridis')
# plt.title('Annual Volatility by Cryptocurrency')
# plt.xlabel('Cryptocurrency')
# plt.ylabel('Annual Volatility (%)')
# plt.show()



'''#2.3.9 Portfolio correlation, heat map'''
''' Optimize the portfolio, through the management of correlation to identify 
 low correlation or negative correlation products to achieve portfolio risk reduction'''
# returns = subset.pivot(index='date', columns='sym', values='return')

# corr_matrix = returns.corr()

# plt.figure(figsize=(10, 8))
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
# plt.title('Cryptocurrency Returns Correlation Heatmap')
# plt.show()


'''2.3.10. Tree chart - Through the tree chart, it can help investors identify the hierarchical 
relationship, similarity and difference between the returns of different cryptocurrencies
, so as to better optimize the investment portfolio for risk management'''

# returns = subset.pivot(index='date', columns='sym', values='return')
# returns = returns.fillna(0)
# Z = linkage(returns.T, method='ward')
# plt.figure(figsize=(12, 8))
# dendrogram(Z, labels=returns.columns, leaf_rotation=90, leaf_font_size=12)
# plt.title('Dendrogram of Cryptocurrency Returns')
# plt.xlabel('Cryptocurrency')
# plt.ylabel('Distance')
# plt.show()


'''# 3 Unstructured data/Sentiment data
#Get the latest cryptocurrency news data from crytocompare and store it in a local file
# ETL, which grabs news and sentiment data'''
import requests
from datetime import datetime
import pandas as pd
import numpy as pf
import os


export_path = r'/Users/guyuchen/Desktop/FINS 5545/Project A WEEK 8/Project A/SentimentData'

'''3.1.1 Unstructured data acquisition'''
def Stage1_CryptoETL_Text(export_path,api_key=None):
    headers= {}
    if api_key and api_key.strip():
        headers['Apikey']= api_key
        
    url = 'https://min-api.cryptocompare.com/data/v2/news/?lang=EN'
    
    response = requests.get(url, headers=headers)
    data = response.json()  
    df = pd.DataFrame(data['Data'])
    df['date'] = pd.to_datetime(df['published_on'], unit='s')
    

    today_date = pd.Timestamp.today().strftime('%d_%m_%Y')
    suffix = '_Sentiment'
    
    df.to_hdf(os.path.join(export_path,today_date + suffix + '.h5'),
              key = 'daily')
  
  
    return df

df = Stage1_CryptoETL_Text(export_path,api_key= None)
# print(df[['title','upvotes','downvotes']])

'''# 3.1.2 After capturing certain news data on cryptocompare and downloading it to the local, 
# the local file is used as the database to train the model'''
# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta
# import os
# import re
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer

# # Read local data
# base_dir = r'/Users/guyuchen/Desktop/FINS 5545/Project A WEEK 8/Project A/SentimentData'
# files = os.listdir(base_dir)

# df = pd.DataFrame()

# #ETL-Open all the files first and read them, converting the data inside, such as the date

# for f in files:
#     # print(f)
#     dfsub = pd.read_hdf(os.path.join(base_dir, f))
#     dfsub['date'] = pd.to_datetime(dfsub['published_on'], unit='s')
#     df = pd.concat([df, dfsub], axis=0)


# df['datem'] = pd.to_datetime(df['date']).dt.date

# df = df.sort_values(['datem'])

# # Drop the 'Unnamed: 0' column if it exists
# if 'Unnamed: 0' in df.columns:
#     df.drop(['Unnamed: 0'], axis=1, inplace=True)

# # Handle missing values if any (optional, based on data)
# df.fillna('', inplace=True)

# # Save to CSV
# output_path = r'/Users/guyuchen/Desktop/FINS 5545/Project A WEEK 8/Project A/SentimentData/Stage_1_Time_Series_Sentiment.csv'
# df.to_csv(output_path, index=False)


# import libraries
# import pandas as pd
# import numpy as np
# import nltk
# from collections import Counter
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import classification_report
# from tqdm import tqdm
# from wordcloud import WordCloud
# import matplotlib.pyplot as plt
# import os
# import re

# tqdm.pandas()

# export_path = r'/Users/guyuchen/Desktop/FINS 5545/Project A WEEK 8/Project A'

# file_name = 'Stage2_Text_Data.csv'  # 文件名

# df = pd.read_csv(r'/Users/guyuchen/Desktop/FINS 5545/Project A WEEK 8/Project A/SentimentData/Stage_1_Time_Series_Sentiment.csv')

# df['datem'] = pd.to_datetime(df['datem'])




# # =============================================================================
# # 2: Feature Engineering
# # =============================================================================

# # Select necessary columns for analyses #
# select_cols = ['datem', 'date', 'id', 'title', 'body', 'categories']
# df = df[select_cols]

# def preprocess_text(text):
#     if not isinstance(text, str):
#         return ""
    
#     # 移除URL、提及和标签
#     text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
#     text = re.sub(r'\@\w+|\#','', text)
    
#     # 分词
#     tokens = word_tokenize(text)
    
#     # 移除停用词
#     filtered_tokens = [token for token in tokens if token.lower() not in stopwords.words('english')]
    
#     # 移除特殊字符和标点符号
#     filtered_tokens = [re.sub(r'[^A-Za-z0-9]+', '', token) for token in filtered_tokens]
    
#     # 词形还原
#     lemmatizer = WordNetLemmatizer()
#     lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    
#     # 将处理后的tokens重新组合成字符串
#     processed_text = ' '.join(lemmatized_tokens)
#     return processed_text

# # 将列转换为字符串类型并填充NaNs为空字符串
# df['title'] = df['title'].astype(str).fillna('')
# df['body'] = df['body'].astype(str).fillna('')

# # 对 'title' 和 'body' 列应用文本预处理函数
# df['title_clean'] = df['title'].progress_apply(preprocess_text)
# df['body_clean'] = df['body'].progress_apply(preprocess_text)

# # =============================================================================
# # 3: Export to file
# # =============================================================================
# df.to_csv(os.path.join(export_path, file_name))


# print(f"Data has been successfully saved to {output_path}")






# '''# feature engineering - data cleaning & filtering
# # Data cleansing and filtering of unstructured data'''


# export_path = r'/Users/guyuchen/Desktop/FINS 5545/Project A WEEK 8/Project A/SentimentData/' 
# file_name = 'Stage2_Text_Data.csv'  # 文件名
# df = pd.read_csv(r'/Users/guyuchen/Desktop/FINS 5545/Project A WEEK 8/Project A/SentimentData/Stage_1_Time_Series_Sentiment.csv')
# df['datem'] = pd.to_datetime(df['datem'])

# 3.1.3 ETL - Check and remove duplicate data
# df = df.set_index(['id', 'date'])
# df.index.duplicated().sum()
# df = df.drop_duplicates()
# print(df)
# print(df.iloc[:, 3:6])


'''#3.2. feature engineering for unstructured data
# Check each column name first'''
# print(df.columns)

#3.2 Select the desired features
# Examine the combinations of different features, what information you get,
# and whether you can build a useful combination
# subset1 = df[['title', 'body', 'upvotes', 'downvotes']]
# # print(subset1)
# # print(subset1[['upvotes', 'downvotes']].value_counts())

# subset2 = df[['guid', 'published_on', 'imageurl']]
# # print(subset2)

# subset3 = df[['tags', 'lang']]
# # print(subset3)

# subset4 = df[['categories', 'source']]
# # print(subset4)

# # Filter only news data that contains BTC
# subset5 = df[df['categories'].notnull()]
# # print(subset5['categories'].str.contains('BTC'))
# subset5 = subset5.loc[subset5['categories'].str.contains('BTC', na=False)]
# # print(subset5)



# def preprocess_text(text):
#     stop_words = set(stopwords.words('english'))  
#     custom_stop_words = {'crypto', 'cryptocurrency', 'blockchain','market','bitcoin','price','new'}  
#     stop_words = stop_words.union(custom_stop_words)  
#     lemmatizer = WordNetLemmatizer()  
#     tokens = word_tokenize(text)  
#     tokens = [word for word in tokens if word not in stop_words]  
#     tokens = [lemmatizer.lemmatize(word) for word in tokens]  
#     return ' '.join(tokens)  

# def plot_fear_and_greed_index(index, dates):

#     plt.figure(figsize=(12, 6))
#     plt.plot(dates, index, label='Fear and Greed Sentiment Index')
#     plt.xlabel('Date')
#     plt.ylabel('Fear and Greed Index')
#     plt.title('Fear and Greed Sentiment Index Over Time')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig('.Sentiment_index.png') 
#     plt.show()





import requests
from datetime import datetime
import pandas as pd
import numpy as np
import os
import nltk
from collections import Counter
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt
import re

# 下载必要的 NLTK 资源
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

tqdm.pandas()

export_path = r'/Users/guyuchen/Desktop/FINS 5545/Project A WEEK 8/Project A/SentimentData'

# 确保导出路径存在
if not os.path.exists(export_path):
    os.makedirs(export_path)

def fetch_crypto_news(export_path, api_key=None):
    headers = {}
    if api_key and api_key.strip():
        headers['Apikey'] = api_key

    url = 'https://min-api.cryptocompare.com/data/v2/news/?lang=EN'
    response = requests.get(url, headers=headers)
    data = response.json()
    
    if 'Data' not in data:
        raise KeyError('Data key not found in the response')

    df = pd.DataFrame(data['Data'])
    
    if 'published_on' in df.columns:
        df['date'] = pd.to_datetime(df['published_on'], unit="s")
    elif 'published_on' in data['Data'][0]:
        df['date'] = pd.to_datetime([item['published_on'] for item in data['Data']], unit="s")
    else:
        raise KeyError('published_on')

    today_date = pd.Timestamp.today().strftime('%d_%m_%Y')
    suffix = '_Sentiment'
    
    output_file_h5 = os.path.join(export_path, today_date + suffix + '.h5')
    df.to_hdf(output_file_h5, key='daily')
             
    return df

# 获取数据
fetch_crypto_news(export_path, api_key=None)

def merge_files_in_directory(directory):
    # 获取目录下所有的 .csv 和 .h5 文件
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    h5_files = [f for f in os.listdir(directory) if f.endswith('.h5')]

    df_list = []

    # 读取所有的 .csv 文件
    for f in csv_files:
        file_path = os.path.join(directory, f)
        df = pd.read_csv(file_path)
        df_list.append(df)

    # 读取所有的 .h5 文件
    for f in h5_files:
        file_path = os.path.join(directory, f)
        df = pd.read_hdf(file_path)
        df_list.append(df)

    # 合并所有的数据
    if df_list:
        combined_df = pd.concat(df_list, ignore_index=True)
    else:
        combined_df = pd.DataFrame()

    return combined_df

# 合并数据
combined_df = merge_files_in_directory(export_path)

# 特征工程
select_cols = ['datem', 'date', 'id', 'title', 'body', 'categories']
combined_df['datem'] = pd.to_datetime(combined_df['date']).dt.date
combined_df = combined_df.sort_values(['datem'])
combined_df = combined_df[select_cols]

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#','', text)
    
    tokens = word_tokenize(text)
    filtered_tokens = [token for token in tokens if token.lower() not in stopwords.words('english')]
    filtered_tokens = [re.sub(r'[^A-Za-z0-9]+', '', token) for token in filtered_tokens]
    
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    
    processed_text = ' '.join(lemmatized_tokens)
    return processed_text

combined_df['title'] = combined_df['title'].astype(str).fillna('')
combined_df['body'] = combined_df['body'].astype(str).fillna('')

combined_df['title_clean'] = combined_df['title'].progress_apply(preprocess_text)
combined_df['body_clean'] = combined_df['body'].progress_apply(preprocess_text)

# 保存最终合并后的数据
output_file_stage2_csv = os.path.join(export_path, 'Stage2_Text_Data.csv')
output_file_stage2_h5 = os.path.join(export_path, 'Stage2_Text_Data.h5')

combined_df.to_csv(output_file_stage2_csv, index=False)
combined_df.to_hdf(output_file_stage2_h5, key='df', mode='w')

print(f"数据已成功保存到 {output_file_stage2_csv} 和 {output_file_stage2_h5}")