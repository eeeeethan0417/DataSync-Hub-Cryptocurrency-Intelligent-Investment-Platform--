git #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 00:15:35 2024

@author: guyuchen
"""

import requests
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import minimize
import os
import json


''' station 1，Access the CryptoCompare API to get historical
 data for a specified cryptocurrency'''

# 1.1

import requests
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os

# Function to fetch crypto data
def fetch_crypto_data(symbol, api_key, limit=2000):
    if api_key.strip():  # Check if api key is not empty or just whitespace
        headers = {'Apikey': api_key}  # The header with your API key
    else:
        headers = {}  # Empty headers if api key is empty
    
    url = f'https://min-api.cryptocompare.com/data/v2/histoday?fsym={symbol}&tsym=USD&limit={limit}'
    response = requests.get(url, headers=headers)
    data = response.json()
    return data['Data']['Data']




'''Get and display the names of the top 100 cryptocurrencies by 
current market capitalization. Focus on understanding the most 
important cryptocurrency information, and make market analysis and 
investment decisions based on the information.'''


# 1.2
def Top_Market_Cap():
    url = 'https://min-api.cryptocompare.com/data/top/mktcapfull?limit=100&tsym=USD'
    response = requests.get(url)
    data = response.json()
    pairs = data['Data']
    return pd.DataFrame(pairs)

df = Top_Market_Cap()
df['Name'] = df['CoinInfo'].apply(lambda x: x['Name'])
top_tickers = df['Name']
# print(top_tickers)

'''By accessing the CryptoCompare API, get historical price data for multiple 
cryptocurrencies and consolidate this data into a single data box and save it as CSV
'''
def main(limit=2000, api_key=''):
    # List of cryptocurrency pairs
    crypto_pairs = list(top_tickers.values)
    crypto_pairs = ['BTC', 'ETH', 'BNB', 'XRP', 'TONCOIN', 'ADA', 'DOGE', 'TRX', 'ONDO', 'SHIB', 'OP']
    # Create an empty DataFrame to store the data
    df = pd.DataFrame()

    # Fetch historical data for each pair
    for symbol in crypto_pairs:
        # print(f"Fetching data for {symbol}...")
        data = pd.DataFrame(fetch_crypto_data(symbol, api_key=api_key, limit=limit))  # Limiting to 30 days
        # ETL - transform
        data['date'] = pd.to_datetime(data['time'], unit='s')
        data.drop(['time'], inplace=True, axis=1)
        data['symbol'] = symbol
        df = pd.concat([df, data], axis=0)

    return df

# ETL - load
# df = main(api_key="")
# df.to_csv('crypto_full100.csv')



# 1.3
'''# station 2 DATA CLEAN：
Cleans and processes historical price information extracted from cryptocurrency data, especially closing prices.'''

# df = pd.read_csv('crypto_full100.csv', index_col=0)
df = pd.read_csv('crypto.csv', index_col=0, parse_dates=True)
# print(df)
# print(df.columns)

df = df[['date', 'close', 'sym']]
# print(df)

rets = []

for i in df.groupby('sym'):
    ser = pd.Series(i[1]['close'].values, i[1]['date'].values)
    ser.name = i[0]
    rets.append(ser)

ret_df = pd.concat(rets, axis=1)
# print(ret_df)
ret_df = ret_df.pct_change().dropna()
# print(ret_df)
ret_df.replace([np.inf, -np.inf], 0, inplace=True)
# print(ret_df)


'''Prepare data and set paths for subsequent analysis or saving.'''


df = ret_df.copy()
exportpath = r'/Users/guyuchen/Desktop/FINS 5545/Project B/data'






# A function to get cryptocurrency data
def fetch_crypto_data(symbol, api_key, limit=2000):
    headers = {'Apikey': api_key} if api_key.strip() else {}
    url = f'https://min-api.cryptocompare.com/data/v2/histoday?fsym={symbol}&tsym=USD&limit={limit}'
    response = requests.get(url, headers=headers)
    data = response.json()
    return data['Data']['Data']

# Get the top 100 cryptocurrencies ranked by market capitalization
def top_market_cap():
    url = 'https://min-api.cryptocompare.com/data/top/mktcapfull?limit=100&tsym=USD'
    response = requests.get(url)
    data = response.json()
    pairs = data['Data']
    return pd.DataFrame(pairs)


#Obtain historical price data for multiple cryptocurrencies as an experimental example
def main(limit=2000, api_key=''):
    df = top_market_cap()
    df['Name'] = df['CoinInfo'].apply(lambda x: x['Name'])
    top_tickers = df['Name']
    
    crypto_pairs = ['BTC', 'ETH', 'BNB', 'XRP', 'TONCOIN', 'ADA', 'DOGE', 'TRX', 'ONDO', 'SHIB', 'OP']
    data = pd.DataFrame()

    for symbol in crypto_pairs:
        print(f" Getting data for {symbol}...")
        crypto_data = pd.DataFrame(fetch_crypto_data(symbol, api_key=api_key, limit=limit))
        crypto_data['date'] = pd.to_datetime(crypto_data['time'], unit='s')
        crypto_data.drop(['time'], inplace=True, axis=1)
        crypto_data['symbol'] = symbol
        data = pd.concat([data, crypto_data], axis=0)

    data.to_csv('crypto_full100.csv')
    return data

# Clean and process historical price data
def clean_data():
    df = pd.read_csv('crypto_full100.csv', index_col=0)
    df = df[['date', 'close', 'symbol']]
    rets = []

    for name, group in df.groupby('symbol'):
        series = pd.Series(group['close'].values, index=group['date'].values)
        series.name = name
        rets.append(series)

    ret_df = pd.concat(rets, axis=1).pct_change().dropna()
    ret_df.replace([np.inf, -np.inf], 0, inplace=True)
    return ret_df










'''Station 3 Model Design'''

# 2.1
# Calculate portfolio performance
def portfolio_performance(weights, mean_returns, cov_matrix):
    weights = np.array(weights)
    returns = np.dot(weights, mean_returns)
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return returns, std

# 2.2
# Optimization function
def statistics(weights, mean_returns, cov_matrix):
    weights = np.array(weights)
    pret = np.sum(mean_returns * weights) * 252
    pvol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return np.array([pret, pvol, pret / pvol])

# 2.3
# Maximum Sharp ratio portfolio
def min_func_sharpe(weights, mean_returns, cov_matrix):
    return -statistics(weights, mean_returns, cov_matrix)[2]

# 2.4
# Minimum variance portfolio
def min_func_variance(weights, mean_returns, cov_matrix):
    return statistics(weights, mean_returns, cov_matrix)[1] ** 2

# 2.5
# Portfolio optimization
def optimize_portfolio(mean_returns, cov_matrix, objective_function):
    noa = len(mean_returns)
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    bounds = tuple((0, 1) for _ in range(noa))
    initial_guess = noa * [1. / noa,]
    result = minimize(objective_function, initial_guess, args=(mean_returns, cov_matrix),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result


# 2.6
# Risk Aversion Portfolio
'''This code implements a function that optimizes risk in an individual portfolio. 
It defines two key functions: person_risk and optimize_personal_risk. The person_risk f
unction calculates a risk-adjusted target value based on the given portfolio weight, the expected 
return rate of the asset and the covariance matrix, combined with the investor's risk appetite.'''

def person_risk(weights, mean_returns, cov_matrix, risk):
    stats = statistics(weights, mean_returns, cov_matrix)
    return -stats[0] + (10 - risk) * stats[1]

def optimize_personal_risk(mean_returns, cov_matrix, risk):
    noa = len(mean_returns)
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    bounds = tuple((0, 1) for _ in range(noa))
    initial_guess = noa * [1. / noa,]
    result = minimize(person_risk, initial_guess, args=(mean_returns, cov_matrix, risk),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result

# 2.7Risk Parity Portfolio

'''Reduce overall portfolio risk by ensuring that all assets contribute roughly the same amount 
of risk to the portfolio'''

def risk_parity_objective(weights, cov_matrix, risk_budget):
    weights = np.array(weights)
    portfolio_var = np.dot(weights.T, np.dot(cov_matrix, weights))
    marginal_contribs = np.dot(cov_matrix, weights)
    risk_contribs = weights * marginal_contribs
    risk_budget_term = np.sum((risk_contribs / portfolio_var - risk_budget) ** 2)
    return risk_budget_term

def optimize_risk_parity(cov_matrix, risk_budget):
    noa = len(risk_budget)
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    bounds = tuple((0, 1) for _ in range(noa))
    initial_guess = noa * [1. / noa,]
    result = minimize(risk_parity_objective, initial_guess, args=(cov_matrix, risk_budget),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result


# 2.8
# Mean-Variance Optimization
def min_variance_target_return(mean_returns, cov_matrix, target_return):
    noa = len(mean_returns)
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                   {'type': 'eq', 'fun': lambda x: portfolio_performance(x, mean_returns, cov_matrix)[0] - target_return}]
    bounds = tuple((0, 1) for _ in range(noa))
    initial_guess = noa * [1. / noa,]
    result = minimize(portfolio_volatility, initial_guess, args=(mean_returns, cov_matrix),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def portfolio_volatility(weights, mean_returns, cov_matrix):
    weights = np.array(weights)
    return portfolio_performance(weights, mean_returns, cov_matrix)[1]


# 2.9
# Maximum Diversification Portfolio


'''Reduce the risk of the overall portfolio by increasing the diversification between different assets. 
This approach aims to use the ircorrelation between assets to reduce overall volatility and thus achieve 
more stable investment returns.'''


def max_diversification_objective(weights, cov_matrix):
    weights = np.array(weights)
    individual_volatilities = np.sqrt(np.diag(cov_matrix))
    weighted_volatility = np.dot(weights, individual_volatilities)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    diversification_ratio = weighted_volatility / portfolio_volatility
    return -diversification_ratio

def optimize_max_diversification(cov_matrix):
    noa = cov_matrix.shape[0]
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    bounds = tuple((0, 1) for _ in range(noa))
    initial_guess = noa * [1. / noa,]
    result = minimize(max_diversification_objective, initial_guess, args=(cov_matrix,),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result



# 3.Efficient Frontier
# 3.1
def plot_efficient_frontier(mean_returns, cov_matrix, num_portfolios=20000):
    results = np.zeros((3, num_portfolios))
    weights_record = []

    for i in range(num_portfolios):
        weights = np.random.random(len(mean_returns))
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_return, portfolio_risk = portfolio_performance(weights, mean_returns, cov_matrix)
        results[0, i] = portfolio_return
        results[1, i] = portfolio_risk
        results[2, i] = (portfolio_return - 0) / portfolio_risk

    optimal_weights_sharpe = optimize_portfolio(mean_returns, cov_matrix, min_func_sharpe)['x']
    optimal_weights_volatility = optimize_portfolio(mean_returns, cov_matrix, min_func_variance)['x']
    target_return = 0.2
    optimal_weights_mean_variance = min_variance_target_return(mean_returns, cov_matrix, target_return)['x']
    risk_budget = np.array([1 / len(mean_returns)] * len(mean_returns))
    optimal_weights_risk_parity = optimize_risk_parity(cov_matrix, risk_budget)['x']
    optimal_weights_max_diversification = optimize_max_diversification(cov_matrix)['x']
# 3.2
    def specific_portfolio_performance(weights):
        return portfolio_performance(weights, mean_returns, cov_matrix)

    max_sharpe_return, max_sharpe_risk = specific_portfolio_performance(optimal_weights_sharpe)
    min_volatility_return, min_volatility_risk = specific_portfolio_performance(optimal_weights_volatility)
    mean_variance_return, mean_variance_risk = specific_portfolio_performance(optimal_weights_mean_variance)
    risk_parity_return, risk_parity_risk = specific_portfolio_performance(optimal_weights_risk_parity)
    max_diversification_return, max_diversification_risk = specific_portfolio_performance(optimal_weights_max_diversification)

    equal_weights = len(mean_returns) * [1. / len(mean_returns)]
    equal_weights_return, equal_weights_risk = specific_portfolio_performance(equal_weights)

    plt.figure(figsize=(14, 8))
    scatter = plt.scatter(results[1, :], results[0, :], c=results[2, :], cmap='viridis', marker='o')
    plt.scatter(max_sharpe_risk, max_sharpe_return, marker='*', color='r', s=200, label='Max Sharpe Ratio')
    plt.scatter(mean_variance_risk, mean_variance_return, marker='*', color='g', s=200, label='Mean Variance RA=0.1')
    plt.scatter(risk_parity_risk, risk_parity_return, marker='*', color='orange', s=200, label='Risk Parity')
    plt.scatter(max_diversification_risk, max_diversification_return, marker='*', color='cyan', s=200, label='Max Diversification')
    plt.scatter(equal_weights_risk, equal_weights_return, marker='x', color='purple', s=200, label='Equal Weights')
    plt.scatter(min_volatility_risk, min_volatility_return, marker='*', color='y', s=200, label='Min Volatility')
    plt.title('Mean-Variance Efficient Frontier')
    plt.xlabel('Standard Deviation (Risk)')
    plt.ylabel('Return')
    plt.colorbar(scatter, label='Sharpe Ratio')
    plt.legend()
    plt.savefig('./Mean-Variance Efficient Frontier.png')
    plt.show()

    return results, weights_record, [optimal_weights_sharpe, optimal_weights_volatility, optimal_weights_mean_variance, optimal_weights_risk_parity, optimal_weights_max_diversification]




def plot_additional_graphs(df, mean_returns, cov_matrix, results, weights_record, optimal_weights, exportpath, column_names):
    
    optimal_weights_sharpe, optimal_weights_volatility, optimal_weights_mean_variance, optimal_weights_risk_parity, optimal_weights_max_diversification = optimal_weights

    
    weights_df = pd.DataFrame({
        'MSR Weights': optimal_weights_sharpe,
        'MV Weights': optimal_weights_volatility,
        'MVRA Weights': optimal_weights_mean_variance,
        'RP Weights': optimal_weights_risk_parity,
        'MD Weights': optimal_weights_max_diversification
    }, index=column_names)

    weights_df.index.rename('Cryptos', inplace=True)

   # 3.3
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(results[1], results[0], c=results[2], cmap='viridis')
    cbar = plt.colorbar(scatter)
    cbar.set_label('Sharpe Ratio', fontsize=14)
    cbar.ax.tick_params(labelsize=14)
    plt.scatter(optimal_weights_sharpe, optimal_weights_volatility, marker='*', color='#FFA500', s=200, label='Max Sharpe Ratio')
    plt.scatter(optimal_weights_volatility, optimal_weights_volatility, marker='*', color='b', s=200, label='Min Volatility')
    plt.scatter(optimal_weights_mean_variance, optimal_weights_mean_variance, marker='*', color='g', s=200, label='Mean Variance RA = 0.1')
    plt.scatter(optimal_weights_risk_parity, optimal_weights_risk_parity, marker='*', color='r', s=200, label='Risk Parity')
    plt.title('Mean-Variance Efficient Frontier', fontsize=14)
    plt.xlabel('Standard Deviation', fontsize=14)
    plt.ylabel('Return', fontsize=14)
    plt.legend(loc='upper left', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    filename = 'mean_variance_efficient_frontier.png'
    plt.savefig(os.path.join(exportpath, filename))   
    plt.show()

# 3.4
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.2
    index = range(len(weights_df['MSR Weights']))

    bar1 = plt.bar(index, weights_df['MSR Weights'], bar_width, label='MSR Weights')
    bar2 = plt.bar([i + bar_width for i in index], weights_df['MV Weights'], bar_width, label='MV Weights')
    bar3 = plt.bar([i + bar_width * 2 for i in index], weights_df['MVRA Weights'], bar_width, label='MVRA Weights')
    bar4 = plt.bar([i + bar_width * 3 for i in index], weights_df['RP Weights'], bar_width, label='RP Weights')

    plt.xlabel('Cryptos', fontsize=14)
    plt.ylabel('Weights', fontsize=14)
    plt.title('Optimal Portfolio Weights Comparison', fontsize=15)
    plt.xticks([i + bar_width * 1.5 for i in index], weights_df.index, fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc='upper left', fontsize=14)

    filename = 'optimal_weights.png'
    plt.savefig(os.path.join(exportpath, filename))   
    plt.show()


# 3.5
    def create_pie_chart(weights, title, exportpath, filename):
        filtered_weights = weights[weights > 0]
        colors = plt.cm.tab20.colors
        translucent_colors = [(r, g, b, 0.5) for r, g, b in colors]
        plt.figure(figsize=(8, 8))
        plt.pie(filtered_weights, labels=filtered_weights.index, autopct='%1.1f%%', startangle=140, colors=translucent_colors, textprops={'fontsize': 14})
        plt.title(title, fontsize=20, pad=40)
        plt.axis('equal')
        plt.savefig(os.path.join(exportpath, filename))
        plt.show()

    create_pie_chart(weights_df['MSR Weights'], 'Maximum Sharpe Ratio Weights', exportpath, 'pie_chart_msr.png')
    create_pie_chart(weights_df['MV Weights'], 'Minimum Variance Weights', exportpath, 'pie_chart_mv.png')
    create_pie_chart(weights_df['MVRA Weights'], 'Minimum Variance with Risk Aversion Weights', exportpath, 'pie_chart_mvra.png')
    create_pie_chart(weights_df['RP Weights'], 'Risk Parity Weights', exportpath, 'pie_chart_rp.png')
    create_pie_chart(weights_df['MD Weights'], 'Max Div. Weights', exportpath, 'pie_chart_md.png')

 
    portfolio_returns = df.dot(weights_df)


    portfolio_returns.columns = ['MSR Portfolio', 'MV Portfolio', 'MVRA Portfolio', 'RP Portfolio', 'MD Portfolio']


    cumulative_returns = (1 + portfolio_returns).cumprod() - 1

    plt.figure(figsize=(10, 8))
    for column in cumulative_returns.columns:
        plt.plot(cumulative_returns.index, cumulative_returns[column], label=column)
# 3.6
    plt.title('Cumulative Optimal Portfolio Returns Over Time', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Cumulative Return', fontsize=14)
    plt.legend(title='Portfolio', fontsize=14, title_fontsize=14)
    plt.xticks(rotation=45, fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()

    filename = 'cumulative_optimal_portfolio_return.png'
    plt.savefig(os.path.join(exportpath, filename))   
    plt.show()


    descriptive_stats = (portfolio_returns * 100).describe().transpose().round(3)
    descriptive_stats['Sharpe'] = (descriptive_stats['mean'] / descriptive_stats['std']).round(3)

    cols = list(descriptive_stats.columns)
    sharpe_col = cols.pop(cols.index('Sharpe'))
    cols.insert(3, sharpe_col)
    descriptive_stats = descriptive_stats[cols]


    descriptive_stats.to_csv(os.path.join(exportpath, 'optimal_descriptive_statistics.csv'))
    weights_df.to_csv(os.path.join(exportpath, 'optimal_weights.csv'))

    return weights_df, portfolio_returns, descriptive_stats




# 4.1
def project_B_main(api_key):
    df = clean_data()
    mean_returns = df.mean() * 252
    cov_matrix = df.cov() * 252


    results, weights_record, optimal_weights = plot_efficient_frontier(mean_returns, cov_matrix)


    exportpath = '/Users/guyuchen/Desktop/FINS 5545/Project B/data'


    os.makedirs(exportpath, exist_ok=True)


    
    # for i in range(1, 10):
    #     risk = i
    #     print(f"****Risk level {risk}******")
    #     opt_risk = optimize_personal_risk(mean_returns, cov_matrix, risk)
    #     print("***Personal Risk Optimization Weights***")
    #     print(opt_risk['x'].round(3))
    #     print(statistics(opt_risk['x'], mean_returns, cov_matrix).round(3))
        





    weights_df, portfolio_returns, descriptive_stats = plot_additional_graphs(df, mean_returns, cov_matrix, results, weights_record, optimal_weights, exportpath, df.columns)


    model_results = {
        'max_sharpe': optimal_weights[0].tolist(),
        'min_var': optimal_weights[1].tolist(),
        'mean_variance': optimal_weights[2].tolist(),
        'risk_parity': optimal_weights[3].tolist(),
        'max_diversification': optimal_weights[4].tolist()
    }

    with open(os.path.join(exportpath, 'first_optimisation.json'), 'w') as f:
        json.dump(model_results, f)


    with open(os.path.join(exportpath, 'first_optimisation.json')) as f:
        weights = json.load(f)

    # print(weights)

    weight_adj = mean_returns.rank()
    weight_adj = pd.Series([1 / len(weight_adj) if i > len(weight_adj) // 2 else 0 for i in weight_adj], index=mean_returns.index)

    weights_df = pd.DataFrame()

    for t in weights:
        ser = pd.Series(weights[t])
        ser.fillna(0, inplace=True)  # 处理NaN值，替换为0
        new_weights = ser + weight_adj
        new_weights = pd.Series([i / np.sum(new_weights) for i in new_weights], new_weights.index)

        weights_df[t] = new_weights

    # print(weights_df)
    weights_df.to_csv(os.path.join(exportpath, 'adjusted_weights.csv'))


project_B_main(api_key="")








'''
5. sentiment design
'''

# import libraries
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import requests
from datetime import datetime
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os
import json
tqdm.pandas()

# 5.1

df = pd.read_csv(r'/Users/guyuchen/Desktop/FINS 5545/Project A WEEK 8/Project A/SentimentData/Stage2_Text_Data.csv')
if 'Unnamed: 0' in df.columns:
    df.drop(['Unnamed: 0'], axis=1, inplace=True)
df['datem'] = pd.to_datetime(df['datem'])
# print(df)
# print(df['categories'])


analyzer = SentimentIntensityAnalyzer()


def get_sentiment_scores(text):
    if not isinstance(text, str):
        return pd.Series([0, 0, 0, 0], index=['pos', 'neg', 'neu', 'compound'])
    scores = analyzer.polarity_scores(text)
    return pd.Series([scores['pos'], scores['neg'], scores['neu'], scores['compound']],
                      index=['pos', 'neg', 'neu', 'compound'])

# Apply sentiment analysis function to the 'title_clean' column
df[['pos_title', 'neg_title', 'neu_title', 'compound_title']] = \
    df['title_clean'].progress_apply(get_sentiment_scores)
print(df)

df_title = df[['datem', 'title_clean', 'pos_title', 'neg_title', 'neu_title', 'compound_title']]


# 5.2
# Daily Sentiment Index
# SentimentIdx = df_title.groupby("datem")[['compound_title']].mean()
# SentimentIdx.index.rename('date', inplace=True)
# SentimentIdx_Title_Comp = SentimentIdx.copy()

# # Interpret and visualize the results
# # Complex Emotion Index
# ax = SentimentIdx.plot(figsize=(10, 6), linewidth=2, legend=False)

# # Add title and tag
# plt.title('Daily Average Crypto News Sentiment Index -- Compound', fontsize=16)
# plt.xlabel('Date', fontsize=14)
# plt.ylabel('Sentiment Index (Title)', fontsize=14)

# # Add legend
# ax.legend(['Normalized Sentiment'], fontsize=14, title_fontsize=14)


# plt.grid(True, which='both', linestyle='--', linewidth=0.5)


# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)

# plt.tight_layout()
# plt.show()


# 5.3
# # Positive Sentiment Index #
# SentimentIdx_Pos = df_title.groupby("datem")[['pos_title']].mean()

# # Directly plot daily emotion index
# ax = SentimentIdx_Pos.plot(figsize=(10, 6), linewidth=2, legend=False)

# # Add title and tag
# plt.title('Daily Average Crypto News Sentiment Percentage -- Positive', fontsize=16)
# plt.xlabel('Date', fontsize=14)
# plt.ylabel('(% ) Positive Sentiment', fontsize=14)


# ax.legend(['(%) Positive Sentiment'], fontsize=14, title_fontsize=14)


# plt.grid(True, which='both', linestyle='--', linewidth=0.5)


# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)

# plt.tight_layout()
# plt.show()





# 5.4
# Apply the function to the 'body_clean' column and create new columns
df[['pos_body', 'neg_body', 'neu_body', 'compound_body']] = \
    df['body_clean'].progress_apply(get_sentiment_scores)

df_body = df[['datem', 'body_clean', 
              'pos_body', 'neg_body', 'neu_body', 'compound_body']]

# Compound Sentiment Index #
# Plot the sentiment index
SentimentIdx = df_body.groupby("datem")[['compound_body']].mean()
SentimentIdx_Body_Comp = SentimentIdx.copy()

# # # Plot the daily compound sentiment index
# ax = SentimentIdx.plot(figsize=(10, 6), linewidth=2, legend=False)

# # Add title and labels
# plt.title('Daily Average Crypto News Sentiment Index -- Compound', fontsize=16)
# plt.xlabel('Date', fontsize=14)
# plt.ylabel('Sentiment Index (Body)', fontsize=14)

# # Add legend
# ax.legend(['Normalized Sentiment'], fontsize=14, title_fontsize=14)

# # Add grid lines
# plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# # Improve x and y ticks
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)

# # Display the plot
# plt.tight_layout()
# plt.show()


# # # # regression
# import statsmodels.api as sm

# # print(SentimentIdx)
# SentimentIdx = SentimentIdx.pct_change().dropna()
# # print(SentimentIdx)

# # print(ret_df)
# ret_df.index = pd.to_datetime(ret_df.index)
# sent_cryp = SentimentIdx.join(ret_df, how='left')
# # print(sent_cryp)

# pred_rets = []

# for col in sent_cryp.columns[1:]:
#     y = sent_cryp[col]
#     X = sent_cryp['compound_body']
#     model = sm.OLS(y, X)
#     results = model.fit()
#     y_pred = results.predict(X.iloc[-1])  # even mean can be considered
#     pred_rets.append(y_pred)

# # pred_rets = pd.Series([i[0] for i in pred_rets], ret_df.columns)
# print(pred_rets)

# weight_adj = pred_rets.rank()
# weight_adj = pd.Series([1/len(weight_adj) if i > len(weight_adj) // 2 else 0 
#                         for i in weight_adj], ret_df.columns)
# # print(weight_adj)





 # 6.1

import statsmodels.api as sm
import pandas as pd

# Ensure that the index of ret_df is in date format
ret_df.index = pd.to_datetime(ret_df.index)


# print("SentimentIdx before pct_change:")
# print(SentimentIdx.head())


SentimentIdx = SentimentIdx.pct_change().dropna()


# print("SentimentIdx after pct_change:")
# print(SentimentIdx.head())


# print("ret_df:")
# print(ret_df.head())

# Merge sentiment index with return rate data
sent_cryp = SentimentIdx.join(ret_df, how='left').dropna()


# print("sent_cryp after join and dropna:")
# print(sent_cryp.head())

pred_rets = []

# Perform regression analysis on the combined data
for col in sent_cryp.columns[1:]:
    y = sent_cryp[col]
    X = sent_cryp[['compound_body']]  
    X = sm.add_constant(X)  
    model = sm.OLS(y, X)
    results = model.fit()
    y_pred = results.predict(X.iloc[-1].values.reshape(1, -1))  
    pred_rets.append(y_pred[0])  

# Convert the prediction result to Series and set the column name with index ret_df
pred_rets = pd.Series(pred_rets, index=ret_df.columns)
# print(pred_rets)


weights = (pred_rets.rank() > len(pred_rets) // 2).astype(int)
weights = weights / weights.sum()  


# print(weights)





exportpath = '/Users/guyuchen/Desktop/FINS 5545/Project B/data'
os.makedirs(exportpath, exist_ok=True)


with open(os.path.join(exportpath, 'first_optimisation.json')) as f:
    weights = json.load(f)


# print("Original weights from JSON:")
# print(weights)


weights_df = pd.DataFrame(weights)


weights_df = weights_df.applymap(lambda x: 0 if abs(x) < 1e-10 else x)


# print("Processed weights:")
# print(weights_df)


mean_returns = pd.Series(np.random.random(weights_df.shape[0]), index=weights_df.index)


weight_adj = mean_returns.rank()
weight_adj = pd.Series([1 / len(weight_adj) if i > len(weight_adj) // 2 else 0 for i in weight_adj], index=weights_df.index)


adjusted_weights_df = pd.DataFrame()

for column in weights_df:
    ser = weights_df[column]
    new_weights = ser + weight_adj
    new_weights = pd.Series([i / np.sum(new_weights) for i in new_weights], new_weights.index)
    adjusted_weights_df[column] = new_weights


# print("Adjusted weights:")
# print(adjusted_weights_df)


adjusted_weights_df.to_csv(os.path.join(exportpath, 'adjusted_weights.csv'))
# print(f"Adjusted weights CSV file saved at: {os.path.join(exportpath, 'adjusted_weights.csv')}")


