from typing import List, Tuple
from datetime import datetime, timedelta
import yfinance as yf
import talib as ta
from pandas_datareader import data as pdr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix
from cvxopt.solvers import qp


class StockFilter:
    def __init__(self, stocks: List[str], start_date: str, end_date: str):
        self.stocks = stocks
        self.start_date = start_date
        self.end_date = end_date

    def get_technical_indicators(self, stock: str) -> pd.DataFrame:
        # Get Stock Price Data
        yf.pdr_override()
        data = pdr.get_data_yahoo(stock, start=self.start_date, end=self.end_date)
        # Create 20-day Moving Average
        data['MA20'] = ta.SMA(data['Close'], timeperiod=20)
        # Create 50-day Moving Average
        data['MA50'] = ta.SMA(data['Close'], timeperiod=50)
        # Calculate RSI
        data['RSI'] = ta.RSI(data['Close'], timeperiod=14)
        return data

    def get_fundamentals(self, stock: str) -> Tuple[float, float]:
        ticker = yf.Ticker(stock)
        # Get P/E Ratio
        pe_ratio = ticker.info['trailingPE']
        # Get Debt to equity Ratio
        debt_to_equity = ticker.info['debtToEquity']
        return pe_ratio, debt_to_equity

    def filter(self) -> List[str]:
        filtered_stocks = []
        for stock in self.stocks:
            try:
                # Get Technical Indicators for the stock
                data = self.get_technical_indicators(stock)
                # Check if the latest Close price is above both Moving Averages
                if data['Close'].iloc[-1] > data['MA20'].iloc[-1] and \
                        data['Close'].iloc[-1] > data['MA50'].iloc[-1]:
                    # Check if the RSI is above 50
                    if data['RSI'].iloc[-1] > 50:
                        # Get Fundamentals
                        pe_ratio, debt_to_equity = self.get_fundamentals(stock)
                        # Check if P/E Ratio is below 25 and Debt to Asset ratio is below 0.5
                        if pe_ratio < 50 and debt_to_equity < 50:
                            filtered_stocks.append(stock)
            except:
                pass
        return filtered_stocks


class Portfolio:
    def __init__(self, filtered_stocks: List[str], start_date: str, end_date: str, initial_capital: float):
        self.stocks = filtered_stocks
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.data = self.get_data()

    def get_data(self) -> pd.DataFrame:
        # Get Stock Price Data
        yf.pdr_override()
        data = pdr.get_data_yahoo(self.stocks, start=self.start_date, end=self.end_date)
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
        data = data.dropna()
        return data

    def get_returns(self) -> pd.DataFrame:
        # Calculate Daily Returns
        self.data['Returns'] = self.data['Close'].pct_change()
        self.data = self.data.dropna()
        return self.data

    def rebalance_portfolio(self, weights: dict) -> None:
        self.data['Weight'] = 0
        for stock, weight in weights.items():
            self.data.loc[self.data.index, (stock, 'Weight')] = weight

        # Calculate Daily Returns weighted by weights
        self.data['Weighted Returns'] = self.data['Weight'] * self.get_returns()['Returns']
        self.data = self.data.dropna()

        # Calculate Aggregate Portfolio Returns
        self.data['Portfolio Returns'] = self.data.groupby(level=0)['Weighted Returns'].sum()
        self.data = self.data.dropna()

    def get_volatility(self) -> None:
        # Calculate 20-day Historical Volatility
        self.data['HV20'] = np.sqrt(
            self.data.groupby(level=0)['Returns'].apply(lambda x: x.rolling(window=20).var(ddof=0)))

    def get_optimal_weights(self) -> dict:
        # Find Optimal Weights using Mean Variance Optimization
        self.get_volatility()
        stock_volatility = self.data.groupby(level=0)['HV20'].mean()
        covariance_matrix = self.data['Returns'].unstack(0).cov()
        mean_returns = self.data['Returns'].unstack(0).mean()
        weights = self.optimal_weights(mean_returns, covariance_matrix, stock_volatility)
        return weights

    def optimal_weights(self, mean_returns: pd.Series, covariance_matrix: pd.DataFrame, stock_volatility: pd.Series) -> dict:
        # Mean Variance Optimization with Volatility Constraint
        mean_return_array = np.array(mean_returns)
        cov_matrix = np.array(covariance_matrix)
        stock_volatility_array = np.array(stock_volatility)

        n = len(mean_returns)
        P = matrix(cov_matrix)
        q = matrix(np.zeros((n, 1)))
        A = matrix(np.vstack((mean_return_array.reshape(1, -1), np.ones((1, n))))) # stacking mean return and all ones rows
        b = matrix(np.array([0.02, 1])) # lower and upper bounds on the expected returns
        G = matrix(-np.diag(stock_volatility_array)) # diagonal matrix with elements = -1/volatility
        h = matrix(np.zeros((n, 1)))

        sol = qp(P, q, G, h, A, b)
        weights = list(sol['x'])
        weights = [round(w, 3) for w in weights]
        weights_dict = dict(zip(mean_returns.index, weights))
        return weights_dict

    def backtest(self) -> None:
        self.get_returns()

        # Rebalance Portfolio every week
        start_date = self.data.index.min() + timedelta(days=7)
        end_date = self.data.index.max()
        while start_date <= end_date:
            # Select only the relevant data
            data_week = self.data.loc[(self.data.index >= start_date-timedelta(days=7)) &
                                      (self.data.index < start_date)]
            weights = self.get_optimal_weights()
            self.rebalance_portfolio(weights)
            start_date += timedelta(days=7)

        # Calculate Cumulative Portfolio Returns
        self.data['Cumulative Returns'] = (1 + self.data['Portfolio Returns']).cumprod() - 1
        return self.data


def plot_cumulative_returns(data: pd.DataFrame) -> None:
    plt.plot(data['Cumulative Returns'])
    plt.show()


if __name__ == "__main__":
    start_date = '2022-12-31'
    end_date = '2023-05-05'
    initial_capital = 1000000

    # Filter Stocks based on Technical Indicators and Fundamentals
    stocks = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
    stocks = list(stocks['Symbol'].values)
    filtered_stocks = StockFilter(stocks, start_date, end_date).filter()

    # Create Portfolio
    portfolio = Portfolio(filtered_stocks, start_date, end_date, initial_capital)
    portfolio_data = portfolio.backtest()

    # Plot Cumulative Returns
    plot_cumulative_returns(portfolio_data)

