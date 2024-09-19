import yfinance as yf
import pandas as pd
import pandas_datareader.data as web
import streamlit as st
from matplotlib import pyplot as plt
import numpy as np
from pandas import to_datetime
import requests
from pandas_datareader.compat import is_number
from pandas.util._decorators import deprecate_kwarg         
from pandas_datareader.av.forex import AVForexReader          
from pandas_datareader.av.quotes import AVQuotesReader       
from pandas_datareader.av.sector import AVSectorPerformanceReader       
from pandas_datareader.av.time_series import AVTimeSeriesReader         
from distutils.version import LooseVersion                               
from functools import reduce                                             
from io import StringIO                                                  
from urllib.error import HTTPError


import appdirs as ad
ad.user_cache_dir = lambda *args: "/tmp"

class TimeValueofMoney:
    def __init__(self, pv, fv, periods, ticker, start_date, end_date, manual_fed_rate=None, manual_inflation=None):
        self.pv = pv
        self.fv = fv
        self.periods = periods
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.manual_fed_rate = manual_fed_rate  # Accept manual Fed rate input
        self.manual_inflation = manual_inflation  # Accept manual inflation rate input
        
        # Fetch and store data
        self.stock_data = self.get_stock_data()
        self.fed_rate_data = self.get_fed_rate_data()
        
        # Calculate averages
        self.annualized_stock_return = self.calculate_stock_return()
        
        # If manual rate provided, use it; otherwise calculate average Fed rate
        if self.manual_fed_rate is not None:
            self.annualized_fed_rate = self.manual_fed_rate / 100
        else:
            self.annualized_fed_rate = self.calculate_average_fed_rate()

        # Calculate real rate of return based on inflation
        self.annualized_real_rate = self.calculate_real_rate(self.annualized_stock_return / 100)

    def calculate_real_rate(self, nominal_rate):
        """Calculate the real rate of return using the nominal rate and inflation."""
        if self.manual_inflation is not None:
            inflation_rate = self.manual_inflation / 100
        else:
            inflation_rate = 0  # Default to 0 if no inflation is provided

        real_rate = (1 + nominal_rate) / (1 + inflation_rate) - 1
        return real_rate

    def calculate_future_value_simple(self, rate, periods):
        """Calculate FV without compounding (simple interest)."""
        return self.pv * (1 + rate)
    
    def calculate_present_value_simple(self, rate, periods):
        """Calculate PV without compounding (simple interest)."""
        return self.fv / (1 + rate)
    
    def calculate_future_value_compounded(self, rate, periods):
        """Calculate FV with compounding."""
        return self.pv * (1 + rate) ** periods
    
    def calculate_present_value_compounded(self, rate, periods):
        """Calculate PV with compounding."""
        return self.fv / (1 + rate) ** periods
    
    def get_stock_data(self):
        stock = yf.Ticker(self.ticker)
        data = stock.history(start=self.start_date, end=self.end_date, interval='1mo')
        return data['Close']
    
    def calculate_stock_return(self):
        returns = self.stock_data.pct_change().dropna()
        avg_return = returns.mean() * 100  # Convert to percentage
        annualized_return = (1 + avg_return / 100) ** 12 - 1
        return annualized_return * 100  # Convert back to percentage
    
    def get_fed_rate_data(self):
        fed_rate = web.DataReader("FEDFUNDS", "fred", self.start_date, self.end_date)
        return fed_rate
    
    def calculate_average_fed_rate(self):
        avg_rate = self.fed_rate_data.mean().values[0]
        annualized_fed = (1 + avg_rate / 100) ** 12 - 1
        return annualized_fed * 100  # Convert back to percentage
    
    def get_calculations(self):
        # Simple interest calculations based on annualized stock return and Fed rate
        stock_pv_simple = self.calculate_present_value_simple(self.annualized_stock_return / 100, self.periods)
        stock_fv_simple = self.calculate_future_value_simple(self.annualized_stock_return / 100, self.periods)
        fed_pv_simple = self.calculate_present_value_simple(self.annualized_fed_rate, self.periods)
        fed_fv_simple = self.calculate_future_value_simple(self.annualized_fed_rate, self.periods)
        
        # Compound interest calculations using annualized stock return
        stock_fv_compounded = self.calculate_future_value_compounded(self.annualized_stock_return / 100, self.periods)
        stock_pv_compounded = self.calculate_present_value_compounded(self.annualized_stock_return / 100, self.periods)
        
        # Compound interest calculations using annualized Fed rate
        fed_fv_compounded = self.calculate_future_value_compounded(self.annualized_fed_rate, self.periods)
        fed_pv_compounded = self.calculate_present_value_compounded(self.annualized_fed_rate, self.periods)

        # Real rate calculations
        real_pv_simple = self.calculate_present_value_simple(self.annualized_real_rate, self.periods)
        real_fv_simple = self.calculate_future_value_simple(self.annualized_real_rate, self.periods)
        real_fv_compounded = self.calculate_future_value_compounded(self.annualized_real_rate, self.periods)
        real_pv_compounded = self.calculate_present_value_compounded(self.annualized_real_rate, self.periods)

        return {
            'Future Value - Stock Return': stock_fv_simple,
            'Present Value - Stock Return': stock_pv_simple,
            'Future Value - Fed Rate': fed_fv_simple,
            'Present Value - Fed Rate': fed_pv_simple,
            'Future Value (Compounded) - Stock Return': stock_fv_compounded,
            'Present Value (Compounded) - Stock Return': stock_pv_compounded,
            'Future Value (Compounded) - Fed Rate': fed_fv_compounded,
            'Present Value (Compounded) - Fed Rate': fed_pv_compounded,
            'Future Value (Simple) - Real Rate': real_fv_simple,
            'Present Value (Simple) - Real Rate': real_pv_simple,
            'Future Value (Compounded) - Real Rate': real_fv_compounded,
            'Present Value (Compounded) - Real Rate': real_pv_compounded,
            'Annualized Stock Return (%)': self.annualized_stock_return,
            'Annualized Fed Rate (%)': self.annualized_fed_rate,
            'Real Rate of Return (%)': self.annualized_real_rate * 100  # Convert to percentage for display
        }

# Streamlit app code
def main():
    st.title("Time Value of Money Calculator")

    # User inputs
    pv = st.number_input("Present Value", value=1000)
    fv = st.number_input("Future Value", value=1500)
    periods = st.number_input("Number of Periods (Years)", value=10)
    ticker = st.text_input("Stock Ticker", value='AAPL')
    start_date = st.date_input("Start Date", value=pd.to_datetime('2020-01-01'))
    end_date = st.date_input("End Date", value=pd.to_datetime('2024-01-01'))

    # Manual Fed rate and inflation rate inputs
    manual_fed_rate = st.number_input("Manual Fed Rate (%) (Leave blank to use fetched rate)", value=0.0)
    manual_inflation = st.number_input("Manual Inflation Rate (%) (Leave blank for 0)", value=0.0)

    if manual_fed_rate == 0.0:
        manual_fed_rate = None  # If no manual input, we will use fetched rate
    if manual_inflation == 0.0:
        manual_inflation = None  # Default to 0 if no inflation is provided

    if st.button("Calculate"):
        # Create an instance of the TimeValueofMoney class
        tvom = TimeValueofMoney(pv, fv, periods, ticker, start_date, end_date, manual_fed_rate, manual_inflation)

        # Get the calculations
        calculations = tvom.get_calculations()

        # Display the results
        for key, value in calculations.items():
            st.write(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")

if __name__ == "__main__":
    main()