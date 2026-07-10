import yfinance as yf
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

class StockDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.data) - self.seq_len
        
    def __getitem__(self, idx):
        # Extract a temporal sequence of length seq_len
        seq = self.data[idx : idx + self.seq_len]
        return torch.FloatTensor(seq)

def get_data(ticker="TSLA", seq_len=24, batch_size=32):
    """
    Downloads historical data from yfinance, calculates daily returns,
    and normalizes them between -1 and 1.
    """
    print(f"Downloading 5 years of data for {ticker}...")
    # Fetch 5 years of daily data
    df = yf.download(ticker, period="5y", interval="1d")
    
    # Use 'Close' price to learn the temporal dynamics of price movements
    prices = df['Close'].values.reshape(-1, 1)
    
    # Calculate daily returns: (P_t - P_{t-1}) / P_{t-1}
    # Returns are stationary, which makes it much easier for GANs to model 
    # than raw prices that drift indefinitely.
    returns = np.diff(prices, axis=0) / prices[:-1]
    
    # Normalize returns between -1 and 1 using MinMaxScaler
    # This aligns the data scale with the Generator's Tanh activation output
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_returns = scaler.fit_transform(returns)
    
    dataset = StockDataset(scaled_returns, seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    return dataloader, scaler, scaled_returns
