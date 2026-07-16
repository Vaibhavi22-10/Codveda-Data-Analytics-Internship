import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Load dataset
df = pd.read_csv("2) Stock Prices Data Set.csv")

# Convert date column
df['date'] = pd.to_datetime(df['date'])

# Select AAPL stock
df = df[df['symbol'] == 'AAPL']

# Sort by date
df = df.sort_values('date')

# Set date as index
df.set_index('date', inplace=True)

# Closing Price Plot
plt.figure(figsize=(10,5))
plt.plot(df['close'])
plt.title("AAPL Closing Price")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.savefig("closing_price.png")
plt.show()

# 7-Day Moving Average
df['MA7'] = df['close'].rolling(window=7).mean()

plt.figure(figsize=(10,5))
plt.plot(df['close'], label='Close Price')
plt.plot(df['MA7'], label='7 Day Moving Average')
plt.legend()
plt.title("AAPL Moving Average")
plt.savefig("moving_average.png")
plt.show()

# Time Series Decomposition
decomposition = seasonal_decompose(
    df['close'],
    model='additive',
    period=30
)

decomposition.plot()
plt.savefig("decomposition.png")
plt.show()

print("Time Series Analysis Completed Successfully!")