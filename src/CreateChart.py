import numpy as np
import pandas as pd
import mplfinance as fplt 


excel_data = pd.read_excel('Full_Sample.xlsx')
df = pd.DataFrame(excel_data, columns=['DATE', 'OPEN','HIGH','LOW','CLOSE','TREND'])
df.rename(columns={'DATE': 'Date', 'OPEN': 'Open', 'HIGH': 'High', 'LOW': 'Low', 'CLOSE': 'Close'},inplace=True)
Trend_Index = df.loc[df['TREND'].notnull()].index.tolist()
for i in range(0,len(Trend_Index)):
    Trend_df = pd.DataFrame(df.iloc[Trend_Index[i]:Trend_Index[i+1],:])
    Trend_df.index = pd.DatetimeIndex(Trend_df['Date']) 
    fplt.plot(
                Trend_df,
                type='candle',
                title=Trend_df.iloc[0,5],
                ylabel='Price ($)'
            )
    fplt.show()