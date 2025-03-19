import pandas as pd
import numpy as np

def load_stock_data():
    # 读取股票数据
    stock_data = pd.read_csv('4.csv')
    # 将date和time合并为datetime
    stock_data['datetime'] = pd.to_datetime(
        stock_data['date'].astype(str) + 
        stock_data['time'].astype(str).str.zfill(4),
        format='%Y%m%d%H%M'
    )
    return stock_data

def calculate_monthly_volatility(stock_data):
    """
    计算月度波动率
    """
    # 获取每天的最后一个价格作为日收盘价
    daily_data = stock_data.groupby(stock_data['datetime'].dt.date).last()
    
    # 计算日收益率
    daily_data['returns'] = np.log(daily_data['close'] / daily_data['close'].shift(1))
    
    # 按月计算波动率
    monthly_vol = daily_data.groupby(pd.Grouper(freq='M', key='datetime')).agg({
        'returns': lambda x: np.std(x) * np.sqrt(252)  # 年化
    }).reset_index()
    
    monthly_vol.rename(columns={'returns': 'volatility'}, inplace=True)
    monthly_vol['datetime'] = pd.to_datetime(monthly_vol['datetime'])
    
    return monthly_vol

def main():
    print("1. 加载股票数据...")
    stock_data = load_stock_data()
    
    print("2. 计算月度波动率...")
    volatility_data = calculate_monthly_volatility(stock_data)
    
    print("3. 保存结果...")
    volatility_data.to_csv('monthly_volatility.csv', index=False)
    print("月度波动率已保存到 monthly_volatility.csv")

if __name__ == "__main__":
    main() 