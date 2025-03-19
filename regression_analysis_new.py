import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_data():
    # 读取模糊度数据
    ambiguity_data = pd.read_csv('monthly_ambiguity.csv')
    ambiguity_data['month'] = pd.to_datetime(ambiguity_data['month'])
    
    # 读取波动率数据
    volatility_data = pd.read_csv('monthly_volatility.csv')
    volatility_data['datetime'] = pd.to_datetime(volatility_data['datetime'])
    
    return ambiguity_data, volatility_data

def regression_analysis(data):
    """
    进行回归分析
    """
    # 准备数据
    X = sm.add_constant(data['ambiguity'])
    y = data['volatility']
    
    # 运行回归
    model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 12})  # 使用HAC标准误
    
    # 异方差性检验
    from statsmodels.stats.diagnostic import het_white
    white_test = het_white(model.resid, X)
    
    # 自相关检验
    from statsmodels.stats.stattools import durbin_watson
    dw_stat = durbin_watson(model.resid)
    
    return model, white_test, dw_stat

def plot_results(data, model):
    """
    可视化回归结果
    """
    plt.style.use('seaborn')
    fig = plt.figure(figsize=(15, 10))
    
    # 1. 散点图和回归线
    ax1 = plt.subplot(221)
    sns.regplot(x='ambiguity', y='volatility', data=data, ax=ax1)
    ax1.set_title('Ambiguity vs Volatility')
    
    # 添加回归方程和R方
    equation = f'y = {model.params[1]:.4f}x + {model.params[0]:.4f}'
    r2 = f'R² = {model.rsquared:.4f}'
    p_value = f'p-value = {model.pvalues[1]:.4f}'
    ax1.text(0.05, 0.95, f'{equation}\n{r2}\n{p_value}', 
             transform=ax1.transAxes, 
             verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.8))
    
    # 2. 残差图
    ax2 = plt.subplot(222)
    sns.scatterplot(x=model.fittedvalues, y=model.resid, ax=ax2)
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_title('Residuals vs Fitted')
    ax2.set_xlabel('Fitted values')
    ax2.set_ylabel('Residuals')
    
    # 3. QQ图
    ax3 = plt.subplot(223)
    stats.probplot(model.resid, dist="norm", plot=ax3)
    ax3.set_title('Q-Q Plot')
    
    # 4. 残差时间序列图
    ax4 = plt.subplot(224)
    plt.plot(data.index, model.resid)
    ax4.axhline(y=0, color='r', linestyle='--')
    ax4.set_title('Residuals over Time')
    
    plt.tight_layout()
    plt.show()

def main():
    print("1. 加载数据...")
    ambiguity_data, volatility_data = load_data()
    
    print("2. 合并数据...")
    merged_data = pd.merge(
        ambiguity_data,
        volatility_data,
        left_on='month',
        right_on='datetime',
        how='inner'
    )
    
    print("3. 进行回归分析...")
    model, white_test, dw_stat = regression_analysis(merged_data)
    
    print("\n回归分析结果:")
    print(model.summary())
    
    print("\n异方差性检验 (White Test):")
    print(f"统计量: {white_test[0]}")
    print(f"p值: {white_test[1]}")
    
    print("\nDurbin-Watson检验:")
    print(f"DW统计量: {dw_stat}")
    
    print("\n4. 生成可视化图表...")
    plot_results(merged_data, model)
    
    print("\n5. 保存结果...")
    results = pd.DataFrame({
        'month': merged_data['month'],
        'ambiguity': merged_data['ambiguity'],
        'volatility': merged_data['volatility'],
        'fitted_values': model.fittedvalues,
        'residuals': model.resid
    })
    results.to_csv('regression_results.csv', index=False)
    print("回归分析结果已保存到 regression_results.csv")

if __name__ == "__main__":
    main() 