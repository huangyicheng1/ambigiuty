import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class AmbiguityMeasurement:
    def __init__(self, file_path, bins_count=70, bin_width=0.002):
        """
        初始化函数
        
        参数:
        file_path: 数据文件路径
        bins_count: 区间数量(70个区间，每个宽度0.2%)
        bin_width: 区间宽度(0.002 = 0.2%)
        """
        self.file_path = file_path
        self.bins_count = bins_count
        self.bin_width = bin_width
        # 创建返回率的分割点 (-7%到7%范围)
        self.return_points = np.linspace(-0.07, 0.07, bins_count + 1)
        # 每日至少要有这么多条记录才计算
        self.min_records_per_day = 120  # 假设至少需要2小时的数据
        # 每月至少要有这么多个交易日才计算
        self.min_days_per_month = 10
        
        # 创建输出目录
        self.output_dir = "ambiguity_results"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def load_data(self):
        print("正在加载数据...")
        self.df = pd.read_csv(self.file_path)
        
        # 显示数据的前几行
        print("数据样例:")
        print(self.df.head())
        
        # 转换日期和时间
        print("处理日期和时间...")
        self.df['date'] = pd.to_datetime(self.df['date'].astype(str), format='%Y%m%d')
        self.df['time'] = self.df['time'].astype(str).apply(lambda x: f"{int(x):04d}")
        
        # 创建完整的datetime
        self.df['datetime'] = pd.to_datetime(
            self.df['date'].dt.strftime('%Y-%m-%d') + ' ' + 
            self.df['time'].str.zfill(4).str[:2] + ':' + 
            self.df['time'].str.zfill(4).str[2:],
            format='%Y-%m-%d %H:%M'
        )
        
        # 仅保留交易时间(9:30-11:30, 13:00-15:00)
        def is_trading_hour(dt):
            hour = dt.hour
            minute = dt.minute
            return (
                (hour == 9 and minute >= 30) or 
                (hour == 10) or 
                (hour == 11 and minute <= 30) or
                (hour >= 13 and hour < 15)
            )
        
        trading_mask = self.df['datetime'].apply(is_trading_hour)
        filtered_count = len(self.df) - trading_mask.sum()
        if filtered_count > 0:
            print(f"过滤了{filtered_count}条非交易时段数据")
        
        self.df = self.df[trading_mask]
        self.df.set_index('datetime', inplace=True)
        
        # 按日期排序
        self.df = self.df.sort_index()
        
        # 统计每天的记录数量
        daily_counts = self.df.groupby(self.df.index.date).size()
        print(f"每日记录数量统计:\n{daily_counts.describe()}")
        
        # 可视化每日记录数量分布
        plt.figure(figsize=(12, 6))
        daily_counts.plot(kind='bar', alpha=0.7)
        plt.axhline(y=self.min_records_per_day, color='r', linestyle='--', label=f'最小阈值({self.min_records_per_day}条)')
        plt.title('每日交易记录数量')
        plt.ylabel('记录数')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/daily_records_count.png")
        plt.close()
        
        # 剔除记录太少的交易日
        valid_days = daily_counts[daily_counts >= self.min_records_per_day].index
        invalid_days = len(daily_counts) - len(valid_days)
        if invalid_days > 0:
            print(f"剔除了{invalid_days}个记录不足的交易日")
        
        date_array = np.array([d.date() for d in self.df.index])
        self.df = self.df[np.in1d(date_array, valid_days)]
        
        print(f"数据加载完成，共 {len(self.df)} 条记录")
        print(f"数据时间范围: {self.df.index.min()} 到 {self.df.index.max()}")
        print(f"有效交易日数量: {len(valid_days)}")
    
    def calculate_returns(self):
        print("\n计算收益率...")
        # 按日期分组计算收益率，避免跨日计算
        self.df['date_only'] = self.df.index.date
        returns = []
        
        for date, group in self.df.groupby('date_only'):
            # 对每个交易日内的数据计算收益率
            group = group.sort_index()  # 确保按时间排序
            group_returns = np.log(group['close'] / group['close'].shift(1))
            returns.append(group_returns)
        
        # 合并所有收益率结果
        self.df['return'] = pd.concat(returns)
        self.df.drop('date_only', axis=1, inplace=True)
        
        # 移除首个值和异常值
        self.df['return'] = self.df['return'].replace([np.inf, -np.inf], np.nan)
        self.df = self.df.dropna(subset=['return'])
        
        # 过滤极端收益率
        lower_bound = -0.1  # -10%
        upper_bound = 0.1   # +10%
        extreme_count = ((self.df['return'] < lower_bound) | (self.df['return'] > upper_bound)).sum()
        if extreme_count > 0:
            print(f"检测到{extreme_count}条极端收益率，进行过滤")
        
        self.df = self.df[(self.df['return'] >= lower_bound) & (self.df['return'] <= upper_bound)]
        
        print("\n收益率统计:")
        print(self.df['return'].describe())
        
        # 绘制收益率分布
        plt.figure(figsize=(12, 8))
        
        # 主图: 收益率分布直方图
        plt.subplot(2, 1, 1)
        sns.histplot(self.df['return'], bins=100, kde=True)
        plt.title('一分钟收益率分布')
        plt.xlabel('收益率')
        plt.ylabel('频率')
        plt.grid(True, alpha=0.3)
        
        # 子图: QQ图检验正态性
        plt.subplot(2, 2, 3)
        from scipy import stats
        stats.probplot(self.df['return'], dist="norm", plot=plt)
        plt.title('收益率QQ图')
        
        # 子图: 波动率聚类
        plt.subplot(2, 2, 4)
        rolling_std = self.df.groupby(self.df.index.date)['return'].std()
        rolling_std.plot(kind='line', marker='o', markersize=3)
        plt.title('每日收益率波动率')
        plt.xlabel('日期')
        plt.ylabel('标准差')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/return_analysis.png")
        plt.close()
    
    def calculate_daily_cdfs(self):
        """计算每日的CDF差值"""
        print("\n计算每日CDF差值...")
        self.daily_cdfs = []
        
        # 引入需要的统计函数
        from scipy import stats
        
        # 使用分位数估计来避免异常值影响
        def robust_statistics(returns, q_low=0.05, q_high=0.95):
            """使用截断数据计算稳健的均值和标准差"""
            q_low_val = np.quantile(returns, q_low)
            q_high_val = np.quantile(returns, q_high)
            trimmed = returns[(returns >= q_low_val) & (returns <= q_high_val)]
            return np.mean(trimmed), np.std(trimmed)
        
        for date, group in self.df.groupby(pd.Grouper(freq='D')):
            if not group.empty and len(group) >= self.min_records_per_day:
                returns = group['return'].values
                
                # 计算每日收益率的统计特性
                mu, sigma = np.mean(returns), np.std(returns)
                # 稳健的统计量，可选用
                robust_mu, robust_sigma = robust_statistics(returns)
                
                # 选择使用哪种统计量计算CDF
                # mu, sigma = robust_mu, robust_sigma
                
                if sigma > 0:  # 避免除以零
                    # 计算每个点的CDF值
                    cdfs = norm.cdf(self.return_points, mu, sigma)
                    # 计算CDF差值
                    cdf_diffs = np.diff(cdfs)
                    
                    self.daily_cdfs.append({
                        'date': date,
                        'cdfs': cdfs.tolist(),  # 转换为列表
                        'cdf_diffs': cdf_diffs.tolist(),  # 转换为列表
                        'count': len(returns),
                        'mu': mu,
                        'sigma': sigma,
                        # 添加更多统计量
                        'robust_mu': robust_mu,
                        'robust_sigma': robust_sigma,
                        'skew': stats.skew(returns),
                        'kurt': stats.kurtosis(returns),
                        'min': np.min(returns),
                        'max': np.max(returns),
                        'median': np.median(returns)
                    })
        
        # 转换为DataFrame
        self.daily_cdfs_df = pd.DataFrame(self.daily_cdfs)
        self.daily_cdfs_df.set_index('date', inplace=True)
        print(f"共计算了 {len(self.daily_cdfs)} 个交易日的CDF")
        
        # 绘制每日波动率时间序列
        plt.figure(figsize=(12, 6))
        self.daily_cdfs_df['sigma'].plot(style='o-', alpha=0.7)
        plt.title('每日收益率波动率')
        plt.xlabel('日期')
        plt.ylabel('波动率')
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{self.output_dir}/daily_volatility.png")
        plt.close()
        
        # 保存每日统计量
        stats_df = self.daily_cdfs_df[['count', 'mu', 'sigma', 'robust_mu', 'robust_sigma', 
                                      'skew', 'kurt', 'min', 'max', 'median']]
        stats_df.to_csv(f"{self.output_dir}/daily_statistics.csv")
        print(f"每日统计数据已保存至 {self.output_dir}/daily_statistics.csv")
    
    def calculate_monthly_ambiguity(self):
        print("\n计算月度模糊度...")
        self.monthly_ambiguity = []
        
        # 按月分组
        monthly_groups = self.daily_cdfs_df.groupby(pd.Grouper(freq='M'))
        
        for month, month_group in monthly_groups:
            if not month_group.empty and len(month_group) >= self.min_days_per_month:
                try:
                    # 提取当月所有天的CDF和CDF差值
                    cdfs_list = np.array([np.array(cdfs) for cdfs in month_group['cdfs']])
                    cdf_diffs_list = np.array([np.array(diffs) for diffs in month_group['cdf_diffs']])
                    
                    # 计算第一个区间
                    first_cdfs = cdfs_list[:, 0]
                    E_first = np.mean(first_cdfs)
                    Var_first = np.var(first_cdfs, ddof=1) if len(first_cdfs) > 1 else 0
                    
                    # 计算中间区间
                    E_diffs = np.mean(cdf_diffs_list, axis=0)
                    Var_diffs = np.var(cdf_diffs_list, axis=0, ddof=1) if len(cdf_diffs_list) > 1 else np.zeros_like(E_diffs)
                    middle_sum = np.sum(E_diffs * Var_diffs)
                    
                    # 计算最后一个区间
                    last_cdfs = 1 - cdfs_list[:, -1]
                    E_last = np.mean(last_cdfs)
                    Var_last = np.var(last_cdfs, ddof=1) if len(last_cdfs) > 1 else 0
                    
                    # 计算总模糊度 (标准化系数调整为0.2%)
                    ambiguity = (E_first * Var_first + middle_sum + E_last * Var_last) / (self.bin_width * (1 - self.bin_width))
                    
                    # 保存月度结果
                    self.monthly_ambiguity.append({
                        'month': month,
                        'ambiguity': ambiguity,
                        'trading_days': len(month_group),
                        'avg_mu': month_group['mu'].mean(),
                        'avg_sigma': month_group['sigma'].mean(),
                        'first_component': E_first * Var_first,
                        'middle_component': middle_sum,
                        'last_component': E_last * Var_last,
                        'avg_skew': month_group['skew'].mean(),
                        'avg_kurt': month_group['kurt'].mean()
                    })
                    
                    # 诊断信息 (仅显示第一个月)
                    if len(self.monthly_ambiguity) == 1:
                        print("\n第一个月的诊断信息:")
                        print(f"月份: {month}")
                        print(f"交易日数: {len(month_group)}")
                        print(f"第一个区间: E={E_first:.6f}, Var={Var_first:.6f}, 产品={E_first * Var_first:.6f}")
                        print(f"中间区间和: {middle_sum:.6f}")
                        print(f"最后区间: E={E_last:.6f}, Var={Var_last:.6f}, 产品={E_last * Var_last:.6f}")
                        print(f"标准化因子 1/(w(1-w)): {1/(self.bin_width * (1 - self.bin_width)):.6f}")
                        print(f"最终模糊度: {ambiguity:.6f}")
                except Exception as e:
                    print(f"处理{month}时出错: {str(e)}")
        
        self.monthly_ambiguity_df = pd.DataFrame(self.monthly_ambiguity)
        print(f"\n共计算了 {len(self.monthly_ambiguity)} 个月的模糊度")
        
        if not self.monthly_ambiguity_df.empty:
            print("\n模糊度统计:")
            print(self.monthly_ambiguity_df['ambiguity'].describe())
            
            # 保存月度模糊度数据
            self.monthly_ambiguity_df.to_csv(f"{self.output_dir}/monthly_ambiguity.csv", index=False)
            print(f"月度模糊度数据已保存至 {self.output_dir}/monthly_ambiguity.csv")
    
    def analyze_ambiguity(self):
        """进一步分析模糊度"""
        if self.monthly_ambiguity_df.empty:
            print("没有足够的数据进行分析")
            return
        
        print("\n进行模糊度分析...")
        
        # 计算模糊度与其他指标的相关性
        corr_matrix = self.monthly_ambiguity_df[
            ['ambiguity', 'avg_sigma', 'avg_skew', 'avg_kurt', 'trading_days']
        ].corr()
        
        print("\n相关性矩阵:")
        print(corr_matrix)
        
        # 保存相关性热图
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('模糊度与其他指标的相关性')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/ambiguity_correlations.png")
        plt.close()
        
        # 模糊度的时间序列分析
        try:
            from statsmodels.tsa.stattools import adfuller
            adf_result = adfuller(self.monthly_ambiguity_df['ambiguity'].dropna())
            print("\nADF测试结果:")
            print(f'ADF统计量: {adf_result[0]:.4f}')
            print(f'p值: {adf_result[1]:.4f}')
            print(f'临界值: 1%: {adf_result[4]["1%"]:.4f}, 5%: {adf_result[4]["5%"]:.4f}, 10%: {adf_result[4]["10%"]:.4f}')
            is_stationary = adf_result[1] < 0.05
            print(f'序列{"" if is_stationary else "不"}平稳')
        except Exception as e:
            print(f"ADF测试出错: {str(e)}")
        
        # 绘制模糊度分解组件
        plt.figure(figsize=(12, 10))
        
        # 1. 模糊度时间序列
        plt.subplot(3, 1, 1)
        plt.plot(self.monthly_ambiguity_df['month'], 
                self.monthly_ambiguity_df['ambiguity'], 
                marker='o', 
                linestyle='-', 
                color='blue',
                markersize=4)
        plt.title('月度市场模糊度', fontsize=12)
        plt.xlabel('时间')
        plt.ylabel('模糊度')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 2. 模糊度与波动率的关系
        plt.subplot(3, 1, 2)
        plt.scatter(self.monthly_ambiguity_df['avg_sigma'], 
                  self.monthly_ambiguity_df['ambiguity'],
                  alpha=0.6,
                  color='red')
        
        # 添加回归线
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            self.monthly_ambiguity_df['avg_sigma'], 
            self.monthly_ambiguity_df['ambiguity']
        )
        x = np.linspace(
            self.monthly_ambiguity_df['avg_sigma'].min(),
            self.monthly_ambiguity_df['avg_sigma'].max(),
            100
        )
        plt.plot(x, intercept + slope*x, 'k--', 
                label=f'y = {slope:.4f}x + {intercept:.4f}\n$R^2$ = {r_value**2:.4f}')
        
        plt.title('模糊度 vs 收益率波动率', fontsize=12)
        plt.xlabel('日均波动率')
        plt.ylabel('模糊度')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 3. 模糊度分布
        plt.subplot(3, 1, 3)
        sns.histplot(self.monthly_ambiguity_df['ambiguity'], bins=20, kde=True)
        plt.title('模糊度分布', fontsize=12)
        plt.xlabel('模糊度')
        plt.ylabel('频率')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/ambiguity_analysis.png", dpi=300)
        plt.close()
        
        # 高模糊度时期分析
        high_ambiguity = self.monthly_ambiguity_df[
            self.monthly_ambiguity_df['ambiguity'] > 
            self.monthly_ambiguity_df['ambiguity'].quantile(0.9)
        ].sort_values('ambiguity', ascending=False)
        
        print("\n高模糊度时期 (前10%):")
        print(high_ambiguity[['month', 'ambiguity', 'avg_sigma', 'avg_skew', 'avg_kurt']].head(10))
    
    def plot_results(self):
        """绘制更多分析图表"""
        if self.monthly_ambiguity_df.empty:
            print("没有足够的数据绘制图表")
            return
        
        print("\n绘制更多分析图表...")
        
        # 1. 模糊度分解
        plt.figure(figsize=(12, 6))
        components = pd.DataFrame({
            'First Component': self.monthly_ambiguity_df['first_component'],
            'Middle Component': self.monthly_ambiguity_df['middle_component'],
            'Last Component': self.monthly_ambiguity_df['last_component']
        }, index=self.monthly_ambiguity_df['month'])
        
        components.plot(kind='area', stacked=True, alpha=0.7)
        plt.title('模糊度组成成分')
        plt.xlabel('时间')
        plt.ylabel('贡献值')
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{self.output_dir}/ambiguity_components.png")
        plt.close()
        
        # 2. 模糊度的季节性分析
        self.monthly_ambiguity_df['year'] = pd.DatetimeIndex(self.monthly_ambiguity_df['month']).year
        self.monthly_ambiguity_df['month_num'] = pd.DatetimeIndex(self.monthly_ambiguity_df['month']).month
        
        plt.figure(figsize=(14, 6))
        seasonal = self.monthly_ambiguity_df.pivot_table(
            index='month_num', 
            columns='year', 
            values='ambiguity', 
            aggfunc='mean'
        )
        
        # 绘制热图
        sns.heatmap(seasonal, cmap='YlOrRd', annot=True, fmt='.2f')
        plt.title('模糊度的季节性模式')
        plt.xlabel('年份')
        plt.ylabel('月份')
        plt.savefig(f"{self.output_dir}/ambiguity_seasonality.png")
        plt.close()
    
    def run_analysis(self):
        try:
            start_time = datetime.now()
            print(f"分析开始: {start_time}")
            
            self.load_data()
            self.calculate_returns()
            self.calculate_daily_cdfs()
            self.calculate_monthly_ambiguity()
            self.analyze_ambiguity()
            self.plot_results()
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds() / 60
            print(f"\n分析完成! 用时: {duration:.2f}分钟")
            print(f"结果保存在目录: {os.path.abspath(self.output_dir)}")
            
            # 生成简要报告
            with open(f"{self.output_dir}/analysis_summary.txt", 'w', encoding='utf-8') as f:
                f.write("模糊度分析报告\n")
                f.write("=" * 50 + "\n\n")
                
                f.write(f"分析时间: {end_time}\n")
                f.write(f"数据文件: {self.file_path}\n")
                f.write(f"区间数量: {self.bins_count}, 区间宽度: {self.bin_width*100:.1f}%\n\n")
                
                f.write(f"数据时间范围: {self.df.index.min()} 到 {self.df.index.max()}\n")
                f.write(f"有效交易日数量: {len(self.daily_cdfs)}\n")
                f.write(f"计算的月份数量: {len(self.monthly_ambiguity)}\n\n")
                
                if not self.monthly_ambiguity_df.empty:
                    f.write("月度模糊度统计:\n")
                    desc = self.monthly_ambiguity_df['ambiguity'].describe().to_string()
                    f.write(desc + "\n\n")
                    
                    f.write("高模糊度时期(前5名):\n")
                    top5 = self.monthly_ambiguity_df.sort_values('ambiguity', ascending=False).head(5)
                    for _, row in top5.iterrows():
                        f.write(f"{row['month'].strftime('%Y-%m')}: {row['ambiguity']:.4f}\n")
                    
                    f.write("\n相关性分析:\n")
                    corr = self.monthly_ambiguity_df[['ambiguity', 'avg_sigma']].corr().iloc[0,1]
                    f.write(f"模糊度与波动率相关性: {corr:.4f}\n")
                
                f.write("\n" + "=" * 50 + "\n")
                f.write("分析完成!\n")
            
            print(f"分析报告已保存至 {self.output_dir}/analysis_summary.txt")
            
        except Exception as e:
            print(f"分析过程中出错: {str(e)}")
            import traceback
            traceback.print_exc()


def main():
    # 使用数据文件路径
    file_path = '4.csv'
    
    # 可以根据需要调整参数
    analyzer = AmbiguityMeasurement(
        file_path=file_path,
        bins_count=70,     # 区间数量
        bin_width=0.002    # 区间宽度(0.2%)
    )
    
    analyzer.run_analysis()


if __name__ == "__main__":
    main()