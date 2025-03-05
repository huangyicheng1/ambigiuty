import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from datetime import datetime

class AmbiguityMeasurement:
    """
    市场模糊度测量类
    基于收益率分布的不确定性来测量市场模糊度
    使用CDF（累积分布函数）的方差来衡量市场的不确定性
    
    主要步骤：
    1. 计算每日对数收益率
    2. 假设收益率服从正态分布，计算CDF
    3. 计算CDF的方差来衡量市场的不确定性
    4. 通过月度汇总得到市场模糊度指标
    """
    def __init__(self, file_path, bins_count=70, bin_width=0.001):
        """
        初始化模糊度测量器
        :param file_path: 数据文件路径，包含 date, time, open, close 等列
        :param bins_count: 收益率区间的数量，用于划分CDF的计算区间
        :param bin_width: 每个区间的宽度，用于标准化模糊度计算
        """
        self.file_path = file_path
        self.bins_count = bins_count
        self.bin_width = bin_width
        # 创建返回率的分割点，范围从-3.5%到3.5%
        # 这个范围覆盖了大多数正常市场条件下的收益率
        self.return_points = np.linspace(-0.035, 0.035, bins_count + 1)
    
    def load_data(self):
        """
        加载并预处理数据
        将日期和时间合并为datetime格式，便于后续时间序列分析
        
        数据要求：
        - date列：格式为YYYYMMDD的字符串
        - time列：格式为HHMM的字符串
        - open列：开盘价
        - close列：收盘价
        """
        print("正在加载数据...")
        self.df = pd.read_csv(self.file_path)
        
        # 转换日期和时间格式
        self.df['date'] = pd.to_datetime(self.df['date'].astype(str), format='%Y%m%d')
        self.df['time'] = self.df['time'].astype(str).apply(lambda x: f"{int(x):04d}")
        
        # 创建完整的datetime索引
        self.df['datetime'] = pd.to_datetime(
            self.df['date'].dt.strftime('%Y-%m-%d') + ' ' + 
            self.df['time'].str.zfill(4).str[:2] + ':' + 
            self.df['time'].str.zfill(4).str[2:],
            format='%Y-%m-%d %H:%M'
        )
        
        self.df.set_index('datetime', inplace=True)
        print(f"数据加载完成，共 {len(self.df)} 条记录")
        print(f"数据时间范围: {self.df.index.min()} 到 {self.df.index.max()}")
        
    def calculate_returns(self):
        """
        计算对数收益率
        使用公式: r = ln(close/open)
        其中：
        - r 是对数收益率
        - close 是收盘价
        - open 是开盘价
        
        对数收益率的优点：
        1. 更好的统计特性
        2. 可加性
        3. 更接近正态分布
        """
        print("\n计算收益率...")
        self.df['return'] = np.log(self.df['close'] / self.df['open'])
        # 处理异常值
        self.df['return'] = self.df['return'].replace([np.inf, -np.inf], np.nan)
        self.df = self.df.dropna(subset=['return'])
        
        print("\n收益率统计:")
        print(self.df['return'].describe())
        
    def calculate_daily_cdfs(self):
        """
        计算每日的CDF（累积分布函数）差值
        使用正态分布假设：
        - 假设每日收益率服从正态分布 N(μ, σ²)
        - 计算每个时间点的CDF值
        - 计算相邻CDF值的差值
        
        计算步骤：
        1. 对每个交易日的数据：
           - 计算收益率均值和标准差
           - 计算CDF值：F(x) = Φ((x-μ)/σ)
           - 计算CDF差值：ΔF(x) = F(x+Δx) - F(x)
        2. 保存每日的CDF和差值数据
        """
        print("\n计算每日CDF差值...")
        self.daily_cdfs = []
        
        for date, group in self.df.groupby(pd.Grouper(freq='D')):
            if not group.empty and len(group) > 10:  # 确保有足够的数据点
                returns = group['return'].values
                mu = np.mean(returns)  # 计算均值
                sigma = np.std(returns)  # 计算标准差
                
                if sigma > 0:  # 避免除以零
                    # 计算每个点的CDF值：F(x) = Φ((x-μ)/σ)
                    # 其中Φ是标准正态分布的CDF
                    cdfs = norm.cdf(self.return_points, mu, sigma)
                    # 计算CDF差值：ΔF(x) = F(x+Δx) - F(x)
                    cdf_diffs = np.diff(cdfs)
                    
                    self.daily_cdfs.append({
                        'date': date,
                        'cdfs': cdfs.tolist(),
                        'cdf_diffs': cdf_diffs.tolist(),
                        'count': len(returns),
                        'mu': mu,
                        'sigma': sigma
                    })
        
        self.daily_cdfs_df = pd.DataFrame(self.daily_cdfs)
        print(f"共计算了 {len(self.daily_cdfs)} 个交易日的CDF")
        
    def calculate_monthly_ambiguity(self):
        """
        计算月度模糊度
        模糊度计算公式：
        A = (E[F(x₁)]Var[F(x₁)] + Σ(E[ΔF(x)]Var[ΔF(x)]) + E[1-F(xₙ)]Var[1-F(xₙ)]) / (w(1-w))
        
        其中：
        - F(x) 是CDF值
        - ΔF(x) 是CDF差值
        - E[·] 表示期望
        - Var[·] 表示方差
        - w 是区间宽度
        - x₁ 是第一个区间点
        - xₙ 是最后一个区间点
        
        模糊度的含义：
        1. 第一个项：衡量分布左尾的不确定性
        2. 中间项：衡量分布中间部分的不确定性
        3. 最后项：衡量分布右尾的不确定性
        4. 分母项：标准化因子，使不同区间的模糊度可比
        """
        print("\n计算月度模糊度...")
        self.monthly_ambiguity = []
        
        for month, month_group in self.daily_cdfs_df.groupby(
            pd.Grouper(key='date', freq='M')):
            
            if not month_group.empty and len(month_group) > 5:  # 确保有足够的数据点
                # 提取当月所有天的CDF和CDF差值
                cdfs_list = np.array([np.array(cdfs) for cdfs in month_group['cdfs']])
                cdf_diffs_list = np.array([np.array(diffs) for diffs in month_group['cdf_diffs']])
                
                # 计算第一个区间的期望和方差
                first_cdfs = cdfs_list[:, 0]
                E_first = np.mean(first_cdfs)  # E[F(x₁)]
                Var_first = np.var(first_cdfs, ddof=1) if len(first_cdfs) > 1 else 0  # Var[F(x₁)]
                
                # 计算中间区间的期望和方差
                E_diffs = np.mean(cdf_diffs_list, axis=0)  # E[ΔF(x)]
                Var_diffs = np.var(cdf_diffs_list, axis=0, ddof=1) if len(cdf_diffs_list) > 1 else np.zeros_like(E_diffs)  # Var[ΔF(x)]
                middle_sum = np.sum(E_diffs * Var_diffs)  # Σ(E[ΔF(x)]Var[ΔF(x)])
                
                # 计算最后一个区间的期望和方差
                last_cdfs = 1 - cdfs_list[:, -1]
                E_last = np.mean(last_cdfs)  # E[1-F(xₙ)]
                Var_last = np.var(last_cdfs, ddof=1) if len(last_cdfs) > 1 else 0  # Var[1-F(xₙ)]
                
                # 计算总模糊度：A = (E[F(x₁)]Var[F(x₁)] + Σ(E[ΔF(x)]Var[ΔF(x)]) + E[1-F(xₙ)]Var[1-F(xₙ)]) / (w(1-w))
                ambiguity = (E_first * Var_first + middle_sum + E_last * Var_last) / (self.bin_width * (1 - self.bin_width))
                
                if len(self.monthly_ambiguity) == 0:  # 打印第一个月的详细信息
                    print("\n第一个月的诊断信息:")
                    print(f"第一个区间: E={E_first:.6f}, Var={Var_first:.6f}")
                    print(f"中间区间和: {middle_sum:.6f}")
                    print(f"最后区间: E={E_last:.6f}, Var={Var_last:.6f}")
                    print(f"标准化因子 1/(w(1-w)): {1/(self.bin_width * (1 - self.bin_width)):.6f}")
                    print(f"最终模糊度: {ambiguity:.6f}")
                
                self.monthly_ambiguity.append({
                    'month': month,
                    'ambiguity': ambiguity,
                    'trading_days': len(month_group),
                    'avg_mu': month_group['mu'].mean(),
                    'avg_sigma': month_group['sigma'].mean()
                })
        
        self.monthly_ambiguity_df = pd.DataFrame(self.monthly_ambiguity)
        print(f"\n共计算了 {len(self.monthly_ambiguity)} 个月的模糊度")
        print("\n模糊度统计:")
        print(self.monthly_ambiguity_df['ambiguity'].describe())
        
    def plot_results(self):
        """
        绘制分析图表
        包括：
        1. 月度模糊度时间序列：展示模糊度随时间的变化趋势
        2. 模糊度与波动率的关系散点图：分析模糊度与市场波动的关系
        3. 模糊度分布直方图：了解模糊度的统计分布特征
        """
        print("\n绘制分析图表...")
        
        plt.rcParams['figure.figsize'] = [15, 12]
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.grid'] = True
        
        fig = plt.figure()
        
        # 1. 模糊度时间序列
        ax1 = plt.subplot(3, 1, 1)
        ax1.plot(self.monthly_ambiguity_df['month'], 
                self.monthly_ambiguity_df['ambiguity'], 
                marker='o', 
                linestyle='-', 
                color='blue',
                markersize=4)
        ax1.set_title('月度市场模糊度', fontsize=12, pad=10)
        ax1.set_xlabel('时间')
        ax1.set_ylabel('模糊度')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # 2. 模糊度与波动率的关系
        ax2 = plt.subplot(3, 1, 2)
        ax2.scatter(self.monthly_ambiguity_df['avg_sigma'], 
                   self.monthly_ambiguity_df['ambiguity'],
                   alpha=0.6,
                   color='red')
        ax2.set_title('模糊度与收益率波动率的关系', fontsize=12, pad=10)
        ax2.set_xlabel('平均日波动率')
        ax2.set_ylabel('模糊度')
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # 3. 模糊度分布
        ax3 = plt.subplot(3, 1, 3)
        ax3.hist(self.monthly_ambiguity_df['ambiguity'], 
                bins=20, 
                alpha=0.7,
                color='green',
                edgecolor='black')
        ax3.set_title('模糊度分布', fontsize=12, pad=10)
        ax3.set_xlabel('模糊度')
        ax3.set_ylabel('频率')
        ax3.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('ambiguity_analysis.png', dpi=300, bbox_inches='tight')
        print("图表已保存为 'ambiguity_analysis.png'")
        
    def save_results(self):
        """
        保存计算结果到CSV文件
        包含以下列：
        - month: 月份
        - ambiguity: 模糊度值
        - trading_days: 交易天数
        - avg_mu: 平均收益率
        - avg_sigma: 平均波动率
        """
        print("\n保存计算结果...")
        self.monthly_ambiguity_df.to_csv('monthly_ambiguity.csv', 
                                       index=False, 
                                       encoding='utf-8-sig')
        print("结果已保存为 'monthly_ambiguity.csv'")
        
    def run_analysis(self):
        """
        运行完整的分析流程
        按顺序执行：
        1. 加载数据
        2. 计算收益率
        3. 计算每日CDF
        4. 计算月度模糊度
        5. 绘制图表
        6. 保存结果
        """
        self.load_data()
        self.calculate_returns()
        self.calculate_daily_cdfs()
        self.calculate_monthly_ambiguity()
        self.plot_results()
        self.save_results()

def main():
    """
    主函数
    创建模糊度测量器实例并运行分析
    """
    analyzer = AmbiguityMeasurement('4.csv')
    analyzer.run_analysis()

if __name__ == "__main__":
    main() 