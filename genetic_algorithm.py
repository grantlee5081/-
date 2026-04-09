"""
genetic_algorithm.py
────────────────────
遺傳演算法（Genetic Algorithm, GA）引擎

職責：
  1. 自動搜尋最佳技術因子參數組合（染色體編碼）
  2. 對股票池進行回測，計算適應度
  3. 透過選擇、交叉、變異迭代演化，收斂至近似最優解
  4. 輸出最佳參數，並對股票池各標的打分排序

演化流程
────────
初始化種群
    ↓
[循環 N 代]
  1. 評估適應度（回測每條染色體）
  2. 精英保留（前 elite_ratio 的個體直接進入下一代）
  3. 錦標賽選擇（tournament selection）
  4. 均勻交叉（uniform crossover）
  5. 高斯變異（Gaussian mutation）
    ↓
返回最佳染色體（策略參數）

染色體結構（12 個基因）
────────────────────────
index | 參數名稱        | 範圍
------+-----------------+----------
  0   | ma_short        | [5, 30]   短期均線周期
  1   | ma_long         | [20,120]  長期均線周期
  2   | ma_weight       | [0, 1]    MA 信號權重
  3   | rsi_period      | [7, 21]   RSI 計算周期
  4   | rsi_ob          | [65, 85]  超買閾值
  5   | rsi_os          | [15, 35]  超賣閾值
  6   | rsi_weight      | [0, 1]    RSI 信號權重
  7   | bb_period       | [10, 30]  布林帶周期
  8   | bb_std          | [1.5,3.0] 布林帶標準差倍數
  9   | bb_weight       | [0, 1]    BB 信號權重
 10   | buy_threshold   | [0.1,0.5] 買入信號閾值
 11   | sell_threshold  | [-0.5,-0.1] 賣出信號閾值
"""

import random
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from technical_factors import TechnicalFactors
from performance_metrics import PerformanceMetrics


# ── 參數邊界定義（有序字典確保索引對齊） ──────────────────────
PARAM_BOUNDS: dict[str, tuple] = {
    'ma_short':       (5,    30,   'int'),
    'ma_long':        (20,   120,  'int'),
    'ma_weight':      (0.0,  1.0,  'float'),
    'rsi_period':     (7,    21,   'int'),
    'rsi_ob':         (65.0, 85.0, 'float'),
    'rsi_os':         (15.0, 35.0, 'float'),
    'rsi_weight':     (0.0,  1.0,  'float'),
    'bb_period':      (10,   30,   'int'),
    'bb_std':         (1.5,  3.0,  'float'),
    'bb_weight':      (0.0,  1.0,  'float'),
    'buy_threshold':  (0.1,  0.5,  'float'),
    'sell_threshold': (-0.5, -0.1, 'float'),
}

N_GENES = len(PARAM_BOUNDS)
PARAM_KEYS = list(PARAM_BOUNDS.keys())


@dataclass
class Individual:
    """
    種群中的一個個體（一組策略參數）。

    genes   : 長度 N_GENES 的 numpy 陣列，儲存原始浮點數值
    fitness : 適應度分數，初始化為 -999
    """
    genes: np.ndarray
    fitness: float = field(default=-999.0)


class GeneticAlgorithm:
    """
    遺傳演算法引擎。

    使用範例
    --------
    ga = GeneticAlgorithm(population_size=50, generations=50)
    best_params = ga.evolve(stock_data_dict, verbose=True)
    scores = ga.score_stocks(stock_data_dict, best_params)
    """

    def __init__(self,
                 population_size: int = 50,
                 generations: int = 50,
                 crossover_rate: float = 0.80,
                 mutation_rate: float = 0.15,
                 tournament_size: int = 5,
                 elite_ratio: float = 0.10):
        """
        Parameters
        ----------
        population_size : 每代種群數量
        generations     : 最大演化代數
        crossover_rate  : 兩親本產生子代的機率
        mutation_rate   : 每個基因位點發生變異的機率
        tournament_size : 錦標賽每場參賽個體數
        elite_ratio     : 精英保留比例（直接複製至下一代）
        """
        self.pop_size = population_size
        self.generations = generations
        self.cr = crossover_rate
        self.mr = mutation_rate
        self.tourn_size = tournament_size
        self.elite_n = max(1, int(population_size * elite_ratio))

        self.best_params: dict = {}
        self.fitness_history: list[dict] = []

    # ── 染色體編解碼 ──────────────────────────────────────────

    def _random_individual(self) -> Individual:
        """
        隨機生成一個合法的個體。

        合法性約束：ma_short < ma_long（短均線周期必須小於長均線）
        """
        genes = np.zeros(N_GENES)
        for i, (key, (lo, hi, dtype)) in enumerate(PARAM_BOUNDS.items()):
            if dtype == 'int':
                genes[i] = float(random.randint(int(lo), int(hi)))
            else:
                genes[i] = random.uniform(lo, hi)

        # 強制約束：ma_short < ma_long
        genes = self._enforce_constraints(genes)
        return Individual(genes=genes)

    @staticmethod
    def _enforce_constraints(genes: np.ndarray) -> np.ndarray:
        """確保 ma_short < ma_long，並將所有基因 clip 回邊界內。"""
        g = genes.copy()
        for i, (key, (lo, hi, dtype)) in enumerate(PARAM_BOUNDS.items()):
            g[i] = np.clip(g[i], lo, hi)
            if dtype == 'int':
                g[i] = round(g[i])

        # ma_short（索引0）必須嚴格小於 ma_long（索引1）
        if g[0] >= g[1]:
            g[0] = max(5.0, g[1] // 2)
        return g

    @staticmethod
    def decode(genes: np.ndarray) -> dict:
        """
        將基因陣列解碼為策略參數字典。
        確保整數型參數以 int 返回。
        """
        params = {}
        for i, (key, (lo, hi, dtype)) in enumerate(PARAM_BOUNDS.items()):
            val = float(np.clip(genes[i], lo, hi))
            params[key] = int(round(val)) if dtype == 'int' else val

        if params['ma_short'] >= params['ma_long']:
            params['ma_short'] = max(5, params['ma_long'] // 2)
        return params

    # ── 適應度評估 ────────────────────────────────────────────

    def _backtest_one(self, prices: pd.Series, params: dict) -> float:
        """
        對單一股票進行向量化回測，回傳適應度分數。

        交易規則（向量化狀態機）：
          - 當前一期信號 > buy_threshold  且 目前空倉 → 買入（持倉=1）
          - 當前一期信號 < sell_threshold 且 目前持倉 → 賣出（持倉=0）
          - 持倉期間：策略日報酬 = 股票日報酬
          - 空倉期間：策略日報酬 = 0

        Notes
        -----
        狀態機本質上是序列依賴的，此處採用 Python 迴圈。
        GA 評估時約 50 支股 × 500 天 = 25,000 次迭代，速度可接受。
        """
        min_bars = params['ma_long'] + 10
        if len(prices) < min_bars:
            return -999.0

        # 計算綜合技術信號
        signal = TechnicalFactors.composite_signal(
            prices,
            ma_short=params['ma_short'],
            ma_long=params['ma_long'],
            ma_weight=params['ma_weight'],
            rsi_period=params['rsi_period'],
            rsi_ob=params['rsi_ob'],
            rsi_os=params['rsi_os'],
            rsi_weight=params['rsi_weight'],
            bb_period=params['bb_period'],
            bb_std=params['bb_std'],
            bb_weight=params['bb_weight'],
        )

        buy_thr = params['buy_threshold']
        sell_thr = params['sell_threshold']
        sig_arr = signal.values
        price_arr = prices.values

        # 向量化狀態機：計算每日持倉（0 或 1）
        n = len(sig_arr)
        position = np.zeros(n, dtype=np.float64)
        in_pos = False
        for i in range(n):
            if not in_pos and sig_arr[i] > buy_thr:
                in_pos = True
            elif in_pos and sig_arr[i] < sell_thr:
                in_pos = False
            position[i] = 1.0 if in_pos else 0.0

        # 日報酬率（使用 numpy 直接計算，比 pct_change 更快）
        daily_ret = np.zeros(n)
        daily_ret[1:] = (price_arr[1:] - price_arr[:-1]) / price_arr[:-1]

        # 策略報酬 = 持倉狀態（前一日） × 當日報酬
        strategy_ret = pd.Series(
            position[:-1] * daily_ret[1:],
            index=prices.index[1:]
        )

        return PerformanceMetrics.fitness_score(strategy_ret)

    def _evaluate_population(self,
                              population: list[Individual],
                              stock_data: dict[str, pd.DataFrame]) -> None:
        """
        評估種群中每個個體的適應度。

        對股票池中所有股票取平均適應度作為個體分數。
        平均化的目的：避免策略只對單一股票過擬合。
        """
        for ind in population:
            params = self.decode(ind.genes)
            total, count = 0.0, 0
            for code, df in stock_data.items():
                if 'Close' not in df.columns:
                    continue
                prices = df['Close'].dropna()
                f = self._backtest_one(prices, params)
                if f > -999.0:
                    total += f
                    count += 1
            ind.fitness = total / count if count > 0 else -1.0

    # ── 遺傳算子 ──────────────────────────────────────────────

    def _tournament_select(self, population: list[Individual]) -> Individual:
        """
        錦標賽選擇（Tournament Selection）。

        從種群中隨機抽取 tournament_size 個個體，
        返回其中適應度最高者。
        相比輪盤賭選擇，錦標賽選擇對噪聲更具魯棒性。
        """
        contenders = random.sample(population,
                                   min(self.tourn_size, len(population)))
        return max(contenders, key=lambda x: x.fitness)

    def _crossover(self,
                   p1: Individual,
                   p2: Individual) -> tuple[Individual, Individual]:
        """
        均勻交叉（Uniform Crossover）。

        對每個基因位，以 50% 機率從父本1或父本2繼承。
        相比單點交叉，均勻交叉探索空間能力更強。
        若隨機數 > crossover_rate，則直接複製雙親（不交叉）。
        """
        if random.random() > self.cr:
            return Individual(p1.genes.copy()), Individual(p2.genes.copy())

        mask = np.random.rand(N_GENES) < 0.5
        g1 = np.where(mask, p1.genes, p2.genes)
        g2 = np.where(mask, p2.genes, p1.genes)

        g1 = self._enforce_constraints(g1)
        g2 = self._enforce_constraints(g2)
        return Individual(g1), Individual(g2)

    def _mutate(self, ind: Individual) -> Individual:
        """
        高斯變異（Gaussian Mutation）。

        對每個基因以 mutation_rate 的機率加入高斯噪聲。
        噪聲標準差 = 該基因範圍的 10%（自適應步長）。
        確保變異後的基因仍在合法範圍內。
        """
        genes = ind.genes.copy()
        for i, (key, (lo, hi, dtype)) in enumerate(PARAM_BOUNDS.items()):
            if random.random() < self.mr:
                noise = np.random.normal(0, (hi - lo) * 0.10)
                genes[i] = np.clip(genes[i] + noise, lo, hi)

        genes = self._enforce_constraints(genes)
        return Individual(genes)

    # ── 主演化流程 ────────────────────────────────────────────

    def evolve(self,
               stock_data: dict[str, pd.DataFrame],
               verbose: bool = True) -> dict:
        """
        執行遺傳演算法，返回最佳策略參數。

        Parameters
        ----------
        stock_data : {股票代號: OHLCV DataFrame} 字典
        verbose    : 每 10 代顯示一次進度

        Returns
        -------
        dict：最佳策略參數，可直接傳入 score_stocks() 使用
        """
        print(f"\n  種群大小: {self.pop_size}  |  演化代數: {self.generations}")
        print(f"  交叉率: {self.cr}  |  變異率: {self.mr}\n")

        # ── 初始化種群 ──
        population = [self._random_individual() for _ in range(self.pop_size)]

        global_best: Individual | None = None
        self.fitness_history = []

        for gen in range(self.generations):
            # 1. 評估適應度
            self._evaluate_population(population, stock_data)

            # 2. 排序（降序）
            population.sort(key=lambda x: x.fitness, reverse=True)

            gen_best = population[0]
            gen_avg = np.mean([ind.fitness for ind in population])

            # 3. 更新全局最佳
            if global_best is None or gen_best.fitness > global_best.fitness:
                global_best = Individual(gen_best.genes.copy(),
                                         fitness=gen_best.fitness)

            self.fitness_history.append({
                'generation': gen + 1,
                'best_fitness': gen_best.fitness,
                'avg_fitness': gen_avg,
            })

            if verbose and ((gen + 1) % 10 == 0 or gen == 0):
                print(f"  第 {gen+1:3d} 代 | "
                      f"最佳: {gen_best.fitness:+.4f} | "
                      f"平均: {gen_avg:+.4f}")

            # ── 產生下一代 ──
            elites = [Individual(ind.genes.copy(), ind.fitness)
                      for ind in population[:self.elite_n]]

            next_gen: list[Individual] = elites[:]
            while len(next_gen) < self.pop_size:
                parent1 = self._tournament_select(population)
                parent2 = self._tournament_select(population)
                child1, child2 = self._crossover(parent1, parent2)
                next_gen.append(self._mutate(child1))
                if len(next_gen) < self.pop_size:
                    next_gen.append(self._mutate(child2))

            population = next_gen

        # ── 最終輸出 ──
        best_params = self.decode(global_best.genes)
        self.best_params = best_params
        print(f"\n  演化完成。最佳適應度: {global_best.fitness:+.4f}")
        return best_params

    # ── 選股評分 ──────────────────────────────────────────────

    def score_stocks(self,
                     stock_data: dict[str, pd.DataFrame],
                     params: dict) -> dict[str, float]:
        """
        用最佳參數對股票池中每支股票打分。

        評分公式：
          score = 0.50 × 近 5 日平均信號
                + 0.50 × 回測適應度（正規化至 [-1, 1]）

        Returns
        -------
        dict：{股票代號: 綜合評分}，分數越高代表買入優先級越高
        """
        scores: dict[str, float] = {}

        for code, df in stock_data.items():
            if 'Close' not in df.columns:
                continue
            prices = df['Close'].dropna()
            min_bars = params['ma_long'] + 5
            if len(prices) < min_bars:
                continue

            # 計算完整信號序列
            signal = TechnicalFactors.composite_signal(
                prices,
                ma_short=params['ma_short'],
                ma_long=params['ma_long'],
                ma_weight=params['ma_weight'],
                rsi_period=params['rsi_period'],
                rsi_ob=params['rsi_ob'],
                rsi_os=params['rsi_os'],
                rsi_weight=params['rsi_weight'],
                bb_period=params['bb_period'],
                bb_std=params['bb_std'],
                bb_weight=params['bb_weight'],
            )

            # 近 5 日平均信號（反映當前市場狀態）
            recent_signal = float(signal.iloc[-5:].mean())

            # 歷史回測適應度（反映策略在此標的的長期有效性）
            fitness = self._backtest_one(prices, params)
            fitness_norm = float(np.clip(fitness, -1.0, 1.0))

            scores[code] = 0.50 * recent_signal + 0.50 * fitness_norm

        return scores
