"""
genetic_algorithm.py
────────────────────
遺傳演算法（GA）引擎 — v2.0 短線強化版

變更紀錄 v2.0：
  - 新增 SHORT_TERM_PARAM_BOUNDS：縮短 MA / RSI 周期，適合波段操作
  - score_stocks() 新增 short_term 模式，融合 KDJ + 成交量爆發信號
  - 短線評分公式：70% 近 3 日信號 + 30% 回測適應度，反應速度更快
  - 長線評分公式（原始）：50% 近 5 日信號 + 50% 回測適應度

染色體結構（12 個基因）─ 長線模式
────────────────────────────────────
index | 參數名稱        | 範圍
------+-----------------+----------
  0   | ma_short        | [5,  30]
  1   | ma_long         | [20, 120]
  2   | ma_weight       | [0,  1]
  3   | rsi_period      | [7,  21]
  4   | rsi_ob          | [65, 85]
  5   | rsi_os          | [15, 35]
  6   | rsi_weight      | [0,  1]
  7   | bb_period       | [10, 30]
  8   | bb_std          | [1.5,3.0]
  9   | bb_weight       | [0,  1]
 10   | buy_threshold   | [0.1,0.5]
 11   | sell_threshold  | [-0.5,-0.1]

染色體結構（12 個基因）─ 短線模式
────────────────────────────────────
index | 參數名稱        | 範圍（短線）
------+-----------------+----------
  0   | ma_short        | [3,  15]   ← 縮短
  1   | ma_long         | [10, 60]   ← 縮短
  2   | ma_weight       | [0,  0.5]  ← 降低比重
  3   | rsi_period      | [5,  14]   ← 縮短
  4   | rsi_ob          | [60, 80]
  5   | rsi_os          | [20, 40]
  6   | rsi_weight      | [0.3,1.0]  ← 提高下限
  7   | bb_period       | [5,  20]   ← 縮短
  8   | bb_std          | [1.5,2.5]
  9   | bb_weight       | [0,  0.6]
 10   | buy_threshold   | [0.05,0.3] ← 降低門檻（反應更快）
 11   | sell_threshold  | [-0.3,-0.05]
"""

import random
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from joblib import Parallel, delayed

from technical_factors import TechnicalFactors
from performance_metrics import PerformanceMetrics


# ── 長線模式參數邊界（原始）────────────────────────────────────
LONG_TERM_PARAM_BOUNDS: dict[str, tuple] = {
    'ma_short':       (5,    30,    'int'),
    'ma_long':        (20,   120,   'int'),
    'ma_weight':      (0.0,  1.0,   'float'),
    'rsi_period':     (7,    21,    'int'),
    'rsi_ob':         (65.0, 85.0,  'float'),
    'rsi_os':         (15.0, 35.0,  'float'),
    'rsi_weight':     (0.0,  1.0,   'float'),
    'bb_period':      (10,   30,    'int'),
    'bb_std':         (1.5,  3.0,   'float'),
    'bb_weight':      (0.0,  1.0,   'float'),
    'buy_threshold':  (0.10, 0.50,  'float'),
    'sell_threshold': (-0.50, -0.10, 'float'),
}

# ── 短線模式參數邊界（波段/當沖）──────────────────────────────
SHORT_TERM_PARAM_BOUNDS: dict[str, tuple] = {
    'ma_short':       (3,    15,    'int'),    # 更短均線
    'ma_long':        (10,   60,    'int'),    # 壓縮長均線
    'ma_weight':      (0.0,  0.50,  'float'),  # 降低 MA 比重
    'rsi_period':     (5,    14,    'int'),    # 更快速的 RSI
    'rsi_ob':         (60.0, 80.0,  'float'),
    'rsi_os':         (20.0, 40.0,  'float'),
    'rsi_weight':     (0.30, 1.0,   'float'),  # RSI 最低 30%
    'bb_period':      (5,    20,    'int'),    # 更短 BB
    'bb_std':         (1.5,  2.5,   'float'),
    'bb_weight':      (0.0,  0.60,  'float'),
    'buy_threshold':  (0.05, 0.30,  'float'),  # 更靈敏的買入閾值
    'sell_threshold': (-0.30, -0.05, 'float'), # 更靈敏的賣出閾值
}

# 向後相容
PARAM_BOUNDS = LONG_TERM_PARAM_BOUNDS
N_GENES = len(PARAM_BOUNDS)
PARAM_KEYS = list(PARAM_BOUNDS.keys())


@dataclass
class Individual:
    """
    種群中的一個個體（一組策略參數）。

    genes   : 長度 N_GENES 的 numpy 陣列
    fitness : 適應度分數，初始化為 -999
    """
    genes: np.ndarray
    fitness: float = field(default=-999.0)


class GeneticAlgorithm:
    """
    遺傳演算法引擎。

    使用範例
    --------
    ga = GeneticAlgorithm(population_size=50, generations=50, mode='short_term')
    best_params = ga.evolve(stock_data_dict, verbose=True)
    scores = ga.score_stocks(stock_data_dict, best_params)
    """

    def __init__(self,
                 population_size: int = 50,
                 generations: int = 50,
                 crossover_rate: float = 0.80,
                 mutation_rate: float = 0.15,
                 tournament_size: int = 5,
                 elite_ratio: float = 0.10,
                 mode: str = 'short_term'):
        """
        Parameters
        ----------
        population_size : 每代種群數量
        generations     : 最大演化代數
        crossover_rate  : 交叉率
        mutation_rate   : 變異率
        tournament_size : 錦標賽每場參賽個體數
        elite_ratio     : 精英保留比例
        mode            : 'short_term'（波段優先）或 'long_term'（長線持有）
        """
        self.pop_size   = population_size
        self.generations = generations
        self.cr         = crossover_rate
        self.mr         = mutation_rate
        self.tourn_size = tournament_size
        self.elite_n    = max(1, int(population_size * elite_ratio))
        self.mode       = mode

        # 依模式選擇參數邊界
        self._bounds = (SHORT_TERM_PARAM_BOUNDS if mode == 'short_term'
                        else LONG_TERM_PARAM_BOUNDS)

        self.best_params: dict = {}
        self.fitness_history: list[dict] = []

    # ── 染色體編解碼 ──────────────────────────────────────────

    def _random_individual(self) -> Individual:
        genes = np.zeros(N_GENES)
        for i, (key, (lo, hi, dtype)) in enumerate(self._bounds.items()):
            if dtype == 'int':
                genes[i] = float(random.randint(int(lo), int(hi)))
            else:
                genes[i] = random.uniform(lo, hi)
        genes = self._enforce_constraints(genes)
        return Individual(genes=genes)

    def _enforce_constraints(self, genes: np.ndarray) -> np.ndarray:
        g = genes.copy()
        for i, (key, (lo, hi, dtype)) in enumerate(self._bounds.items()):
            g[i] = np.clip(g[i], lo, hi)
            if dtype == 'int':
                g[i] = round(g[i])
        if g[0] >= g[1]:
            g[0] = max(float(list(self._bounds.values())[0][0]), g[1] // 2)
        return g

    def decode(self, genes: np.ndarray) -> dict:
        """將基因陣列解碼為策略參數字典。"""
        params = {}
        for i, (key, (lo, hi, dtype)) in enumerate(self._bounds.items()):
            val = float(np.clip(genes[i], lo, hi))
            params[key] = int(round(val)) if dtype == 'int' else val
        if params['ma_short'] >= params['ma_long']:
            params['ma_short'] = max(3, params['ma_long'] // 2)
        return params

    # ── 適應度評估 ────────────────────────────────────────────

    def _backtest_one(self, prices: pd.Series, params: dict) -> float:
        """
        對單一股票進行向量化回測，回傳適應度分數。

        短線模式下，狀態機的持倉判斷更靈敏（門檻更低）。
        """
        min_bars = params['ma_long'] + 10
        if len(prices) < min_bars:
            return -999.0

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

        buy_thr  = params['buy_threshold']
        sell_thr = params['sell_threshold']
        sig_arr  = signal.values
        price_arr = prices.values

        n = len(sig_arr)
        position = np.zeros(n, dtype=np.float64)
        in_pos = False
        for i in range(n):
            if not in_pos and sig_arr[i] > buy_thr:
                in_pos = True
            elif in_pos and sig_arr[i] < sell_thr:
                in_pos = False
            position[i] = 1.0 if in_pos else 0.0

        daily_ret = np.zeros(n)
        daily_ret[1:] = (price_arr[1:] - price_arr[:-1]) / price_arr[:-1]

        strategy_ret = pd.Series(
            position[:-1] * daily_ret[1:],
            index=prices.index[1:]
        )
        return PerformanceMetrics.fitness_score(strategy_ret)

    def _eval_one_individual(
        self,
        genes: np.ndarray,
        stock_data: dict[str, pd.DataFrame],
    ) -> float:
        """
        計算單一個體的跨股票平均適應度。
        拆分為獨立函數以供 joblib 並行呼叫。
        """
        params = self.decode(genes)
        total, count = 0.0, 0
        for code, df in stock_data.items():
            if 'Close' not in df.columns:
                continue
            prices = df['Close'].dropna()
            f = self._backtest_one(prices, params)
            if f > -999.0:
                total += f
                count += 1
        return total / count if count > 0 else -1.0

    def _evaluate_population(self,
                              population: list[Individual],
                              stock_data: dict[str, pd.DataFrame]) -> None:
        """
        並行評估種群中每個個體的適應度。

        使用 joblib.Parallel(n_jobs=-1, prefer='threads') 確保：
          - 充分利用多核心 CPU
          - 在 Windows + Streamlit 環境下不會因 multiprocessing spawn 崩潰
          - prefer='threads' 使用執行緒後端，避免 pickle 與 __main__ 衛兵問題
        """
        fitnesses: list[float] = Parallel(
            n_jobs=-1,
            prefer='threads',
        )(
            delayed(self._eval_one_individual)(ind.genes, stock_data)
            for ind in population
        )
        for ind, fit in zip(population, fitnesses):
            ind.fitness = fit

    # ── 遺傳算子 ──────────────────────────────────────────────

    def _tournament_select(self, population: list[Individual]) -> Individual:
        contenders = random.sample(population, min(self.tourn_size, len(population)))
        return max(contenders, key=lambda x: x.fitness)

    def _crossover(self, p1: Individual, p2: Individual) -> tuple[Individual, Individual]:
        if random.random() > self.cr:
            return Individual(p1.genes.copy()), Individual(p2.genes.copy())
        mask = np.random.rand(N_GENES) < 0.5
        g1 = np.where(mask, p1.genes, p2.genes)
        g2 = np.where(mask, p2.genes, p1.genes)
        return Individual(self._enforce_constraints(g1)), Individual(self._enforce_constraints(g2))

    def _mutate(self, ind: Individual) -> Individual:
        genes = ind.genes.copy()
        for i, (key, (lo, hi, dtype)) in enumerate(self._bounds.items()):
            if random.random() < self.mr:
                noise = np.random.normal(0, (hi - lo) * 0.10)
                genes[i] = np.clip(genes[i] + noise, lo, hi)
        return Individual(self._enforce_constraints(genes))

    # ── 主演化流程 ────────────────────────────────────────────

    def evolve(self,
               stock_data: dict[str, pd.DataFrame],
               verbose: bool = True) -> dict:
        """
        執行遺傳演算法，返回最佳策略參數。

        Parameters
        ----------
        stock_data : {股票代號: OHLCV DataFrame}
        verbose    : 每 10 代顯示進度
        """
        mode_label = "短線波段" if self.mode == 'short_term' else "長線持有"
        print(f"\n  模式: {mode_label}  |  種群: {self.pop_size}  |  代數: {self.generations}")
        print(f"  交叉率: {self.cr}  |  變異率: {self.mr}\n")

        population = [self._random_individual() for _ in range(self.pop_size)]
        global_best: Individual | None = None
        self.fitness_history = []

        for gen in range(self.generations):
            self._evaluate_population(population, stock_data)
            population.sort(key=lambda x: x.fitness, reverse=True)

            gen_best = population[0]
            gen_avg  = np.mean([ind.fitness for ind in population])

            if global_best is None or gen_best.fitness > global_best.fitness:
                global_best = Individual(gen_best.genes.copy(), fitness=gen_best.fitness)

            self.fitness_history.append({
                'generation':   gen + 1,
                'best_fitness': gen_best.fitness,
                'avg_fitness':  gen_avg,
            })

            if verbose and ((gen + 1) % 10 == 0 or gen == 0):
                print(f"  第 {gen+1:3d} 代 | 最佳: {gen_best.fitness:+.4f} | 平均: {gen_avg:+.4f}")

            elites  = [Individual(ind.genes.copy(), ind.fitness) for ind in population[:self.elite_n]]
            next_gen: list[Individual] = elites[:]
            while len(next_gen) < self.pop_size:
                c1, c2 = self._crossover(
                    self._tournament_select(population),
                    self._tournament_select(population),
                )
                next_gen.append(self._mutate(c1))
                if len(next_gen) < self.pop_size:
                    next_gen.append(self._mutate(c2))
            population = next_gen

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

        長線模式：
          score = 0.50 × 近 5 日平均信號 + 0.50 × 回測適應度

        短線模式（v2.0 新增）：
          base  = 0.70 × 近 3 日平均信號 + 0.30 × 回測適應度
          boost = KDJ 信號均值 × 0.25 + 成交量爆發均值 × 0.15
          score = 0.60 × base + 0.40 × boost
          （KDJ 與量能最多貢獻 40% 的評分調整）

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

            # 基礎技術信號（MA + RSI + BB）
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

            fitness     = self._backtest_one(prices, params)
            fitness_norm = float(np.clip(fitness, -1.0, 1.0))

            if self.mode == 'short_term':
                # 短線：用近 3 日信號，並融合 KDJ + 成交量
                recent_signal = float(signal.iloc[-3:].mean())
                base_score    = 0.70 * recent_signal + 0.30 * fitness_norm

                # 計算 KDJ 與量能信號（需要 H/L/V 欄位）
                boost = 0.0
                if all(c in df.columns for c in ('High', 'Low', 'Close')):
                    try:
                        high   = df['High'].reindex(prices.index).ffill()
                        low    = df['Low'].reindex(prices.index).ffill()
                        s_kdj  = TechnicalFactors.kdj_signal(high, low, prices)
                        kdj_val = float(s_kdj.iloc[-3:].mean())
                        boost  += 0.60 * kdj_val   # KDJ 權重 60%
                    except Exception:
                        pass

                if 'Volume' in df.columns:
                    try:
                        volume = df['Volume'].reindex(prices.index).fillna(0)
                        s_vol  = TechnicalFactors.volume_burst_signal(prices, volume)
                        vol_val = float(s_vol.iloc[-3:].mean())
                        boost  += 0.40 * vol_val   # 量能權重 40%
                    except Exception:
                        pass

                scores[code] = 0.60 * base_score + 0.40 * boost

            else:
                # 長線：原始邏輯
                recent_signal = float(signal.iloc[-5:].mean())
                scores[code]  = 0.50 * recent_signal + 0.50 * fitness_norm

        return scores
