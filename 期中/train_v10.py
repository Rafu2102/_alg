"""
🚀 Snake AI V10.0 "Phoenix" - 訓練腳本
========================================

這個檔案負責訓練 AI 模型，使用 PPO（Proximal Policy Optimization）演算法。

主要功能：
1. 設定訓練超參數
2. 建立並行環境（32 個貪吃蛇同時訓練）
3. 課程學習（Curriculum Learning）：分階段增加難度
4. 自動儲存模型和恢復訓練

技術特點：
- PPO 演算法：穩定且高效的強化學習方法
- 課程學習：從簡單到困難，循序漸進
- TF32 加速：利用 RTX 40 系列的硬體加速
- 並行訓練：32 個環境同時運行，大幅提升效率
"""

import os
import sys
import multiprocessing

# ==================== 硬體優化設定 ====================
# 這些環境變數用於優化多執行緒效能

# 限制各種數學庫只使用 1 個執行緒
# 因為我們用多進程（32 個環境），不需要每個進程再開多執行緒
os.environ["OMP_NUM_THREADS"] = "1"          # OpenMP
os.environ["MKL_NUM_THREADS"] = "1"          # Intel MKL
os.environ["NUMEXPR_NUM_THREADS"] = "1"      # NumExpr
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Tokenizers

# Windows 多進程修復
# Windows 需要使用 'spawn' 方式創建子進程
if sys.platform == 'win32':
    multiprocessing.set_start_method('spawn', force=True)

import time
from datetime import datetime, timedelta
import numpy as np
import torch

# ==================== GPU 優化設定 ====================
# TF32 是 NVIDIA Ampere 架構（RTX 30/40 系列）的加速功能
# 可以在幾乎不損失精度的情況下提升訓練速度

torch.backends.cudnn.benchmark = True          # cuDNN 自動尋找最佳卷積演算法
torch.backends.cuda.matmul.allow_tf32 = True   # 允許 TF32 矩陣乘法
torch.backends.cudnn.allow_tf32 = True         # 允許 TF32 卷積
torch.set_float32_matmul_precision('high')     # 高精度矩陣運算
torch.set_num_threads(1)                       # PyTorch 只用 1 個執行緒

# 匯入強化學習相關套件
from sb3_contrib import MaskablePPO           # 支援動作遮罩的 PPO
from sb3_contrib.common.wrappers import ActionMasker  # 動作遮罩包裝器
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed, get_schedule_fn

# 匯入我們的遊戲環境
from snake_env_v10 import SnakeEnvV10

# ==================== PPO 超參數設定 ====================
# 這些參數控制 AI 的學習行為

N_ENVS = 32           # 並行環境數量（同時玩 32 局遊戲）
N_STEPS = 4096        # 每次收集多少步經驗再學習
BATCH_SIZE = 16384    # 每批次用多少樣本訓練
GAMMA = 0.999         # 折扣因子：越接近 1 越重視長期獎勵
GAE_LAMBDA = 0.95     # GAE（廣義優勢估計）的 λ 參數
VF_COEF = 0.5         # Value Function 損失函數的權重
CLIP_RANGE = 0.2      # PPO 裁剪範圍，防止策略更新太大
N_EPOCHS = 10         # 每批資料學習幾次
MAX_GRAD_NORM = 0.5   # 梯度裁剪，防止梯度爆炸

"""
超參數解釋：

N_ENVS = 32：
- 同時運行 32 個貪吃蛇遊戲
- 每步收集 32 份經驗，大幅提升資料效率

N_STEPS = 4096：
- 每個環境跑 4096 步後才更新模型
- 更長的軌跡 = 更準確的優勢估計

GAMMA = 0.999：
- 接近 1 代表非常重視未來獎勵
- 貪吃蛇需要長遠規劃，所以用高 gamma

N_EPOCHS = 10：
- 同一批資料反覆學習 10 次
- 太高會過擬合，太低學不夠
"""


def linear_schedule(lr_start, lr_end):
    """
    線性學習率排程
    
    什麼是學習率排程？
    - 學習率控制每次更新的「步伐大小」
    - 訓練初期用大學習率，快速進步
    - 訓練後期用小學習率，精細調整
    
    這個函數返回一個排程函數，根據訓練進度返回對應的學習率。
    
    參數：
    - lr_start: 初始學習率
    - lr_end: 結束學習率
    
    返回：
    - schedule: 排程函數，接受 progress_remaining（1.0 → 0.0）
    """
    def schedule(progress_remaining):
        # progress_remaining: 1.0（開始）→ 0.0（結束）
        # 線性插值：lr = lr_end + (lr_start - lr_end) * progress
        return lr_end + (lr_start - lr_end) * progress_remaining
    return schedule


# 全域隨機種子（用於可重現性）
BASE_SEED = 12345

# ==================== 課程學習設計 ====================
# 課程學習 (Curriculum Learning) 是一種訓練策略：
# 從簡單的任務開始，逐漸增加難度

STAGES = [
    # ==================== Stage A: 基礎奠定期 ====================
    # 目標：讓 AI 學會基本技能（吃食物、避免撞牆）
    {
        "name": "v10_stage_a",        # 階段名稱
        "grid_size": 20,              # 網格大小
        "steps": 30_000_000,          # 訓練步數（3000 萬）
        "lr_start": 3e-4,             # 初始學習率 0.0003
        "lr_end": 2e-4,               # 結束學習率 0.0002
        "ent_coef": 0.03,             # 熵係數（鼓勵探索）
        "start_length": 3,            # 總是從長度 3 開始
        "endgame_prob": 0.0,          # 不做 Endgame 訓練
        "target_length": 200          # 目標長度
    },
    
    # ==================== Stage B: 平滑過渡期 ====================
    # 目標：開始接觸中盤局面，但不要太激進
    {
        "name": "v10_stage_b",
        "grid_size": 20,
        "steps": 70_000_000,          # 7000 萬步
        "lr_start": 2e-4,
        "lr_end": 1e-4,
        "ent_coef": 0.025,            # 稍微提高探索
        "start_length": 50,           # 20% 機率從長度 50 開始
        "endgame_prob": 0.2,          # Endgame 出現機率
        "target_length": 350
    },
    
    # ==================== Stage C: 衝刺期 ====================
    # 目標：挑戰滿分 400
    {
        "name": "v10_final",
        "grid_size": 20,
        "steps": 100_000_000,         # 1 億步
        "lr_start": 1.5e-4,
        "lr_end": 5e-5,
        "ent_coef": 0.02,
        "start_length": 100,          # 25% 機率從長度 100 開始
        "endgame_prob": 0.25,
        "target_length": 400
    }
]

"""
課程學習的重要性：

為什麼不能一開始就練困難任務？
- AI 一開始什麼都不會
- 如果直接給困難任務，會學不到東西
- 循序漸進才能穩定進步

每個階段的設計邏輯：
- Stage A：打好基礎，學會生存
- Stage B：開始接觸中盤，但 80% 時間複習基礎
- Stage C：挑戰 Endgame，衝刺滿分
"""

# 神經網路架構
POLICY_KWARGS = dict(
    net_arch=[dict(pi=[256, 256], vf=[256, 256])]
)
"""
神經網路架構說明：

pi=[256, 256]：Policy 網路
- 兩層全連接層，每層 256 個神經元
- 輸入：26 維觀察向量
- 輸出：4 個動作的機率分佈

vf=[256, 256]：Value 網路
- 同樣是兩層 256 神經元
- 輸出：當前狀態的價值估計

為什麼用 256？
- 26 維輸入不需要太大的網路
- V9.0 用 512 反而效果更差（過度設計）
"""


# ==================== 終端機顏色（美化輸出）====================
class C:
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    END = '\033[0m'
    MAGENTA = '\033[95m'

def clear():
    """清除終端機畫面"""
    os.system('cls' if os.name == 'nt' else 'clear')

def mask_fn(env):
    """動作遮罩函數：返回當前可用的動作"""
    return env.action_masks()

def make_env(grid_size, rank, seed=0, start_length=3, endgame_prob=0.0):
    """
    環境工廠函數
    
    這個函數創建一個貪吃蛇環境。
    SubprocVecEnv 會調用這個函數 N_ENVS 次，創建多個環境。
    
    參數：
    - grid_size: 網格大小
    - rank: 環境編號（0 到 N_ENVS-1）
    - seed: 隨機種子
    - start_length: 課程學習的起始長度
    - endgame_prob: Endgame 出現機率
    
    返回：
    - _init: 初始化函數
    """
    def _init():
        # 設定這個環境的隨機種子（每個環境不同）
        set_random_seed(seed + rank)
        import random as py_random
        py_random.seed(seed + rank)
        np.random.seed(seed + rank)
        
        # 創建遊戲環境
        env = SnakeEnvV10(
            grid_size=grid_size,
            default_start_length=start_length,
            endgame_prob=endgame_prob
        )
        # 包裝動作遮罩
        env = ActionMasker(env, mask_fn)
        return env
    return _init


class V10ProgressCallback(BaseCallback):
    """
    V10.0 訓練進度回呼
    
    這個類別用於：
    1. 顯示訓練進度（漂亮的終端機介面）
    2. 記錄統計資料（最佳長度、平均長度等）
    3. 自動儲存里程碑模型（達到 50, 100, 200... 時）
    """
    
    def __init__(self, stage_name, total_steps, target_length=400, ent_coef=0.01, verbose=0):
        super().__init__(verbose)
        self.stage_name = stage_name
        self.total_steps = total_steps
        self.target_length = target_length
        self.ent_coef = ent_coef
        
        # 統計資料
        self.episode_lengths = []  # 每局遊戲的蛇長度
        self.fallback_counts = []  # Fallback 觸發次數
        self.best_length = 3       # 歷史最佳長度
        self.best_avg = 0          # 歷史最佳平均
        self.generation = 0        # 總遊戲局數
        self.start_time = time.time()
        self.last_display = 0
        
        # 里程碑（達到這些長度時儲存模型）
        self.milestones = [50, 100, 150, 200, 250, 300, 350, 375, 400]
        self.achieved = set()  # 已達成的里程碑
        
    def _on_step(self):
        """
        每步訓練都會調用這個方法
        
        注意：只在 episode 結束時記錄長度（done=True）
        """
        dones = self.locals.get('dones', [])
        infos = self.locals.get('infos', [])
        
        for done, info in zip(dones, infos):
            if done:  # 只在遊戲結束時記錄
                length = info.get('length', 0)
                if length > 0:
                    self.episode_lengths.append(length)
                    self.generation += 1
                    
                    if 'fallback_count' in info:
                        self.fallback_counts.append(info['fallback_count'])
                    
                    # 檢查新紀錄
                    if length > self.best_length:
                        self.best_length = length
                        
                        # 檢查里程碑
                        for m in self.milestones:
                            if length >= m and m not in self.achieved:
                                self.achieved.add(m)
                                self.model.save(f"checkpoints/{self.stage_name}_milestone_{m}")
                                print(f"\n{C.GREEN}🏆 里程碑達成: {m}! 已儲存!{C.END}\n")
        
        # 記錄到 TensorBoard
        if len(self.fallback_counts) > 0:
            fallback_mean = np.mean(self.fallback_counts[-100:])
            self.logger.record("custom/fallback_mean", fallback_mean)
        
        if len(self.episode_lengths) > 0:
            recent_avg = np.mean(self.episode_lengths[-500:])
            self.logger.record("custom/length_avg_500", recent_avg)
            
            # 儲存最佳平均模型
            if recent_avg > self.best_avg and len(self.episode_lengths) >= 500:
                self.best_avg = recent_avg
                self.model.save(f"checkpoints/{self.stage_name}_best_avg_{int(recent_avg)}")
                self.logger.record("custom/best_avg", self.best_avg)
            
            # 計算達標百分比
            recent = self.episode_lengths[-500:]
            pct_target = sum(1 for l in recent if l >= self.target_length) / len(recent) * 100
            self.logger.record("custom/pct_target", pct_target)
        
        # 定期更新顯示
        now = time.time()
        if now - self.last_display >= 1.0:
            self._display()
            self.last_display = now
        
        return True
    
    def _display(self):
        """顯示漂亮的訓練進度介面"""
        clear()
        
        elapsed = time.time() - self.start_time
        steps = self.num_timesteps
        progress = min(100, steps / self.total_steps * 100)
        
        recent = self.episode_lengths[-500:] if self.episode_lengths else [3]
        avg = np.mean(recent)
        max_r = max(recent) if recent else 3
        p90 = np.percentile(recent, 90) if len(recent) >= 10 else max_r
        
        sps = steps / elapsed if elapsed > 0 else 0
        eta = (self.total_steps - steps) / sps if sps > 0 else 0
        
        # GPU 資訊
        gpu = "CPU"
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)[:20]
            mem = torch.cuda.memory_allocated() / 1024**3
            gpu = f"{name} | {mem:.1f}GB"
        
        # 進度條
        bar_len = 50
        filled = int(bar_len * progress / 100)
        bar = '█' * filled + '░' * (bar_len - filled)
        
        # 里程碑顯示
        ms = ""
        for m in self.milestones:
            if m in self.achieved:
                ms += f"{C.GREEN}[{m}]{C.END} "
            elif m <= self.best_length:
                ms += f"{C.YELLOW}~{m}{C.END} "
            else:
                ms += f"{C.DIM}{m}{C.END} "
        
        print(f"""
{C.CYAN}╔{'═'*78}╗{C.END}
{C.CYAN}║{C.END}      {C.BOLD}{C.MAGENTA}SNAKE AI V10.0 "PHOENIX" - {self.stage_name.upper()}{C.END}              {C.CYAN}║{C.END}
{C.CYAN}╠{'═'*78}╣{C.END}
{C.CYAN}║{C.END}  {C.DIM}GPU: {gpu:<60}{C.END}  {C.CYAN}║{C.END}
{C.CYAN}║{C.END}  {C.DIM}Net: 256×256 | Envs: {N_ENVS} | Batch: {BATCH_SIZE} | ent: {self.ent_coef}{C.END}      {C.CYAN}║{C.END}
{C.CYAN}╠{'─'*78}╣{C.END}
{C.CYAN}║{C.END}                                                                              {C.CYAN}║{C.END}
{C.CYAN}║{C.END}  {C.YELLOW}📊 訓練狀態{C.END}                                                       {C.CYAN}║{C.END}
{C.CYAN}║{C.END}  Generation:     {C.GREEN}{self.generation:>15,}{C.END}                                          {C.CYAN}║{C.END}
{C.CYAN}║{C.END}  Total Steps:    {C.GREEN}{steps:>15,}{C.END}                                          {C.CYAN}║{C.END}
{C.CYAN}║{C.END}  Speed:          {C.GREEN}{sps:>15,.0f}{C.END} steps/s                               {C.CYAN}║{C.END}
{C.CYAN}║{C.END}                                                                              {C.CYAN}║{C.END}
{C.CYAN}║{C.END}  {C.YELLOW}🎯 表現 (目標: {self.target_length}){C.END}                                       {C.CYAN}║{C.END}
{C.CYAN}║{C.END}  Best Length:    {C.GREEN}{self.best_length:>15}{C.END}                                          {C.CYAN}║{C.END}
{C.CYAN}║{C.END}  Avg (500):      {C.GREEN}{avg:>15.1f}{C.END}                                          {C.CYAN}║{C.END}
{C.CYAN}║{C.END}  Max Recent:     {C.GREEN}{max_r:>15}{C.END}                                          {C.CYAN}║{C.END}
{C.CYAN}║{C.END}  P90 (Top 10%):  {C.GREEN}{p90:>15.1f}{C.END}                                          {C.CYAN}║{C.END}
{C.CYAN}║{C.END}                                                                              {C.CYAN}║{C.END}
{C.CYAN}║{C.END}  {C.YELLOW}🏆 里程碑{C.END}                                                            {C.CYAN}║{C.END}
{C.CYAN}║{C.END}  {ms:<76}{C.CYAN}║{C.END}
{C.CYAN}║{C.END}                                                                              {C.CYAN}║{C.END}
{C.CYAN}║{C.END}  {C.YELLOW}⏱️ 時間{C.END}                                                                  {C.CYAN}║{C.END}
{C.CYAN}║{C.END}  Elapsed:        {str(timedelta(seconds=int(elapsed))):>15}                                  {C.CYAN}║{C.END}
{C.CYAN}║{C.END}  ETA:            {str(timedelta(seconds=int(eta))):>15}                                  {C.CYAN}║{C.END}
{C.CYAN}║{C.END}                                                                              {C.CYAN}║{C.END}
{C.CYAN}║{C.END}  {C.YELLOW}📈 進度{C.END}                                                              {C.CYAN}║{C.END}
{C.CYAN}║{C.END}  [{bar}] {progress:>5.1f}%        {C.CYAN}║{C.END}
{C.CYAN}║{C.END}                                                                              {C.CYAN}║{C.END}
{C.CYAN}╠{'─'*78}╣{C.END}
{C.CYAN}║{C.END}  {C.RED}按 Ctrl+C 暫停並儲存{C.END}                                          {C.CYAN}║{C.END}
{C.CYAN}╚{'═'*78}╝{C.END}
""")


def main():
    """
    主訓練流程
    
    流程：
    1. 初始化 GPU 和隨機種子
    2. 對每個 Stage：
       a. 創建並行環境
       b. 載入或創建模型
       c. 訓練指定步數
       d. 儲存模型
    3. 完成
    """
    print(f"{C.MAGENTA}{'='*60}{C.END}")
    print(f"{C.MAGENTA}  SNAKE AI V10.0 \"PHOENIX\" - 課程學習訓練{C.END}")
    print(f"{C.MAGENTA}{'='*60}{C.END}")
    print(f"{C.CYAN}  26維obs | 256×256 MLP | TF32 | 32 envs{C.END}")
    
    # ==================== 設定全域隨機種子 ====================
    print(f"{C.DIM}  設定全域種子: {BASE_SEED}{C.END}")
    set_random_seed(BASE_SEED)
    import random as py_random
    py_random.seed(BASE_SEED)
    np.random.seed(BASE_SEED)
    torch.manual_seed(BASE_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(BASE_SEED)
    
    # GPU 檢查
    if torch.cuda.is_available():
        print(f"{C.GREEN}[✓] GPU: {torch.cuda.get_device_name(0)}{C.END}")
        device = "cuda"
        torch.backends.cudnn.benchmark = True
        torch.set_num_threads(1)
    else:
        print(f"{C.YELLOW}[!] 沒有 GPU，使用 CPU{C.END}")
        device = "cpu"
    
    # 創建 checkpoints 目錄
    os.makedirs("checkpoints", exist_ok=True)
    
    # ==================== Checkpoint 自動恢復 ====================
    def find_latest_checkpoint(stage_name):
        """尋找某個階段的最新 checkpoint"""
        import glob
        import re
        
        pattern = f"checkpoints/{stage_name}_*_steps.zip"
        checkpoints = glob.glob(pattern)
        
        if not checkpoints:
            return None, 0
        
        max_steps = 0
        latest_path = None
        for cp in checkpoints:
            match = re.search(r'_(\d+)_steps\.zip$', cp)
            if match:
                steps = int(match.group(1))
                if steps > max_steps:
                    max_steps = steps
                    latest_path = cp.replace('.zip', '')
        
        return latest_path, max_steps
    
    def find_vecnorm_for_checkpoint(checkpoint_path):
        """尋找對應的 VecNormalize 檔案"""
        import os
        stage_name = os.path.basename(checkpoint_path).split('_')[0] + '_' + os.path.basename(checkpoint_path).split('_')[1] + '_' + os.path.basename(checkpoint_path).split('_')[2]
        vecnorm_path = f"checkpoints/{stage_name}_vecnorm.pkl"
        if os.path.exists(vecnorm_path):
            return vecnorm_path
        for stage in STAGES:
            stage_vecnorm = f"checkpoints/{stage['name']}_vecnorm.pkl"
            if os.path.exists(stage_vecnorm):
                return stage_vecnorm
        return None
    
    current_model_path = None
    current_vecnorm_path = None
    
    # ==================== 主訓練迴圈 ====================
    for i, stage in enumerate(STAGES):
        name = stage['name']
        grid_size = stage['grid_size']
        steps = stage['steps']
        lr_start = stage['lr_start']
        lr_end = stage['lr_end']
        ent_coef = stage['ent_coef']
        start_length = stage['start_length']
        endgame_prob = stage['endgame_prob']
        target_length = stage['target_length']
        
        # 檢查是否有 checkpoint 可以恢復
        checkpoint_path, steps_completed = find_latest_checkpoint(name)
        remaining_steps = max(0, steps - steps_completed)
        
        # 跳過已完成的階段
        if os.path.exists(f"checkpoints/{name}.zip") and os.path.exists(f"checkpoints/{name}_vecnorm.pkl"):
            print(f"\n{C.GREEN}[✓] 階段 {name} 已完成，跳過...{C.END}")
            current_model_path = f"checkpoints/{name}"
            current_vecnorm_path = f"checkpoints/{name}_vecnorm.pkl"
            continue
        
        # 從 checkpoint 恢復
        if checkpoint_path and steps_completed > 0:
            print(f"\n{C.YELLOW}[!] 找到 checkpoint: {checkpoint_path} ({steps_completed:,} 步已完成){C.END}")
            print(f"{C.YELLOW}    繼續訓練，剩餘 {remaining_steps:,} 步...{C.END}")
            current_model_path = checkpoint_path
            found_vecnorm = find_vecnorm_for_checkpoint(checkpoint_path)
            if found_vecnorm:
                current_vecnorm_path = found_vecnorm
        else:
            remaining_steps = steps
        
        print(f"\n{C.CYAN}{'='*60}{C.END}")
        print(f"{C.CYAN}🚀 階段 {i+1}/3: {name.upper()}{C.END}")
        print(f"{C.CYAN}   網格: {grid_size}x{grid_size} | 步數: {remaining_steps:,} | LR: {lr_start}→{lr_end}{C.END}")
        print(f"{C.CYAN}   Endgame: {endgame_prob*100:.0f}% | 起始長度: {start_length} | ent: {ent_coef}{C.END}")
        print(f"{C.CYAN}{'='*60}{C.END}")
        
        # ==================== 創建並行環境 ====================
        print(f"{C.DIM}  創建 {N_ENVS} 個並行環境...{C.END}")
        env = SubprocVecEnv([
            make_env(grid_size, k, seed=BASE_SEED, 
                    start_length=start_length, endgame_prob=endgame_prob) 
            for k in range(N_ENVS)
        ])
        env = VecMonitor(env)  # 監控環境（記錄統計）
        
        # 載入或創建 VecNormalize
        if current_vecnorm_path and os.path.exists(current_vecnorm_path):
            print(f"{C.GREEN}[✓] 載入 VecNormalize: {current_vecnorm_path}{C.END}")
            env = VecNormalize.load(current_vecnorm_path, env)
            env.training = True
        else:
            print(f"{C.YELLOW}  創建新的 VecNormalize...{C.END}")
            env = VecNormalize(env, norm_obs=True, norm_reward=True, 
                              clip_obs=10.0, clip_reward=100.0)
        
        # ==================== 創建或載入模型 ====================
        lr_schedule = linear_schedule(lr_start, lr_end)
        
        if current_model_path is None:
            print(f"{C.YELLOW}  創建新模型 (256×256)...{C.END}")
            model = MaskablePPO(
                "MlpPolicy",
                env,
                verbose=0,
                learning_rate=lr_schedule,
                batch_size=BATCH_SIZE,
                n_steps=N_STEPS,
                gamma=GAMMA,
                gae_lambda=GAE_LAMBDA,
                ent_coef=ent_coef,
                vf_coef=VF_COEF,
                clip_range=CLIP_RANGE,
                n_epochs=N_EPOCHS,
                max_grad_norm=MAX_GRAD_NORM,
                target_kl=0.03,  # 防止更新太激進
                policy_kwargs=POLICY_KWARGS,
                device=device,
                tensorboard_log="./snake_v10_logs/"
            )
        else:
            print(f"{C.GREEN}[✓] 載入模型: {current_model_path}{C.END}")
            model = MaskablePPO.load(current_model_path, env=env, device=device)
            
            # 更新學習率和熵係數
            model.lr_schedule = get_schedule_fn(lr_schedule)
            model.ent_coef = ent_coef
            print(f"{C.GREEN}  LR: {lr_start}→{lr_end} | ent_coef: {ent_coef}{C.END}")
        
        # ==================== 設定回呼函數 ====================
        callbacks = [
            V10ProgressCallback(name, remaining_steps, 
                               target_length=target_length, ent_coef=ent_coef),
            CheckpointCallback(
                save_freq=2_000_000 // N_ENVS,  # 每 200 萬步儲存
                save_path="./checkpoints/",
                name_prefix=name
            )
        ]
        
        print(f"{C.GREEN}[✓] 開始訓練!{C.END}")
        time.sleep(1)
        
        # ==================== 訓練 ====================
        try:
            model.learn(
                total_timesteps=remaining_steps,
                callback=callbacks,
                progress_bar=False,
                reset_num_timesteps=(steps_completed == 0)
            )
        except KeyboardInterrupt:
            print(f"\n{C.YELLOW}暫停中，正在儲存...{C.END}")
        
        # ==================== 儲存 ====================
        save_path = f"checkpoints/{name}"
        vecnorm_path = f"checkpoints/{name}_vecnorm.pkl"
        
        model.save(save_path)
        env.save(vecnorm_path)
        
        current_model_path = save_path
        current_vecnorm_path = vecnorm_path
        
        env.close()
        
        print(f"{C.GREEN}[✓] 階段 {i+1} 完成! 已儲存至 {save_path}{C.END}")
        
        # 最終儲存（加時間戳）
        if i == len(STAGES) - 1:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            model.save(f"checkpoints/v10_final_{ts}")
            print(f"{C.GREEN}[✓] 最終模型已儲存: v10_final_{ts}.zip{C.END}")
    
    print(f"\n{C.MAGENTA}{'='*60}{C.END}")
    print(f"{C.MAGENTA}  V10.0 訓練完成!{C.END}")
    print(f"{C.MAGENTA}{'='*60}{C.END}")
    print(f"\n{C.CYAN}執行: python watch_v10.py{C.END}")


if __name__ == "__main__":
    multiprocessing.freeze_support()  # Windows 多進程支援
    main()
