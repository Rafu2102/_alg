"""
🐍 Snake AI V10.0 "Phoenix" - 涅槃重生環境
============================================

這是貪吃蛇 AI 的核心遊戲環境，使用 OpenAI Gymnasium 標準介面。
AI 透過這個環境學習如何玩貪吃蛇遊戲。

主要功能：
1. 定義遊戲規則（蛇的移動、吃食物、死亡判定）
2. 生成 26 維觀察向量，供 AI 理解當前遊戲狀態
3. 計算獎勵信號，引導 AI 學習正確行為
4. 提供動作遮罩，防止 AI 做出必死的動作

技術特點：
- 26 維觀察空間（AI 的「眼睛」）
- Hamiltonian Path 整合（保證能走完全場的安全路徑）
- BFS/Flood Fill 演算法（計算可達區域）
- 動作遮罩（過濾危險動作）
"""

# ==================== 匯入必要的套件 ====================
import gymnasium as gym          # OpenAI 的遊戲環境標準介面
from gymnasium import spaces     # 定義觀察空間和動作空間
import numpy as np               # 數值計算
from collections import deque    # 雙端佇列，用於蛇身（頭尾操作都是 O(1)）
import random                    # 隨機數生成
import math                      # 數學函數

# 從工具檔案匯入核心演算法函數
from snake_utils_v10 import (
    make_serpentine_path,           # 生成 Hamiltonian Cycle（首尾相鄰的完整巡迴）
    make_endgame_start,             # 生成 Endgame 起始狀態（用於課程學習）
    create_bfs_buffers,             # 預分配 BFS 緩衝區（避免重複分配記憶體）
    get_flood_fill_area_buffered,   # 計算某方向的可達空間大小
    compute_reachable_mask_buffered, # 計算從尾巴可達的所有格子
)

# ==================== 遊戲常數設定 ====================
GRID_SIZE_DEFAULT = 20  # 預設網格大小 20x20，共 400 格
CELL_SIZE = 30          # 每格的像素大小（用於渲染）

# 顏色定義（RGB 格式，用於遊戲畫面渲染）
BG = (13, 27, 42)              # 背景色（深藍色）
GRID_COLOR = (40, 55, 75)      # 網格線顏色
SNAKE_COLOR = (126, 200, 227)  # 蛇身顏色（青色）
SNAKE_HEAD_COLOR = (167, 215, 197)  # 蛇頭顏色（淺綠色）
FOOD_COLOR = (244, 166, 160)   # 食物顏色（粉紅色）


class SnakeEnvV10(gym.Env):
    """
    V10.0 貪吃蛇環境 - 涅槃重生版
    
    這是一個符合 OpenAI Gymnasium 標準的遊戲環境類別。
    AI 訓練框架（如 Stable Baselines3）會透過這個介面與遊戲互動。
    
    核心方法：
    - reset(): 重置遊戲，開始新的一局
    - step(action): 執行一個動作，返回新狀態和獎勵
    - _get_observation(): 生成 26 維觀察向量
    - action_masks(): 返回哪些動作是安全的
    """
    
    # Gymnasium 標準元數據
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    # ==================== 動作定義 ====================
    # 將動作編號映射到座標變化 (行變化, 列變化)
    # 例如：動作 0 (UP) = 行 -1，列不變
    MOVES = {
        0: (-1, 0),  # UP    (上): 行減少
        1: (1, 0),   # DOWN  (下): 行增加
        2: (0, -1),  # LEFT  (左): 列減少
        3: (0, 1)    # RIGHT (右): 列增加
    }
    
    # 相反方向對照表（用於防止 180 度轉彎）
    # 蛇不能直接回頭，否則會撞到自己的脖子
    OPPOSITE = {
        0: 1,  # UP 的相反是 DOWN
        1: 0,  # DOWN 的相反是 UP
        2: 3,  # LEFT 的相反是 RIGHT
        3: 2   # RIGHT 的相反是 LEFT
    }
    
    DEBUG_MODE = False  # 除錯模式開關
    
    def __init__(self, render_mode=None, grid_size=GRID_SIZE_DEFAULT,
                 default_start_length=3, endgame_prob=0.0):
        """
        初始化遊戲環境
        
        參數：
        - render_mode: 渲染模式，"human" 表示顯示視窗
        - grid_size: 網格大小，預設 20x20
        - default_start_length: 課程學習中的預設起始長度
        - endgame_prob: 從長蛇開始的機率（用於 Endgame 訓練）
        """
        super().__init__()
        self.render_mode = render_mode
        
        # 課程學習參數：控制訓練的難度
        self.default_start_length = default_start_length  # 起始長度
        self.endgame_prob = endgame_prob  # Endgame 出現機率
        
        # ==================== 定義觀察空間和動作空間 ====================
        # 觀察空間：26 維向量，每個值在 [0, 1] 之間
        # 這是 AI 的「眼睛」，透過這 26 個數字理解遊戲狀態
        self.observation_space = spaces.Box(
            low=0,      # 最小值
            high=1,     # 最大值
            shape=(26,), # 26 維向量
            dtype=np.float32
        )
        
        # 動作空間：4 個離散動作（上、下、左、右）
        self.action_space = spaces.Discrete(4)
        
        # ==================== 初始化遊戲網格 ====================
        self.grid_size = grid_size
        # 網格陣列：0 = 空格，1 = 蛇身
        # 使用 int8 節省記憶體
        self.grid_array = np.zeros((grid_size, grid_size), dtype=np.int8)
        self.N = grid_size * grid_size  # 總格子數 = 400
        
        # ==================== 生成 Hamiltonian Cycle ====================
        # Hamiltonian Cycle 是一條經過所有格子恰好一次，且首尾相鄰的巡迴路徑
        # 如果蛇沿著這條路走，保證不會自撞，且可以用 (idx+1) % N 循環
        # path_coords: 按順序排列的座標列表
        # hc_idx: 每個座標在 cycle 中的索引（用於快速查詢）
        self.path_coords, self.hc_idx = make_serpentine_path(grid_size)
        
        # 驗證 HC path 是否完整
        assert len(self.path_coords) == self.N, f"HC path 不完整: {len(self.path_coords)} vs {self.N}"
        
        # ==================== 預分配 BFS 緩衝區 ====================
        # 為了效能，預先分配 BFS 演算法需要的記憶體
        # 避免每次計算都重新分配，大幅提升速度
        self._bfs_buffers = create_bfs_buffers(grid_size)
        
        # Flood Fill 快取（-1 表示尚未計算）
        self.flood_val_cache = np.full((grid_size, grid_size), -1, dtype=np.int32)
        
        # ==================== 飢餓限制設定 ====================
        # 如果蛇太久沒吃到食物，就會「餓死」
        # 這是為了防止 AI 學會「躺平」（一直繞圈不吃食物）
        self.base_starvation_limit = self.N * 4  # 1600 步
        
        # Sigmoid 獎勵排程參數（用於動態調整獎勵）
        self.sig_midpoint = self.N * 0.25
        self.sig_scale = self.sig_midpoint * 0.2
        self.lazy_threshold = max(50, int(self.N * 0.25))
        
        # ==================== 遊戲狀態變數 ====================
        self.snake = None          # 蛇身（deque，頭在前尾在後）
        self.food = None           # 食物位置 (row, col)
        self.direction = 0         # 當前方向（0=上, 1=下, 2=左, 3=右）
        self.score = 0             # 分數（吃到的食物數量）
        self.steps = 0             # 總步數
        self.steps_without_food = 0  # 連續沒吃到食物的步數
        
        # 快取版本控制（用於判斷是否需要重新計算）
        self._board_version = 0
        self._reach_cache_version = -1
        
        # 統計資訊
        self.fallback_count = 0    # Fallback 觸發次數
        self.ignored_180_count = 0  # 被忽略的 180 度轉彎次數
        
        # 渲染相關
        self.window = None
        self.clock = None
        
    def set_grid_size(self, size):
        """
        課程學習：動態調整網格大小
        
        參數：
        - size: 新的網格大小
        """
        self.grid_size = size
        self.grid_array = np.zeros((size, size), dtype=np.int8)
        self.N = size * size
        
        # 重新生成 HC cycle
        self.path_coords, self.hc_idx = make_serpentine_path(size)
        
        # 重新分配緩衝區
        self._bfs_buffers = create_bfs_buffers(size)
        self.flood_val_cache = np.full((size, size), -1, dtype=np.int32)
        
        # 更新限制（🔧 V10.12 修復：正確更新 base_starvation_limit）
        self.base_starvation_limit = self.N * 4
        self.sig_midpoint = self.N * 0.25
        self.sig_scale = self.sig_midpoint * 0.2
        self.lazy_threshold = max(50, int(self.N * 0.25))
        
        # 使快取失效
        self._board_version = 0
        self._reach_cache_version = -1
        
        print(f"🗺️  地圖大小調整: {size}x{size} | 飢餓限制: {self.base_starvation_limit}")

    def _get_dynamic_starvation_limit(self):
        """
        動態飢餓極限（V10.10 新增）
        
        原理：
        - 蛇短（<300）：維持高壓（1600步），強迫高效率
        - 蛇長（>=300）：給予額外時間，讓蛇有機會「解開」複雜的身體
        
        這是因為當蛇很長時，可能需要很多步才能繞過自己的身體到達食物
        """
        if self.snake is None:
            return self.base_starvation_limit
        
        current_len = len(self.snake)
        
        if current_len < 300:
            return self.base_starvation_limit  # 1600 步
        else:
            # 超過 300 後，每多 1 格給 100 步額外時間
            # 例如長度 380：1600 + 80*100 = 9600 步
            extra_time = (current_len - 300) * 100
            return self.base_starvation_limit + extra_time

    def _ensure_reach_cache(self):
        """
        確保可達性快取是最新的
        
        使用反向 BFS 從尾巴出發，計算所有可以到達尾巴的格子
        這用於判斷某個移動是否會把蛇困住
        """
        # 如果快取是最新的，直接返回
        if self._reach_cache_version == self._board_version:
            return
        
        if self.snake is None:
            return
        
        # 從尾巴開始做 BFS
        tail = self.snake[-1]
        compute_reachable_mask_buffered(
            self.grid_array, 
            tail[0], tail[1],  # 尾巴座標
            self._bfs_buffers['reachable'],  # 結果存在這個陣列
            self._bfs_buffers['queue_r'], 
            self._bfs_buffers['queue_c']
        )
        self._reach_cache_version = self._board_version

    def reset(self, seed=None, options=None):
        """
        重置遊戲環境，開始新的一局
        
        這是 Gymnasium 標準介面的一部分。
        每次遊戲結束後會調用這個方法開始新遊戲。
        
        參數：
        - seed: 隨機種子（用於可重現性）
        - options: 額外選項，可指定 start_length
        
        返回：
        - observation: 26 維觀察向量
        - info: 額外資訊字典
        """
        super().reset(seed=seed)
        
        # 設定隨機種子
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # ==================== 課程學習：決定起始長度 ====================
        if options is not None and "start_length" in options:
            # 如果有明確指定，使用指定值
            start_length = options["start_length"]
        elif self.endgame_prob > 0.0 and self.default_start_length > 3:
            # 根據機率決定是否從長蛇開始（Endgame 訓練）
            if np.random.random() < self.endgame_prob:
                # 加入隨機抖動 ±10，避免 AI 過擬合特定長度
                jitter = random.randint(-10, 10)
                start_length = max(10, self.default_start_length + jitter)
            else:
                start_length = 3  # 正常開始
        else:
            start_length = 3  # 預設長度
        
        # ==================== 生成蛇身和食物 ====================
        if start_length > 3:
            # Endgame 起始：沿 HC path 生成長蛇
            self.snake, _, self.food = make_endgame_start(
                self.grid_size, start_length, 
                self.path_coords, self.hc_idx,
                self.grid_array  # 就地寫入
            )
            # 根據 HC path 計算初始方向
            head = self.snake[0]
            head_hc = self.hc_idx[head[0], head[1]]
            next_hc = (head_hc + 1) % self.N
            next_pos = self.path_coords[next_hc]
            dr = next_pos[0] - head[0]
            dc = next_pos[1] - head[1]
            if dr == -1: self.direction = 0    # UP
            elif dr == 1: self.direction = 1   # DOWN
            elif dc == -1: self.direction = 2  # LEFT
            else: self.direction = 3           # RIGHT
        else:
            # 正常起始：沿 HC path 的前三格
            self.grid_array.fill(0)  # 清空網格
            self.snake = deque([
                self.path_coords[2],  # 頭
                self.path_coords[1],  # 身體
                self.path_coords[0],  # 尾
            ])
            # 在網格中標記蛇身
            for r, c in self.snake:
                self.grid_array[r, c] = 1
            
            # 計算初始方向
            head_pos = self.path_coords[2]
            next_pos = self.path_coords[3]
            dr = next_pos[0] - head_pos[0]
            dc = next_pos[1] - head_pos[1]
            if dr == -1: self.direction = 0
            elif dr == 1: self.direction = 1
            elif dc == -1: self.direction = 2
            else: self.direction = 3
            
            self._spawn_food()  # 生成食物
            
        # 重置遊戲狀態
        self.score = 0
        self.steps = 0
        self.steps_without_food = 0
        self.fallback_count = 0
        self.ignored_180_count = 0
        
        # 使快取失效
        self._board_version += 1
        self._reach_cache_version = -1
        self.flood_val_cache.fill(-1)
        
        return self._get_observation(), {"length": len(self.snake), "fallback_count": self.fallback_count}

    def action_masks(self):
        """
        動作遮罩（Action Masking）
        
        這個函數返回一個布林陣列，指示哪些動作是「安全」的。
        AI 只能選擇被標記為 True 的動作。
        
        遮罩邏輯：
        1. 禁止 180 度轉彎（會撞到脖子）
        2. 禁止撞牆
        3. 禁止撞到自己（尾巴除外，因為尾巴會移動）
        4. 禁止走進「死路」（BFS 無法到達尾巴的位置）
        
        如果所有動作都被禁止，啟用 Smart Fallback：
        選擇可達空間最大的方向（「死得最慢」）
        
        返回：
        - masks: [True, True, True, True] 格式的布林列表
        """
        # 確保可達性快取是最新的
        self._ensure_reach_cache()
        reachable_mask = self._bfs_buffers['reachable']
        
        masks = [True, True, True, True]  # 預設全部可行
        head = self.snake[0]
        tail = self.snake[-1]
        opposite = self.OPPOSITE.get(self.direction)  # 相反方向
        
        for i in range(4):  # 檢查四個方向
            # A. 禁止 180 度轉彎
            if i == opposite:
                masks[i] = False
                continue
            
            # 計算該方向的新位置
            dr, dc = self.MOVES[i]
            nr, nc = head[0] + dr, head[1] + dc
            
            # B. 邊界檢查（禁止撞牆）
            if not (0 <= nr < self.grid_size and 0 <= nc < self.grid_size):
                masks[i] = False
                continue
            
            # C. 障礙物檢查（禁止自撞，但尾巴位置是安全的）
            # 因為當蛇移動時，尾巴會離開原位
            if self.grid_array[nr, nc] == 1 and (nr, nc) != tail:
                masks[i] = False
                continue
            
            # D. BFS 可達性檢查（禁止死路）
            # 如果從該位置無法到達尾巴，代表會被困住
            if reachable_mask[nr, nc] == 0:
                masks[i] = False
                
        # ==================== Smart Fallback ====================
        # 如果所有方向都被禁止，選擇「死得最慢」的方向
        if not any(masks):
            areas = []
            for i in range(4):
                dr, dc = self.MOVES[i]
                nr, nc = head[0] + dr, head[1] + dc
                
                # 檢查是否物理上可行
                if (nr < 0 or nr >= self.grid_size or nc < 0 or nc >= self.grid_size or 
                    (self.grid_array[nr, nc] == 1 and (nr, nc) != tail)):
                    areas.append(-1)  # 不可行
                else:
                    # 計算該方向的可達空間大小
                    if self.flood_val_cache[nr, nc] != -1:
                        area = self.flood_val_cache[nr, nc]  # 使用快取
                    else:
                        area = get_flood_fill_area_buffered(
                            self.grid_array, nr, nc, tail[0], tail[1],
                            self._bfs_buffers['visited'], 
                            self._bfs_buffers['queue_r'], 
                            self._bfs_buffers['queue_c']
                        )
                        self.flood_val_cache[nr, nc] = area  # 存入快取
                    areas.append(area)
            
            # 選擇空間最大的方向
            best_idx = int(np.argmax(np.array(areas)))
            if areas[best_idx] > -1:
                masks[best_idx] = True
                self.fallback_count += 1
            else:
                # 終極 Fallback：放寬可達性要求
                relaxed = [False, False, False, False]
                for i in range(4):
                    if i == opposite:
                        continue
                    dr, dc = self.MOVES[i]
                    nr, nc = head[0] + dr, head[1] + dc
                    if not (0 <= nr < self.grid_size and 0 <= nc < self.grid_size):
                        continue
                    if self.grid_array[nr, nc] == 1 and (nr, nc) != tail:
                        continue
                    relaxed[i] = True
                
                if any(relaxed):
                    masks = relaxed
                    self.fallback_count += 1
                # 如果還是沒有，蛇已經完全被困住了
                
        return masks

    def step(self, action):
        """
        執行一個動作
        
        這是 Gymnasium 標準介面的核心方法。
        AI 透過這個方法與遊戲互動。
        
        參數：
        - action: 動作編號 (0=上, 1=下, 2=左, 3=右)
        
        返回：
        - observation: 新的 26 維觀察向量
        - reward: 獎勵值
        - terminated: 遊戲是否結束
        - truncated: 是否被截斷
        - info: 額外資訊
        """
        action = int(action)
        
        # ==================== 180 度保護 ====================
        # 如果 AI 嘗試回頭，忽略這個動作，改為繼續前進
        if action == self.OPPOSITE.get(self.direction):
            self.ignored_180_count += 1
            action = self.direction
        
        # 更新步數計數器
        self.steps += 1
        self.steps_without_food += 1
        self.direction = action
        
        # ==================== 計算新位置 ====================
        dr, dc = self.MOVES[action]
        prev_head = self.snake[0]
        new_head = (prev_head[0] + dr, prev_head[1] + dc)
        tail = self.snake[-1]
        
        # ==================== 死亡檢查 ====================
        
        # 1. 撞牆檢查
        if not (0 <= new_head[0] < self.grid_size and 0 <= new_head[1] < self.grid_size):
            return self._get_observation(), -10.0, True, False, {"length": len(self.snake), "fallback_count": self.fallback_count}
        
        # 2. 自撞檢查（尾巴位置除外）
        if self.grid_array[new_head[0], new_head[1]] == 1 and new_head != tail:
            return self._get_observation(), -10.0, True, False, {"length": len(self.snake), "fallback_count": self.fallback_count}
        
        # ==================== 蛇還活著，執行移動 ====================
        # 將新頭部加入蛇身（O(1) 操作）
        self.snake.appendleft(new_head)
        self.grid_array[new_head[0], new_head[1]] = 1
        
        # 使快取失效（因為盤面改變了）
        self._board_version += 1
        self._reach_cache_version = -1
        self.flood_val_cache.fill(-1)
        
        reward = 0.0
        length = len(self.snake)
        
        # ==================== 獎勵計算 ====================
        # V10.3 "Pure Hustle" 模式：固定獎勵值
        food_w = 1.0  # 吃食物獎勵
        
        if new_head == self.food:
            # 吃到食物！
            self.score += 1
            self.steps_without_food = 0  # 重置飢餓計數器
            self._spawn_food()  # 生成新食物
            reward = food_w  # +1.0
        else:
            # 沒吃到食物，移除尾巴（蛇保持原長度）
            old_tail = self.snake.pop()  # O(1) 操作
            if old_tail != new_head:  # 避免吃到尾巴時的錯誤
                self.grid_array[old_tail[0], old_tail[1]] = 0
            
            # 每步小懲罰，鼓勵效率（避免 AI 躺平繞圈）
            reward -= 0.02
        
        # ==================== 飢餓檢查 ====================
        # 使用動態極限（長蛇給予更多時間）
        current_limit = self._get_dynamic_starvation_limit()
        if self.steps_without_food > current_limit:
            return self._get_observation(), -10.0, True, False, {"length": length, "fallback_count": self.fallback_count}
        
        truncated = False
        
        # 除錯模式：驗證資料一致性
        if self.DEBUG_MODE:
            grid_count = np.sum(self.grid_array == 1)
            snake_len = len(self.snake)
            assert grid_count == snake_len, f"資料不一致: 網格={grid_count}, 蛇={snake_len}"
            assert len(set(self.snake)) == snake_len, "蛇身有重複座標！"
            
        return self._get_observation(), reward, False, truncated, {"length": length, "fallback_count": self.fallback_count}

    def _get_observation(self):
        """
        生成 26 維觀察向量
        
        這是 AI 的「眼睛」，透過這 26 個數字理解遊戲狀態。
        所有值都正規化到 [0, 1] 範圍。
        
        26 維觀察空間結構：
        [0-3]   四方向危險偵測（是否會死？）
        [4-7]   四方向食物方向（食物在哪？）
        [8-11]  四方向 Flood Fill 面積（該方向有多少空間？）
        [12-15] 四方向 BFS 可達性（能否從尾巴到達？）
        [16]    飢餓進度（多久沒吃到食物？）
        [17]    頭部可達總面積
        [18-21] 四方向空間損失警告
        [22-25] Hamiltonian Cycle 特徵
        
        返回：
        - obs: 26 維 float32 向量
        """
        head = self.snake[0]
        tail = self.snake[-1]
        
        # 確保可達性快取是最新的
        self._ensure_reach_cache()
        reachable_mask = self._bfs_buffers['reachable']
        
        obs = np.zeros(26, dtype=np.float32)
        
        # 智能截斷閾值：至少看 64 格，或蛇身長度的 1.5 倍
        smart_limit = max(64, int(len(self.snake) * 1.5))
        tail_r, tail_c = tail[0], tail[1]
        
        # ==================== 四方向特徵 ====================
        for i in range(4):
            dr, dc = self.MOVES[i]
            nr, nc = head[0] + dr, head[1] + dc
            valid = (0 <= nr < self.grid_size and 0 <= nc < self.grid_size)
            
            # [0-3] 危險偵測：該方向是否會死？
            danger = not valid or (self.grid_array[nr, nc] == 1 and (nr, nc) != tail)
            obs[i] = 1.0 if danger else 0.0
            
            # [4-7] 食物方向：食物相對於頭的位置
            if self.food:
                if i == 0: is_dir = self.food[0] < head[0]    # 上：食物在上方？
                elif i == 1: is_dir = self.food[0] > head[0]  # 下：食物在下方？
                elif i == 2: is_dir = self.food[1] < head[1]  # 左：食物在左邊？
                else: is_dir = self.food[1] > head[1]         # 右：食物在右邊？
                obs[4 + i] = 1.0 if is_dir else 0.0
            
            # [8-11] Flood Fill 面積：該方向有多少可用空間？
            if valid and reachable_mask[nr, nc] == 1:
                if self.flood_val_cache[nr, nc] != -1:
                    area = self.flood_val_cache[nr, nc]  # 使用快取
                else:
                    area = get_flood_fill_area_buffered(
                        self.grid_array, nr, nc, tail_r, tail_c,
                        self._bfs_buffers['visited'], 
                        self._bfs_buffers['queue_r'], 
                        self._bfs_buffers['queue_c'],
                        limit=smart_limit  # 智能截斷
                    )
                    self.flood_val_cache[nr, nc] = area
                obs[8 + i] = min(1.0, area / smart_limit)  # 正規化
            else:
                obs[8 + i] = 0.0
            
            # [12-15] BFS 可達性：能否從該位置到達尾巴？
            if valid:
                obs[12 + i] = float(reachable_mask[nr, nc])
            else:
                obs[12 + i] = 0.0
                
        # [16] 飢餓進度：已多久沒吃到食物？
        current_limit = self._get_dynamic_starvation_limit()
        obs[16] = min(1.0, self.steps_without_food / current_limit)
        
        # [17] 頭部可達總面積
        head_area = self._compute_head_reachable_area(smart_limit)
        obs[17] = min(1.0, head_area / smart_limit)
        
        # [18-21] 空間損失警告：該方向會減少多少可達空間？
        for i in range(4):
            if obs[i] == 0.0:  # 如果該方向不危險
                dr, dc = self.MOVES[i]
                nr, nc = head[0] + dr, head[1] + dc
                
                if self.flood_val_cache[nr, nc] != -1:
                    next_area = self.flood_val_cache[nr, nc]
                else:
                    next_area = get_flood_fill_area_buffered(
                        self.grid_array, nr, nc, tail_r, tail_c,
                        self._bfs_buffers['visited'], 
                        self._bfs_buffers['queue_r'], 
                        self._bfs_buffers['queue_c'],
                        limit=smart_limit
                    )
                    self.flood_val_cache[nr, nc] = next_area
                
                # 如果空間損失超過 1 格，發出警告
                obs[18 + i] = 1.0 if next_area < (head_area - 1) else 0.0
            else:
                obs[18 + i] = 1.0  # 被阻擋 = 空間歸零
            
        # ==================== Hamiltonian Cycle 特徵 ====================
        # 這些特徵幫助 AI 理解「安全路徑」的資訊
        head_hc = self.hc_idx[head[0], head[1]]  # 頭在 HC 的位置
        tail_hc = self.hc_idx[tail[0], tail[1]]  # 尾在 HC 的位置
        
        obs[22] = head_hc / (self.N - 1)  # 正規化的頭 HC 索引
        obs[23] = tail_hc / (self.N - 1)  # 正規化的尾 HC 索引
        
        if self.food:
            food_hc = self.hc_idx[self.food[0], self.food[1]]
            obs[24] = food_hc / (self.N - 1)  # 正規化的食物 HC 索引
            
            # 計算 HC 上的循環距離
            diff = abs(food_hc - head_hc)
            cyclic_dist = min(diff, self.N - diff)
            obs[25] = np.clip(cyclic_dist / (self.N / 2), 0.0, 1.0)
        else:
            obs[24] = 0.0
            obs[25] = 0.0
        
        return obs
    
    def _compute_head_reachable_area(self, limit=0):
        """
        計算從頭部出發能到達多少格子
        
        參數：
        - limit: 智能截斷閾值，0 表示不截斷
        
        返回：
        - area: 可達格子數量
        """
        head = self.snake[0]
        tail = self.snake[-1]
        return int(get_flood_fill_area_buffered(
            self.grid_array, head[0], head[1], tail[0], tail[1],
            self._bfs_buffers['visited'], 
            self._bfs_buffers['queue_r'], 
            self._bfs_buffers['queue_c'],
            limit=limit
        ))

    def _spawn_food(self):
        """
        在隨機空格生成食物
        
        使用 Rejection Sampling：
        1. 隨機選一個位置
        2. 如果是空格就放食物
        3. 否則重試（最多 32 次）
        4. 如果還是失敗，用 argwhere 找所有空格
        
        這個方法在早期（空格多）幾乎一次命中，效率很高
        """
        # 嘗試 32 次隨機選擇
        for _ in range(32):
            r = random.randrange(self.grid_size)
            c = random.randrange(self.grid_size)
            if self.grid_array[r, c] == 0:  # 空格
                self.food = (r, c)
                return
        
        # 保底方案：找出所有空格
        empty = np.argwhere(self.grid_array == 0)
        if len(empty) > 0:
            idx = random.randint(0, len(empty) - 1)
            self.food = (int(empty[idx][0]), int(empty[idx][1]))
        else:
            self.food = None  # 沒有空格了（理論上蛇填滿了整個場地）
    
    # ==================== 輔助方法（供視覺化使用）====================
    def get_snake(self):
        """返回蛇身座標列表"""
        return list(self.snake)
    
    def get_food(self):
        """返回食物座標"""
        return self.food
    
    def get_grid(self):
        """返回網格陣列的副本"""
        return self.grid_array.copy()
    
    def get_direction(self):
        """返回當前方向"""
        return self.direction
    
    def get_hc_idx(self):
        """返回 HC 索引陣列"""
        return self.hc_idx
    
    def get_path_coords(self):
        """返回 HC 路徑座標"""
        return self.path_coords

    def render(self):
        """
        渲染遊戲畫面
        
        使用 Pygame 繪製遊戲視窗。
        只在 render_mode="human" 時執行。
        """
        if self.render_mode != "human":
            return
        
        # Lazy import（延遲載入）：加速環境創建
        import pygame
            
        if self.window is None:
            pygame.init()
            size = self.grid_size * CELL_SIZE
            self.window = pygame.display.set_mode((size, size))
            pygame.display.set_caption("Snake AI V10.0")
            self.clock = pygame.time.Clock()
        
        self.window.fill(BG)  # 填充背景色
        
        # 繪製網格線
        for i in range(self.grid_size + 1):
            pygame.draw.line(self.window, GRID_COLOR, 
                           (i * CELL_SIZE, 0), 
                           (i * CELL_SIZE, self.grid_size * CELL_SIZE))
            pygame.draw.line(self.window, GRID_COLOR, 
                           (0, i * CELL_SIZE), 
                           (self.grid_size * CELL_SIZE, i * CELL_SIZE))
        
        # 繪製蛇身
        for idx, (r, c) in enumerate(self.snake):
            color = SNAKE_HEAD_COLOR if idx == 0 else SNAKE_COLOR
            rect = pygame.Rect(c * CELL_SIZE + 2, r * CELL_SIZE + 2, 
                             CELL_SIZE - 4, CELL_SIZE - 4)
            pygame.draw.rect(self.window, color, rect, border_radius=4)
        
        # 繪製食物
        if self.food:
            r, c = self.food
            rect = pygame.Rect(c * CELL_SIZE + 4, r * CELL_SIZE + 4, 
                             CELL_SIZE - 8, CELL_SIZE - 8)
            pygame.draw.rect(self.window, FOOD_COLOR, rect, border_radius=8)
        
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        """關閉遊戲視窗"""
        if self.window is not None:
            import pygame
            pygame.quit()
            self.window = None
