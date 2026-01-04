"""
🐍 Snake AI V10.0 "Phoenix" - Numba 加速演算法庫
================================================

這個檔案包含所有核心演算法的實現：
1. Hamiltonian Path 生成（保證安全的遍歷路徑）
2. BFS（廣度優先搜尋）可達性計算
3. Flood Fill 區域大小計算

所有演算法都使用 Numba JIT 編譯，達到接近 C++ 的執行速度。

技術特點：
- @njit 裝飾器：Just-In-Time 編譯，速度提升 100 倍
- 預分配緩衝區：避免重複 malloc，減少 GC 開銷
- 智能截斷：提前停止搜尋，節省計算時間
"""

import numpy as np
from numba import njit  # Numba 的 No-Python JIT 編譯器
import random

# =========================================================================
#                     全域常數（供 Numba 編譯時使用）
# =========================================================================

# 四方向鄰居偏移量 (上、下、左、右)
# 使用 numpy 陣列讓 Numba 能高效存取
DR = np.array([-1, 1, 0, 0], dtype=np.int32)  # 行變化
DC = np.array([0, 0, -1, 1], dtype=np.int32)  # 列變化


# =========================================================================
#                 HAMILTONIAN PATH 生成（圖論）
# =========================================================================

def make_serpentine_path(grid_size: int) -> tuple:
    """
    生成真正的 Hamiltonian CYCLE（首尾相鄰）
    
    ⚠️ 重要修復：
    舊版本是 Hamiltonian PATH，最後一格 (19,0) 與第一格 (0,0) 不相鄰！
    這會導致用 (idx+1) % N 計算下一步時出錯，造成 Autopilot 失敗。
    
    新版本是真正的 Hamiltonian CYCLE：
    - 最後一格 (1,0) 與第一格 (0,0) 相鄰
    - 用 (idx+1) % N 永遠可以正確找到「下一個相鄰格子」
    
    路徑結構（4x4 範例）：
    
     0 →  1 →  2 →  3     起點 (0,0)
                    ↓
    15    6 ←  5 ←  4     第 0 欄保留給回程
     ↑    ↓
    14    7 →  8 →  9     蛇形走 1~n-1 欄
     ↑              ↓
    13← 12← 11← 10        
     ↑
     └─ 沿著第 0 欄走回起點
    
    參數：
    - grid_size: 網格大小（必須是偶數）
    
    返回：
    - path_coords: 座標列表，按路徑順序排列
    - hc_idx: 2D 陣列，hc_idx[r][c] = 該座標在 cycle 中的索引
    """
    if grid_size % 2 != 0:
        raise ValueError(
            f"Hamiltonian Cycle 需要偶數網格大小。收到: {grid_size}。"
        )

    N = grid_size * grid_size
    path_coords = []
    hc_idx = np.zeros((grid_size, grid_size), dtype=np.int32)

    idx = 0

    # 步驟 1：第 0 列全走 (0,0) → (0,1) → ... → (0, n-1)
    r = 0
    for c in range(grid_size):
        path_coords.append((r, c))
        hc_idx[r, c] = idx
        idx += 1

    # 步驟 2：第 1 ~ n-1 列蛇形走，但只走第 1 ~ n-1 欄（第 0 欄留給回程）
    for r in range(1, grid_size):
        if r % 2 == 1:
            # 奇數列：從右往左 (n-1 → 1)
            for c in range(grid_size - 1, 0, -1):
                path_coords.append((r, c))
                hc_idx[r, c] = idx
                idx += 1
        else:
            # 偶數列：從左往右 (1 → n-1)
            for c in range(1, grid_size):
                path_coords.append((r, c))
                hc_idx[r, c] = idx
                idx += 1

    # 步驟 3：沿第 0 欄從底往上走回起點 (n-1,0) → (n-2,0) → ... → (1,0)
    for r in range(grid_size - 1, 0, -1):
        path_coords.append((r, 0))
        hc_idx[r, 0] = idx
        idx += 1

    # 驗證：確保走完所有格子
    assert idx == N, f"Hamiltonian Cycle 不完整: {idx} vs {N}"
    
    # 驗證：確保「每一對相鄰節點」都真的相鄰（曼哈頓距離 = 1）
    # 這個完整驗證能抓到任何路徑錯誤，不只是首尾
    for i in range(N):
        r1, c1 = path_coords[i]
        r2, c2 = path_coords[(i + 1) % N]  # 下一格（循環）
        manhattan_dist = abs(r1 - r2) + abs(c1 - c2)
        assert manhattan_dist == 1, \
            f"Cycle 在第 {i} 步不相鄰: {(r1,c1)} → {(r2,c2)}，距離={manhattan_dist}"
    
    return path_coords, hc_idx


def make_endgame_start(grid_size: int, length: int, path_coords: list, hc_idx: np.ndarray, 
                       grid_array: np.ndarray = None) -> tuple:
    """
    生成 Endgame 起始狀態
    
    這是「課程學習」的一部分：
    - 正常遊戲從長度 3 開始
    - Endgame 訓練從長蛇開始（例如長度 200）
    - 這讓 AI 學會處理困難的終盤局面
    
    生成方式（Growing-from-seed）：
    1. 在 HC 路徑上隨機選一個位置作為頭
    2. 沿著 HC 路徑反向生長蛇身
    3. 這樣生成的蛇保證不會自交（因為是沿著 HC 走的）
    
    參數：
    - grid_size: 網格大小
    - length: 目標蛇長度
    - path_coords: HC 路徑座標
    - hc_idx: HC 索引陣列
    - grid_array: 可選，預分配的網格陣列（用於效能優化）
    
    返回：
    - snake: 蛇身 deque（頭在前，尾在後）
    - grid_array: 更新後的網格
    - food: 食物座標
    """
    from collections import deque
    
    N = grid_size * grid_size
    length = min(length, N - 10)  # 至少留 10 格給食物
    
    # 隨機選擇蛇頭位置（在路徑的後半段，確保有空間生長）
    head_idx = random.randint(length, N - 1)
    
    # 沿 HC 路徑反向生成蛇身
    snake_positions = []
    for i in range(length):
        path_idx = (head_idx - i) % N  # 反向走
        snake_positions.append(path_coords[path_idx])
    
    snake = deque(snake_positions)
    
    # 初始化或清空網格
    if grid_array is None:
        grid_array = np.zeros((grid_size, grid_size), dtype=np.int8)
    else:
        grid_array.fill(0)  # 清空（就地操作，不分配新記憶體）
    
    # 在網格中標記蛇身
    for r, c in snake:
        grid_array[r, c] = 1
    
    # 隨機生成食物位置（不能在蛇身上）
    min_distance = 5  # 最小距離，避免開局就吃到
    offset = random.randint(min_distance, N - 1 - min_distance)
    food_idx = (head_idx + offset) % N
    
    forbidden = set(snake)  # 蛇身位置
    
    # 確保食物不在蛇身上
    attempts = 0
    while tuple(path_coords[food_idx]) in forbidden and attempts < N:
        food_idx = (food_idx + 1) % N
        attempts += 1
    
    food = tuple(path_coords[food_idx]) if attempts < N else None
    
    return snake, grid_array, food


# =========================================================================
#                 BFS 與 FLOOD FILL（搜尋演算法）
# =========================================================================

@njit(cache=True)
def compute_reachable_mask(grid, tail_r, tail_c):
    """
    反向 BFS：計算所有可以到達尾巴的格子
    
    這個函數使用 BFS（廣度優先搜尋）從尾巴出發，標記所有可達的格子。
    
    為什麼是「反向」？
    - 我們從尾巴出發，而不是從頭出發
    - 這樣可以判斷：如果蛇頭移動到某個位置，還能不能回到尾巴
    - 如果不能回到尾巴 = 蛇會被困住 = 這是危險的移動
    
    為什麼尾巴特殊？
    - 蛇移動時，尾巴會離開原位
    - 所以尾巴位置視為「可走」
    
    BFS 演算法步驟：
    1. 將起點（尾巴）放入佇列
    2. 從佇列取出一個位置
    3. 將該位置的四個鄰居（如果可走）加入佇列
    4. 重複步驟 2-3 直到佇列為空
    5. 被訪問過的位置就是「可達」的
    
    時間複雜度：O(N)，N = 網格總格子數
    空間複雜度：O(N)
    
    參數：
    - grid: 2D 網格陣列（0=空，1=障礙物）
    - tail_r, tail_c: 尾巴的座標
    
    返回：
    - reachable: 2D 陣列，1=可達，0=不可達
    """
    rows, cols = grid.shape
    reachable = np.zeros((rows, cols), dtype=np.int8)
    
    # 使用固定大小陣列實現佇列（Numba 不支援動態佇列）
    queue_r = np.zeros(rows * cols, dtype=np.int32)  # 行座標佇列
    queue_c = np.zeros(rows * cols, dtype=np.int32)  # 列座標佇列
    q_head = 0     # 佇列頭指標
    q_tail_idx = 0  # 佇列尾指標
    
    # 從尾巴開始 BFS
    queue_r[q_tail_idx] = tail_r
    queue_c[q_tail_idx] = tail_c
    q_tail_idx += 1
    reachable[tail_r, tail_c] = 1
    
    # 四方向鄰居
    dr = DR
    dc = DC
    
    # BFS 主迴圈
    while q_head < q_tail_idx:
        # 取出佇列頭
        r = queue_r[q_head]
        c = queue_c[q_head]
        q_head += 1
        
        # 檢查四個方向
        for i in range(4):
            nr, nc = r + dr[i], c + dc[i]
            
            # 邊界檢查
            if 0 <= nr < rows and 0 <= nc < cols:
                # 如果還沒訪問過
                if reachable[nr, nc] == 0:
                    # 可以走的條件：空格 或 是尾巴位置
                    if grid[nr, nc] == 0 or (nr == tail_r and nc == tail_c):
                        reachable[nr, nc] = 1
                        queue_r[q_tail_idx] = nr
                        queue_c[q_tail_idx] = nc
                        q_tail_idx += 1
                    
    return reachable


@njit(cache=True)
def get_flood_fill_area(grid, start_r, start_c, tail_r, tail_c):
    """
    Flood Fill：計算從某點出發能到達多少格子
    
    與 BFS 可達性的區別：
    - BFS 可達性：判斷「能不能」到達（True/False）
    - Flood Fill：計算「能到達多少格」（數量）
    
    用途：評估某個方向的「空間大小」
    - 空間大 = 有足夠迴旋餘地
    - 空間小 = 容易被困住
    
    這個資訊用於：
    1. Smart Fallback：當 AI 被困住時，選擇最大空間方向
    2. 觀察向量 [8-11]：告訴 AI 每個方向的空間大小
    
    時間複雜度：O(可達格子數)
    
    參數：
    - grid: 2D 網格
    - start_r, start_c: 起點座標
    - tail_r, tail_c: 尾巴座標（視為可走）
    
    返回：
    - count: 可達格子數量
    """
    rows, cols = grid.shape
    
    # 起點不合法
    if start_r < 0 or start_r >= rows or start_c < 0 or start_c >= cols:
        return np.int32(0)
    
    # 起點被阻擋（除非是尾巴）
    if grid[start_r, start_c] == 1 and not (start_r == tail_r and start_c == tail_c):
        return np.int32(0)
    
    # 初始化
    visited = np.zeros((rows, cols), dtype=np.int8)
    queue_r = np.zeros(rows * cols, dtype=np.int32)
    queue_c = np.zeros(rows * cols, dtype=np.int32)
    q_head = 0
    q_tail_idx = 0
    
    # 從起點開始
    queue_r[q_tail_idx] = start_r
    queue_c[q_tail_idx] = start_c
    q_tail_idx += 1
    visited[start_r, start_c] = 1
    count = np.int32(0)
    
    dr = np.array([-1, 1, 0, 0], dtype=np.int32)
    dc = np.array([0, 0, -1, 1], dtype=np.int32)
    
    while q_head < q_tail_idx:
        r = queue_r[q_head]
        c = queue_c[q_head]
        q_head += 1
        count += 1  # 計數
        
        for i in range(4):
            nr, nc = r + dr[i], c + dc[i]
            if 0 <= nr < rows and 0 <= nc < cols:
                if visited[nr, nc] == 0:
                    if grid[nr, nc] == 0 or (nr == tail_r and nc == tail_c):
                        visited[nr, nc] = 1
                        queue_r[q_tail_idx] = nr
                        queue_c[q_tail_idx] = nc
                        q_tail_idx += 1
    
    return count


# =========================================================================
#           高效能版本（使用預分配緩衝區，零記憶體分配）
# =========================================================================

@njit(cache=True)
def compute_reachable_mask_buffered(grid, tail_r, tail_c, reachable_buf, queue_r_buf, queue_c_buf):
    """
    高效能版反向 BFS：使用預分配緩衝區
    
    與 compute_reachable_mask 的邏輯完全相同，
    但使用外部傳入的緩衝區，完全消除 malloc/GC 開銷。
    
    為什麼要這樣？
    - 每步遊戲都要做 BFS
    - 創建陣列需要 malloc，釋放需要 GC
    - 重複使用緩衝區可以省下這些開銷
    - 在我們的測試中，速度提升約 30%
    
    參數：
    - grid: 2D 網格
    - tail_r, tail_c: 尾巴座標
    - reachable_buf: 預分配的結果陣列
    - queue_r_buf, queue_c_buf: 預分配的佇列陣列
    
    返回：
    - reachable_buf（就地修改，也返回引用）
    """
    rows, cols = grid.shape
    
    # 清空緩衝區（向量化操作，比迴圈快）
    reachable_buf[:, :] = 0
    
    q_head = 0
    q_tail_idx = 0
    
    # 從尾巴開始
    queue_r_buf[q_tail_idx] = tail_r
    queue_c_buf[q_tail_idx] = tail_c
    q_tail_idx += 1
    reachable_buf[tail_r, tail_c] = 1
    
    dr = DR
    dc = DC
    
    while q_head < q_tail_idx:
        r = queue_r_buf[q_head]
        c = queue_c_buf[q_head]
        q_head += 1
        
        for i in range(4):
            nr, nc = r + dr[i], c + dc[i]
            
            if 0 <= nr < rows and 0 <= nc < cols:
                if reachable_buf[nr, nc] == 0:
                    if grid[nr, nc] == 0 or (nr == tail_r and nc == tail_c):
                        reachable_buf[nr, nc] = 1
                        queue_r_buf[q_tail_idx] = nr
                        queue_c_buf[q_tail_idx] = nc
                        q_tail_idx += 1
                    
    return reachable_buf


@njit(cache=True)
def get_flood_fill_area_buffered(grid, start_r, start_c, tail_r, tail_c, 
                                  visited_buf, queue_r_buf, queue_c_buf, limit=0):
    """
    高效能版 Flood Fill：使用預分配緩衝區 + 智能截斷
    
    V10.3 新增 limit 參數：
    - limit=0：不截斷，計算完整區域
    - limit>0：當 count >= limit 時提前返回
    
    智能截斷的意義：
    - 如果區域 >= limit，表示「足夠大」
    - 不需要知道精確值，可以提前停止
    - 大幅節省 Endgame 時的計算時間
    
    例如：limit=100，實際區域=300
    - 不截斷：要遍歷 300 格
    - 智能截斷：遍歷 100 格就返回，省 67% 時間
    
    參數：
    - grid: 2D 網格
    - start_r, start_c: 起點座標
    - tail_r, tail_c: 尾巴座標
    - visited_buf: 預分配的訪問標記陣列
    - queue_r_buf, queue_c_buf: 預分配的佇列陣列
    - limit: 智能截斷閾值（0=不截斷）
    
    返回：
    - count: 可達格子數量（可能被截斷於 limit）
    """
    rows, cols = grid.shape
    
    # 起點不合法
    if start_r < 0 or start_r >= rows or start_c < 0 or start_c >= cols:
        return np.int32(0)
    
    # 起點被阻擋
    if grid[start_r, start_c] == 1 and not (start_r == tail_r and start_c == tail_c):
        return np.int32(0)
    
    # 清空緩衝區
    visited_buf[:, :] = 0
    
    q_head = 0
    q_tail_idx = 0
    
    queue_r_buf[q_tail_idx] = start_r
    queue_c_buf[q_tail_idx] = start_c
    q_tail_idx += 1
    visited_buf[start_r, start_c] = 1
    count = np.int32(0)
    
    dr = DR
    dc = DC
    
    while q_head < q_tail_idx:
        r = queue_r_buf[q_head]
        c = queue_c_buf[q_head]
        q_head += 1
        count += 1
        
        # 智能截斷：達到閾值就返回
        if limit > 0 and count >= limit:
            return np.int32(limit)  # 回傳 limit 表示「至少有這麼大」
        
        for i in range(4):
            nr, nc = r + dr[i], c + dc[i]
            if 0 <= nr < rows and 0 <= nc < cols:
                if visited_buf[nr, nc] == 0:
                    if grid[nr, nc] == 0 or (nr == tail_r and nc == tail_c):
                        visited_buf[nr, nc] = 1
                        queue_r_buf[q_tail_idx] = nr
                        queue_c_buf[q_tail_idx] = nc
                        q_tail_idx += 1
    
    return count


# =========================================================================
#                     緩衝區工廠函數
# =========================================================================

def create_bfs_buffers(grid_size: int) -> dict:
    """
    創建 BFS/Flood Fill 所需的預分配緩衝區
    
    這個函數應該在 SnakeEnv.__init__ 中調用一次，
    之後的所有 BFS 操作都重複使用這些緩衝區。
    
    為什麼預分配？
    - 每步遊戲都要做多次 BFS
    - 每次 np.zeros() 都會觸發記憶體分配
    - 預分配後重複使用，省下分配/釋放的開銷
    
    緩衝區說明：
    - reachable: 可達性標記陣列
    - visited: 訪問標記陣列（Flood Fill 用）
    - queue_r, queue_c: BFS 佇列的行列座標
    
    參數：
    - grid_size: 網格大小
    
    返回：
    - dict: 包含所有緩衝區的字典
    """
    N = grid_size * grid_size
    return {
        'reachable': np.zeros((grid_size, grid_size), dtype=np.int8),
        'visited': np.zeros((grid_size, grid_size), dtype=np.int8),
        'queue_r': np.zeros(N, dtype=np.int32),
        'queue_c': np.zeros(N, dtype=np.int32),
    }
