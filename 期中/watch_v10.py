"""
🐍 SNAKE AI V10.0 "PHOENIX" - 豪華觀察器
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Engine: Pygame
基於 V6.0 核心 + V9.0 優化

✓ 26 維觀察空間
✓ HC 索引距離顯示
✓ 粒子特效
✓ VecNormalize 支援
✓ 可調倍速 (1-8 鍵或上下鍵)
✓ 繁體中文鍵盤支援
✓ 400 分勝利畫面
✓ Endgame Autopilot
"""
import pygame
import pygame.gfxdraw
import math
import random
import time
import sys
import os
import numpy as np
import traceback  # V10.14: 用於顯示詳細錯誤

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from snake_env_v10 import SnakeEnvV10

# ══════════════════════════════════════════════════════════════
#                        CONFIGURATION
# ══════════════════════════════════════════════════════════════

DEBUG = False

# Model configs (priority order)
MODEL_CONFIGS = [
    # 🏆 最佳模型 (平均 388 分)
    {"model": "checkpoints/v10_stage_b_best_avg_388", "vecnorm": "checkpoints/v10_stage_b_vecnorm.pkl", "grid_size": 20},
    {"model": "checkpoints/v10_stage_b_milestone_400", "vecnorm": "checkpoints/v10_stage_b_vecnorm.pkl", "grid_size": 20},
    {"model": "checkpoints/v10_stage_b_final", "vecnorm": "checkpoints/v10_stage_b_vecnorm.pkl", "grid_size": 20},
    {"model": "checkpoints/v10_stage_b", "vecnorm": "checkpoints/v10_stage_b_vecnorm.pkl", "grid_size": 20},
    {"model": "checkpoints/v10_stage_a", "vecnorm": "checkpoints/v10_stage_a_vecnorm.pkl", "grid_size": 20},
]

GRID_SIZE = 20
CELL_SIZE = 32
PANEL_WIDTH = 300

DEATH_DELAY = 3.0

# 🔥 速度預設 (可用 1-9, 0 或上下鍵調整) - 最高 50x
SPEED_LEVELS = [
    0.200,   # 1x
    0.150,   # 2x
    0.100,   # 3x
    0.070,   # 4x
    0.050,   # 5x
    0.035,   # 6x
    0.025,   # 7x
    0.018,   # 8x
    0.012,   # 9x
    0.008,   # 10x
    0.006,   # 11x
    0.005,   # 12x
    0.004,   # 13x
    0.0035,  # 14x
    0.003,   # 15x
    0.0025,  # 16x
    0.002,   # 17x
    0.0015,  # 18x
    0.001,   # 19x
    0.0008,  # 20x
    0.0006,  # 21x
    0.0005,  # 22x
    0.0004,  # 23x
    0.00035, # 24x
    0.0003,  # 25x
    0.00025, # 26x
    0.0002,  # 27x
    0.00018, # 28x
    0.00015, # 29x
    0.00012, # 30x
    0.0001,  # 31x
    0.00009, # 32x
    0.00008, # 33x
    0.00007, # 34x
    0.00006, # 35x
    0.00005, # 36x
    0.00004, # 37x
    0.00003, # 38x
    0.00002, # 39x
    0.00001, # 40x
    0.000008,# 41x
    0.000006,# 42x
    0.000005,# 43x
    0.000004,# 44x
    0.000003,# 45x
    0.000002,# 46x
    0.000001,# 47x
    0.0000005,# 48x
    0.0000001,# 49x
    0.0,     # 50x (MAX! 無延遲)
]
DEFAULT_SPEED_LEVEL = 2  # 預設 3 (0.1秒/步)

# ══════════════════════════════════════════════════════════════
#                     COLOR PALETTE
# ══════════════════════════════════════════════════════════════

class Colors:
    BG_DARK = (8, 10, 20)
    GRID_LINE = (25, 35, 55)
    GRID_GLOW = (40, 60, 100)
    
    # 🐍 蛇身漸變色
    SNAKE_HEAD = (50, 255, 100)      # 亮綠色頭部
    SNAKE_BODY_START = (0, 220, 255)  # 青色
    SNAKE_BODY_END = (120, 80, 200)   # 紫色
    SNAKE_TAIL = (255, 80, 120)       # 粉紅色尾巴
    
    # 🍎 食物
    FOOD_CORE = (255, 50, 150)
    FOOD_GLOW = (255, 100, 200)
    FOOD_OUTER = (255, 150, 220)
    
    # ✨ 粒子
    PARTICLE_GOLD = (255, 220, 100)
    PARTICLE_PINK = (255, 100, 200)
    PARTICLE_CYAN = (0, 255, 255)
    PARTICLE_GREEN = (100, 255, 150)
    
    # 🏆 勝利
    VICTORY_GOLD = (255, 215, 0)
    VICTORY_GLOW = (255, 255, 150)
    
    # 📊 文字
    TEXT_WHITE = (240, 245, 255)
    TEXT_CYAN = (0, 220, 255)
    TEXT_DIM = (100, 110, 140)
    TEXT_RED = (255, 80, 80)
    TEXT_GREEN = (80, 255, 120)
    TEXT_YELLOW = (255, 220, 100)
    TEXT_PURPLE = (180, 120, 255)

# ══════════════════════════════════════════════════════════════
#                     PARTICLE SYSTEM
# ══════════════════════════════════════════════════════════════

class Particle:
    def __init__(self, x, y, color, size=4, lifetime=1.0, gravity=200, speed_range=(100, 300)):
        self.x = x
        self.y = y
        self.color = color
        self.size = size
        self.max_life = lifetime
        self.life = lifetime
        
        angle = random.uniform(0, math.pi * 2)
        speed = random.uniform(*speed_range)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.gravity = gravity
    
    def update(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.vy += self.gravity * dt
        self.life -= dt
        return self.life > 0
    
    def draw(self, surface):
        alpha = self.life / self.max_life
        size = max(1, int(self.size * alpha))
        if size > 0:
            # 發光效果
            glow_size = size + 3
            glow_color = (*self.color, int(50 * alpha))
            pygame.gfxdraw.filled_circle(surface, int(self.x), int(self.y), 
                                         glow_size, glow_color)
            pygame.gfxdraw.filled_circle(surface, int(self.x), int(self.y), 
                                         size, (*self.color, int(255 * alpha)))

# 🌟 勝利用的特殊粒子
class VictoryParticle:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.color = random.choice([Colors.VICTORY_GOLD, Colors.PARTICLE_CYAN, Colors.PARTICLE_GREEN])
        self.size = random.randint(3, 8)
        self.life = random.uniform(1.5, 3.0)
        self.max_life = self.life
        
        angle = random.uniform(0, math.pi * 2)
        speed = random.uniform(50, 150)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed - 100  # 向上飄
        self.phase = random.uniform(0, math.pi * 2)
    
    def update(self, dt):
        self.x += self.vx * dt + math.sin(time.time() * 3 + self.phase) * 20 * dt
        self.y += self.vy * dt
        self.vy += 30 * dt  # 輕微下墜
        self.life -= dt
        return self.life > 0
    
    def draw(self, surface):
        alpha = self.life / self.max_life
        size = max(1, int(self.size * (0.5 + 0.5 * alpha)))
        # 閃爍效果
        flicker = 0.7 + 0.3 * math.sin(time.time() * 10 + self.phase)
        color_alpha = int(255 * alpha * flicker)
        pygame.gfxdraw.filled_circle(surface, int(self.x), int(self.y), 
                                     size + 2, (*self.color, int(color_alpha * 0.3)))
        pygame.gfxdraw.filled_circle(surface, int(self.x), int(self.y), 
                                     size, (*self.color, color_alpha))

# ══════════════════════════════════════════════════════════════
#                     MAIN GAME CLASS
# ══════════════════════════════════════════════════════════════

class SnakeGameV10:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Snake AI V10.0 - Phoenix 🐍")
        
        self.grid_size = GRID_SIZE
        self.cell_size = CELL_SIZE
        self.panel_width = PANEL_WIDTH
        
        self.game_width = self.grid_size * self.cell_size
        self.screen_width = self.game_width + self.panel_width
        self.screen_height = self.grid_size * self.cell_size
        
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        
        # Fonts (使用支援中文的字型，備選英文字型)
        # 嘗試微軟正黑體，如果不存在則用 Arial
        try:
            self.font_large = pygame.font.SysFont("Microsoft JhengHei", 28, bold=True)
            self.font_medium = pygame.font.SysFont("Microsoft JhengHei", 20)
            self.font_small = pygame.font.SysFont("Microsoft JhengHei", 15)
            self.font_victory = pygame.font.SysFont("Microsoft JhengHei", 42, bold=True)
        except:
            self.font_large = pygame.font.SysFont("Arial", 28, bold=True)
            self.font_medium = pygame.font.SysFont("Arial", 20)
            self.font_small = pygame.font.SysFont("Arial", 15)
            self.font_victory = pygame.font.SysFont("Arial", 42, bold=True)
        
        # State
        self.particles = []
        self.victory_particles = []
        self.shake_offset = [0, 0]
        self.shake_decay = 0.9
        
        self.generation = 0
        self.best_length = 3
        self.total_score = 0
        self.current_length = 3
        
        self.is_dead = False
        self.death_time = 0
        self.is_paused = False
        self.is_victory = False  # 🏆 勝利狀態
        self.victory_time = 0
        
        # 🔥 速度控制
        self.speed_level = DEFAULT_SPEED_LEVEL
        self.ai_step_delay = SPEED_LEVELS[self.speed_level]
        
        # V6.0 Metrics
        self.reachable_area = 0
        self.hc_distance = 0.0
        self.head_hc_idx = 0
        self.food_hc_idx = 0
        
        # Load model and environment
        self._load_model()
        self._reset()
    
    def _load_model(self):
        """Load AI model with VecNormalize"""
        for config in MODEL_CONFIGS:
            model_path = config["model"] + ".zip"
            vecnorm_path = config["vecnorm"]
            grid_size = config["grid_size"]
            
            if os.path.exists(model_path):
                print(f"Loading model: {model_path}")
                
                self.grid_size = grid_size
                self.game_width = self.grid_size * self.cell_size
                self.screen_width = self.game_width + self.panel_width
                self.screen_height = self.grid_size * self.cell_size
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
                
                # Create environment
                def mask_fn(env):
                    return env.action_masks()
                
                def make_env():
                    env = SnakeEnvV10(grid_size=grid_size)
                    env = ActionMasker(env, mask_fn)
                    return env
                
                self.env = DummyVecEnv([make_env])
                
                # Load VecNormalize if available
                if os.path.exists(vecnorm_path):
                    print(f"Loading VecNormalize: {vecnorm_path}")
                    self.env = VecNormalize.load(vecnorm_path, self.env)
                    self.env.training = False
                    self.env.norm_reward = False
                
                # Get reference to underlying SnakeEnvV10
                self.unwrapped_env = self.env.envs[0].env
                
                # Load model
                self.model = MaskablePPO.load(model_path, env=self.env)
                print(f"Model loaded! Grid: {grid_size}x{grid_size}")
                return
        
        print("找不到已訓練模型! 請先執行 train_v10.py")
        sys.exit(1)
    
    def _reset(self):
        """Reset environment"""
        self.obs = self.env.reset()
        self.is_dead = False
        self.death_time = 0
        self.is_victory = False
        self.victory_time = 0
        self.generation += 1
        self.current_length = len(self.unwrapped_env.snake)
        self.particles.clear()
        self.victory_particles.clear()
    
    def _spawn_particles(self, x, y, count=25):
        """Create eat particles (精美版)"""
        for _ in range(count):
            color = random.choice([Colors.PARTICLE_GOLD, Colors.PARTICLE_PINK, Colors.PARTICLE_CYAN])
            size = random.randint(3, 7)
            self.particles.append(Particle(x, y, color, size=size, lifetime=1.2))
    
    def _spawn_victory_particles(self):
        """持續產生勝利粒子"""
        for _ in range(5):
            x = random.randint(0, self.game_width)
            y = random.randint(0, self.screen_height)
            self.victory_particles.append(VictoryParticle(x, y))
    
    def _add_shake(self, intensity=5):
        """Add screen shake"""
        self.shake_offset = [
            random.uniform(-intensity, intensity),
            random.uniform(-intensity, intensity)
        ]
    
    def _update_particles(self, dt):
        """Update particle system"""
        self.particles = [p for p in self.particles if p.update(dt)]
        self.victory_particles = [p for p in self.victory_particles if p.update(dt)]
        
        self.shake_offset[0] *= self.shake_decay
        self.shake_offset[1] *= self.shake_decay
    
    def _draw_grid(self):
        """Draw background grid (精美版)"""
        self.screen.fill(Colors.BG_DARK)
        
        # 繪製網格線
        for i in range(self.grid_size + 1):
            x = i * self.cell_size + int(self.shake_offset[0])
            y = i * self.cell_size + int(self.shake_offset[1])
            
            # 每 5 格畫一條亮線
            color = Colors.GRID_GLOW if i % 5 == 0 else Colors.GRID_LINE
            pygame.draw.line(self.screen, color,
                           (x, 0), (x, self.game_width))
            pygame.draw.line(self.screen, color,
                           (0, y), (self.game_width, y))
    
    def _draw_snake(self):
        """Draw snake with beautiful gradient (精美版)"""
        snake = list(self.unwrapped_env.snake)
        length = len(snake)
        
        for i, (r, c) in enumerate(snake):
            x = c * self.cell_size + int(self.shake_offset[0])
            y = r * self.cell_size + int(self.shake_offset[1])
            
            if self.is_victory:
                # 🏆 勝利時的彩虹特效
                hue = (time.time() * 100 + i * 10) % 360
                color = self._hsv_to_rgb(hue, 0.8, 1.0)
            elif i == 0:
                # HEAD: 發光效果
                color = Colors.SNAKE_HEAD
                # 畫發光圈
                glow_rect = pygame.Rect(x - 2, y - 2, self.cell_size + 4, self.cell_size + 4)
                pygame.draw.rect(self.screen, (*color, 80), glow_rect, border_radius=8)
            elif i == length - 1:
                # TAIL
                color = Colors.SNAKE_TAIL
            else:
                # BODY: 漸變
                t = i / max(1, length - 1)
                color = self._lerp_color(Colors.SNAKE_BODY_START, Colors.SNAKE_BODY_END, t)
            
            rect = pygame.Rect(x + 2, y + 2, self.cell_size - 4, self.cell_size - 4)
            pygame.draw.rect(self.screen, color, rect, border_radius=6)
            
            # 內部高光
            highlight_rect = pygame.Rect(x + 4, y + 4, self.cell_size - 12, self.cell_size - 12)
            highlight_color = tuple(min(255, c + 40) for c in color)
            pygame.draw.rect(self.screen, highlight_color, highlight_rect, border_radius=4)
    
    def _lerp_color(self, c1, c2, t):
        """顏色插值"""
        return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))
    
    def _hsv_to_rgb(self, h, s, v):
        """HSV to RGB 轉換"""
        c = v * s
        x = c * (1 - abs((h / 60) % 2 - 1))
        m = v - c
        
        if h < 60: r, g, b = c, x, 0
        elif h < 120: r, g, b = x, c, 0
        elif h < 180: r, g, b = 0, c, x
        elif h < 240: r, g, b = 0, x, c
        elif h < 300: r, g, b = x, 0, c
        else: r, g, b = c, 0, x
        
        return (int((r + m) * 255), int((g + m) * 255), int((b + m) * 255))
    
    def _draw_food(self):
        """Draw pulsing food (精美版)"""
        food = self.unwrapped_env.food
        if not food:
            return
        
        r, c = food
        x = c * self.cell_size + self.cell_size // 2 + int(self.shake_offset[0])
        y = r * self.cell_size + self.cell_size // 2 + int(self.shake_offset[1])
        
        # 多層發光效果
        pulse = math.sin(time.time() * 5) * 0.2 + 1.0
        
        # 外圈光暈
        outer_radius = int((self.cell_size // 2) * pulse)
        pygame.gfxdraw.filled_circle(self.screen, x, y, outer_radius + 6, (*Colors.FOOD_OUTER, 30))
        pygame.gfxdraw.filled_circle(self.screen, x, y, outer_radius + 3, (*Colors.FOOD_GLOW, 80))
        pygame.gfxdraw.filled_circle(self.screen, x, y, outer_radius, Colors.FOOD_CORE)
        
        # 核心高光
        pygame.gfxdraw.filled_circle(self.screen, x - 3, y - 3, 3, (255, 255, 255))
    
    def _draw_panel(self):
        """Draw stats panel (精美版)"""
        panel_x = self.game_width
        
        # Panel background with gradient
        for i in range(self.panel_width):
            alpha = 0.95 - i * 0.001
            color = tuple(int(c * alpha) for c in (20, 25, 40))
            pygame.draw.line(self.screen, color, (panel_x + i, 0), (panel_x + i, self.screen_height))
        
        pygame.draw.line(self.screen, Colors.GRID_GLOW,
                        (panel_x, 0), (panel_x, self.screen_height), 2)
        
        y = 20
        
        # Title
        title = self.font_large.render("V10.0 PHOENIX", True, Colors.TEXT_CYAN)
        self.screen.blit(title, (panel_x + 20, y))
        y += 45
        
        # Speed display
        speed_text = f"Speed: {self.speed_level + 1}x"
        speed_surf = self.font_medium.render(speed_text, True, Colors.TEXT_YELLOW)
        self.screen.blit(speed_surf, (panel_x + 20, y))
        y += 35
        
        # Separator
        pygame.draw.line(self.screen, Colors.GRID_GLOW, (panel_x + 20, y), (panel_x + self.panel_width - 20, y))
        y += 15
        
        # Core stats
        stats = [
            ("Gen", f"{self.generation}", Colors.TEXT_WHITE),
            ("Best", f"{self.best_length}", Colors.TEXT_GREEN),
            ("Now", f"{self.current_length}", Colors.TEXT_CYAN),
            ("Score", f"{self.unwrapped_env.score}", Colors.TEXT_YELLOW),
        ]
        
        for label, value, color in stats:
            label_surf = self.font_small.render(label, True, Colors.TEXT_DIM)
            value_surf = self.font_medium.render(value, True, color)
            self.screen.blit(label_surf, (panel_x + 20, y))
            self.screen.blit(value_surf, (panel_x + 120, y))
            y += 28
        
        y += 15
        
        # V10 Metrics section
        pygame.draw.line(self.screen, Colors.GRID_GLOW, (panel_x + 20, y), (panel_x + self.panel_width - 20, y))
        y += 15
        
        v10_title = self.font_medium.render("AI METRICS", True, Colors.TEXT_PURPLE)
        self.screen.blit(v10_title, (panel_x + 20, y))
        y += 30
        
        v10_stats = [
            ("Reachable", f"{self.reachable_area}"),
            ("HC Dist", f"{self.hc_distance:.2f}"),
            ("Fallback", f"{self.unwrapped_env.fallback_count}"),
        ]
        
        for label, value in v10_stats:
            label_surf = self.font_small.render(label, True, Colors.TEXT_DIM)
            value_surf = self.font_small.render(value, True, Colors.TEXT_CYAN)
            self.screen.blit(label_surf, (panel_x + 20, y))
            self.screen.blit(value_surf, (panel_x + 140, y))
            y += 24
        
        # Autopilot status (V10.11: 門檻 380)
        if self.current_length >= 380:
            y += 10
            auto_surf = self.font_medium.render("AUTOPILOT", True, Colors.VICTORY_GOLD)
            self.screen.blit(auto_surf, (panel_x + 20, y))
            y += 25
            remaining = 400 - self.current_length
            remain_surf = self.font_small.render(f"{remaining} to go!", True, Colors.TEXT_GREEN)
            self.screen.blit(remain_surf, (panel_x + 20, y))
        
        # Status
        y = self.screen_height - 130
        pygame.draw.line(self.screen, Colors.GRID_GLOW, (panel_x + 20, y), (panel_x + self.panel_width - 20, y))
        y += 15
        
        if self.is_victory:
            victory_surf = self.font_large.render("PERFECT!", True, Colors.VICTORY_GOLD)
            self.screen.blit(victory_surf, (panel_x + 50, y))
        elif self.is_dead:
            death_surf = self.font_large.render("DEAD", True, Colors.TEXT_RED)
            self.screen.blit(death_surf, (panel_x + 80, y))
            y += 35
            remaining = max(0, DEATH_DELAY - (time.time() - self.death_time))
            timer_surf = self.font_small.render(f"Restart: {remaining:.1f}s", True, Colors.TEXT_DIM)
            self.screen.blit(timer_surf, (panel_x + 50, y))
        elif self.is_paused:
            pause_surf = self.font_large.render("PAUSED", True, Colors.TEXT_YELLOW)
            self.screen.blit(pause_surf, (panel_x + 60, y))
        
        # Controls hint
        y = self.screen_height - 45
        hints = [
            "P/Enter: Pause  R: Restart",
            "1-8/F1-F8: Speed  Space: Skip"
        ]
        for hint in hints:
            hint_surf = self.font_small.render(hint, True, Colors.TEXT_DIM)
            self.screen.blit(hint_surf, (panel_x + 15, y))
            y += 18
    
    def _draw_victory_overlay(self):
        """繪製勝利畫面"""
        if not self.is_victory:
            return
        
        # 半透明覆蓋
        overlay = pygame.Surface((self.game_width, self.screen_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 100))
        self.screen.blit(overlay, (0, 0))
        
        # 勝利文字 (脈動效果)
        pulse = 1.0 + 0.1 * math.sin(time.time() * 3)
        
        # 陰影
        shadow_surf = self.font_victory.render("🏆 PERFECT! 400/400 🏆", True, (0, 0, 0))
        shadow_rect = shadow_surf.get_rect(center=(self.game_width // 2 + 3, self.screen_height // 2 + 3))
        self.screen.blit(shadow_surf, shadow_rect)
        
        # 主文字
        victory_surf = self.font_victory.render("🏆 PERFECT! 400/400 🏆", True, Colors.VICTORY_GOLD)
        victory_rect = victory_surf.get_rect(center=(self.game_width // 2, self.screen_height // 2))
        self.screen.blit(victory_surf, victory_rect)
        
        # 副標題
        sub_surf = self.font_medium.render("AI + HC Hybrid Perfection!", True, Colors.TEXT_CYAN)
        sub_rect = sub_surf.get_rect(center=(self.game_width // 2, self.screen_height // 2 + 50))
        self.screen.blit(sub_surf, sub_rect)
        
        # 產生勝利粒子
        self._spawn_victory_particles()
        
        # 繪製勝利粒子
        for p in self.victory_particles:
            p.draw(self.screen)
    
    def _update_v6_metrics(self):
        """Update V6-specific metrics from observation"""
        if hasattr(self.unwrapped_env, 'snake') and len(self.unwrapped_env.snake) > 0:
            head = self.unwrapped_env.snake[0]
            self.head_hc_idx = self.unwrapped_env.hc_idx[head[0], head[1]]
            
            if self.unwrapped_env.food:
                food = self.unwrapped_env.food
                self.food_hc_idx = self.unwrapped_env.hc_idx[food[0], food[1]]
                
                # Calculate HC distance
                diff = self.food_hc_idx - self.head_hc_idx
                N = self.unwrapped_env.N
                self.hc_distance = (diff + (N - 1)) / (2 * (N - 1))
            
            # Reachable area
            self.reachable_area = self.unwrapped_env._compute_head_reachable_area()
    
    def _compute_flood_area(self, grid, start_r, start_c, tail):
        """
        計算從指定位置出發的可達空間大小（簡易 BFS）
        用於 Autopilot 的智能 Fallback
        """
        from collections import deque
        
        rows, cols = grid.shape
        
        # 邊界檢查
        if start_r < 0 or start_r >= rows or start_c < 0 or start_c >= cols:
            return 0
        
        # 起點被阻擋
        if grid[start_r, start_c] == 1 and (start_r, start_c) != tail:
            return 0
        
        visited = set()
        queue = deque([(start_r, start_c)])
        visited.add((start_r, start_c))
        count = 0
        
        while queue:
            r, c = queue.popleft()
            count += 1
            
            # 限制搜尋深度，避免太慢
            if count > 100:
                return count
            
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                    if grid[nr, nc] == 0 or (nr, nc) == tail:
                        visited.add((nr, nc))
                        queue.append((nr, nc))
        
        return count
    
    def _is_hc_aligned(self):
        """
        🔧 V10.13 新增：檢查蛇身是否沿著 HC 連續排列
        
        條件：每一節的 hc_idx 都剛好是「前一節 + 1」（mod N）
        只有在對齊時，Autopilot 才能安全地「直接走 next HC」
        否則會走到自己身體裡，導致死循環
        
        返回：True = 對齊，可以直走 HC；False = 不對齊，用 fallback
        """
        env = self.unwrapped_env
        snake = list(env.snake)
        N = env.N
        hc = env.hc_idx
        
        # snake[i] 應該是 snake[i+1] 的「下一格」（沿 HC 前進）
        # 也就是 idx(snake[i]) == idx(snake[i+1]) + 1 (mod N)
        for i in range(len(snake) - 1):
            r1, c1 = snake[i]
            r2, c2 = snake[i + 1]
            idx1 = hc[r1, c1]
            idx2 = hc[r2, c2]
            if idx1 != (idx2 + 1) % N:
                return False
        return True
    
    def _get_flood_fallback_action(self, action_masks):
        """
        🔧 V10.15 智能 Fallback：優先走 HC，其次選空間最大
        
        優先級：
        1. HC 下一步可走 → 直接走
        2. 否則選「可達空間大」且「接近 HC 尾巴」的方向
        """
        env = self.unwrapped_env
        head = env.snake[0]
        tail = env.snake[-1]
        grid = env.grid_array
        hc = env.hc_idx
        N = env.N
        
        # 先嘗試走 HC 下一步（最優先）
        head_hc = hc[head[0], head[1]]
        next_hc = (head_hc + 1) % N
        next_pos = env.path_coords[next_hc]
        
        dr = next_pos[0] - head[0]
        dc = next_pos[1] - head[1]
        
        if dr == -1: hc_action = 0
        elif dr == 1: hc_action = 1
        elif dc == -1: hc_action = 2
        else: hc_action = 3
        
        if action_masks[hc_action]:
            return hc_action  # HC 下一步可走，優先
        
        # HC 不可走，選擇「空間大 + 接近尾巴」的方向
        MOVES = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        tail_hc = hc[tail[0], tail[1]]
        
        best_action = None
        best_score = -1
        
        for act_idx in range(4):
            if not action_masks[act_idx]:
                continue
            
            dr, dc = MOVES[act_idx]
            nr, nc = head[0] + dr, head[1] + dc
            
            # 計算可達空間
            area = self._compute_flood_area(grid, nr, nc, tail)
            
            # 計算該方向接近尾巴的程度
            next_cell_hc = hc[nr, nc]
            dist_to_tail = (tail_hc - next_cell_hc) % N
            
            # 分數 = 空間大小（主要）+ 接近尾巴（次要）
            score = area * 100 + (N - dist_to_tail)
            
            if score > best_score:
                best_score = score
                best_action = act_idx
        
        return best_action
    
    def update(self, dt):
        """Main update loop"""
        if self.is_paused or self.is_victory:
            if self.is_victory:
                self._update_particles(dt)
            return
        
        if self.is_dead:
            if time.time() - self.death_time >= DEATH_DELAY:
                self._reset()
            return
        
        # AI step
        old_length = self.current_length
        
        # 🔥 V10.11 Endgame Autopilot（簡單穩定版）
        # 原理：370+ 時優先跟隨 HC，如果被 action_masks 禁止就讓 AI 決定
        AUTOPILOT_THRESHOLD = 380
        action = None
        
        try:
            if self.current_length >= AUTOPILOT_THRESHOLD:
                env = self.unwrapped_env
                head = env.snake[0]
                hc = env.hc_idx
                N = env.N
                
                # 計算 HC 下一步
                current_hc_idx = hc[head[0], head[1]]
                next_hc_idx = (current_hc_idx + 1) % N
                next_pos = env.path_coords[next_hc_idx]
                
                dr = next_pos[0] - head[0]
                dc = next_pos[1] - head[1]
                
                if dr == -1: autopilot_action = 0
                elif dr == 1: autopilot_action = 1
                elif dc == -1: autopilot_action = 2
                else: autopilot_action = 3
                
                # 用 action_masks 檢查是否安全
                action_masks = self.env.env_method("action_masks")[0]
                
                if action_masks[autopilot_action]:
                    # HC 路徑安全，跟隨
                    action = [autopilot_action]
                # 否則讓 AI 決定（不做複雜 fallback）
                        
        except Exception as e:
            print(f"❌ Autopilot error: {e}")
            action = None
        
        # 如果 Autopilot 沒有選擇動作，使用 AI
        if action is None:
            action_masks = self.env.env_method("action_masks")[0]
            action, _ = self.model.predict(self.obs, action_masks=action_masks, deterministic=True)
        
        self.obs, reward, done, info = self.env.step(action)
        
        self.current_length = len(self.unwrapped_env.snake)
        
        if self.current_length > self.best_length:
            self.best_length = self.current_length
        
        # 🏆 勝利檢測
        if self.current_length >= 400:
            self.is_victory = True
            self.victory_time = time.time()
            print("\n🏆🏆🏆 PERFECT GAME! 400/400 ACHIEVED! 🏆🏆🏆\n")
            return
        
        # Eat effect
        if self.current_length > old_length:
            head = self.unwrapped_env.snake[0]
            px = head[1] * self.cell_size + self.cell_size // 2
            py = head[0] * self.cell_size + self.cell_size // 2
            self._spawn_particles(px, py, count=30)
            self._add_shake(10)
        
        # Update metrics
        self._update_v6_metrics()
        
        # Death check
        if done[0]:
            self.is_dead = True
            self.death_time = time.time()
            self._add_shake(20)
        
        self._update_particles(dt)
    
    def handle_input(self):
        """Handle keyboard input (使用 scancode 確保輸入法狀態下也能用)"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            if event.type == pygame.KEYDOWN:
                key = event.key
                # 使用 scancode 作為備用 (不受輸入法影響)
                scancode = event.scancode if hasattr(event, 'scancode') else 0
                
                # Debug: 顯示按下的鍵 (可以幫助調試)
                # print(f"Key: {key}, Scancode: {scancode}")
                
                # === 離開遊戲 ===
                # ESC (scancode 1 on Windows)
                if key == pygame.K_ESCAPE or scancode == 1:
                    return False
                
                # === SPACE 鍵 (特殊處理) ===
                # 死亡/勝利時 = 跳過延遲重啟
                # 正常遊戲時 = 暫停
                is_space = (key == pygame.K_SPACE or scancode == 57)
                if is_space:
                    if self.is_dead or self.is_victory:
                        self._reset()  # 跳過延遲
                    else:
                        self.is_paused = not self.is_paused  # 暫停
                    continue  # 處理完 SPACE 後跳過其他判斷
                
                # === 暫停 (P 和 Enter) ===
                # P (scancode 25), Enter (scancode 28)
                is_pause_key = (key in [pygame.K_p, pygame.K_RETURN, pygame.K_KP_ENTER] or 
                               scancode in [25, 28, 284])
                if is_pause_key:
                    if not self.is_dead and not self.is_victory:
                        self.is_paused = not self.is_paused
                    continue
                
                # === 重啟 ===
                # R (scancode 19)
                if key == pygame.K_r or scancode == 19:
                    self._reset()
                    continue
                
                # === 速度控制 ===
                # 主鍵盤數字鍵 1-8 (scancodes 2-9)
                # 數字鍵 1-9 = 速度 1-9, 0 = 速度 10
                if pygame.K_1 <= key <= pygame.K_9:
                    self.speed_level = key - pygame.K_1
                    self.ai_step_delay = SPEED_LEVELS[self.speed_level]
                    print(f"Speed: {self.speed_level + 1}x")
                elif key == pygame.K_0:  # 0 = 速度 10
                    self.speed_level = 9
                    self.ai_step_delay = SPEED_LEVELS[self.speed_level]
                    print(f"Speed: 10x")
                elif 2 <= scancode <= 11:  # Scancode fallback 1-0
                    self.speed_level = min(scancode - 2, 9)
                    self.ai_step_delay = SPEED_LEVELS[self.speed_level]
                    print(f"Speed: {self.speed_level + 1}x")
                
                # 小鍵盤 1-8
                if pygame.K_KP1 <= key <= pygame.K_KP8:
                    self.speed_level = key - pygame.K_KP1
                    self.ai_step_delay = SPEED_LEVELS[self.speed_level]
                    print(f"Speed: {self.speed_level + 1}x")
                
                # F1-F8 (scancodes 59-66)
                if pygame.K_F1 <= key <= pygame.K_F8:
                    self.speed_level = key - pygame.K_F1
                    self.ai_step_delay = SPEED_LEVELS[self.speed_level]
                    print(f"Speed: {self.speed_level + 1}x")
                elif 59 <= scancode <= 66:  # Scancode fallback
                    self.speed_level = scancode - 59
                    self.ai_step_delay = SPEED_LEVELS[self.speed_level]
                    print(f"Speed: {self.speed_level + 1}x")
                
                # 方向鍵 (Up: 72/328, Down: 80/336, Left: 75/331, Right: 77/333)
                up_codes = [pygame.K_UP, pygame.K_RIGHT]
                down_codes = [pygame.K_DOWN, pygame.K_LEFT]
                up_scancodes = [72, 77, 328, 333]  # Up, Right
                down_scancodes = [80, 75, 336, 331]  # Down, Left
                
                if key in up_codes or scancode in up_scancodes:
                    self.speed_level = min(49, self.speed_level + 1)  # 最高 50x
                    self.ai_step_delay = SPEED_LEVELS[self.speed_level]
                    print(f"Speed: {self.speed_level + 1}x")
                if key in down_codes or scancode in down_scancodes:
                    self.speed_level = max(0, self.speed_level - 1)
                    self.ai_step_delay = SPEED_LEVELS[self.speed_level]
                    print(f"Speed: {self.speed_level + 1}x")
                
                # + / - 鍵
                plus_keys = [pygame.K_PLUS, pygame.K_KP_PLUS, pygame.K_EQUALS]
                minus_keys = [pygame.K_MINUS, pygame.K_KP_MINUS]
                if key in plus_keys or scancode == 13:  # = on main keyboard
                    self.speed_level = min(49, self.speed_level + 1)  # 最高 50x
                    self.ai_step_delay = SPEED_LEVELS[self.speed_level]
                    print(f"Speed: {self.speed_level + 1}x")
                if key in minus_keys or scancode == 12:  # - on main keyboard
                    self.speed_level = max(0, self.speed_level - 1)
                    self.ai_step_delay = SPEED_LEVELS[self.speed_level]
                    print(f"Speed: {self.speed_level + 1}x")
        
        return True
    
    def draw(self):
        """Render frame"""
        self._draw_grid()
        self._draw_snake()
        self._draw_food()
        
        # Particles
        for p in self.particles:
            p.draw(self.screen)
        
        self._draw_panel()
        
        # Victory overlay
        if self.is_victory:
            self._draw_victory_overlay()
        
        pygame.display.flip()
    
    def run(self):
        """Main game loop"""
        print("=" * 60)
        print("  🐍 SNAKE AI V10.0 - PHOENIX")
        print("=" * 60)
        print("操控:")
        print("  P / 空白    - 暫停")
        print("  SPACE       - 跳過死亡/勝利延遲")
        print("  R           - 重新開始")
        print("  1-8 / ↑↓    - 調整速度")
        print("  ESC         - 離開")
        print("=" * 60)
        print(f"目前速度: {self.speed_level + 1}x")
        print("=" * 60)
        
        running = True
        last_step_time = time.time()
        
        while running:
            dt = self.clock.tick(144) / 1000.0  # 144 FPS for high refresh rate monitors
            
            running = self.handle_input()
            
            # AI step delay - 高速模式下每幀執行多步
            current_time = time.time()
            elapsed = current_time - last_step_time
            
            # 計算這一幀應該執行多少步
            if self.ai_step_delay > 0:
                steps_to_run = int(elapsed / self.ai_step_delay)
                steps_to_run = min(steps_to_run, 100)  # 防止卡頓時爆炸
            else:
                # 50x MAX 模式：無延遲，每幀執行最大步數
                steps_to_run = 100
            
            if steps_to_run > 0:
                for _ in range(steps_to_run):
                    if not self.is_dead and not self.is_victory and not self.is_paused:
                        self.update(dt)
                last_step_time = current_time
            else:
                self._update_particles(dt)
            
            self.draw()
        
        pygame.quit()


if __name__ == "__main__":
    game = SnakeGameV10()
    game.run()
