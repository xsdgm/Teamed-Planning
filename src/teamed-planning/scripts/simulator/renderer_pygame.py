"""
renderer_pygame.py - Optimized Pygame renderer with high performance
- Background caching for static elements
- View frustum culling (only draw visible objects)
- Configurable FPS limit
- Hardware acceleration
- Performance monitoring
"""

import numpy as np
from typing import List, Tuple, Optional

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("Warning: pygame not installed. Rendering disabled.")


class PygameRenderer:
    """Optimized Pygame 2D Renderer"""
    
    def __init__(self, world_size: Tuple[float, float], 
                 window_size: Tuple[int, int] = (1200, 1200),
                 target_fps: int = 30):  # Default 30 FPS for rendering
        if not PYGAME_AVAILABLE:
            raise ImportError("pygame is required for rendering")
        
        self.world_size = world_size
        pygame.init()
        # Use matching window/surface pixels. SCALED keeps a fixed logical
        # surface and can letterbox a resized window, complicating picking.
        self.display_flags = pygame.RESIZABLE
        desktop_width, desktop_height = pygame.display.get_desktop_sizes()[0]
        fit = min(1., max(1, desktop_width - 80) / window_size[0],
                  max(1, desktop_height - 120) / window_size[1])
        window_size = tuple(max(1, int(value * fit)) for value in window_size)
        self.screen = pygame.display.set_mode(window_size, self.display_flags)
        window_size = self.screen.get_size()
        self.window_size = window_size
        self.target_fps = target_fps
        self.closed = False
        
        # Initial scale
        self.base_scale_x = window_size[0] / world_size[0]
        self.base_scale_y = window_size[1] / world_size[1]
        self.base_scale = min(self.base_scale_x, self.base_scale_y)
        
        # View parameters
        self.zoom_level = 1.0
        self.scale = self.base_scale * self.zoom_level
        self.offset_x = 0
        self.offset_y = 0
        
        # Track view changes
        self.prev_zoom = self.zoom_level
        self.prev_offset = (self.offset_x, self.offset_y)
        
        # Drag state
        self.dragging = False
        self.drag_start_pos = None
        self.drag_start_offset = None
        
        pygame.display.set_caption("Multi-Robot Sim [Scroll:Zoom Drag:Pan R:Reset F:FPS 1-3:Limit]")
        self.clock = pygame.time.Clock()
        
        # Background cache
        self.background_cache = None
        self.cache_valid = False
        
        # Colors
        self.colors = {
            'background': (245, 245, 220),
            'ground_robot': (255, 68, 68),
            'ground_robot_border': (170, 0, 0),
            'drone': (68, 68, 255),
            'drone_cross': (0, 0, 170),
            'obstacle': (34, 139, 34),
            'obstacle_dark': (46, 139, 87),
            'obstacle_trunk': (139, 69, 19),
            'label_bg': (255, 255, 255),
            'label_border': (0, 0, 0),
            'label_text': (0, 0, 0),
            'grid': (200, 200, 180),
            'velocity_arrow': (255, 200, 0),
        }
        
        # Fonts
        self.font_small = pygame.font.Font(None, 18)
        self.font_medium = pygame.font.Font(None, 24)
        
        # Robot sizes
        self.ground_robot_size = 15
        self.drone_size = 12
        self.drone_arm_length = 20
        self.label_width = 50
        self.label_height = 20
        
        # Render options
        self.show_grid = True
        self.show_labels = True
        self.show_velocity = False
        self.show_fps = True
        
        # Performance tracking
        self.fps_history = []
        
    def world_to_screen(self, pos: np.ndarray) -> Tuple[int, int]:
        """World to screen coordinates"""
        x = pos[0] - self.offset_x
        y = pos[1] - self.offset_y
        screen_x = int(x * self.scale)
        screen_y = int(self.window_size[1] - y * self.scale)
        return (screen_x, screen_y)
    
    def screen_to_world(self, screen_pos: Tuple[int, int]) -> np.ndarray:
        """Screen to world coordinates"""
        screen_x, screen_y = screen_pos
        y = (self.window_size[1] - screen_y) / self.scale
        x = screen_x / self.scale
        return np.array([x + self.offset_x, y + self.offset_y])
    
    def world_length_to_screen(self, length: float) -> int:
        """World length to screen length"""
        return max(1, int(length * self.scale))
    
    def get_visible_bounds(self) -> Tuple[float, float, float, float]:
        """Get visible world bounds (left, right, bottom, top)"""
        margin = 100 / self.scale
        left = self.offset_x - margin
        right = self.offset_x + self.window_size[0] / self.scale + margin
        bottom = self.offset_y - margin
        top = self.offset_y + self.window_size[1] / self.scale + margin
        return (left, right, bottom, top)
    
    def is_visible(self, pos: np.ndarray) -> bool:
        """Check if position is visible"""
        left, right, bottom, top = self.get_visible_bounds()
        return (left <= pos[0] <= right and bottom <= pos[1] <= top)
    
    def zoom(self, zoom_delta: float, mouse_pos: Optional[Tuple[int, int]] = None):
        """Zoom view"""
        if mouse_pos is not None:
            world_pos_before = self.screen_to_world(mouse_pos)
        
        self.zoom_level *= 1.1 ** (zoom_delta / 0.1)
        self.zoom_level = np.clip(self.zoom_level, 0.1, 20.0)
        self.scale = self.base_scale * self.zoom_level
        
        if mouse_pos is not None:
            world_pos_after = self.screen_to_world(mouse_pos)
            self.offset_x += world_pos_before[0] - world_pos_after[0]
            self.offset_y += world_pos_before[1] - world_pos_after[1]
        
        self.cache_valid = False

    def resize(self, window_size):
        """Resize the drawing surface while preserving the view center."""
        window_size = tuple(max(1, int(value)) for value in window_size)
        if window_size == self.window_size:
            return
        center = self.screen_to_world((self.window_size[0] / 2., self.window_size[1] / 2.))
        surface = pygame.display.get_surface()
        if surface is None or surface.get_size() != window_size:
            surface = pygame.display.set_mode(window_size, self.display_flags)
        self.screen = surface
        self.window_size = self.screen.get_size()
        self.base_scale_x = self.window_size[0] / self.world_size[0]
        self.base_scale_y = self.window_size[1] / self.world_size[1]
        self.base_scale = min(self.base_scale_x, self.base_scale_y)
        self.scale = self.base_scale * self.zoom_level
        self.offset_x = center[0] - self.window_size[0] / (2. * self.scale)
        self.offset_y = center[1] - self.window_size[1] / (2. * self.scale)
        self.background_cache = None
        self.cache_valid = False
        self.dragging = False
    
    def reset_view(self):
        """Reset view"""
        self.zoom_level = 1.0
        self.scale = self.base_scale
        self.offset_x = 0
        self.offset_y = 0
        self.cache_valid = False
    
    def render(self, ground_robots, drones, obstacles, mode='human', auto_flip=True):
        """
        Render one frame (optimized)
        
        Args:
            auto_flip: If True, automatically call pygame.display.flip() (default: True)
        """
        if self.closed:
            return
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return
            elif event.type in (pygame.VIDEORESIZE, pygame.WINDOWSIZECHANGED):
                # Pygame 2 resizes the surface automatically. Read the current
                # size rather than replaying stale events from rapid drags.
                self.resize(pygame.display.get_window_size())
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.close()
                    return
                elif event.key == pygame.K_g:
                    self.show_grid = not self.show_grid
                    self.cache_valid = False
                elif event.key == pygame.K_l:
                    self.show_labels = not self.show_labels
                elif event.key == pygame.K_v:
                    self.show_velocity = not self.show_velocity
                elif event.key == pygame.K_f:
                    self.show_fps = not self.show_fps
                elif event.key == pygame.K_r:
                    self.reset_view()
                elif event.key == pygame.K_1:
                    self.target_fps = 30
                elif event.key == pygame.K_2:
                    self.target_fps = 60
                elif event.key == pygame.K_3:
                    self.target_fps = 120
                elif event.key == pygame.K_0:
                    self.target_fps = 0  # Unlimited
            elif event.type == pygame.MOUSEWHEEL:
                self.zoom(0.1 * event.y, pygame.mouse.get_pos())
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self.dragging = True
                    self.drag_start_pos = event.pos
                    self.drag_start_offset = (self.offset_x, self.offset_y)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.dragging = False
            elif event.type == pygame.MOUSEMOTION:
                if self.dragging:
                    dx = event.pos[0] - self.drag_start_pos[0]
                    dy = event.pos[1] - self.drag_start_pos[1]
                    self.offset_x = self.drag_start_offset[0] - dx / self.scale
                    self.offset_y = self.drag_start_offset[1] + dy / self.scale
                    self.cache_valid = False
        
        # Check if view changed
        view_changed = (abs(self.zoom_level - self.prev_zoom) > 1e-6 or
                       abs(self.offset_x - self.prev_offset[0]) > 1e-3 or
                       abs(self.offset_y - self.prev_offset[1]) > 1e-3)
        self.prev_zoom = self.zoom_level
        self.prev_offset = (self.offset_x, self.offset_y)
        
        # Use cached background if valid
        if not self.cache_valid or view_changed:
            self.screen.fill(self.colors['background'])
            if self.show_grid:
                self._draw_grid()
            self._draw_obstacles(obstacles)
            self.background_cache = self.screen.copy()
            self.cache_valid = True
        else:
            if self.background_cache:
                self.screen.blit(self.background_cache, (0, 0))
        
        # Draw robots (always)
        self._draw_ground_robots(ground_robots)
        self._draw_drones(drones)
        self._draw_info_panel(ground_robots, drones, obstacles)
        self._draw_controls_hint()
        if self.show_fps:
            self._draw_fps()
        
        if mode == 'human':
            # Only flip if auto_flip is True
            if auto_flip:
                pygame.display.flip()
            
            if self.target_fps > 0:
                self.clock.tick(self.target_fps)
            else:
                self.clock.tick()
            
            fps = self.clock.get_fps()
            self.fps_history.append(fps)
            if len(self.fps_history) > 30:
                self.fps_history.pop(0)
        elif mode == 'rgb_array':
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))
    
    def _draw_grid(self):
        """Draw grid"""
        spacing = 1000 if self.zoom_level <= 2 else (500 if self.zoom_level <= 5 else 100)
        left, right, bottom, top = self.get_visible_bounds()
        
        x = int(left / spacing) * spacing
        while x <= right:
            sx, _ = self.world_to_screen(np.array([x, 0]))
            if 0 <= sx <= self.window_size[0]:
                pygame.draw.line(self.screen, self.colors['grid'],
                               (sx, 0), (sx, self.window_size[1]), 1)
            x += spacing
        
        y = int(bottom / spacing) * spacing
        while y <= top:
            _, sy = self.world_to_screen(np.array([0, y]))
            if 0 <= sy <= self.window_size[1]:
                pygame.draw.line(self.screen, self.colors['grid'],
                               (0, sy), (self.window_size[0], sy), 1)
            y += spacing
    
    def _draw_obstacles(self, obstacles):
        """Draw obstacles (with culling)"""
        for obs in obstacles:
            if not self.is_visible(obs.position):
                continue
            
            sp = self.world_to_screen(obs.position)
            
            if obs.type == 'circle':
                r = max(2, self.world_length_to_screen(obs.radius))
                pygame.draw.circle(self.screen, self.colors['obstacle'], sp, r)
                pygame.draw.circle(self.screen, self.colors['obstacle_dark'], sp, max(1, int(r*0.7)))
                pygame.draw.circle(self.screen, self.colors['obstacle_trunk'], sp, max(1, int(r*0.15)))
            elif obs.type == 'rect':
                w = max(3, self.world_length_to_screen(obs.width))
                h = max(3, self.world_length_to_screen(obs.height))
                rect = pygame.Rect(sp[0]-w//2, sp[1]-h//2, w, h)
                pygame.draw.rect(self.screen, self.colors['obstacle'], rect)
                pygame.draw.rect(self.screen, self.colors['obstacle_dark'], rect, 2)
    
    def _draw_ground_robots(self, robots):
        """Draw ground robots"""
        for r in robots:
            if not self.is_visible(r.position):
                continue
            
            sp = self.world_to_screen(r.position)
            s = self.ground_robot_size
            h = s/2
            corners = [(-h,-h),(h,-h),(h,h),(-h,h)]
            theta = r.orientation
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            rc = [(sp[0]+cx*cos_t-cy*sin_t, sp[1]-(cx*sin_t+cy*cos_t)) for cx,cy in corners]
            
            pygame.draw.polygon(self.screen, self.colors['ground_robot'], rc)
            pygame.draw.polygon(self.screen, self.colors['ground_robot_border'], rc, 3)
            
            ae = (sp[0]+s*0.6*np.cos(theta), sp[1]-s*0.6*np.sin(theta))
            pygame.draw.line(self.screen, (255,255,255), sp, ae, 3)
            pygame.draw.circle(self.screen, (255,255,255), (int(ae[0]),int(ae[1])), 4)
            
            if self.show_labels:
                self._draw_label(sp, f"G{r.robot_id}", -s*1.2)
            if self.show_velocity:
                self._draw_velocity(r, sp)
    
    def _draw_drones(self, drones):
        """Draw drones"""
        for d in drones:
            if not self.is_visible(d.position):
                continue
            
            sp = self.world_to_screen(d.position)
            c = self.drone_size
            a = self.drone_arm_length
            p = int(c*0.8)
            
            # Arms
            for ang in [45,135,225,315]:
                rad = np.radians(ang)
                ex = sp[0]+int(a*np.cos(rad))
                ey = sp[1]-int(a*np.sin(rad))
                pygame.draw.line(self.screen, (80,80,80), sp, (ex,ey), 3)
                
                # Propellers
                pygame.draw.circle(self.screen, self.colors['drone'], (ex,ey), p)
                pygame.draw.circle(self.screen, (100,100,255), (ex,ey), int(p*0.5))
                pygame.draw.circle(self.screen, (30,30,120), (ex,ey), p, 2)
            
            # Center frame
            cs = int(c*1.2)
            pygame.draw.line(self.screen, self.colors['drone_cross'],
                           (sp[0]-cs,sp[1]), (sp[0]+cs,sp[1]), 4)
            pygame.draw.line(self.screen, self.colors['drone_cross'],
                           (sp[0],sp[1]-cs), (sp[0],sp[1]+cs), 4)
            pygame.draw.circle(self.screen, self.colors['drone_cross'], sp, int(c*0.6))
            pygame.draw.circle(self.screen, (255,255,255), sp, int(c*0.3))
            
            if self.show_labels:
                self._draw_label(sp, f"D{d.robot_id}", -(a+p+5))
    
    def _draw_label(self, sp, text, oy):
        """Draw label"""
        lx = sp[0] - self.label_width//2
        ly = sp[1] + int(oy) - self.label_height//2
        lr = pygame.Rect(lx-3, ly-3, self.label_width+6, self.label_height+6)
        pygame.draw.rect(self.screen, self.colors['label_bg'], lr, 0, 5)
        pygame.draw.rect(self.screen, self.colors['label_border'], lr, 2, 5)
        ts = self.font_medium.render(text, True, self.colors['label_text'])
        tr = ts.get_rect(center=(sp[0], ly+self.label_height//2))
        self.screen.blit(ts, tr)
    
    def _draw_velocity(self, r, sp):
        """Draw velocity arrow"""
        v = r.velocity
        s = np.linalg.norm(v)
        if s > 0.1:
            ang = np.arctan2(v[1], v[0])
            l = min(100, s*20)
            ex = sp[0] + l*np.cos(ang)
            ey = sp[1] - l*np.sin(ang)
            pygame.draw.line(self.screen, self.colors['velocity_arrow'], sp, (int(ex),int(ey)), 2)
    
    def _draw_fps(self):
        """Draw FPS"""
        if self.fps_history:
            fps = sum(self.fps_history)/len(self.fps_history)
            ts = self.font_medium.render(f"FPS:{fps:.0f}", True, (0,200,0))
            self.screen.blit(ts, (self.window_size[0]-90, 10))
    
    def _draw_info_panel(self, gr, dr, obs):
        """Draw info panel"""
        surf = pygame.Surface((280,150), pygame.SRCALPHA)
        pygame.draw.rect(surf, (255,255,255,200), surf.get_rect(), 0, 10)
        pygame.draw.rect(surf, (0,0,0,255), surf.get_rect(), 2, 10)
        self.screen.blit(surf, (10,10))
        
        y = 20
        for t in [f"Ground:{len(gr)}", f"Drones:{len(dr)}", f"Obs:{len(obs)}",
                  f"World:{self.world_size[0]/1000:.1f}x{self.world_size[1]/1000:.1f}km",
                  f"Zoom:{self.zoom_level:.2f}x", f"FPS Limit:{self.target_fps if self.target_fps else 'Max'}"]:
            ts = self.font_small.render(t, True, (0,0,0))
            self.screen.blit(ts, (20,y))
            y += 23
    
    def _draw_controls_hint(self):
        """Draw controls"""
        hints = ["Scroll:Zoom Drag:Pan R:Reset F:FPS", "1:30fps 2:60fps 3:120fps 0:Max", "G:Grid L:Labels V:Vel"]
        y = self.window_size[1]-70
        for h in hints:
            ts = self.font_small.render(h, True, (80,80,80))
            self.screen.blit(ts, (10,y))
            y += 22
    
    def close(self):
        """Close renderer"""
        if PYGAME_AVAILABLE:
            self.closed = True
            pygame.quit()


class DummyRenderer:
    def __init__(self, *args, **kwargs):
        print("Warning: No rendering backend available")
    def render(self, *args, **kwargs):
        return None
    def close(self):
        pass
