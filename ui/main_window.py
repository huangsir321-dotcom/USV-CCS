from __future__ import annotations

import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtGui import QAction, QFont
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QPushButton, QComboBox, QSpinBox, QSlider, QTextEdit, QFileDialog,
    QMessageBox, QGroupBox, QSplitter, QTableWidget, QTableWidgetItem,
    QHeaderView, QCheckBox, QProgressBar, QScrollArea, QApplication,
    QSizePolicy, QTabWidget, QDoubleSpinBox, QFrame
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib import patheffects
from matplotlib.patches import Polygon, Circle, Patch
from matplotlib.lines import Line2D

from usv_sim.scenarios import make_scenario, make_uncoordinated_scenario, add_demo_obstacle
from usv_sim.engine import SimulationEngine
from usv_sim.models import StaticObstacle
from usv_sim.obstacles import distance_to_polygon, point_in_polygon, validate_polygon, circle_polygon_vertices
from usv_sim.scenario_io import load_scenario_json
from usv_sim.dataset import list_cases, load_case, default_dataset_root, GROUP_COUNTS
from usv_sim.ppo_agent import ACTION_NAMES

KNOT_TO_MPS = 0.514444


def risk_label(value: str) -> str:
    return {
        "无危险": "无危险",
        "关注": "关注",
        "碰撞危险": "碰撞危险",
        "紧迫局面": "紧迫局面",
    }.get(value, value or "未知")


def encounter_label(value: str) -> str:
    return {
        "head_on": "对遇",
        "own_overtaking": "本船追越",
        "being_overtaken": "本船被追越",
        "starboard_crossing": "右舷交叉",
        "starboard_beam": "右正横",
        "port_crossing": "左舷交叉",
        "port_beam": "左正横",
        "following": "同向尾随／监测",
        "none": "无活跃会遇",
        "右舷交叉": "右舷交叉",
        "右交叉相遇": "右舷交叉",
        "左交叉相遇": "左舷交叉",
        "右正横交叉相遇": "右正横交叉",
        "左正横交叉相遇": "左正横交叉",
        "右舷对遇": "右舷对遇",
        "左舷对遇": "左舷对遇",
        "本船追越目标船": "本船追越",
        "目标船追越本船": "本船被追越",
        "右舷追越": "右舷追越",
        "左舷追越": "左舷追越",
        "右舷被追越": "右舷被追越",
        "左舷被追越": "左舷被追越",
        "右正横": "右正横",
        "左正横": "左正横",
        "左舷交叉": "左舷交叉",
        "0°对遇": "0°对遇",
    }.get(value, value or "未分类")


def action_label(value: str) -> str:
    return {
        "Maintain / hold strategy": "保持航向/当前避让策略",
        "Maintain course": "保持航向",
        "Starboard turn": "向右转向",
        "Port turn": "向左转向",
        "Reduce speed": "减速",
        "Starboard + reduce speed": "右转并减速",
    }.get(value, value)


def gpu_status_text() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return f"已检测到 CUDA GPU：{torch.cuda.get_device_name(0)}（CUDA {torch.version.cuda}）"
        return "未检测到 CUDA GPU：请先运行 install_cuda_torch.bat 并检查 NVIDIA 驱动"
    except Exception as exc:
        return f"GPU 状态检测失败：{exc}"


class TrainingWorker(QThread):
    output = Signal(str)
    result = Signal(bool, str)

    def __init__(self, command: list[str], cwd: Path, parent=None, task_name: str = "训练"):
        super().__init__(parent)
        self.command = command
        self.cwd = cwd
        self.task_name = task_name
        self.process = None
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True
        if self.process is not None and self.process.poll() is None:
            try:
                self.process.terminate()
            except Exception:
                pass

    def run(self):
        try:
            self.process = subprocess.Popen(
                self.command, cwd=str(self.cwd), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                bufsize=1, universal_newlines=True,
            )
            assert self.process.stdout is not None
            for line in self.process.stdout:
                self.output.emit(line.rstrip())
                if self._stop_requested:
                    self.stop()
            code = self.process.wait()
            if self._stop_requested:
                self.result.emit(False, f"用户已停止{self.task_name}。")
            elif code == 0:
                self.result.emit(True, f"{self.task_name}完成。")
            else:
                self.result.emit(False, f"{self.task_name}进程退出，错误代码：{code}。")
        except Exception as exc:
            self.result.emit(False, f"无法启动{self.task_name}：{exc}")


class SimulationCanvas(FigureCanvas):
    """Interactive map canvas with zoom, drag-pan and polygon drawing."""
    polygon_finished = Signal(object)
    circle_finished = Signal(object, float)
    draft_changed = Signal()
    obstacle_selected = Signal(str)

    def __init__(self, parent=None):
        self.figure = Figure(figsize=(14, 10), dpi=110)
        self.ax = self.figure.add_subplot(111)
        self.figure.subplots_adjust(left=0.075, right=0.985, bottom=0.09, top=0.92)
        super().__init__(self.figure)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.updateGeometry()
        self.toolbar = None
        self._engine = None
        self._user_view = False
        self._view_limits = None
        self._drag_anchor = None
        self._drag_limits = None
        self._drag_moved = False
        self._drawing_polygon = False
        self._draft_points: list[tuple[float, float]] = []
        self._placing_circle = False
        self._circle_radius = 0.0
        self._circle_preview_center: tuple[float, float] | None = None
        self.selected_obstacle_id: str | None = None
        self.mpl_connect("scroll_event", self._on_scroll)
        self.mpl_connect("button_press_event", self._on_button_press)
        self.mpl_connect("motion_notify_event", self._on_motion)
        self.mpl_connect("button_release_event", self._on_button_release)
        self.mpl_connect("draw_event", self._on_draw_event)

    @staticmethod
    def _ship_polygon(x, y, psi, length, beam, scale=1.0):
        # Local bow is +Y. Heading is 0 deg north, clockwise positive.
        # `scale` only changes the map marker's visual size; it does not alter
        # the physical vessel dimensions used by the dynamics or collision model.
        length = float(length) * float(scale)
        beam = float(beam) * float(scale)
        local = [(0, 0.58 * length), (-0.5 * beam, -0.42 * length),
                 (0, -0.28 * length), (0.5 * beam, -0.42 * length)]
        pts = []
        for lx, ly in local:
            gx = x + lx * math.cos(psi) + ly * math.sin(psi)
            gy = y - lx * math.sin(psi) + ly * math.cos(psi)
            pts.append((gx, gy))
        return pts

    def _toolbar_is_active(self):
        mode = getattr(self.toolbar, "mode", "") if self.toolbar is not None else ""
        mode_text = str(mode).upper()
        return bool(mode) and "NONE" not in mode_text

    def _on_draw_event(self, event):
        # Also capture extents changed by Matplotlib's built-in toolbar tools.
        self._view_limits = (self.ax.get_xlim(), self.ax.get_ylim())
        if self._toolbar_is_active():
            self._user_view = True

    def reset_view(self):
        self._user_view = False
        self._view_limits = None
        self._drag_anchor = None
        self._drag_limits = None

    def start_polygon_drawing(self):
        self._drawing_polygon = True
        self._draft_points = []
        # Freeze the currently visible map extent while the user places vertices.
        self._user_view = True
        self._view_limits = (self.ax.get_xlim(), self.ax.get_ylim())
        self.draft_changed.emit()

    def start_circle_placement(self, radius_m: float):
        self._drawing_polygon = False
        self._draft_points = []
        self._placing_circle = True
        self._circle_radius = float(radius_m)
        self._circle_preview_center = None
        self._user_view = True
        self._view_limits = (self.ax.get_xlim(), self.ax.get_ylim())
        self.draft_changed.emit()

    def cancel_circle_placement(self):
        self._placing_circle = False
        self._circle_preview_center = None
        self.draft_changed.emit()

    def cancel_polygon_drawing(self):
        self._drawing_polygon = False
        self._draft_points = []
        self._placing_circle = False
        self._circle_preview_center = None
        self.draft_changed.emit()

    def undo_polygon_vertex(self):
        if self._draft_points:
            self._draft_points.pop()
            self.draft_changed.emit()

    def finish_polygon_drawing(self):
        if len(self._draft_points) < 3:
            return False
        points = list(self._draft_points)
        self._drawing_polygon = False
        self._draft_points = []
        self.draft_changed.emit()
        self.polygon_finished.emit(points)
        return True

    def set_selected_obstacle(self, obstacle_id: str | None):
        self.selected_obstacle_id = obstacle_id

    def _on_scroll(self, event):
        if event.inaxes is not self.ax or event.xdata is None or event.ydata is None:
            return
        if self._toolbar_is_active():
            return
        scale = 0.82 if getattr(event, "step", 0) > 0 else 1.22
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        new_x = (event.xdata - (event.xdata - x0) * scale,
                 event.xdata + (x1 - event.xdata) * scale)
        new_y = (event.ydata - (event.ydata - y0) * scale,
                 event.ydata + (y1 - event.ydata) * scale)
        self.ax.set_xlim(new_x)
        self.ax.set_ylim(new_y)
        self._view_limits = (new_x, new_y)
        self._user_view = True
        self.draw_idle()

    def _on_button_press(self, event):
        if event.inaxes is not self.ax:
            return
        if self._drawing_polygon:
            if event.button == 3:
                self.finish_polygon_drawing()
            elif event.button == 1:
                # A double-click closes the polygon; ordinary clicks add vertices.
                if getattr(event, "dblclick", False):
                    self.finish_polygon_drawing()
                elif event.xdata is not None and event.ydata is not None:
                    self._draft_points.append((float(event.xdata), float(event.ydata)))
                    self.draft_changed.emit()
            return
        if self._placing_circle:
            if event.button == 3:
                self.cancel_circle_placement()
            elif event.button == 1 and event.xdata is not None and event.ydata is not None:
                center = (float(event.xdata), float(event.ydata))
                radius = self._circle_radius
                self._placing_circle = False
                self._circle_preview_center = None
                self.draft_changed.emit()
                self.circle_finished.emit(center, radius)
            return
        if self._toolbar_is_active():
            return
        if event.button not in (1, 2):
            return
        # Clicking inside a polygon selects it instead of starting a pan gesture.
        if event.xdata is not None and event.ydata is not None and self._engine is not None:
            for obstacle in reversed(self._engine.obstacles):
                if (point_in_polygon(event.xdata, event.ydata, obstacle.vertices)
                        or distance_to_polygon(event.xdata, event.ydata, obstacle.vertices) <= 50.0):
                    self.selected_obstacle_id = obstacle.obstacle_id
                    self.obstacle_selected.emit(obstacle.obstacle_id)
                    self.draw_scene(self._engine)
                    return
        self._drag_anchor = self.ax.transData.inverted().transform((event.x, event.y))
        self._drag_limits = (self.ax.get_xlim(), self.ax.get_ylim())
        self._drag_moved = False

    def _on_motion(self, event):
        if self._placing_circle and event.inaxes is self.ax and event.xdata is not None and event.ydata is not None:
            self._circle_preview_center = (float(event.xdata), float(event.ydata))
            if self._engine is not None:
                self.draw_scene(self._engine)
            return
        if self._drag_anchor is None or self._drag_limits is None:
            return
        current = self.ax.transData.inverted().transform((event.x, event.y))
        dx = float(self._drag_anchor[0] - current[0])
        dy = float(self._drag_anchor[1] - current[1])
        if abs(dx) + abs(dy) < 1e-6:
            return
        xlim0, ylim0 = self._drag_limits
        new_x = (xlim0[0] + dx, xlim0[1] + dx)
        new_y = (ylim0[0] + dy, ylim0[1] + dy)
        self.ax.set_xlim(new_x)
        self.ax.set_ylim(new_y)
        self._view_limits = (new_x, new_y)
        self._user_view = True
        self._drag_moved = True
        self.draw_idle()

    def _on_button_release(self, event):
        if self._drag_anchor is not None:
            self._view_limits = (self.ax.get_xlim(), self.ax.get_ylim())
            self._user_view = True
            self._drag_anchor = None
            self._drag_limits = None
        elif self._toolbar_is_active():
            # Capture the axes extent produced by the standard Matplotlib toolbar.
            self._view_limits = (self.ax.get_xlim(), self.ax.get_ylim())
            self._user_view = True

    def draw_scene(self, engine):
        self._engine = engine
        for old_legend in list(self.figure.legends):
            old_legend.remove()
        ax = self.ax
        ax.clear()
        ax.set_facecolor("#071522")
        self.figure.patch.set_facecolor("#0c1b2a")
        ax.grid(True, color="#294052", alpha=0.5, linestyle="--", linewidth=0.7)
        ax.tick_params(colors="#d8e4ed", labelsize=10)
        for spine in ax.spines.values():
            spine.set_color("#557085")
        ax.set_xlabel("East position (m)", color="#d8e4ed", fontsize=11)
        ax.set_ylabel("North position (m)", color="#d8e4ed", fontsize=11)
        source = engine.scenario.metadata.get("source_group", "Demo")
        scene = engine.scenario.metadata.get("scene_id", "Demo")
        ax.set_title(f"Top-down encounter view | {source} / Scene {scene}", color="white", fontsize=14, pad=12)

        ax.plot([engine.route_start_x, engine.route_x], [engine.route_start_y, engine.route_y],
                color="#76b7ff", linestyle="--", linewidth=1.5, label="Planned route")
        if len(engine.own_track) > 1:
            ax.plot([p[0] for p in engine.own_track], [p[1] for p in engine.own_track],
                    color="#5ce1a6", linewidth=2.0, label="Ownship track")
        if engine.last_plan and engine.last_plan.predicted_path:
            ax.plot([p[0] for p in engine.last_plan.predicted_path],
                    [p[1] for p in engine.last_plan.predicted_path],
                    color="#ffd166", linewidth=1.5, linestyle=":", label="MPC prediction")

        own = engine.own
        # Extra-large electric-cyan ownship marker with a bright white halo.
        # The visual scale is deliberately separate from vessel physics.
        own_points = self._ship_polygon(
            own.x, own.y, own.psi, engine.config.length_m, engine.config.beam_m, scale=1.48
        )
        own_patch = Polygon(
            own_points, closed=True, facecolor="#00f5ff", edgecolor="#ffffff",
            linewidth=2.8, zorder=8,
            path_effects=[patheffects.Stroke(linewidth=5.2, foreground="#071522"), patheffects.Normal()],
        )
        ax.add_patch(own_patch)
        assessment_map = {a.target_id: a for a in engine.last_snapshot.assessments}
        for obstacle in engine.obstacles:
            is_selected = obstacle.obstacle_id == self.selected_obstacle_id
            manual = obstacle.obstacle_id.startswith("USR-")
            face = "#4dd0e1" if manual else "#e8b45b"
            edge = "#ffffff" if is_selected else ("#8ff3ff" if manual else "#ffe0a3")
            if obstacle.shape_type == "circle" and obstacle.center is not None and obstacle.radius_m:
                ox, oy = obstacle.center
                ax.add_patch(Circle((ox, oy), obstacle.radius_m, facecolor=face,
                                    edgecolor=edge, alpha=0.42 if is_selected else 0.32,
                                    linewidth=2.6 if is_selected else 1.5, zorder=3))
            else:
                ax.add_patch(Polygon(obstacle.vertices, closed=True, facecolor=face,
                                     edgecolor=edge, alpha=0.42 if is_selected else 0.32,
                                     linewidth=2.6 if is_selected else 1.5, zorder=3))
                ox = sum(v[0] for v in obstacle.vertices) / len(obstacle.vertices)
                oy = sum(v[1] for v in obstacle.vertices) / len(obstacle.vertices)
            ax.text(ox, oy, obstacle.obstacle_id, color="#ffffff" if is_selected else "#ffe0a3",
                    fontsize=8, ha="center", va="center", zorder=4)

        for target in engine.targets:
            a = assessment_map.get(target.target_id)
            points = engine.target_tracks.get(target.target_id, [])
            if len(points) > 1:
                ax.plot([p[0] for p in points], [p[1] for p in points],
                        color="#ff9f68", alpha=0.55, linewidth=1.1, label=f"{target.target_id} track")
            tp = Polygon(self._ship_polygon(target.state.x, target.state.y, target.state.psi,
                                            target.length_m, target.beam_m),
                         closed=True, facecolor="#f97365", edgecolor="white", linewidth=1.0, zorder=5)
            ax.add_patch(tp)
            ax.text(target.state.x, target.state.y, target.target_id, color="white", fontsize=8,
                    ha="center", va="center", zorder=6)
            if a and a.sda_m > 0.0:
                # Draw the Appendix-B SDA itself. SDApref is a separate risk-
                # timing margin and must not inflate the displayed boundary.
                current_range = math.hypot(own.x-target.state.x, own.y-target.state.y)
                already_failed = target.target_id in engine.metrics.sda_violation_targets
                inside_now = current_range < a.sda_m
                predicted_intrusion = a.dcpa_m < a.sda_m and a.tcpa_s > 0.0
                ring_color = ("#ff3b30" if already_failed or inside_now else
                              "#ff9f43" if predicted_intrusion else "#e8b45b")
                ring_style = "-" if already_failed or inside_now else "--"
                ax.add_patch(Circle((target.state.x, target.state.y), a.sda_m, fill=False,
                                    color=ring_color, linestyle=ring_style,
                                    linewidth=1.8 if already_failed or inside_now else 1.25,
                                    alpha=0.88, zorder=4))
                approx_mark = "~" if a.sda_estimated else ""
                ax.text(target.state.x, target.state.y + a.sda_m,
                        f"{approx_mark}SDA {a.sda_m:.0f} m",
                        color=ring_color, fontsize=7, ha="center", va="bottom",
                        bbox=dict(facecolor="#101820", edgecolor="none", alpha=0.65, pad=1.5),
                        zorder=7)

        if self._draft_points:
            draft_x = [p[0] for p in self._draft_points]
            draft_y = [p[1] for p in self._draft_points]
            ax.plot(draft_x, draft_y, color="#ffffff", linewidth=2.0, linestyle="--",
                    marker="o", markersize=5, markerfacecolor="#00e5ff",
                    markeredgecolor="#ffffff", zorder=10, label="Obstacle being drawn")
            if len(self._draft_points) >= 3:
                ax.plot([draft_x[-1], draft_x[0]], [draft_y[-1], draft_y[0]],
                        color="#00e5ff", linewidth=1.0, linestyle=":", zorder=9)
        if self._placing_circle and self._circle_preview_center is not None:
            ax.add_patch(Circle(self._circle_preview_center, self._circle_radius,
                                facecolor="#00e5ff", edgecolor="#ffffff", alpha=0.18,
                                linewidth=1.8, linestyle="--", zorder=10))
            ax.plot([self._circle_preview_center[0]], [self._circle_preview_center[1]],
                    marker="+", color="#ffffff", markersize=9, zorder=11)

        points_x = [own.x, engine.route_start_x]
        points_y = [own.y, engine.route_start_y]
        for px, py in engine.own_track:
            points_x.append(px); points_y.append(py)
        for t in engine.targets:
            points_x.append(t.state.x); points_y.append(t.state.y)
            sda_assessment = assessment_map.get(t.target_id)
            if sda_assessment is not None and sda_assessment.sda_m > 0.0:
                # Keep the complete standard-SDA circle inside the automatic map extent.
                points_x.extend([t.state.x - sda_assessment.sda_m,
                                 t.state.x + sda_assessment.sda_m])
                points_y.extend([t.state.y - sda_assessment.sda_m,
                                 t.state.y + sda_assessment.sda_m])
            for px, py in engine.target_tracks.get(t.target_id, [])[-1200:]:
                points_x.append(px); points_y.append(py)
        for o in engine.obstacles:
            if o.shape_type == "circle" and o.center is not None and o.radius_m:
                ox, oy = o.center
                points_x.extend([ox - o.radius_m, ox + o.radius_m])
                points_y.extend([oy - o.radius_m, oy + o.radius_m])
            else:
                points_x.extend(v[0] for v in o.vertices); points_y.extend(v[1] for v in o.vertices)
        for px, py in self._draft_points:
            points_x.append(px); points_y.append(py)
        if self._placing_circle and self._circle_preview_center is not None:
            cx, cy = self._circle_preview_center
            points_x.extend([cx - self._circle_radius, cx + self._circle_radius])
            points_y.extend([cy - self._circle_radius, cy + self._circle_radius])
        min_x, max_x = min(points_x), max(points_x)
        min_y, max_y = min(points_y), max(points_y)
        bbox = ax.get_position()
        fig_w, fig_h = self.figure.get_size_inches()
        plot_aspect = (bbox.width * fig_w) / max(1e-6, bbox.height * fig_h)
        plot_aspect = max(1.15, min(2.5, float(plot_aspect)))
        y_span = max(3500.0, (max_y - min_y) * 1.10, ((max_x - min_x) / plot_aspect) * 1.10)
        x_span = y_span * plot_aspect
        cx = (min_x + max_x) / 2.0
        cy = (min_y + max_y) / 2.0
        auto_x = (cx - x_span / 2.0, cx + x_span / 2.0)
        auto_y = (cy - y_span / 2.0, cy + y_span / 2.0)
        if self._user_view and self._view_limits is not None:
            ax.set_xlim(self._view_limits[0])
            ax.set_ylim(self._view_limits[1])
        else:
            ax.set_xlim(auto_x)
            ax.set_ylim(auto_y)
            self._view_limits = (auto_x, auto_y)
        ax.set_aspect("equal", adjustable="box")

        # The legend now lives in the right-side information panel, leaving the
        # complete left canvas available to the nautical chart.
        self.draw_idle()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.project_root = Path(__file__).resolve().parents[1]
        self.dataset_root = default_dataset_root()
        self.case_specs = list_cases(self.dataset_root)
        self.case_by_path = {str(spec.path): spec for spec in self.case_specs}
        self.engine: SimulationEngine | None = None
        self.training_worker: TrainingWorker | None = None
        self.evaluation_worker: TrainingWorker | None = None
        self.ppo_checkpoint: Path | None = (self.project_root / "models" / "ppo_policy.pt")
        self.last_evaluation_csv: Path | None = None
        self._manual_obstacle_counter = 0
        self._manual_obstacles: list[StaticObstacle] = []
        self._manual_obstacle_scene_key = None
        self.timer = QTimer(self)
        self.timer.setInterval(120)
        self.timer.timeout.connect(self.on_timer)
        self.sim_speed = 1
        self._filling_targets = False
        self.setWindowTitle("无人船智能避碰仿真系统")
        self.resize(1920, 1200)
        self.setMinimumSize(1280, 820)
        self._build_ui()
        self._create_menu()
        self.populate_cases()
        self.reset_simulation()
        self.setWindowState(self.windowState() | Qt.WindowState.WindowMaximized)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 8, 10, 8)
        root.setSpacing(7)

        title = QLabel("无人船自主避碰仿真实验平台")
        title.setStyleSheet("font-size:20px;font-weight:700;color:#ecf5fb;padding:3px;")
        subtitle = QLabel("COLREGs 规则约束  |  多船风险评估  |  DWA-MPC 局部规划  |  PPO 智能决策  |  船舶运动控制模型")
        subtitle.setStyleSheet("color:#9fb6c8;font-size:11px;padding-left:5px;")
        root.addWidget(title)
        root.addWidget(subtitle)

        controls = QGroupBox("场景与仿真控制")
        cg = QGridLayout(controls)
        cg.setHorizontalSpacing(8)
        cg.setVerticalSpacing(6)
        self.source_combo = QComboBox()
        self.source_combo.addItem("内置示例", "demo")
        self.source_combo.addItem("1T 案例（138）", "1T")
        self.source_combo.addItem("2T 案例（99）", "2T")
        self.source_combo.addItem("3T 案例（31）", "3T")
        self.source_combo.addItem("全部上传案例（268）", "ALL")
        self.source_combo.setCurrentIndex(1)
        self.source_combo.currentIndexChanged.connect(self.populate_cases)
        self.case_combo = QComboBox()
        self.case_combo.setMinimumWidth(175)
        self.demo_targets_combo = QComboBox()
        self.demo_targets_combo.addItems(["1 艘目标船", "2 艘目标船", "3 艘目标船"])
        self.demo_targets_combo.setCurrentIndex(0)
        self.own_length_spin = QSpinBox()
        self.own_length_spin.setRange(50, 500)
        self.own_length_spin.setSingleStep(10)
        self.own_length_spin.setValue(200)
        self.own_length_spin.setSuffix(" m")
        self.target_length_spin = QSpinBox()
        self.target_length_spin.setRange(50, 500)
        self.target_length_spin.setSingleStep(10)
        self.target_length_spin.setValue(100)
        self.target_length_spin.setSuffix(" m")
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["1x", "2x", "4x", "8x"])
        self.speed_combo.currentIndexChanged.connect(lambda i: setattr(self, "sim_speed", [1, 2, 4, 8][i]))
        self.obstacle_check = QCheckBox("示例多边形障碍物")
        self.load_case_btn = QPushButton("加载选中案例")
        self.start_btn = QPushButton("开始仿真")
        self.pause_btn = QPushButton("暂停")
        self.step_btn = QPushButton("单步运行")
        self.reset_btn = QPushButton("重置")
        self.export_btn = QPushButton("导出日志")
        self.clear_track_btn = QPushButton("清除轨迹")
        self.load_case_btn.clicked.connect(self.reset_simulation)
        self.start_btn.clicked.connect(self.start_simulation)
        self.pause_btn.clicked.connect(self.pause_simulation)
        self.step_btn.clicked.connect(self.single_step)
        self.reset_btn.clicked.connect(self.reset_simulation)
        self.export_btn.clicked.connect(self.export_log)
        self.clear_track_btn.clicked.connect(self.clear_tracks)

        cg.addWidget(QLabel("案例集"), 0, 0); cg.addWidget(self.source_combo, 0, 1)
        cg.addWidget(QLabel("场景"), 1, 0); cg.addWidget(self.case_combo, 1, 1)
        length_row = QHBoxLayout()
        length_row.setContentsMargins(0, 0, 0, 0)
        length_row.setSpacing(6)
        length_row.addWidget(QLabel("本船")); length_row.addWidget(self.own_length_spin, 1)
        length_row.addWidget(QLabel("目标船")); length_row.addWidget(self.target_length_spin, 1)
        cg.addLayout(length_row, 2, 0, 1, 2)
        playback_row = QHBoxLayout()
        playback_row.setContentsMargins(0, 0, 0, 0)
        playback_row.setSpacing(6)
        playback_row.addWidget(QLabel("目标船数量")); playback_row.addWidget(self.demo_targets_combo, 1)
        playback_row.addWidget(QLabel("速度")); playback_row.addWidget(self.speed_combo, 1)
        cg.addLayout(playback_row, 3, 0, 1, 2)
        cg.addWidget(self.obstacle_check, 4, 0, 1, 2)
        cg.addWidget(self.load_case_btn, 5, 0, 1, 2)
        cg.addWidget(self.start_btn, 6, 0); cg.addWidget(self.pause_btn, 6, 1)
        cg.addWidget(self.step_btn, 7, 0); cg.addWidget(self.reset_btn, 7, 1)
        cg.addWidget(self.export_btn, 8, 0); cg.addWidget(self.clear_track_btn, 8, 1)
        cg.setColumnStretch(0, 1); cg.setColumnStretch(1, 2)
        training = QGroupBox("PPO 强化学习训练 | 上传案例数据集")
        tg = QGridLayout(training)
        self.train_steps_spin = QSpinBox()
        self.train_steps_spin.setRange(100, 500000)
        self.train_steps_spin.setSingleStep(5000)
        self.train_steps_spin.setValue(150000)
        self.training_device_combo = QComboBox()
        self.training_device_combo.addItem("CUDA GPU（推荐）", "cuda")
        self.training_device_combo.addItem("CPU（备用）", "cpu")
        self.gpu_status_label = QLabel(gpu_status_text())
        self.gpu_status_label.setWordWrap(True)
        self.gpu_status_label.setStyleSheet("color:#9fd6b8;font-size:10px;")
        self.train_button = QPushButton("开始 PPO 训练")
        self.train_button.clicked.connect(self.toggle_training)
        self.load_ppo_button = QPushButton("加载 PPO 模型")
        self.load_ppo_button.clicked.connect(self.load_ppo)
        self.unload_ppo_button = QPushButton("取消加载 PPO 模型")
        self.unload_ppo_button.clicked.connect(self.unload_ppo)
        self.unload_ppo_button.setEnabled(False)
        self.ppo_model_status = QLabel("当前模型：规则策略（未加载 PPO）")
        self.ppo_model_status.setWordWrap(True)
        self.ppo_model_status.setStyleSheet("color:#a9c4d8;font-size:10px;")
        self.train_progress = QProgressBar()
        self.train_progress.setRange(0, 100)
        self.train_progress.setValue(0)
        self.train_status = QLabel("就绪。案例按几何分组划分训练／验证／测试集，仅训练集参与学习；默认不进行教师预热。PPO 网络可选 CUDA，候选预测仍在 CPU 运行。")
        self.train_status.setStyleSheet("color:#a9c4d8;font-size:10px;")
        self.train_status.setWordWrap(True)
        tg.addWidget(QLabel("训练交互步数"), 0, 0); tg.addWidget(self.train_steps_spin, 0, 1)
        tg.addWidget(QLabel("训练设备"), 1, 0); tg.addWidget(self.training_device_combo, 1, 1)
        tg.addWidget(self.train_button, 2, 0, 1, 2)
        tg.addWidget(self.load_ppo_button, 3, 0); tg.addWidget(self.unload_ppo_button, 3, 1)
        tg.addWidget(self.train_progress, 4, 0, 1, 2)
        tg.addWidget(self.gpu_status_label, 5, 0, 1, 2)
        tg.addWidget(self.ppo_model_status, 6, 0, 1, 2)
        tg.addWidget(self.train_status, 7, 0, 1, 2)

        evaluation = QGroupBox("训练权重批量评估 | 1T / 2T / 3T 案例")
        eg = QGridLayout(evaluation)
        self.eval_group_combo = QComboBox()
        self.eval_group_combo.addItem("独立测试集：1T", "1T")
        self.eval_group_combo.addItem("独立测试集：2T", "2T")
        self.eval_group_combo.addItem("独立测试集：3T", "3T")
        self.eval_group_combo.addItem("独立测试集：全部组别", "ALL")
        self.eval_max_steps_spin = QSpinBox()
        self.eval_max_steps_spin.setRange(100, 30000)
        self.eval_max_steps_spin.setSingleStep(500)
        self.eval_max_steps_spin.setValue(6000)
        self.eval_start_btn = QPushButton("开始批量评估并保存 CSV")
        self.eval_stop_btn = QPushButton("停止评估")
        self.eval_stop_btn.setEnabled(False)
        self.eval_progress = QProgressBar()
        self.eval_progress.setRange(0, 100)
        self.eval_progress.setValue(0)
        self.eval_status = QLabel("加载新框架权重后，按权重保存的独立测试集评估；结果按案例逐行写入 CSV。全部案例对照可使用命令行 --split all。")
        self.eval_status.setWordWrap(True)
        self.eval_status.setStyleSheet("color:#a9c4d8;font-size:10px;")
        self.eval_start_btn.clicked.connect(self.start_evaluation)
        self.eval_stop_btn.clicked.connect(self.stop_evaluation)
        eg.addWidget(QLabel("评估案例组"), 0, 0, 1, 2)
        eg.addWidget(self.eval_group_combo, 1, 0, 1, 2)
        eg.addWidget(QLabel("单案例最大步数"), 2, 0); eg.addWidget(self.eval_max_steps_spin, 2, 1)
        eg.addWidget(self.eval_start_btn, 3, 0, 1, 2)
        eg.addWidget(self.eval_stop_btn, 4, 0, 1, 2)
        eg.addWidget(self.eval_progress, 5, 0, 1, 2)
        eg.addWidget(self.eval_status, 6, 0, 1, 2)

        top_tabs = QTabWidget()
        top_tabs.addTab(controls, "场景 / 仿真")
        top_tabs.addTab(training, "PPO / GPU")
        top_tabs.addTab(evaluation, "批量评估")
        splitter = QSplitter(Qt.Orientation.Horizontal)
        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)
        plot_layout.setContentsMargins(0, 0, 4, 0)
        self.canvas = SimulationCanvas(self)
        self.plot_toolbar = NavigationToolbar(self.canvas, self)
        self.canvas.toolbar = self.plot_toolbar
        self.canvas.polygon_finished.connect(self.add_manual_obstacle)
        self.canvas.circle_finished.connect(self.add_manual_circle_obstacle)
        self.canvas.draft_changed.connect(self.on_polygon_draft_changed)
        self.canvas.obstacle_selected.connect(self.on_obstacle_selected)
        # Matplotlib toolbar icons remain standard; their tooltips are translated.
        for action in self.plot_toolbar.actions():
            tip = action.toolTip()
            translations = {
                "Home": "恢复初始视图", "Back": "后退", "Forward": "前进",
                "Pan": "平移", "Zoom": "框选缩放", "Configure subplots": "调整子图",
                "Save the figure": "保存地图图片",
            }
            if tip in translations:
                action.setToolTip(translations[tip])
            if tip == "Home":
                action.triggered.connect(self.reset_map_view)
        plot_layout.addWidget(self.plot_toolbar)
        chart_row_widget = QWidget()
        chart_row = QHBoxLayout(chart_row_widget)
        chart_row.setContentsMargins(0, 0, 0, 0)
        chart_row.setSpacing(5)
        chart_row.addWidget(self.canvas, 1)

        map_legend_panel = QFrame()
        map_legend_panel.setObjectName("mapLegendPanel")
        map_legend_panel.setFixedWidth(156)
        map_legend_panel.setMaximumHeight(226)
        legend_layout = QVBoxLayout(map_legend_panel)
        legend_layout.setContentsMargins(8, 8, 7, 7)
        legend_layout.setSpacing(4)
        legend_title = QLabel("海图图例")
        legend_title.setStyleSheet("font-size:12px;font-weight:700;color:#f1f7fb;padding-bottom:3px;")
        legend_layout.addWidget(legend_title)
        map_legend_label = QLabel(
            '<span style="color:#76b7ff">━━</span> 规划航线<br>'
            '<span style="color:#00f5ff">▲</span> 本船<br>'
            '<span style="color:#5ce1a6">━━</span> 本船轨迹<br>'
            '<span style="color:#f97365">▲</span> 目标船<br>'
            '<span style="color:#ff9f68">━━</span> 目标船轨迹<br>'
            '<span style="color:#ffd166">┈┈</span> MPC 预测<br>'
            '<span style="color:#e8b45b">■</span> 示例障碍物<br>'
            '<span style="color:#4dd0e1">■</span> 手动障碍物<br>'
            '<span style="color:#e8b45b">┈┈</span> 标准 SDA 边界（圆半径）<br>'
            '<span style="color:#ff3b30">━━</span> 已侵入 SDA（本回合失败）'
        )
        map_legend_label.setTextFormat(Qt.TextFormat.RichText)
        map_legend_label.setWordWrap(True)
        map_legend_label.setStyleSheet("color:#dce8f1;font-size:10px;line-height:1.35;padding:1px;")
        legend_layout.addWidget(map_legend_label)
        legend_layout.addStretch(1)
        chart_row.addWidget(map_legend_panel, 0, Qt.AlignmentFlag.AlignTop)
        plot_layout.addWidget(chart_row_widget, 1)
        splitter.addWidget(plot_panel)

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(5, 5, 5, 5)
        rv.setSpacing(7)
        # All controls/settings/status are placed on the right; the left side is
        # reserved for a large map canvas.
        top_tabs.setMinimumHeight(300)
        rv.addWidget(top_tabs)

        map_control_group = QGroupBox("海图操作与障碍物")
        map_control_layout = QGridLayout(map_control_group)
        self.draw_obstacle_btn = QPushButton("绘制多边形")
        self.place_circle_obstacle_btn = QPushButton("放置圆形障碍物")
        self.circle_radius_spin = QDoubleSpinBox()
        self.circle_radius_spin.setRange(10.0, 10000.0)
        self.circle_radius_spin.setSingleStep(50.0)
        self.circle_radius_spin.setDecimals(1)
        self.circle_radius_spin.setValue(300.0)
        self.circle_radius_spin.setSuffix(" m")
        self.finish_obstacle_btn = QPushButton("完成并放置")
        self.undo_obstacle_vertex_btn = QPushButton("撤销顶点")
        self.cancel_obstacle_btn = QPushButton("取消绘制/放置")
        self.delete_obstacle_btn = QPushButton("删除选中障碍物")
        self.clear_manual_obstacles_btn = QPushButton("清空手动障碍物")
        self.reset_map_view_btn = QPushButton("重置海图视野")
        self.finish_obstacle_btn.setEnabled(False)
        self.undo_obstacle_vertex_btn.setEnabled(False)
        self.cancel_obstacle_btn.setEnabled(False)
        self.delete_obstacle_btn.setEnabled(False)
        self.draw_obstacle_btn.clicked.connect(self.start_polygon_drawing)
        self.place_circle_obstacle_btn.clicked.connect(self.start_circle_obstacle_placement)
        self.finish_obstacle_btn.clicked.connect(self.finish_polygon_drawing)
        self.undo_obstacle_vertex_btn.clicked.connect(self.canvas.undo_polygon_vertex)
        self.cancel_obstacle_btn.clicked.connect(self.cancel_polygon_drawing)
        self.delete_obstacle_btn.clicked.connect(self.delete_selected_obstacle)
        self.clear_manual_obstacles_btn.clicked.connect(self.clear_manual_obstacles)
        self.reset_map_view_btn.clicked.connect(self.reset_map_view)
        map_control_layout.addWidget(self.draw_obstacle_btn, 0, 0)
        map_control_layout.addWidget(self.place_circle_obstacle_btn, 0, 1)
        map_control_layout.addWidget(QLabel("圆形半径"), 1, 0)
        map_control_layout.addWidget(self.circle_radius_spin, 1, 1)
        map_control_layout.addWidget(self.finish_obstacle_btn, 2, 0)
        map_control_layout.addWidget(self.undo_obstacle_vertex_btn, 2, 1)
        map_control_layout.addWidget(self.cancel_obstacle_btn, 3, 0, 1, 2)
        map_control_layout.addWidget(self.delete_obstacle_btn, 4, 0, 1, 2)
        map_control_layout.addWidget(self.clear_manual_obstacles_btn, 5, 0)
        map_control_layout.addWidget(self.reset_map_view_btn, 5, 1)
        self.obstacle_draw_hint = QLabel("海图操作：滚轮缩放，按住左键拖动。多边形逐点绘制；圆形输入半径后点击海图确定圆心。")
        self.obstacle_draw_hint.setWordWrap(True)
        self.obstacle_draw_hint.setStyleSheet("color:#9fb6c8;font-size:10px;padding:2px;")
        map_control_layout.addWidget(self.obstacle_draw_hint, 6, 0, 1, 2)
        rv.addWidget(map_control_group)

        self.status_label = QLabel("系统状态：待机")
        self.status_label.setStyleSheet("font-size:13px;font-weight:700;color:#7ee0b1;padding:3px;")
        self.own_status_label = QLabel("本船状态：等待加载场景")
        self.own_status_label.setWordWrap(True)
        self.own_status_label.setStyleSheet("background:#102638;color:#e3f2fa;border:1px solid #345066;border-radius:5px;padding:7px;font-size:11px;")
        self.metrics_label = QLabel("仿真指标将在此处显示。")
        self.metrics_label.setWordWrap(True)
        self.metrics_label.setStyleSheet("color:#c7d8e4;padding:2px;font-size:10px;")
        runtime_group = QGroupBox("实时状态")
        runtime_layout = QVBoxLayout(runtime_group)
        runtime_layout.setContentsMargins(7, 8, 7, 7)
        runtime_layout.setSpacing(4)
        runtime_layout.addWidget(self.status_label)
        runtime_layout.addWidget(self.own_status_label)
        runtime_layout.addWidget(self.metrics_label)
        rv.addWidget(runtime_group)

        detail_tabs = QTabWidget()
        target_group = QWidget()
        target_group_layout = QVBoxLayout(target_group)
        target_group_layout.setContentsMargins(3, 4, 3, 3)
        target_layout = target_group_layout
        self.targets_table = QTableWidget(0, 6)
        self.targets_table.setHorizontalHeaderLabels(["目标船", "船长(m)", "对地航速(kn)", "对地航向(°)", "船首向(°)", "风险等级"])
        self.targets_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.targets_table.verticalHeader().setVisible(False)
        self.targets_table.setMinimumHeight(115)
        self.targets_table.setMaximumHeight(180)
        self.targets_table.cellChanged.connect(self.on_target_cell_changed)
        target_layout.addWidget(self.targets_table)
        target_hint = QLabel("可编辑目标船的船长，范围为 50～500 m。")
        target_hint.setWordWrap(True)
        target_hint.setStyleSheet("color:#9fb6c8;font-size:9px;")
        target_layout.addWidget(target_hint)
        detail_tabs.addTab(target_group, "目标船")

        risk_group = QWidget()
        risk_layout = QVBoxLayout(risk_group)
        risk_layout.setContentsMargins(3, 4, 3, 3)
        self.risk_table = QTableWidget(0, 11)
        self.risk_table.setHorizontalHeaderLabels(["目标船", "案例标签 / 程序识别", "避让责任", "距离(m)", "DCPA(m)", "TCPA(s)", "SDA(m)", "DCRC估算(m)", "DCQS估算(m)", "追越通过侧", "SDA通过位置/来源"])
        self.risk_table.setWordWrap(True)
        self.risk_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.risk_table.verticalHeader().setVisible(False)
        self.risk_table.setMinimumHeight(120)
        self.risk_table.setMaximumHeight(210)
        self.risk_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        risk_layout.addWidget(self.risk_table)
        detail_tabs.addTab(risk_group, "碰撞风险")

        log_group = QWidget()
        log_layout = QVBoxLayout(log_group)
        log_layout.setContentsMargins(3, 4, 3, 3)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(140)
        self.log_text.setMaximumHeight(240)
        log_layout.addWidget(self.log_text)
        detail_tabs.addTab(log_group, "运行日志")
        detail_tabs.setMinimumHeight(245)
        rv.addWidget(detail_tabs, 1)
        rv.addStretch(1)

        right_scroll.setWidget(right)
        right_scroll.setMinimumWidth(370)
        right_scroll.setMaximumWidth(470)
        splitter.addWidget(right_scroll)
        splitter.setHandleWidth(7)
        splitter.setStretchFactor(0, 8)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([1540, 380])
        root.addWidget(splitter, 1)

        self.setStyleSheet("""
            QMainWindow,QWidget { background:#0c1b2a; color:#dce8f1; }
            QGroupBox { color:#e1edf5; font-weight:600; border:1px solid #36536a;
                        border-radius:6px; margin-top:8px; padding-top:8px; }
            QGroupBox::title { subcontrol-origin:margin; left:10px; padding:0 4px; }
            QPushButton { background:#1c4661; color:white; border:1px solid #47728f;
                          border-radius:5px; padding:6px 10px; min-height:22px; }
            QPushButton:hover { background:#286184; }
            QPushButton:disabled { background:#253744; color:#7b8f9d; }
            QComboBox,QSpinBox,QDoubleSpinBox { background:#142b3d; color:white; border:1px solid #45657c;
                                 padding:5px; border-radius:4px; min-height:22px; }
            QTableWidget { background:#102638; color:#e1edf5; gridline-color:#345066;
                           border:1px solid #345066; font-size:10px; }
            QHeaderView::section { background:#19364b; color:#e1edf5; padding:5px;
                                   border:1px solid #345066; font-size:9px; }
            QTextEdit { background:#102638; color:#bed0dc; border:1px solid #345066; }
            QProgressBar { background:#102638; color:#e1edf5; border:1px solid #345066;
                           border-radius:4px; text-align:center; min-height:18px; }
            QProgressBar::chunk { background:#2b9a77; border-radius:3px; }
            QScrollArea { border:0px; }
            QFrame#mapLegendPanel { background:#102638; border:1px solid #45657c; border-radius:7px; }
            QTabWidget::pane { border:1px solid #36536a; border-radius:5px; top:-1px; }
            QTabBar::tab { background:#142b3d; color:#bcd0dd; padding:7px 10px; margin-right:2px;
                           border:1px solid #36536a; border-bottom:none; border-top-left-radius:4px; border-top-right-radius:4px; }
            QTabBar::tab:selected { background:#1c4661; color:#ffffff; font-weight:700; }
        """)

    def _create_menu(self):
        file_menu = self.menuBar().addMenu("文件")
        load_action = QAction("加载 PPO 模型...", self)
        load_action.triggered.connect(self.load_ppo)
        file_menu.addAction(load_action)
        unload_action = QAction("取消加载 PPO 模型", self)
        unload_action.triggered.connect(self.unload_ppo)
        file_menu.addAction(unload_action)
        scenario_action = QAction("加载 JSON 场景...", self)
        scenario_action.triggered.connect(self.load_scenario_file)
        file_menu.addAction(scenario_action)
        file_menu.addSeparator()
        exit_action = QAction("退出", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        help_menu = self.menuBar().addMenu("帮助")
        about = QAction("关于系统与模型限制", self)
        about.triggered.connect(self.show_about)
        help_menu.addAction(about)

    def populate_cases(self, *_):
        source = self.source_combo.currentData()
        self.case_combo.blockSignals(True)
        self.case_combo.clear()
        if source == "demo":
            self.case_combo.addItem("内置混合会遇场景", None)
            self.case_combo.setEnabled(False)
            self.demo_targets_combo.setEnabled(True)
        else:
            self.case_combo.setEnabled(True)
            self.demo_targets_combo.setEnabled(False)
            group = None if source == "ALL" else source
            specs = list_cases(self.dataset_root, group)
            for spec in specs:
                self.case_combo.addItem(f"{spec.group} | 场景 {spec.scene_id}", str(spec.path))
            if specs:
                self.case_combo.setCurrentIndex(0)
        self.case_combo.blockSignals(False)

    def _current_scene_key(self):
        source = self.source_combo.currentData()
        if source == "demo":
            return ("demo", int(self.demo_targets_combo.currentIndex() + 1))
        return (str(source), str(self.case_combo.currentData() or ""))

    def _scenario_from_controls(self):
        source = self.source_combo.currentData()
        own_len = self.own_length_spin.value()
        target_len = self.target_length_spin.value()
        if source == "demo":
            count = self.demo_targets_combo.currentIndex() + 1
            scenario = make_scenario(count, own_len)
            for target in scenario.targets:
                target.length_m = float(target_len)
                target.beam_m = max(8.0, 0.16 * target.length_m)
            if self.obstacle_check.isChecked():
                scenario = add_demo_obstacle(scenario)
            return scenario
        spec_path = self.case_combo.currentData()
        spec = self.case_by_path.get(str(spec_path)) if spec_path else None
        if spec is None:
            raise ValueError("请先选择一个数据集案例。")
        scenario = load_case(spec, own_length_m=own_len, default_target_length_m=target_len)
        if self.obstacle_check.isChecked():
            scenario = add_demo_obstacle(scenario)
        return scenario

    def reset_simulation(self):
        self.timer.stop()
        if hasattr(self, "canvas"):
            self.cancel_polygon_drawing()
            self.canvas.set_selected_obstacle(None)
            self.canvas.reset_view()
        try:
            scene_key = self._current_scene_key()
            if self._manual_obstacle_scene_key != scene_key:
                self._manual_obstacles = []
                self._manual_obstacle_scene_key = scene_key
            scenario = self._scenario_from_controls()
            # Keep manually placed polygons when resetting the same selected scene.
            scenario.obstacles.extend(o.copy() for o in self._manual_obstacles)
            checkpoint = self.ppo_checkpoint
            self.engine = SimulationEngine(scenario, str(checkpoint) if checkpoint else None)
            self._manual_obstacle_counter = 0
            self.log_text.clear()
            if scenario.metadata.get("source_group"):
                self.log_text.append(f"已加载 {scenario.name}，目标船数量：{len(scenario.targets)}")
                self.log_text.append("数据字段：经纬度、对地航速、对地航向、DCPA、TCPA、方位及相对运动。")
                labels = scenario.metadata.get("meeting_situations", [])
                stages = scenario.metadata.get("stage_labels", [])
                if labels:
                    self.log_text.append("原始会遇标签：" + " | ".join(encounter_label(x) for x in labels))
                if stages:
                    stage_text = ["碰撞危险" if x == "碰撞危险" else "无危险" if x in ("没有危险", "无危险") else x for x in stages]
                    self.log_text.append("原始风险阶段标签：" + " | ".join(stage_text))
                self.log_text.append("原始案例未提供船舶长度，长度由界面参数设定。")
            else:
                self.log_text.append(f"已加载内置示例，目标船数量：{len(scenario.targets)}")
            if self.engine.policy.available:
                self.ppo_model_status.setText(f"当前模型：PPO 已加载 — {Path(self.engine.policy.checkpoint).name}")
                self.unload_ppo_button.setEnabled(True)
                self.eval_start_btn.setEnabled(True)
            else:
                self.ppo_model_status.setText("当前模型：规则策略（未加载 PPO）")
                self.unload_ppo_button.setEnabled(False)
                self.eval_start_btn.setEnabled(False)
                self.log_text.append("未加载 PPO 权重，当前使用规则策略。训练完成或加载模型后可启用 PPO。")
                if self.engine.policy.load_error:
                    self.log_text.append(self.engine.policy.load_error)
            self.update_view()
        except Exception as exc:
            QMessageBox.critical(self, "场景加载错误", str(exc))

    def start_simulation(self):
        if self.engine:
            self.engine.running = True
            self.timer.start()
            self.status_label.setText("系统状态：运行中")

    def pause_simulation(self):
        self.timer.stop()
        if self.engine:
            self.engine.running = False
        self.status_label.setText("系统状态：已暂停")

    def single_step(self):
        if not self.engine:
            return
        self.engine.step()
        self.update_view()

    def on_timer(self):
        if not self.engine or not self.engine.running:
            self.timer.stop()
            return
        for _ in range(self.sim_speed):
            self.engine.step()
            if not self.engine.running:
                break
        self.update_view()

    def start_polygon_drawing(self):
        if not self.engine:
            QMessageBox.warning(self, "无法绘制", "请先加载一个仿真场景。")
            return
        if self.timer.isActive():
            self.pause_simulation()
        # Avoid simultaneous selection-rectangle/pan gestures while placing vertices.
        for action in self.plot_toolbar.actions():
            if action.isCheckable() and action.isChecked():
                action.trigger()
        self.canvas.start_polygon_drawing()
        self.draw_obstacle_btn.setEnabled(False)
        self.place_circle_obstacle_btn.setEnabled(False)
        self.finish_obstacle_btn.setEnabled(True)
        self.undo_obstacle_vertex_btn.setEnabled(True)
        self.cancel_obstacle_btn.setEnabled(True)
        self.obstacle_draw_hint.setText("绘制模式：在海图上逐点单击多边形顶点；至少 3 个顶点后点击“完成并放置”，也可双击或右键结束。")
        self.log_text.append("已进入多边形障碍物绘制模式；仿真已暂停，地图滚轮缩放仍可使用。")

    def start_circle_obstacle_placement(self):
        if not self.engine:
            QMessageBox.warning(self, "无法放置", "请先加载一个仿真场景。")
            return
        if self.timer.isActive():
            self.pause_simulation()
        for action in self.plot_toolbar.actions():
            if action.isCheckable() and action.isChecked():
                action.trigger()
        radius = float(self.circle_radius_spin.value())
        self.canvas.start_circle_placement(radius)
        self.draw_obstacle_btn.setEnabled(False)
        self.place_circle_obstacle_btn.setEnabled(False)
        self.finish_obstacle_btn.setEnabled(False)
        self.undo_obstacle_vertex_btn.setEnabled(False)
        self.cancel_obstacle_btn.setEnabled(True)
        self.obstacle_draw_hint.setText(f"圆形放置模式：半径 {radius:g} m。将鼠标移动到海图预览圆形，单击确定圆心；右键可取消。")
        self.log_text.append(f"已进入圆形障碍物放置模式，设定半径 {radius:g} m；仿真已暂停。")

    def add_manual_circle_obstacle(self, center, radius_m):
        if not self.engine:
            return
        cx, cy = float(center[0]), float(center[1])
        radius = float(radius_m)
        if not (math.isfinite(cx) and math.isfinite(cy) and math.isfinite(radius)) or radius <= 0:
            QMessageBox.warning(self, "圆形障碍物无效", "圆心坐标必须有效，半径必须大于 0。")
            self.cancel_polygon_drawing()
            return
        # The circumscribed boundary ensures the MPC's polygon-distance checks
        # never underestimate the user-specified circle radius.
        try:
            vertices = circle_polygon_vertices((cx, cy), radius, segments=64)
        except ValueError as exc:
            QMessageBox.warning(self, "圆形障碍物无效", str(exc))
            self.cancel_polygon_drawing()
            return
        existing_ids = {o.obstacle_id for o in self.engine.obstacles}
        while True:
            self._manual_obstacle_counter += 1
            obstacle_id = f"USR-OBS-{self._manual_obstacle_counter:03d}"
            if obstacle_id not in existing_ids:
                break
        obstacle = StaticObstacle(obstacle_id, vertices, "circle", (cx, cy), radius)
        self.engine.obstacles.append(obstacle)
        self._manual_obstacles.append(obstacle.copy())
        self._manual_obstacle_scene_key = self._current_scene_key()
        self.engine.scenario.obstacles = self.engine.obstacles
        self.canvas.set_selected_obstacle(obstacle_id)
        self.draw_obstacle_btn.setEnabled(True)
        self.place_circle_obstacle_btn.setEnabled(True)
        self.finish_obstacle_btn.setEnabled(False)
        self.undo_obstacle_vertex_btn.setEnabled(False)
        self.cancel_obstacle_btn.setEnabled(False)
        self.delete_obstacle_btn.setEnabled(True)
        self.obstacle_draw_hint.setText(f"已放置圆形障碍物 {obstacle_id}：半径 {radius:g} m；点击圆形内部可选中并删除。")
        self.log_text.append(f"已放置圆形障碍物 {obstacle_id}，圆心=({cx:.1f}, {cy:.1f}) m，半径={radius:g} m；已接入 MPC 预测避障及碰撞检测。")
        self.update_view()

    def finish_polygon_drawing(self):
        if not self.canvas.finish_polygon_drawing():
            QMessageBox.information(self, "多边形顶点不足", "至少单击 3 个不同顶点后才能放置障碍物。")

    def cancel_polygon_drawing(self):
        self.canvas.cancel_polygon_drawing()
        self.draw_obstacle_btn.setEnabled(True)
        self.place_circle_obstacle_btn.setEnabled(True)
        self.finish_obstacle_btn.setEnabled(False)
        self.undo_obstacle_vertex_btn.setEnabled(False)
        self.cancel_obstacle_btn.setEnabled(False)
        self.obstacle_draw_hint.setText("已取消绘制。滚轮缩放，按住鼠标左键拖动；点击障碍物内部可选中并删除。")
        if self.engine:
            self.update_view()

    def on_polygon_draft_changed(self):
        drawing = bool(self.canvas._drawing_polygon)
        self.undo_obstacle_vertex_btn.setEnabled(drawing and bool(self.canvas._draft_points))
        if self.engine:
            self.canvas.draw_scene(self.engine)

    def add_manual_obstacle(self, vertices):
        if not self.engine:
            return
        valid, reason = validate_polygon(vertices)
        if not valid:
            QMessageBox.warning(self, "多边形无效", reason)
            self.draw_obstacle_btn.setEnabled(True)
            self.finish_obstacle_btn.setEnabled(False)
            self.undo_obstacle_vertex_btn.setEnabled(False)
            self.cancel_obstacle_btn.setEnabled(False)
            self.obstacle_draw_hint.setText("多边形无效，请重新绘制。")
            return
        clean_vertices = [(float(x), float(y)) for x, y in vertices]
        if len(clean_vertices) > 3 and math.dist(clean_vertices[0], clean_vertices[-1]) < 1e-6:
            clean_vertices.pop()
        existing_ids = {o.obstacle_id for o in self.engine.obstacles}
        while True:
            self._manual_obstacle_counter += 1
            obstacle_id = f"USR-OBS-{self._manual_obstacle_counter:03d}"
            if obstacle_id not in existing_ids:
                break
        obstacle = StaticObstacle(obstacle_id, clean_vertices)
        self.engine.obstacles.append(obstacle)
        self._manual_obstacles.append(obstacle.copy())
        self._manual_obstacle_scene_key = self._current_scene_key()
        # The engine and scenario share this obstacle list; keep the linkage explicit
        # so planner and scenario export both see the manually placed obstruction.
        self.engine.scenario.obstacles = self.engine.obstacles
        self.canvas.set_selected_obstacle(obstacle_id)
        self.draw_obstacle_btn.setEnabled(True)
        self.place_circle_obstacle_btn.setEnabled(True)
        self.finish_obstacle_btn.setEnabled(False)
        self.undo_obstacle_vertex_btn.setEnabled(False)
        self.cancel_obstacle_btn.setEnabled(False)
        self.delete_obstacle_btn.setEnabled(True)
        self.obstacle_draw_hint.setText(f"已放置 {obstacle_id}（{len(clean_vertices)} 个顶点）；点击多边形可选中，随后可删除。")
        self.log_text.append(f"已放置手动障碍物 {obstacle_id}，顶点数={len(clean_vertices)}；该障碍物已加入 MPC 预测避障与碰撞检测。")
        self.update_view()

    def on_obstacle_selected(self, obstacle_id: str):
        self.delete_obstacle_btn.setEnabled(True)
        self.obstacle_draw_hint.setText(f"当前选中障碍物：{obstacle_id}。点击“删除选中障碍物”可移除。")

    def delete_selected_obstacle(self):
        if not self.engine or not self.canvas.selected_obstacle_id:
            QMessageBox.information(self, "未选择障碍物", "请先在地图上单击障碍物内部进行选中。")
            return
        obstacle_id = self.canvas.selected_obstacle_id
        before = len(self.engine.obstacles)
        self.engine.obstacles[:] = [o for o in self.engine.obstacles if o.obstacle_id != obstacle_id]
        self._manual_obstacles = [o for o in self._manual_obstacles if o.obstacle_id != obstacle_id]
        self.engine.scenario.obstacles = self.engine.obstacles
        if len(self.engine.obstacles) == before:
            self.log_text.append(f"未找到障碍物 {obstacle_id}，可能已被移除。")
        else:
            self.log_text.append(f"已删除障碍物 {obstacle_id}。")
        self.canvas.set_selected_obstacle(None)
        self.delete_obstacle_btn.setEnabled(False)
        self.obstacle_draw_hint.setText("障碍物已删除。点击其他障碍物可重新选中。")
        self.update_view()

    def clear_manual_obstacles(self):
        if not self.engine:
            return
        before = len(self.engine.obstacles)
        self.engine.obstacles[:] = [o for o in self.engine.obstacles if not o.obstacle_id.startswith("USR-")]
        removed = before - len(self.engine.obstacles)
        self._manual_obstacles = []
        self.engine.scenario.obstacles = self.engine.obstacles
        if self.canvas.selected_obstacle_id and self.canvas.selected_obstacle_id.startswith("USR-"):
            self.canvas.set_selected_obstacle(None)
        self.delete_obstacle_btn.setEnabled(bool(self.canvas.selected_obstacle_id))
        self.log_text.append(f"已清空手动放置障碍物：{removed} 个；示例障碍物保留。")
        self.obstacle_draw_hint.setText(f"已清空 {removed} 个手动障碍物；示例障碍物保留。")
        self.update_view()

    def reset_map_view(self):
        if not hasattr(self, "canvas"):
            return
        self.canvas.reset_view()
        if self.engine:
            self.canvas.draw_scene(self.engine)

    def update_view(self):
        if not self.engine:
            return
        e = self.engine
        self.canvas.draw_scene(e)
        risk = max(e.last_snapshot.assessments, key=lambda a: a.risk_score) if e.last_snapshot.assessments else None
        system_risk = risk_label(risk.risk_level) if risk else "无危险"
        odd_text = "ODD 降级" if e.odd_status.degraded else "ODD 正常"
        state_text = "紧迫局面" if e.last_snapshot.emergency else "风险监测" if e.last_snapshot.max_risk_score > 0.1 else "正常航行"
        if e.metrics.sda_violation_count:
            failed_targets = ",".join(e.metrics.sda_violation_targets)
            state_text = f"SDA已侵入：避碰失败（{failed_targets}）"
        self.status_label.setText(f"系统状态：{state_text}  |  {odd_text}")
        own_vx = e.own.u * math.sin(e.own.psi) + e.own.v * math.cos(e.own.psi) + e.environment.current_x_mps
        own_vy = e.own.u * math.cos(e.own.psi) - e.own.v * math.sin(e.own.psi) + e.environment.current_y_mps
        own_sog_mps = math.hypot(own_vx, own_vy)
        own_sog_kn = own_sog_mps / KNOT_TO_MPS
        own_cog_deg = math.degrees(math.atan2(own_vx, own_vy)) % 360.0
        own_heading_deg = math.degrees(e.own.psi) % 360.0
        self.own_status_label.setText(
            f"本船状态（OV）\n"
            f"船长：{e.config.length_m:.0f} m   |   对地航速 SOG：{own_sog_kn:.2f} kn ({own_sog_mps:.2f} m/s)\n"
            f"任务目标航速：{e.cruise_speed_mps / KNOT_TO_MPS:.2f} kn ({e.cruise_speed_mps:.2f} m/s)   |   航速误差：{e.cruise_speed_error_mps:.2f} m/s\n"
            f"对地航向 COG：{own_cog_deg:06.2f}°   |   船首向 Heading：{own_heading_deg:06.2f}°\n"
            f"舵角：{math.degrees(e.own.delta):+.2f}°   |   艏摇角速度：{math.degrees(e.own.r):+.3f}°/s\n"
            f"位置：东向 {e.own.x:.0f} m，北向 {e.own.y:.0f} m"
        )
        plan_message = e.last_plan.message if e.last_plan else "尚未生成规划结果"
        ppo_state = "PPO 已加载" if e.policy.available else "规则策略（未加载 PPO）"
        odd_reasons = "；".join(e.odd_status.reasons) if e.odd_status.reasons else "无"
        predicted_clearance_text = (f"{e.last_plan.minimum_clearance_m:.1f} m"
                                    if e.last_plan is not None else "尚未生成")
        target_course_text = (f"{math.degrees(e.planner.last_course_cmd_rad) % 360.0:.1f}°"
                              if e.planner.last_course_cmd_rad is not None else "尚未生成")
        metric_min = (f"{e.metrics.min_separation_m:.1f} m"
                      if math.isfinite(e.metrics.min_separation_m) else "暂无")
        route_cross_track = e.route_cross_track_distance_m()
        route_tolerance = e.route_recovery_tolerance_m
        sda_margin_text = (f"{e.metrics.minimum_domain_clearance_m:.1f} m"
                           if math.isfinite(e.metrics.minimum_domain_clearance_m) else "暂无")
        maneuver_start_text = (f"最近避碰机动启动：{e.last_maneuver_start_time_s:.1f} s"
                               if e.last_maneuver_start_time_s is not None
                               else "最近避碰机动启动：尚未触发")
        self.metrics_label.setText(
            f"仿真时间：{e.metrics.elapsed_s:.1f} s\n"
            f"重点目标：{risk.target_id if risk else '无'} | 最高风险：{system_risk}（{e.last_snapshot.max_risk_score:.2f}）\n"
            f"本回合实测最小船间距离：{metric_min}\n"
            f"SDA 判定：{'失败' if e.metrics.sda_violation_count else '未侵入'} | 侵入目标：{', '.join(e.metrics.sda_violation_targets) or '无'} | 最小 SDA 裕度：{sda_margin_text}\n"
            f"船舶碰撞次数：{e.metrics.collision_count} | 障碍物接触次数：{e.metrics.obstacle_collision_count}\n"
            f"局部规划：{plan_message}\n"
            f"策略模式：{ppo_state} | 决策来源：{e.last_policy_source}\n"
            f"提议/执行动作：{action_label(ACTION_NAMES[e.last_action])} / {action_label(ACTION_NAMES[e.last_executed_action])} | 阶段：{e.phase}\n"
            f"PPO 战术权限：{'开放' if e.ppo_action_authorized else '受规则/MPC接管'} | 原因：{e.policy_gate_reason}\n"
            f"{maneuver_start_text}\n"
            f"可行候选：{sum(c.feasible for c in e.candidates)}/{len(e.candidates)} | 安全干预：{'是' if e.last_intervention else '否'}\n"
            f"航速控制：{'锁定任务目标航速' if e.cruise_speed_locked else '允许战术调速'} | 回航航速误差容差：{e.route_recovery_speed_tolerance_mps:.2f} m/s\n"
            f"干预原因：{'；'.join(e.intervention_reasons) or '无'}\n"
            f"期望航向指令：{target_course_text} | 预测最小安全裕度：{predicted_clearance_text}\n"
            f"航线横向偏差：{route_cross_track:.1f} m / 回归阈值：{route_tolerance:.1f} m（船长的 0.1 倍）\n"
            f"回航完成航向误差：{e.route_heading_error_deg:.1f}° / 容差上限：{math.degrees(e.route_recovery_heading_tolerance_rad):.1f}°（仅用于回航完成判定，非最大转向角）；稳定计数：{e._route_recovery_stable_steps}/{e.route_recovery_stable_required}\n"
            f"航行阶段：{e.navigation_stage} | 已回归航线：{'是' if e.metrics.route_returned else '否'}\n"
            f"ODD 状态：{odd_reasons}"
        )

        self._filling_targets = True
        self.targets_table.blockSignals(True)
        self.targets_table.setRowCount(len(e.targets))
        risk_by_id = {a.target_id: a for a in e.last_snapshot.assessments}
        for row, target in enumerate(e.targets):
            assessment = risk_by_id.get(target.target_id)
            target_vx = target.state.u * math.sin(target.state.psi) + e.environment.current_x_mps
            target_vy = target.state.u * math.cos(target.state.psi) + e.environment.current_y_mps
            target_sog_kn = math.hypot(target_vx, target_vy) / KNOT_TO_MPS
            target_cog_deg = math.degrees(math.atan2(target_vx, target_vy)) % 360.0
            target_heading_deg = math.degrees(target.state.psi) % 360.0
            values = [
                target.target_id,
                f"{target.length_m:.1f}",
                f"{target_sog_kn:.2f}",
                f"{target_cog_deg:.2f}",
                f"{target_heading_deg:.2f}",
                risk_label(assessment.risk_level) if assessment else "无危险",
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if col == 1:
                    item.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsEditable)
                else:
                    item.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
                self.targets_table.setItem(row, col, item)
        self.targets_table.blockSignals(False)
        self._filling_targets = False

        self.risk_table.setRowCount(len(e.last_snapshot.assessments))
        source_meeting_labels = e.scenario.metadata.get("meeting_situations", [])
        for row, a in enumerate(e.last_snapshot.assessments):
            target_index = int(a.target_id.replace("TV", "")) - 1 if a.target_id.startswith("TV") else -1
            source_label = (source_meeting_labels[target_index]
                            if 0 <= target_index < len(source_meeting_labels) else "未提供")
            encounter_text = f"案例：{source_label or '未提供'}\n识别：{encounter_label(a.encounter)}"
            sda_position = {"bow": "目标船首向通过", "stern": "目标船尾向通过",
                            "generic": "通用", "conservative_max": "首/尾取大（保守）"}.get(
                                a.sda_pass_position, a.sda_pass_position)
            sda_source = ("无对应表项" if a.sda_m <= 0 else
                          "插值/外推估算" if a.sda_estimated else "附录B表值")
            values = [a.target_id, encounter_text,
                      "让路船" if a.own_give_way else "直航船/保持监测",
                      f"{a.range_m:.0f}", f"{a.dcpa_m:.0f}", f"{a.tcpa_s:.1f}",
                      ("—" if a.sda_m <= 0 else
                       f"~{a.sda_m:.0f}" if a.sda_estimated else f"{a.sda_m:.0f}"),
                      (f"{a.dcrc_m:.0f}" if math.isfinite(a.dcrc_m) else "—"),
                      (f"{a.dcqs_m:.0f}" if math.isfinite(a.dcqs_m) else "—"),
                      ({"port": "左舷", "starboard": "右舷"}.get(a.pass_side, "未指定")),
                      f"{sda_position} / {sda_source}"]
            for col, value in enumerate(values):
                self.risk_table.setItem(row, col, QTableWidgetItem(str(value)))
        self.risk_table.resizeRowsToContents()

        if e.logs and (len(e.logs) == 1 or len(e.logs) % 10 == 0):
            rec = e.logs[-1]
            self.log_text.append(
                f"时间={rec['time_s']:.0f}s | 策略={rec['intent']} | 风险={risk_label(rec['risk'])} | "
                f"本船航速={rec['speed_mps'] / KNOT_TO_MPS:.2f} kn，船首向={rec['heading_deg']:.1f}° | 预测安全={rec['plan_safe']}"
            )
        if e.metrics.destination_reached:
            self.timer.stop()
            if e.metrics.sda_violation_count:
                self.status_label.setText(
                    "系统状态：已到达终点，但实际会遇距离低于 SDA，避碰失败"
                )
            else:
                self.status_label.setText("系统状态：到达航线终点，未侵入 SDA")
            e.running = False

    def on_target_cell_changed(self, row: int, column: int):
        if self._filling_targets or column != 1 or not self.engine or row >= len(self.engine.targets):
            return
        item = self.targets_table.item(row, column)
        if item is None:
            return
        try:
            length = float(item.text())
            if not 50.0 <= length <= 500.0:
                raise ValueError("目标船长必须在 50～500 m 之间。")
            target = self.engine.targets[row]
            target.length_m = length
            target.beam_m = max(8.0, 0.16 * length)
            self.engine.last_snapshot = self.engine.risk_engine.assess(self.engine.own, self.engine.config, self.engine.targets)
            self.log_text.append(f"已将 {target.target_id} 船长更新为 {length:.0f} m。")
            self.update_view()
        except Exception as exc:
            QMessageBox.warning(self, "目标船长设置错误", str(exc))
            self.update_view()

    def export_log(self):
        if not self.engine:
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出仿真日志", "simulation_log.json", "JSON 文件 (*.json)")
        if path:
            self.engine.export_log(path)
            QMessageBox.information(self, "导出完成", f"仿真日志已保存至：\n{path}")

    def load_scenario_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择 JSON 场景", str(self.project_root / "configs"), "JSON 场景 (*.json)")
        if not path:
            return
        try:
            scenario = load_scenario_json(path)
            checkpoint = str(self.ppo_checkpoint) if self.ppo_checkpoint else None
            self.timer.stop()
            self.cancel_polygon_drawing()
            self._manual_obstacles = []
            self._manual_obstacle_scene_key = ("json", str(Path(path).resolve()))
            self.engine = SimulationEngine(scenario, checkpoint)
            self.canvas.reset_view()
            self.canvas.set_selected_obstacle(None)
            self.log_text.clear()
            self.log_text.append(f"已加载自定义 JSON 场景：{scenario.name}，目标船数量：{len(scenario.targets)}")
            self.update_view()
        except Exception as exc:
            QMessageBox.critical(self, "场景加载失败", str(exc))

    def load_ppo(self):
        start_dir = str(self.ppo_checkpoint.parent if self.ppo_checkpoint else self.project_root / "models")
        path, _ = QFileDialog.getOpenFileName(self, "选择 PPO 模型文件", start_dir, "PyTorch 模型 (*.pt *.pth)")
        if path and self.engine:
            from usv_sim.ppo_agent import PPOPolicy
            candidate = PPOPolicy(path)
            if candidate.available:
                self.ppo_checkpoint = Path(path)
                self.engine.policy = candidate
                self.ppo_model_status.setText(f"当前模型：PPO 已加载 — {self.ppo_checkpoint.name}")
                self.unload_ppo_button.setEnabled(True)
                self.log_text.append(f"已加载 PPO 模型：{Path(path).name}")
                self.eval_status.setText(f"当前评估权重：{self.ppo_checkpoint}")
                self.eval_start_btn.setEnabled(True)
                QMessageBox.information(self, "PPO 模型", "模型加载成功，可用于当前仿真和批量评估。")
            else:
                QMessageBox.warning(self, "PPO 模型", f"{candidate.load_error}\n旧五动作模型需按新候选框架重新训练。\n当前策略和评估权重未更改。")
            self.update_view()

    def unload_ppo(self):
        """Unload the in-memory PPO policy and keep the checkpoint file intact."""
        if self.evaluation_worker and self.evaluation_worker.isRunning():
            QMessageBox.warning(self, "正在批量评估", "请先停止正在运行的批量评估，再取消加载 PPO 模型。")
            return
        from usv_sim.ppo_agent import PPOPolicy
        self.ppo_checkpoint = None
        if self.engine:
            self.engine.policy = PPOPolicy(None)
        self.ppo_model_status.setText("当前模型：规则策略（PPO 已取消加载）")
        self.unload_ppo_button.setEnabled(False)
        self.eval_start_btn.setEnabled(False)
        self.eval_status.setText("PPO 权重已取消加载；如需评估，请重新加载训练权重。")
        self.train_status.setText("PPO 模型已从当前仿真卸载，当前使用规则策略。磁盘上的权重文件未删除。")
        self.log_text.append("已取消加载 PPO 模型；当前仿真切换为规则策略，原权重文件保留在磁盘。")
        self.update_view()

    def clear_tracks(self):
        if not self.engine:
            return
        self.engine.clear_tracks()
        self.canvas.draw_scene(self.engine)
        self.log_text.append("已清除本船和所有目标船的历史轨迹；从当前时刻开始重新记录。")

    def start_evaluation(self):
        if self.evaluation_worker and self.evaluation_worker.isRunning():
            return
        if self.ppo_checkpoint is None:
            QMessageBox.warning(self, "缺少 PPO 权重", "当前没有加载 PPO 模型，请先点击“加载 PPO 模型”。")
            return
        checkpoint = Path(self.ppo_checkpoint).expanduser()
        if not checkpoint.is_file():
            QMessageBox.warning(self, "缺少 PPO 权重", "请先点击“加载 PPO 模型”，选择训练得到的 .pt/.pth 权重文件。")
            return
        if not self.engine or not self.engine.policy.available or Path(self.engine.policy.checkpoint or "") != checkpoint:
            from usv_sim.ppo_agent import PPOPolicy
            policy = PPOPolicy(str(checkpoint))
            if not policy.available:
                QMessageBox.warning(self, "PPO 权重无效", "所选权重无法加载，不能开始评估。")
                return
            if self.engine:
                self.engine.policy = policy
        group = self.eval_group_combo.currentData()
        group_name = {"1T": "1T", "2T": "2T", "3T": "3T", "ALL": "全部案例"}.get(group, str(group))
        default_name = f"ppo_evaluation_{group}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        results_dir = self.project_root / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        output_path, _ = QFileDialog.getSaveFileName(
            self, "保存 PPO 批量评估结果", str(results_dir / default_name), "CSV 文件 (*.csv)"
        )
        if not output_path:
            return
        if not output_path.lower().endswith(".csv"):
            output_path += ".csv"
        self.last_evaluation_csv = Path(output_path)
        script = self.project_root / "evaluate_ppo.py"
        cmd = [sys.executable, "-u", str(script), "--checkpoint", str(checkpoint),
               "--group", str(group), "--dataset-root", str(self.dataset_root),
               "--out", output_path, "--max-steps", str(self.eval_max_steps_spin.value()),
               "--own-length", str(self.own_length_spin.value()),
               "--target-length", str(self.target_length_spin.value())]
        self.evaluation_worker = TrainingWorker(cmd, self.project_root, self, task_name="评估")
        self.evaluation_worker.output.connect(self.on_evaluation_output)
        self.evaluation_worker.result.connect(self.on_evaluation_result)
        self.eval_progress.setValue(0)
        self.eval_start_btn.setEnabled(False)
        self.eval_stop_btn.setEnabled(True)
        self.eval_status.setText(f"正在评估 {group_name}，每个案例最多 {self.eval_max_steps_spin.value()} 步；结果将实时写入 CSV。")
        self.log_text.append(f"开始 PPO 批量评估：组别={group_name}，权重={checkpoint.name}。")
        self.evaluation_worker.start()

    def stop_evaluation(self):
        if self.evaluation_worker and self.evaluation_worker.isRunning():
            self.evaluation_worker.stop()
            self.eval_stop_btn.setEnabled(False)
            self.eval_status.setText("正在停止评估进程……")

    def on_evaluation_output(self, line: str):
        match = re.search(r"EVAL_PROGRESS\s+(\d+)\s+(\d+)", line)
        if match:
            done, total = int(match.group(1)), max(1, int(match.group(2)))
            self.eval_progress.setValue(min(100, int(done * 100 / total)))
            self.eval_status.setText(f"批量评估进度：{done}/{total}（{done / total:.1%}）")
        elif line.startswith("EVAL_CASE"):
            self.log_text.append(line)
        elif line.startswith("EVAL_CSV "):
            self.last_evaluation_csv = Path(line.split(" ", 1)[1].strip())
            self.eval_status.setText(f"CSV 正在保存：{self.last_evaluation_csv}")
        elif line.startswith("EVAL_COMPLETE"):
            self.eval_status.setText(line.replace("EVAL_COMPLETE", "评估完成"))
            self.log_text.append(line)
        elif line.startswith("ERROR"):
            self.eval_status.setText(line)
            self.log_text.append(line)
        elif line.strip():
            self.log_text.append(line)

    def on_evaluation_result(self, success: bool, message: str):
        self.eval_start_btn.setEnabled(True)
        self.eval_stop_btn.setEnabled(False)
        if success:
            self.eval_progress.setValue(100)
            if self.last_evaluation_csv and self.last_evaluation_csv.is_file():
                self.eval_status.setText(f"评估完成，CSV 已保存：{self.last_evaluation_csv}")
                QMessageBox.information(self, "批量评估完成", f"{message}\n\n结果文件：\n{self.last_evaluation_csv}")
            else:
                self.eval_status.setText(message)
                QMessageBox.information(self, "批量评估完成", message)
        else:
            partial_note = ""
            if self.last_evaluation_csv and self.last_evaluation_csv.is_file():
                partial_note = f"\n\n已完成案例的部分结果已保存在：\n{self.last_evaluation_csv}"
            self.eval_status.setText(message + (f" 部分结果：{self.last_evaluation_csv}" if partial_note else ""))
            self.log_text.append(message + partial_note)
            QMessageBox.warning(self, "批量评估未完成", message + partial_note)

    def toggle_training(self):
        if self.training_worker and self.training_worker.isRunning():
            self.training_worker.stop()
            self.train_button.setEnabled(False)
            self.train_status.setText("正在停止训练进程……")
            return
        steps = self.train_steps_spin.value()
        script = self.project_root / "train_ppo.py"
        checkpoint = self.project_root / "models" / "ppo_policy.pt"
        device = self.training_device_combo.currentData()
        if device == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    QMessageBox.warning(
                        self, "未检测到 CUDA GPU",
                        "当前 Python 环境中的 PyTorch 不支持 CUDA。\n\n"
                        "请先在当前 Anaconda 环境执行 install_cuda_torch.bat，并确认 NVIDIA 驱动已安装。\n"
                        "为避免误以为正在使用 GPU，本次不会自动回退到 CPU。"
                    )
                    return
            except Exception as exc:
                QMessageBox.critical(self, "GPU 检测失败", str(exc))
                return
        cmd = [sys.executable, "-u", str(script), "--steps", str(steps),
               "--device", str(device), "--episode-steps", "120", "--max-episode-steps", "1500",
               "--dataset-root", str(self.dataset_root), "--out", str(checkpoint)]
        self.training_worker = TrainingWorker(cmd, self.project_root, self)
        self.training_worker.output.connect(self.on_training_output)
        self.training_worker.result.connect(self.on_training_result)
        self.train_progress.setValue(0)
        self.train_button.setText("停止训练")
        self.train_button.setEnabled(True)
        self.train_status.setText("训练已启动：按几何分组划分数据，仅训练集参与学习；2秒控制／10秒决策。")
        self.log_text.append(f"已启动 PPO 训练：{steps} 个交互步，案例数量={len(self.case_specs)}。")
        self.training_worker.start()

    def on_training_output(self, line: str):
        match = re.search(r"PROGRESS\s+(\d+)\s+(\d+)", line)
        if match:
            done, total = int(match.group(1)), max(1, int(match.group(2)))
            self.train_progress.setValue(min(100, int(100 * done / total)))
            display = f"PPO 训练进度：{done:,} / {total:,} 环境交互步"
            self.train_status.setText(display)
        elif line.startswith("WARMUP_DATA"):
            display = f"规则预训练：已处理 {line.split()[-1]} 个案例"
            self.train_status.setText(display)
        elif line.startswith("WARMUP_EPOCH"):
            m = re.search(r"WARMUP_EPOCH\s+(\d+/\d+)\s+loss=([0-9.eE+-]+)", line)
            display = f"规则预训练轮次：{m.group(1)}，损失值：{m.group(2)}" if m else "正在进行规则预训练……"
            self.train_status.setText(display)
        elif line.startswith("DEVICE cuda"):
            display = "训练设备：CUDA GPU — " + line.split("|", 2)[1].strip()
            self.train_status.setText(display)
        elif line.startswith("DEVICE cpu"):
            display = "训练设备：CPU（未启用 GPU）"
            self.train_status.setText(display)
        elif line.startswith("DATASET"):
            m = re.search(r"total=(\d+)\s+1T=(\d+)\s+2T=(\d+)\s+3T=(\d+)", line)
            display = (f"训练案例：共 {m.group(1)} 个（1T：{m.group(2)}，2T：{m.group(3)}，3T：{m.group(4)}）"
                       if m else "正在读取训练案例……")
            self.train_status.setText(display)
        elif line.startswith("TRAINING_START"):
            display = "PPO 仿真训练开始。"
            self.train_status.setText(display)
        elif line.startswith("TRAINING_COMPLETE"):
            display = "PPO 训练完成，模型已保存。"
            self.train_status.setText(display)
        elif line.startswith("NOTE source CSV"):
            display = "说明：上传 CSV 是初始会遇状态，不是连续时间轨迹。"
        else:
            display = line
        self.log_text.append(display)

    def on_training_result(self, success: bool, message: str):
        self.train_button.setText("开始 PPO 训练")
        self.train_button.setEnabled(True)
        self.train_status.setText(message)
        if success:
            self.train_progress.setValue(100)
            if self.engine:
                from usv_sim.ppo_agent import PPOPolicy
                checkpoint = self.project_root / "models" / "ppo_policy.pt"
                self.ppo_checkpoint = checkpoint
                self.engine.policy = PPOPolicy(str(checkpoint))
                self.ppo_model_status.setText(f"当前模型：PPO 已加载 — {checkpoint.name}" if self.engine.policy.available
                                              else f"模型加载被拒绝：{self.engine.policy.load_error}")
                self.unload_ppo_button.setEnabled(self.engine.policy.available)
                self.eval_start_btn.setEnabled(self.engine.policy.available)
            self.log_text.append("PPO 权重已生成；加载状态见模型栏。")
            self.update_view()
        else:
            self.log_text.append(message)

    def show_about(self):
        QMessageBox.information(self, "关于仿真系统",
            "本软件为研究型软件在环仿真原型。\n\n"
            "CSV 是会遇初态，并非专家轨迹。案例按几何分组划分训练／验证／测试集，只有训练集用于 PPO。75 个候选采用 5° 间隔覆盖左右转向与三种速度模式；默认无教师预热。\n\n"
            "本船和目标船长度可分别配置为 50～500 m。上传案例未包含船长和船宽，因此需由界面配置或训练时随机化。\n\n"
            "当前船舶模型为包含舵机、艏摇和航速滞后的简化 3 自由度模型，尚未使用具体船舶试验数据标定为 MMG 模型。本软件不属于经过认证的航行或避碰系统。")

    def closeEvent(self, event):
        self.timer.stop()
        if self.training_worker and self.training_worker.isRunning():
            self.training_worker.stop()
            self.training_worker.wait(3000)
        if self.evaluation_worker and self.evaluation_worker.isRunning():
            self.evaluation_worker.stop()
            self.evaluation_worker.wait(3000)
        super().closeEvent(event)
