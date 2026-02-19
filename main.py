"""Windows 离线视频管理小工具主程序。

该程序使用 PyQt6 构建桌面界面，支持：
1. 首级目录视频扫描（按修改时间倒序）
2. 评分（91~100）与标签多选
3. 异步生成并缓存截图预览
4. 重命名（固定模板）/删除/播放/上下切换
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import string
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from PyQt6.QtCore import QObject, QRunnable, Qt, QThreadPool, pyqtSignal
from PyQt6.QtGui import QKeySequence, QPixmap, QShortcut
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

# ==================== 配置区（可按需修改） ====================
DEFAULT_DIR = r"D:\Videos\Inbox"
POTPLAYER_EXE = r"C:\Program Files\DAUM\PotPlayer\PotPlayerMini64.exe"
FFMPEG_EXE = "ffmpeg"  # 可改为绝对路径，例如 r"C:\ffmpeg\bin\ffmpeg.exe"
FFPROBE_EXE = "ffprobe"  # 可改为绝对路径，例如 r"C:\ffmpeg\bin\ffprobe.exe"

VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm"}
SCREENSHOT_COUNT = 12
TAGS = [
    "占位标签1",
    "占位标签2",
    "占位标签3",
    "占位标签4",
    "占位标签5",
    "占位标签6",
    "占位标签7",
    "占位标签8",
    "占位标签9",
    "占位标签10",
]
MULTI_TAG_MODE = True

STATE_FILE = "state.json"
CACHE_DIR = ".cache_screens"
GRID_COLUMNS = 4
CODE_LENGTH = 6
# ============================================================


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


@dataclass
class VideoItem:
    """视频文件结构体，承载列表展示所需信息。"""

    path: Path
    mtime: float
    size: int


class StateManager:
    """管理评分、标签、重命名信息的本地 JSON 状态。"""

    def __init__(self, state_path: Path) -> None:
        self.state_path = state_path
        self.state: dict[str, dict] = {}
        self.load()

    def load(self) -> None:
        """从磁盘加载状态文件；不存在时初始化为空。"""

        if not self.state_path.exists():
            self.state = {}
            return
        try:
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            logging.warning("读取 state.json 失败，将使用空状态：%s", exc)
            self.state = {}

    def save(self) -> None:
        """将内存中的状态写回磁盘。"""

        self.state_path.write_text(
            json.dumps(self.state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def get(self, path: Path) -> dict:
        """读取某个视频对应状态，缺省返回空字典。"""

        return self.state.get(str(path.resolve()), {})

    def update(self, path: Path, payload: dict) -> None:
        """更新某个视频的状态并立即持久化。"""

        self.state[str(path.resolve())] = payload
        self.save()

    def remove(self, path: Path) -> None:
        """移除某个视频状态（例如删除文件后）。"""

        self.state.pop(str(path.resolve()), None)
        self.save()

    def move_key(self, old_path: Path, new_path: Path) -> None:
        """重命名后同步迁移状态 key，避免丢失历史记录。"""

        old_key = str(old_path.resolve())
        new_key = str(new_path.resolve())
        if old_key in self.state:
            self.state[new_key] = self.state.pop(old_key)
            self.save()


class WorkerSignals(QObject):
    """截图线程信号：完成、失败、状态提示。"""

    finished = pyqtSignal(str, list)
    error = pyqtSignal(str)
    status = pyqtSignal(str)


class ScreenshotWorker(QRunnable):
    """后台线程：生成或读取视频截图缓存，避免阻塞 UI。"""

    def __init__(self, video_path: Path, cache_root: Path, count: int) -> None:
        super().__init__()
        self.video_path = video_path
        self.cache_root = cache_root
        self.count = count
        self.signals = WorkerSignals()

    def run(self) -> None:
        """执行截图流程：计算 key、探测时长、按时间采样、调用 ffmpeg。"""

        try:
            key = build_cache_key(self.video_path)
            cache_dir = self.cache_root / key
            cache_dir.mkdir(parents=True, exist_ok=True)

            cached = sorted(cache_dir.glob("*.jpg"))
            if len(cached) == self.count:
                self.signals.finished.emit(str(self.video_path), [str(p) for p in cached])
                return

            # 缓存数量不完整时先清理，保证显示结果一致。
            for file in cache_dir.glob("*.jpg"):
                file.unlink(missing_ok=True)

            duration = probe_duration(self.video_path)
            if duration <= 0:
                raise RuntimeError("无法获取视频时长，请检查 ffprobe 是否可用")

            points = sample_points(duration, self.count)
            output_files: list[str] = []
            for idx, sec in enumerate(points, start=1):
                out_file = cache_dir / f"{idx:02d}.jpg"
                self.signals.status.emit(f"正在生成截图 {idx}/{self.count} ...")
                cmd = [
                    FFMPEG_EXE,
                    "-y",
                    "-ss",
                    f"{sec:.3f}",
                    "-i",
                    str(self.video_path),
                    "-frames:v",
                    "1",
                    "-q:v",
                    "2",
                    str(out_file),
                ]
                run_subprocess(cmd)
                output_files.append(str(out_file))

            self.signals.finished.emit(str(self.video_path), output_files)
        except Exception as exc:  # noqa: BLE001
            self.signals.error.emit(str(exc))


def run_subprocess(cmd: list[str]) -> None:
    """执行子进程命令，失败时抛出带 stderr 的异常。"""

    completed = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "子进程执行失败")


def probe_duration(video_path: Path) -> float:
    """使用 ffprobe 获取视频总时长（秒）。"""

    cmd = [
        FFPROBE_EXE,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    completed = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "ffprobe 执行失败")
    return float(completed.stdout.strip())


def sample_points(duration: float, count: int) -> list[float]:
    """按 5%~95% 区间均匀采样时间点，避免片头片尾无效帧。"""

    start = duration * 0.05
    end = duration * 0.95
    if count <= 1:
        return [(start + end) / 2]
    step = (end - start) / (count - 1)
    return [start + i * step for i in range(count)]


def build_cache_key(video_path: Path) -> str:
    """根据“绝对路径 + mtime_ns”构建稳定缓存 key。"""

    stat = video_path.stat()
    raw = f"{video_path.resolve()}|{stat.st_mtime_ns}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()  # noqa: S324


def sanitize_for_windows(name: str) -> str:
    """清理 Windows 文件名非法字符，避免重命名失败。"""

    cleaned = re.sub(r'[\\/:*?"<>|]', "_", name.strip())
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned


def random_code(length: int = CODE_LENGTH) -> str:
    """生成 6 位随机大写字母数字 code。"""

    alphabet = string.ascii_uppercase + string.digits
    return "".join(random.choices(alphabet, k=length))


class MainWindow(QMainWindow):
    """主窗口：负责 UI 组装、事件绑定与业务调度。"""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("自用视频管理小软件")
        self.resize(1500, 900)

        self.current_dir = Path(DEFAULT_DIR)
        # all_video_items 保存完整扫描结果，filtered_video_items 保存过滤后的可见列表。
        self.all_video_items: list[VideoItem] = []
        self.filtered_video_items: list[VideoItem] = []
        self.current_index = -1
        self.current_score: int | None = None
        self.current_tags: list[str] = []

        self.cache_root = Path(CACHE_DIR)
        self.cache_root.mkdir(exist_ok=True)
        self.state_manager = StateManager(Path(STATE_FILE))
        self.thread_pool = QThreadPool.globalInstance()

        self.score_buttons: dict[int, QPushButton] = {}
        self.tag_buttons: dict[str, QPushButton] = {}

        self._build_ui()
        self._bind_shortcuts()
        self.scan_directory(self.current_dir)

    def _build_ui(self) -> None:
        """构建主界面布局：左列表 + 右信息/截图/操作区。"""

        container = QWidget()
        root_layout = QHBoxLayout(container)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root_layout.addWidget(splitter)

        # 左侧：视频列表
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        self.btn_choose_dir = QPushButton("选择文件夹")
        self.btn_choose_dir.clicked.connect(self.on_choose_dir)
        left_layout.addWidget(self.btn_choose_dir)

        # 左侧过滤输入框：可按“文件名/评分/标签”关键字过滤当前目录视频。
        self.edt_filter = QLineEdit()
        self.edt_filter.setPlaceholderText("过滤：输入文件名、评分或标签关键字")
        self.edt_filter.textChanged.connect(self.on_filter_changed)
        left_layout.addWidget(self.edt_filter)

        self.list_widget = QListWidget()
        self.list_widget.currentRowChanged.connect(self.on_row_changed)
        left_layout.addWidget(self.list_widget)

        splitter.addWidget(left_widget)

        # 右侧：详情、截图、操作
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        self.lbl_info = QLabel("当前视频信息")
        self.lbl_info.setWordWrap(True)
        right_layout.addWidget(self.lbl_info)

        self.lbl_status = QLabel("就绪")
        self.lbl_status.setStyleSheet("color: #666;")
        right_layout.addWidget(self.lbl_status)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.screenshot_container = QWidget()
        self.screenshot_grid = QGridLayout(self.screenshot_container)
        self.scroll_area.setWidget(self.screenshot_container)
        right_layout.addWidget(self.scroll_area, stretch=1)

        # 评分按钮
        score_layout = QHBoxLayout()
        score_layout.addWidget(QLabel("评分："))
        for score in range(91, 101):
            btn = QPushButton(str(score))
            btn.setCheckable(True)
            btn.clicked.connect(self._build_score_handler(score))
            score_layout.addWidget(btn)
            self.score_buttons[score] = btn
        right_layout.addLayout(score_layout)

        # 标签按钮
        tag_layout = QVBoxLayout()
        tag_layout.addWidget(QLabel("标签："))
        # 标签容器改为网格：根据宽度自适应换行，最多显示三行。
        self.tag_container = QWidget()
        self.tag_grid = QGridLayout(self.tag_container)
        self.tag_grid.setContentsMargins(0, 0, 0, 0)
        self.tag_grid.setHorizontalSpacing(8)
        self.tag_grid.setVerticalSpacing(6)
        for tag in TAGS:
            btn = QPushButton(tag)
            btn.setCheckable(True)
            btn.clicked.connect(self._build_tag_handler(tag))
            self.tag_buttons[tag] = btn
        tag_layout.addWidget(self.tag_container)
        right_layout.addLayout(tag_layout)

        # 操作按钮
        action_layout = QHBoxLayout()
        self.btn_play = QPushButton("播放")
        self.btn_play.clicked.connect(self.on_play)
        action_layout.addWidget(self.btn_play)

        self.btn_delete = QPushButton("删除")
        self.btn_delete.clicked.connect(self.on_delete)
        action_layout.addWidget(self.btn_delete)

        self.btn_prev = QPushButton("上一个")
        self.btn_prev.clicked.connect(self.on_prev)
        action_layout.addWidget(self.btn_prev)

        self.btn_confirm = QPushButton("确认")
        self.btn_confirm.clicked.connect(self.on_confirm)
        action_layout.addWidget(self.btn_confirm)

        self.btn_next = QPushButton("下一个")
        self.btn_next.clicked.connect(self.on_next)
        action_layout.addWidget(self.btn_next)

        right_layout.addLayout(action_layout)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(1, 3)

        self.setCentralWidget(container)
        # 初始化完成后执行一次标签排版，确保首次显示即按宽度换行。
        self.arrange_tag_buttons()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        """窗口尺寸变化时，实时调整标签按钮的自适应换行布局。"""

        super().resizeEvent(event)
        self.arrange_tag_buttons()

    def arrange_tag_buttons(self) -> None:
        """将标签按钮按可用宽度自动换行，最多三行。"""

        if not hasattr(self, "tag_grid"):
            return

        # 清空旧布局项（按钮对象保留并复用）。
        while self.tag_grid.count():
            item = self.tag_grid.takeAt(0)
            widget = item.widget()
            if widget:
                widget.setParent(None)

        available = max(self.tag_container.width(), 360)
        row = 0
        col = 0
        row_width = 0
        spacing = self.tag_grid.horizontalSpacing() or 8

        for tag in TAGS:
            btn = self.tag_buttons[tag]
            btn_w = btn.sizeHint().width()

            # 当前行放不下则换行；最多三行，超出后继续塞到第三行。
            if col > 0 and row_width + spacing + btn_w > available:
                if row < 2:
                    row += 1
                    col = 0
                    row_width = 0
                else:
                    col += 1
                    self.tag_grid.addWidget(btn, row, col)
                    continue

            self.tag_grid.addWidget(btn, row, col)
            row_width = btn_w if col == 0 else row_width + spacing + btn_w
            col += 1

    def _bind_shortcuts(self) -> None:
        """绑定常用快捷键，提高人工筛选效率。"""

        QShortcut(QKeySequence(Qt.Key.Key_Left), self, activated=self.on_prev)
        QShortcut(QKeySequence(Qt.Key.Key_Right), self, activated=self.on_next)
        QShortcut(QKeySequence(Qt.Key.Key_Return), self, activated=self.on_confirm)
        QShortcut(QKeySequence(Qt.Key.Key_Enter), self, activated=self.on_confirm)

        # 1~0 快速评分，对应 91~100。
        key_map = [
            (Qt.Key.Key_1, 91),
            (Qt.Key.Key_2, 92),
            (Qt.Key.Key_3, 93),
            (Qt.Key.Key_4, 94),
            (Qt.Key.Key_5, 95),
            (Qt.Key.Key_6, 96),
            (Qt.Key.Key_7, 97),
            (Qt.Key.Key_8, 98),
            (Qt.Key.Key_9, 99),
            (Qt.Key.Key_0, 100),
        ]
        for key, score in key_map:
            QShortcut(QKeySequence(key), self, activated=self._build_score_handler(score))

    def scan_directory(self, directory: Path) -> None:
        """扫描首级目录视频文件并刷新左侧列表。"""

        if not directory.exists() or not directory.is_dir():
            self.status_message(f"目录不存在：{directory}")
            return

        self.current_dir = directory
        self.all_video_items = []
        for item in directory.iterdir():
            if item.is_file() and item.suffix.lower() in VIDEO_EXTS:
                stat = item.stat()
                self.all_video_items.append(VideoItem(path=item, mtime=stat.st_mtime, size=stat.st_size))

                # 重新扫描时：若文件名符合命名模板，则自动回填评分/标签到 state。
                self.ingest_state_from_filename(item)

        self.all_video_items.sort(key=lambda x: x.mtime, reverse=True)
        self.apply_filter()

        if self.filtered_video_items:
            self.list_widget.setCurrentRow(0)
        else:
            self.current_index = -1
            self.lbl_info.setText("未找到视频文件")
            self.clear_screenshots()

    def ingest_state_from_filename(self, path: Path) -> None:
        """从文件名解析评分/标签并写入状态；不符合格式则跳过。"""

        # 匹配：{score}-{tags}-{code}{ext}，其中 score/tags 可为空。
        match = re.match(r"^(?P<score>\d*)-(?P<tags>.*)-(?P<code>[A-Z0-9]{6})$", path.stem)
        if not match:
            return

        score_raw = match.group("score")
        tags_raw = match.group("tags")
        code = match.group("code")

        # 分数仅接受 91~100；否则视为不合法格式，不导入。
        score: int | None = None
        if score_raw:
            try:
                parsed = int(score_raw)
                if 91 <= parsed <= 100:
                    score = parsed
                else:
                    return
            except ValueError:
                return

        tags = [tag for tag in tags_raw.split("_") if tag] if tags_raw else []
        self.state_manager.update(
            path,
            {
                "score": score,
                "tags": tags,
                "code": code,
                "renamed_to": str(path),
            },
        )

    def on_filter_changed(self, _text: str) -> None:
        """过滤输入变化时，实时刷新左侧可见视频列表。"""

        self.apply_filter()

    def apply_filter(self) -> None:
        """按关键字过滤视频：匹配文件名、评分文本、标签文本。"""

        keyword = self.edt_filter.text().strip().lower() if hasattr(self, "edt_filter") else ""
        if not keyword:
            self.filtered_video_items = list(self.all_video_items)
        else:
            filtered: list[VideoItem] = []
            for video in self.all_video_items:
                state = self.state_manager.get(video.path)
                score = state.get("score")
                tags = state.get("tags", [])
                haystack = " ".join(
                    [video.path.name.lower(), str(score or "").lower(), "_".join(tags).lower()]
                )
                if keyword in haystack:
                    filtered.append(video)
            self.filtered_video_items = filtered

        self.refresh_list_widget()
        if self.filtered_video_items:
            self.list_widget.setCurrentRow(0)
        else:
            self.current_index = -1
            self.lbl_info.setText("过滤后无匹配视频")
            self.clear_screenshots()

    def refresh_list_widget(self) -> None:
        """根据当前视频集合重绘列表项（含评分/标签前缀）。"""

        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        for video in self.filtered_video_items:
            state = self.state_manager.get(video.path)
            score = state.get("score")
            tags = state.get("tags", [])
            tag_text = "_".join(tags) if tags else ""
            prefix = f"[{score or ''}][{tag_text}] "
            mtime_text = datetime.fromtimestamp(video.mtime).strftime("%Y-%m-%d %H:%M:%S")
            size_mb = video.size / 1024 / 1024
            text = f"{prefix}{video.path.name} | {mtime_text} | {size_mb:.2f} MB"
            item = QListWidgetItem(text)
            self.list_widget.addItem(item)
        self.list_widget.blockSignals(False)

    def on_choose_dir(self) -> None:
        """打开目录选择框并触发重新扫描。"""

        selected = QFileDialog.getExistingDirectory(self, "选择视频目录", str(self.current_dir))
        if selected:
            self.scan_directory(Path(selected))

    def on_row_changed(self, row: int) -> None:
        """当用户切换列表选中项时，加载对应状态与截图。"""

        if row < 0 or row >= len(self.filtered_video_items):
            return

        self.save_current_state()
        self.current_index = row
        current = self.filtered_video_items[row]
        self.load_state_for_video(current.path)
        self.update_video_info(current)
        self.load_screenshots_async(current.path)

    def load_state_for_video(self, path: Path) -> None:
        """把 state.json 中保存的评分/标签恢复到按钮状态。"""

        state = self.state_manager.get(path)
        self.current_score = state.get("score")
        self.current_tags = list(state.get("tags", []))
        self.sync_score_buttons()
        self.sync_tag_buttons()

    def save_current_state(self) -> None:
        """切换视频前保存当前评分/标签，防止操作丢失。"""

        if self.current_index < 0 or self.current_index >= len(self.filtered_video_items):
            return
        path = self.filtered_video_items[self.current_index].path
        old = self.state_manager.get(path)
        payload = {
            "score": self.current_score,
            "tags": self.current_tags,
            "code": old.get("code"),
            "renamed_to": old.get("renamed_to"),
        }
        self.state_manager.update(path, payload)

    def update_video_info(self, video: VideoItem) -> None:
        """刷新右上角当前视频基础信息。"""

        mtime_text = datetime.fromtimestamp(video.mtime).strftime("%Y-%m-%d %H:%M:%S")
        size_mb = video.size / 1024 / 1024
        self.lbl_info.setText(
            f"路径：{video.path}\n"
            f"时间：{mtime_text}\n"
            f"大小：{size_mb:.2f} MB\n"
            f"评分：{self.current_score if self.current_score else ''}\n"
            f"标签：{'_'.join(self.current_tags)}"
        )

    def _build_score_handler(self, score: int) -> Callable[[], None]:
        """创建评分按钮点击处理函数，供循环绑定使用。"""

        def handler() -> None:
            self.current_score = score
            self.sync_score_buttons()
            self.refresh_current_info_only()

        return handler

    def _build_tag_handler(self, tag: str) -> Callable[[], None]:
        """创建标签按钮点击处理函数，支持多选/单选切换。"""

        def handler() -> None:
            if MULTI_TAG_MODE:
                if tag in self.current_tags:
                    self.current_tags.remove(tag)
                else:
                    self.current_tags.append(tag)
            else:
                self.current_tags = [] if tag in self.current_tags else [tag]
            self.sync_tag_buttons()
            self.refresh_current_info_only()

        return handler

    def sync_score_buttons(self) -> None:
        """根据 current_score 刷新评分按钮高亮状态。"""

        for score, btn in self.score_buttons.items():
            btn.blockSignals(True)
            btn.setChecked(score == self.current_score)
            btn.blockSignals(False)

    def sync_tag_buttons(self) -> None:
        """根据 current_tags 刷新标签按钮选中状态。"""

        for tag, btn in self.tag_buttons.items():
            btn.blockSignals(True)
            btn.setChecked(tag in self.current_tags)
            btn.blockSignals(False)

    def refresh_current_info_only(self) -> None:
        """局部刷新当前视频信息，不触发截图重载。"""

        if self.current_index < 0 or self.current_index >= len(self.filtered_video_items):
            return
        self.update_video_info(self.filtered_video_items[self.current_index])

    def load_screenshots_async(self, path: Path) -> None:
        """启动后台任务生成截图，并展示加载状态。"""

        self.clear_screenshots()
        self.status_message("正在加载截图...")

        worker = ScreenshotWorker(path, self.cache_root, SCREENSHOT_COUNT)
        worker.signals.status.connect(self.status_message)
        worker.signals.error.connect(self.on_screenshot_error)
        worker.signals.finished.connect(self.on_screenshot_finished)
        self.thread_pool.start(worker)

    def on_screenshot_finished(self, video_path: str, image_files: list[str]) -> None:
        """后台截图完成后回填到网格区域。"""

        # 如果用户已切换到其他视频，则忽略过期任务结果。
        if self.current_index < 0 or self.filtered_video_items[self.current_index].path != Path(video_path):
            return

        self.clear_screenshots()
        row = 0
        col = 0
        for img in image_files:
            label = QLabel()
            label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            pix = QPixmap(img)
            if not pix.isNull():
                label.setPixmap(
                    pix.scaled(
                        280,
                        170,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
            self.screenshot_grid.addWidget(label, row, col)
            col += 1
            if col >= GRID_COLUMNS:
                col = 0
                row += 1

        self.status_message("截图加载完成")

    def on_screenshot_error(self, message: str) -> None:
        """截图失败时给出非致命提示，程序继续可用。"""

        self.clear_screenshots()
        err = QLabel(f"截图生成失败：{message}\n请检查 ffmpeg/ffprobe 配置。")
        err.setStyleSheet("color: #c00;")
        self.screenshot_grid.addWidget(err, 0, 0)
        self.status_message("截图生成失败")

    def clear_screenshots(self) -> None:
        """清空截图网格中的旧控件。"""

        while self.screenshot_grid.count():
            item = self.screenshot_grid.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

    def on_play(self) -> None:
        """调用 PotPlayer 播放当前视频文件。"""

        current = self.get_current_video()
        if current is None:
            return
        if not Path(POTPLAYER_EXE).exists():
            QMessageBox.warning(self, "提示", f"PotPlayer 不存在，请配置 POTPLAYER_EXE：\n{POTPLAYER_EXE}")
            return
        try:
            subprocess.Popen([POTPLAYER_EXE, str(current.path)])
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "播放失败", str(exc))

    def on_delete(self) -> None:
        """强约束：点击后直接删除当前视频，不弹确认框。"""

        current = self.get_current_video()
        if current is None:
            return

        try:
            current.path.unlink()
            self.state_manager.remove(current.path)
            self.clean_cache_for_video(current.path)
            removed_index = self.current_index
            # 从全量列表移除当前文件，再按过滤条件重建可见列表。
            self.all_video_items = [v for v in self.all_video_items if v.path != current.path]
            self.apply_filter()

            if self.filtered_video_items:
                next_index = min(removed_index, len(self.filtered_video_items) - 1)
                self.list_widget.setCurrentRow(next_index)
            else:
                self.current_index = -1
                self.lbl_info.setText("视频已删空")
                self.clear_screenshots()
            self.status_message("删除成功")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "删除失败", str(exc))

    def clean_cache_for_video(self, path: Path) -> None:
        """删除指定视频在缓存目录中的历史截图（可选清理）。"""

        try:
            key = build_cache_key(path)
            cache_dir = self.cache_root / key
            if cache_dir.exists():
                for f in cache_dir.glob("*"):
                    f.unlink(missing_ok=True)
                cache_dir.rmdir()
        except Exception as exc:  # noqa: BLE001
            logging.warning("清理截图缓存失败：%s", exc)

    def on_confirm(self) -> None:
        """确认：先重命名当前视频，再自动跳到下一个。"""

        # 新规则：确认按钮始终进入下一个；仅当“评分+标签”都已填写时才重命名。
        self.rename_current_video_if_needed(require_complete_fields=True)
        self.goto_next_video()

    def on_next(self) -> None:
        """下一个：按需求先尝试重命名，再切换到下一个。"""

        # 新规则：下一个按钮始终切换；仅当“评分+标签”都已填写时才重命名。
        self.rename_current_video_if_needed(require_complete_fields=True)
        self.goto_next_video()

    def on_prev(self) -> None:
        """上一个：切换到上一个；当评分+标签完整时先执行重命名。"""

        if not self.filtered_video_items:
            return
        # 新规则：上一个按钮在“评分+标签”都已填写时，同样执行重命名。
        self.rename_current_video_if_needed(require_complete_fields=True)
        target = max(self.current_index - 1, 0)
        self.list_widget.setCurrentRow(target)

    def goto_next_video(self) -> None:
        """切换到列表中的下一个视频。"""

        if not self.filtered_video_items:
            return
        target = min(self.current_index + 1, len(self.filtered_video_items) - 1)
        self.list_widget.setCurrentRow(target)

    def rename_current_video_if_needed(self, require_complete_fields: bool = False) -> bool:
        """执行强约束重命名：{score}-{tags}-{code}{ext}。

        Args:
            require_complete_fields: 为 True 时，要求评分和标签都已填写才执行重命名。
        """

        current = self.get_current_video()
        if current is None:
            return False

        old_path = current.path
        old_state = self.state_manager.get(old_path)

        # 新规则开关：若要求“字段完整”但评分或标签缺失，则直接跳过重命名。
        if require_complete_fields and (self.current_score is None or not self.current_tags):
            self.status_message("评分或标签未完整填写，已跳过重命名")
            return False

        score_text = str(self.current_score) if self.current_score is not None else ""
        tags_text = "_".join(sanitize_for_windows(tag) for tag in self.current_tags) if self.current_tags else ""

        # 若已有 code，则复用；无则新生成，避免重复 rename 造成额外变化。
        code = old_state.get("code") or random_code()

        new_name = f"{score_text}-{tags_text}-{code}{old_path.suffix}"
        new_path = old_path.with_name(new_name)

        # 如果名字本来就一致，仅写状态即可。
        if new_path == old_path:
            self.state_manager.update(
                old_path,
                {
                    "score": self.current_score,
                    "tags": self.current_tags,
                    "code": code,
                    "renamed_to": str(old_path),
                },
            )
            self.refresh_list_widget()
            self.list_widget.setCurrentRow(self.current_index)
            return True

        # 冲突时重试生成 code，直到找到可用文件名。
        retry = 0
        while new_path.exists() and retry < 20:
            code = random_code()
            new_name = f"{score_text}-{tags_text}-{code}{old_path.suffix}"
            new_path = old_path.with_name(new_name)
            retry += 1

        if new_path.exists():
            QMessageBox.warning(self, "重命名失败", "多次重试后仍存在重名文件")
            return False

        try:
            old_path.rename(new_path)
            # 同步更新全量列表与过滤列表中的路径引用，保持 UI/状态一致。
            for collection in (self.all_video_items, self.filtered_video_items):
                for item in collection:
                    if item.path == old_path:
                        item.path = new_path
            self.state_manager.move_key(old_path, new_path)
            self.state_manager.update(
                new_path,
                {
                    "score": self.current_score,
                    "tags": self.current_tags,
                    "code": code,
                    "renamed_to": str(new_path),
                },
            )
            self.refresh_list_widget()
            self.list_widget.setCurrentRow(self.current_index)
            self.status_message(f"已重命名为：{new_path.name}")
            return True
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "重命名失败", str(exc))
            return False

    def get_current_video(self) -> VideoItem | None:
        """返回当前选中的视频对象，无选中时返回 None。"""

        if self.current_index < 0 or self.current_index >= len(self.filtered_video_items):
            return None
        return self.filtered_video_items[self.current_index]

    def status_message(self, text: str) -> None:
        """更新状态栏文本（非阻塞提示）。"""

        self.lbl_status.setText(text)


def main() -> None:
    """程序入口：创建 QApplication 并启动主窗口。"""

    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
