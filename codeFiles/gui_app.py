#!/usr/bin/env python3
"""
AI Fish Migration Detector – GUI
Pipeline:
  1. Barrel / fisheye correction  (fix_barrel.py)
  2. HDR tone-mapping + CLAHE     (fix_hdr.py)
  3. YOLO fish counting           (count.py)

Usage:
    python3 codeFiles/gui_app.py
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import subprocess
import threading
import os
import sys
import tempfile

try:
    import cv2
    import numpy as np
    from PIL import Image, ImageTk
    HAS_VIDEO = True
except ImportError:
    HAS_VIDEO = False

current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

# All GUI-generated output videos go here
GUI_OUT_DIR = os.path.join(project_root, "GUI_processed_videos")
os.makedirs(GUI_OUT_DIR, exist_ok=True)

# ── Design System ─────────────────────────────────────────────────────────────
# Dark navy theme with vibrant accent colors
BG           = "#0f1623"       # deep navy background
SIDEBAR_BG   = "#141d2e"       # slightly lighter for sidebar
CARD         = "#1a2540"       # card background
CARD_INNER   = "#10192a"       # input/preview area
CARD_HOVER   = "#1f2d4e"       # card hover state

BORDER       = "#2a3a5c"       # subtle border
BORDER_FOCUS = "#3b82f6"       # focused border

ACCENT       = "#3b82f6"       # vivid blue accent
ACCENT_DARK  = "#2563eb"       # darker blue
ACCENT_GLOW  = "#1d4ed8"       # deep blue for pressed

GREEN        = "#10b981"       # emerald green
GREEN_DARK   = "#059669"       # darker green
GREEN_GLOW   = "#047857"

AMBER        = "#f59e0b"       # status amber
RED          = "#ef4444"       # error red

TEXT_PRIMARY = "#f0f4ff"       # near-white primary text
TEXT_SECOND  = "#8b9ec5"       # secondary muted text
TEXT_ACCENT  = "#60a5fa"       # blue-tinted accent text
TEXT_GREEN   = "#34d399"       # green status text
TEXT_MUTED   = "#4a5a7a"       # very muted/disabled text

PREVIEW_W    = 460
PREVIEW_H    = 270

FONT_FAMILY  = "Helvetica"
# ─────────────────────────────────────────────────────────────────────────────


def _make_gradient_photo(w, h, hex_top, hex_bot, width_px=1):
    """Create a very thin vertical gradient PhotoImage for button backgrounds."""
    img = tk.PhotoImage(width=w, height=h)
    r1, g1, b1 = int(hex_top[1:3], 16), int(hex_top[3:5], 16), int(hex_top[5:7], 16)
    r2, g2, b2 = int(hex_bot[1:3], 16), int(hex_bot[3:5], 16), int(hex_bot[5:7], 16)
    data = []
    for row in range(h):
        t = row / max(h - 1, 1)
        r = int(r1 + (r2 - r1) * t)
        g = int(g1 + (g2 - g1) * t)
        b = int(b1 + (b2 - b1) * t)
        hex_col = f"#{r:02x}{g:02x}{b:02x}"
        data.append(" ".join([hex_col] * w))
    img.put("\n".join("{" + row + "}" for row in data))
    return img


class RoundedButton(tk.Canvas):
    """
    Rounded-rectangle button drawn on a Canvas.
    Uses after_idle() for the initial draw so the widget
    is fully realized before any Tcl canvas commands are issued
    (avoids the macOS 'invalid command name' crash).
    """
    RADIUS = 10

    def __init__(self, parent, text, command=None,
                 bg=ACCENT, bg_hover=None, bg_disabled=None,
                 fg=TEXT_PRIMARY,
                 height=46, **kw):
        super().__init__(parent, height=height, bd=0,
                         highlightthickness=0,
                         bg=SIDEBAR_BG, cursor="hand2", **kw)
        self._text        = text
        self._command     = command
        self._bg          = bg
        self._bg_hover    = bg_hover or bg
        # disabled: caller can pass an explicit colour; fall back to a
        # darkened shade of the button's own colour so text stays readable
        self._bg_disabled = bg_disabled or self._darken(bg, 0.45)
        self._fg          = fg
        self._enabled     = True
        self._hovering    = False

        # Defer draw until widget is realized
        self.after_idle(self._draw)
        self.bind("<Configure>", lambda e: self._draw())
        self.bind("<Enter>",     self._on_enter)
        self.bind("<Leave>",     self._on_leave)
        self.bind("<Button-1>", self._on_press)

    # ── Drawing ───────────────────────────────────────────────────────────────

    @staticmethod
    def _darken(hex_color, factor):
        """Return hex_color darkened by factor (0=black, 1=unchanged)."""
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        return f"#{int(r*factor):02x}{int(g*factor):02x}{int(b*factor):02x}"

    def _fill_color(self):
        if not self._enabled:
            return self._bg_disabled
        return self._bg_hover if self._hovering else self._bg

    def _text_color(self):
        return self._fg   # always white — readable on any fill

    def _draw(self):
        try:
            self.delete("all")
        except tk.TclError:
            return   # widget not yet realized; will retry via <Configure>
        w = self.winfo_width()  or int(self["width"]  or 200)
        h = self.winfo_height() or int(self["height"] or 46)
        r = self.RADIUS
        fill = self._fill_color()

        # rounded rectangle via overlapping shapes
        self.create_arc(0, 0, 2*r, 2*r, start=90, extent=90,   fill=fill, outline=fill)
        self.create_arc(w-2*r, 0, w, 2*r, start=0, extent=90,  fill=fill, outline=fill)
        self.create_arc(w-2*r, h-2*r, w, h, start=270, extent=90, fill=fill, outline=fill)
        self.create_arc(0, h-2*r, 2*r, h, start=180, extent=90, fill=fill, outline=fill)
        self.create_rectangle(r, 0, w-r, h,   fill=fill, outline=fill)
        self.create_rectangle(0, r, w,   h-r, fill=fill, outline=fill)

        self.create_text(w // 2, h // 2, text=self._text,
                         fill=self._text_color(),
                         font=(FONT_FAMILY, 12, "bold"))

    # ── Interaction ───────────────────────────────────────────────────────────

    def _on_enter(self, _):
        if self._enabled:
            self._hovering = True
            self._draw()

    def _on_leave(self, _):
        self._hovering = False
        self._draw()

    def _on_press(self, _):
        if self._enabled and self._command:
            self._command()

    def set_state(self, enabled: bool):
        self._enabled = enabled
        self._hovering = False
        self._draw()



class SeparatorLine(tk.Frame):
    """A thin horizontal separator line."""
    def __init__(self, parent, color=BORDER, **kw):
        super().__init__(parent, bg=color, height=1, bd=0,
                         highlightthickness=0, **kw)


class BadgeLabel(tk.Frame):
    """A small pill/badge label with colored background."""
    def __init__(self, parent, text, bg_color=ACCENT, fg_color=TEXT_PRIMARY, **kw):
        super().__init__(parent, bg=bg_color, bd=0, highlightthickness=0, **kw)
        tk.Label(self, text=text, bg=bg_color, fg=fg_color,
                 font=(FONT_FAMILY, 8, "bold"),
                 padx=6, pady=2).pack()


class VideoPlayer:
    """
    Renders video frames into a tk.Label at (speed-adjusted) native frame rate,
    looping until stop() is called.
    All UI updates happen on the main thread via root.after().
    """

    def __init__(self, root, label, width=PREVIEW_W, height=PREVIEW_H):
        self.root    = root
        self.label   = label
        self.width   = width
        self.height  = height
        self.speed   = 1.0
        self._cap    = None
        self._active = False
        self._fps    = 30.0
        self._photo  = None

    def play(self, path, loop=True):
        self.stop()
        if not HAS_VIDEO:
            return
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return
        self._cap    = cap
        self._loop   = loop
        self._active = True
        self._fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._step()

    def stop(self):
        self._active = False
        if self._cap:
            self._cap.release()
            self._cap = None

    def _step(self):
        if not self._active or self._cap is None:
            return
        ret, frame = self._cap.read()
        if not ret:
            if self._loop:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.root.after(self._frame_delay(), self._step)
            else:
                self.stop()
            return
        self._show_bgr(frame)
        self.root.after(self._frame_delay(), self._step)

    def _frame_delay(self):
        return max(1, int(1000 / (self._fps * max(0.1, self.speed))))

    def _show_bgr(self, frame):
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img   = Image.fromarray(rgb)
        img.thumbnail((self.width, self.height), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(img)
        self.label.config(image=self._photo, text="", compound=tk.CENTER)

    def show_jpeg_bytes(self, data):
        if not HAS_VIDEO:
            return
        arr   = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is not None:
            self._show_bgr(frame)


# ─────────────────────────────────────────────────────────────────────────────

class VideoProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Fish Migration Detector")
        self.root.geometry("1000x680")
        self.root.resizable(True, True)
        self.root.minsize(900, 620)
        self.root.configure(bg=BG)

        self.selected_file_path = None
        self.hdr_output_path    = None
        self.processing         = False
        self.player             = None

        self._configure_styles()
        self._build_ui()

    # ── ttk style configuration ───────────────────────────────────────────────

    def _configure_styles(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        # Progress bar – dark track, vivid blue fill
        style.configure("Dark.Horizontal.TProgressbar",
                        troughcolor=CARD_INNER,
                        background=ACCENT,
                        borderwidth=0,
                        thickness=4)

        # Scale (speed slider)
        style.configure("Dark.Horizontal.TScale",
                        background=CARD,
                        troughcolor=CARD_INNER,
                        sliderthickness=14)

    # ── Top-level layout ──────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Header bar ────────────────────────────────────────────────────────
        header = tk.Frame(self.root, bg=SIDEBAR_BG, height=64)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        inner_hdr = tk.Frame(header, bg=SIDEBAR_BG)
        inner_hdr.pack(fill=tk.Y, side=tk.LEFT, padx=24, pady=0)

        # Fish icon badge
        badge_frame = tk.Frame(inner_hdr, bg=ACCENT, padx=8, pady=4)
        badge_frame.pack(side=tk.LEFT, anchor="w", pady=14)
        tk.Label(badge_frame, text="🐟", bg=ACCENT,
                 font=(FONT_FAMILY, 16)).pack(side=tk.LEFT)

        title_block = tk.Frame(inner_hdr, bg=SIDEBAR_BG)
        title_block.pack(side=tk.LEFT, padx=10)
        tk.Label(title_block, text="AI Fish Migration Detector",
                 bg=SIDEBAR_BG, fg=TEXT_PRIMARY,
                 font=(FONT_FAMILY, 17, "bold")).pack(anchor="w")
        tk.Label(title_block, text="Automated fisheye correction · HDR enhancement · YOLO fish counting",
                 bg=SIDEBAR_BG, fg=TEXT_SECOND,
                 font=(FONT_FAMILY, 9)).pack(anchor="w")

        # Thin accent line at bottom of header
        tk.Frame(self.root, bg=ACCENT, height=2).pack(fill=tk.X)

        # ── Body ──────────────────────────────────────────────────────────────
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill=tk.BOTH, expand=True, padx=20, pady=16)

        # Left sidebar
        sidebar = tk.Frame(body, bg=SIDEBAR_BG, width=310, bd=0,
                           highlightthickness=1, highlightbackground=BORDER)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 16))
        sidebar.pack_propagate(False)

        # Right content
        content = tk.Frame(body, bg=BG)
        content.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._build_sidebar(sidebar)
        self._build_output_panel(content)

    # ── Sidebar ───────────────────────────────────────────────────────────────

    def _build_sidebar(self, parent):
        # Section: Upload
        self._sidebar_section_label(parent, "1  ·  Select Video")
        self._build_upload_area(parent)
        SeparatorLine(parent).pack(fill=tk.X, padx=16, pady=12)

        # Section: Processing
        self._sidebar_section_label(parent, "2  ·  Apply Pipeline")
        self._build_pipeline_buttons(parent)
        SeparatorLine(parent).pack(fill=tk.X, padx=16, pady=12)

        # Section: Counting
        self._sidebar_section_label(parent, "3  ·  Count Fish")
        self._build_count_button(parent)
        SeparatorLine(parent).pack(fill=tk.X, padx=16, pady=12)

        # Status strip at the bottom
        self._build_status_strip(parent)

    def _sidebar_section_label(self, parent, text):
        tk.Label(parent, text=text,
                 bg=SIDEBAR_BG, fg=TEXT_ACCENT,
                 font=(FONT_FAMILY, 9, "bold"),
                 anchor="w").pack(fill=tk.X, padx=18, pady=(14, 6))

    def _build_upload_area(self, parent):
        outer = tk.Frame(parent, bg=SIDEBAR_BG)
        outer.pack(fill=tk.X, padx=14)

        drop = tk.Frame(outer, bg=CARD_INNER, bd=0,
                        highlightthickness=2, highlightbackground=BORDER,
                        height=120)
        drop.pack(fill=tk.X)
        drop.pack_propagate(False)

        # Icon row
        icon_lbl = tk.Label(drop, text="⬆", bg=CARD_INNER, fg=ACCENT,
                            font=(FONT_FAMILY, 26))
        icon_lbl.pack(pady=(16, 4))

        self.drop_title = tk.Label(drop, text="Click to upload video",
                                   bg=CARD_INNER, fg=TEXT_PRIMARY,
                                   font=(FONT_FAMILY, 10, "bold"))
        self.drop_title.pack()

        self.drop_sub = tk.Label(drop, text="MP4 · AVI · MOV · MKV  (max 500 MB)",
                                 bg=CARD_INNER, fg=TEXT_SECOND,
                                 font=(FONT_FAMILY, 8))
        self.drop_sub.pack(pady=(2, 0))

        for w in (drop, icon_lbl, self.drop_title, self.drop_sub):
            w.bind("<Button-1>", lambda e: self._select_file())
            w.bind("<Enter>", lambda e: drop.config(
                highlightbackground=ACCENT))
            w.bind("<Leave>", lambda e: drop.config(
                highlightbackground=BORDER))
            w.configure(cursor="hand2")

        self._drop_widget = drop

    def _build_pipeline_buttons(self, parent):
        btn_frame = tk.Frame(parent, bg=SIDEBAR_BG)
        btn_frame.pack(fill=tk.X, padx=14)

        self.run_btn = RoundedButton(
            btn_frame,
            text="▶   Run Correction Pipeline",
            command=self._run_pipeline,
            bg=ACCENT, bg_hover="#5ca0ff",
        )
        self.run_btn.pack(fill=tk.X, pady=(0, 8))
        self.run_btn.set_state(False)

        # Progress bar (hidden until running)
        self.progress = ttk.Progressbar(btn_frame, orient=tk.HORIZONTAL,
                                        mode="indeterminate",
                                        style="Dark.Horizontal.TProgressbar")

    def _build_count_button(self, parent):
        btn_frame = tk.Frame(parent, bg=SIDEBAR_BG)
        btn_frame.pack(fill=tk.X, padx=14)

        # ── Model selector ────────────────────────────────────────────────────
        tk.Label(btn_frame, text="MODEL",
                 bg=SIDEBAR_BG, fg=TEXT_MUTED,
                 font=(FONT_FAMILY, 7, "bold")).pack(anchor="w", pady=(0, 2))

        # Discover .pt files in project root; default = current_fish_model.pt
        pt_files = sorted(
            f for f in os.listdir(project_root) if f.endswith(".pt")
        )
        default_model = "extraLarge.pt"
        if default_model not in pt_files:
            pt_files.insert(0, default_model)

        self.model_var = tk.StringVar(value=default_model)
        model_menu = ttk.Combobox(
            btn_frame,
            textvariable=self.model_var,
            values=pt_files,
            state="readonly",
            font=(FONT_FAMILY, 9),
        )
        model_menu.pack(fill=tk.X, pady=(0, 10))

        # ── Confidence slider ─────────────────────────────────────────────────
        conf_header = tk.Frame(btn_frame, bg=SIDEBAR_BG)
        conf_header.pack(fill=tk.X)

        tk.Label(conf_header, text="CONFIDENCE THRESHOLD",
                 bg=SIDEBAR_BG, fg=TEXT_MUTED,
                 font=(FONT_FAMILY, 7, "bold")).pack(side=tk.LEFT)

        self.conf_label = tk.Label(conf_header, text="0.25",
                                   bg=SIDEBAR_BG, fg=TEXT_ACCENT,
                                   font=(FONT_FAMILY, 9, "bold"))
        self.conf_label.pack(side=tk.RIGHT)

        self.conf_var = tk.DoubleVar(value=0.25)
        conf_slider = ttk.Scale(
            btn_frame, from_=0.25, to=0.99,
            variable=self.conf_var, orient=tk.HORIZONTAL,
            command=self._on_conf_change,
            style="Dark.Horizontal.TScale",
        )
        conf_slider.pack(fill=tk.X, pady=(4, 10))

        # ── Count button ──────────────────────────────────────────────────────
        self.count_btn = RoundedButton(
            btn_frame,
            text="▶  Run Fish Count",
            command=self._run_count,
            bg=GREEN, bg_hover="#30d49a",
        )
        self.count_btn.pack(fill=tk.X)
        self.count_btn.set_state(False)

        # ── Informational hints ───────────────────────────────────────────────
        hints_frame = tk.Frame(btn_frame, bg=SIDEBAR_BG)
        hints_frame.pack(fill=tk.X, pady=(10, 0))

        hint1 = (
            "1: Larger models offer better accuracy and counting "
            "while smaller models will run quicker."
        )
        tk.Label(hints_frame, text=hint1,
                 bg=SIDEBAR_BG, fg=TEXT_SECOND,
                 font=(FONT_FAMILY, 8),
                 wraplength=265, justify=tk.LEFT,
                 anchor="w").pack(fill=tk.X, pady=(0, 4))

        hint2 = (
            "2: The confidence threshold determines how confident the model "
            "must be before it makes a detection. 0.25 is the lowest "
            "confidence while 0.99 is the highest."
        )
        tk.Label(hints_frame, text=hint2,
                 bg=SIDEBAR_BG, fg=TEXT_SECOND,
                 font=(FONT_FAMILY, 8),
                 wraplength=265, justify=tk.LEFT,
                 anchor="w").pack(fill=tk.X)

    def _build_status_strip(self, parent):
        strip = tk.Frame(parent, bg=SIDEBAR_BG)
        strip.pack(side=tk.BOTTOM, fill=tk.X, padx=14, pady=10)

        SeparatorLine(strip).pack(fill=tk.X, pady=(0, 8))

        row = tk.Frame(strip, bg=SIDEBAR_BG)
        row.pack(fill=tk.X)

        self._dot = tk.Label(row, text="●", bg=SIDEBAR_BG, fg=GREEN,
                             font=(FONT_FAMILY, 10))
        self._dot.pack(side=tk.LEFT)

        self.status_var = tk.StringVar(value="Ready")
        tk.Label(row, textvariable=self.status_var,
                 bg=SIDEBAR_BG, fg=TEXT_SECOND,
                 font=(FONT_FAMILY, 9), anchor="w").pack(side=tk.LEFT, padx=(4, 0))

    # ── Output panel (right) ──────────────────────────────────────────────────

    def _build_output_panel(self, parent):
        # Panel card
        panel = tk.Frame(parent, bg=CARD, bd=0,
                         highlightthickness=1, highlightbackground=BORDER)
        panel.pack(fill=tk.BOTH, expand=True)

        # ── Panel header ──────────────────────────────────────────────────────
        ph = tk.Frame(panel, bg=CARD)
        ph.pack(fill=tk.X, padx=18, pady=(14, 0))

        tk.Label(ph, text="Output Preview",
                 bg=CARD, fg=TEXT_PRIMARY,
                 font=(FONT_FAMILY, 13, "bold")).pack(side=tk.LEFT)

        self._live_badge = BadgeLabel(ph, "● LIVE", bg_color="#7c3aed",
                                      fg_color="#e9d5ff")
        # shown only during counting

        SeparatorLine(panel, color=BORDER).pack(fill=tk.X, padx=0, pady=(10, 0))

        # ── Video preview ──────────────────────────────────────────────────────
        preview_wrap = tk.Frame(panel, bg=CARD)
        preview_wrap.pack(fill=tk.BOTH, expand=True, padx=18, pady=12)

        preview_frame = tk.Frame(preview_wrap, bg=CARD_INNER, bd=0,
                                 highlightthickness=1, highlightbackground=BORDER,
                                 width=PREVIEW_W, height=PREVIEW_H)
        preview_frame.pack(fill=tk.BOTH, expand=True)
        preview_frame.pack_propagate(False)

        self.preview_label = tk.Label(
            preview_frame,
            bg=CARD_INNER,
            fg=TEXT_MUTED,
            text="Output video will appear here after processing",
            font=(FONT_FAMILY, 10),
            wraplength=260,
        )
        self.preview_label.place(relx=0.5, rely=0.5, anchor="center")

        self.player = VideoPlayer(self.root, self.preview_label)

        # ── Controls row ──────────────────────────────────────────────────────
        SeparatorLine(panel, color=BORDER).pack(fill=tk.X)

        controls = tk.Frame(panel, bg=CARD)
        controls.pack(fill=tk.X, padx=18, pady=10)

        # Speed control
        speed_col = tk.Frame(controls, bg=CARD)
        speed_col.pack(side=tk.LEFT, fill=tk.X, expand=True)

        speed_header = tk.Frame(speed_col, bg=CARD)
        speed_header.pack(fill=tk.X)

        tk.Label(speed_header, text="PLAYBACK SPEED",
                 bg=CARD, fg=TEXT_MUTED,
                 font=(FONT_FAMILY, 7, "bold")).pack(side=tk.LEFT)

        self.speed_label = tk.Label(speed_header, text="1.00×",
                                    bg=CARD, fg=TEXT_ACCENT,
                                    font=(FONT_FAMILY, 9, "bold"))
        self.speed_label.pack(side=tk.RIGHT)

        self.speed_var = tk.DoubleVar(value=1.0)
        speed_slider = ttk.Scale(
            speed_col, from_=0.25, to=2.0,
            variable=self.speed_var, orient=tk.HORIZONTAL,
            command=self._on_speed_change,
            style="Dark.Horizontal.TScale",
        )
        speed_slider.pack(fill=tk.X, pady=(4, 0))

        # ── Fish count display ────────────────────────────────────────────────
        SeparatorLine(panel, color=BORDER).pack(fill=tk.X)

        count_row = tk.Frame(panel, bg=CARD)
        count_row.pack(fill=tk.X, padx=18, pady=12)

        tk.Label(count_row, text="🐟  TOTAL FISH COUNT",
                 bg=CARD, fg=TEXT_MUTED,
                 font=(FONT_FAMILY, 8, "bold")).pack(side=tk.LEFT, anchor="s",
                                                     pady=(0, 2))

        self.count_var = tk.StringVar(value="—")
        self._count_display = tk.Label(count_row, textvariable=self.count_var,
                                       bg=CARD, fg=TEXT_ACCENT,
                                       font=(FONT_FAMILY, 26, "bold"))
        self._count_display.pack(side=tk.RIGHT)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _set_status(self, text, color=TEXT_SECOND, dot_color=None):
        self.status_var.set(text)
        if dot_color:
            self._dot.config(fg=dot_color)

    def _on_conf_change(self, val):
        self.conf_label.config(text=f"{float(val):.2f}")

    def _on_speed_change(self, val):
        speed = float(val)
        self.speed_label.config(text=f"{speed:.2f}×")
        if self.player:
            self.player.speed = speed

    def _select_file(self):
        if self.processing:
            return
        filetypes = (
            ("Video files", "*.mp4 *.avi *.mov *.mkv"),
            ("All files",   "*.*"),
        )
        filename = filedialog.askopenfilename(
            title="Open a video file",
            initialdir=project_root,
            filetypes=filetypes,
        )
        if filename:
            self.selected_file_path = filename
            short = os.path.basename(filename)
            self.drop_title.config(text=f"✔  {short}", fg=TEXT_GREEN)
            self.drop_sub.config(text="Click to change file")
            self._drop_widget.config(highlightbackground=GREEN)
            self.run_btn.set_state(True)
            self._set_status("Video loaded — ready to run.", dot_color=GREEN)

    # ── Pipeline (barrel → HDR) ───────────────────────────────────────────────

    def _run_pipeline(self):
        if not self.selected_file_path or self.processing:
            return
        self.processing      = True
        self.hdr_output_path = None
        self.run_btn.set_state(False)
        self.count_btn.set_state(False)
        self.progress.pack(fill=tk.X, pady=(6, 0))
        self.progress.start(10)
        self._set_status("Stage 1/2 — Barrel correction…", dot_color=AMBER)

        t = threading.Thread(target=self._pipeline_thread,
                             args=(self.selected_file_path,))
        t.daemon = True
        t.start()

    def _pipeline_thread(self, input_video):
        barrel_script = os.path.join(current_dir, "fix_barrel.py")
        hdr_script    = os.path.join(current_dir, "fix_hdr.py")

        # Derive a clean base name from the original input file
        orig_base = os.path.splitext(os.path.basename(input_video))[0]
        hdr_out   = os.path.join(GUI_OUT_DIR, f"{orig_base}_corrected.mp4")

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4", prefix="barrel_tmp_")
        os.close(tmp_fd)

        try:
            r1 = subprocess.run(
                [sys.executable, barrel_script, input_video, tmp_path],
                cwd=project_root, capture_output=True, text=True,
            )
            if r1.returncode != 0:
                err = r1.stderr[-400:] if r1.stderr else "Unknown error"
                self.root.after(0, self._finish_pipeline, False,
                                f"Stage 1 (barrel correction) failed:\n{err}", None)
                return

            self.root.after(0, lambda: self._set_status(
                "Stage 2/2 — HDR tone-mapping…", dot_color=AMBER))

            # Pass the explicit output path so fix_hdr.py writes to GUI_processed_videos/
            r2 = subprocess.run(
                [sys.executable, hdr_script, tmp_path, hdr_out],
                cwd=project_root, capture_output=True, text=True,
            )
            if r2.returncode != 0:
                err = r2.stderr[-400:] if r2.stderr else "Unknown error"
                self.root.after(0, self._finish_pipeline, False,
                                f"Stage 2 (HDR enhancement) failed:\n{err}", None)
                return

            # Verify the file exists (fix_hdr.py still prints the path; just cross-check)
            final_hdr = hdr_out if os.path.exists(hdr_out) else None

            self.root.after(0, self._finish_pipeline, True,
                            "Pipeline complete!", final_hdr)

        except Exception as exc:
            self.root.after(0, self._finish_pipeline, False, str(exc), None)
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    def _finish_pipeline(self, success, message, hdr_output_path=None):
        self.processing = False
        self.progress.stop()
        self.progress.pack_forget()
        self.run_btn.set_state(True)

        if success:
            self.hdr_output_path = hdr_output_path
            if hdr_output_path and os.path.exists(hdr_output_path):
                self.count_btn.set_state(True)
            self._set_status("Pipeline complete — output saved.", dot_color=GREEN)
            messagebox.showinfo(
                "Pipeline Complete",
                "✔ Both stages finished successfully.\n\n"
                f"The enhanced video is in 'GUI_processed_videos/'.\n"
                "You may now run the fish count.",
            )
        else:
            self._set_status("Error during processing.", dot_color=RED)
            messagebox.showerror("Processing Error",
                                 f"Failed to process video.\n\n{message}")

    # ── Fish counting (live stream from count.py) ─────────────────────────────

    def _run_count(self):
        if not self.hdr_output_path or self.processing:
            return
        self.processing = True
        self.run_btn.set_state(False)
        self.count_btn.set_state(False)
        self.count_var.set("—")
        self.player.stop()
        self.progress.pack(fill=tk.X, pady=(6, 0))
        self.progress.start(10)
        self._set_status("Counting fish — live view active…", dot_color="#7c3aed")
        self._live_badge.pack(side=tk.RIGHT)

        t = threading.Thread(target=self._count_thread,
                             args=(self.hdr_output_path,))
        t.daemon = True
        t.start()

    def _count_thread(self, hdr_video):
        count_script = os.path.join(current_dir, "count.py")
        # Strip any suffix tags (_corrected etc.) to recover the clean original base
        base     = os.path.splitext(os.path.basename(hdr_video))[0]
        base     = base.removesuffix("_corrected")   # Python 3.9+
        out_path = os.path.join(GUI_OUT_DIR, f"{base}_counted.mp4")

        try:
            proc = subprocess.Popen(
                [sys.executable, "-u", count_script, hdr_video,
                 "--output", out_path,
                 "--no-show", "--gui-stream",
                 "--conf", str(round(self.conf_var.get(), 2)),
                 "--model", self.model_var.get()],
                cwd=project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            while True:
                header = proc.stdout.read(4)
                if not header or len(header) < 4:
                    break
                frame_len = int.from_bytes(header, "big")

                frame_data = bytearray()
                while len(frame_data) < frame_len:
                    chunk = proc.stdout.read(frame_len - len(frame_data))
                    if not chunk:
                        break
                    frame_data.extend(chunk)

                if len(frame_data) < frame_len:
                    break

                self.root.after(0, self.player.show_jpeg_bytes, bytes(frame_data))

            proc.wait()
            err_txt = proc.stderr.read().decode(errors="replace")

            # Parse the final in-count and out-count emitted by count.py
            in_count = out_count = None
            for line in err_txt.splitlines():
                if line.startswith("FINAL_IN_COUNT:"):
                    try:
                        in_count = int(line.split(":", 1)[1].strip())
                    except ValueError:
                        pass
                elif line.startswith("FINAL_OUT_COUNT:"):
                    try:
                        out_count = int(line.split(":", 1)[1].strip())
                    except ValueError:
                        pass

            if in_count is not None or out_count is not None:
                total = (in_count or 0) + (out_count or 0)
            self.root.after(0, self.count_var.set, str(total))
            if proc.returncode != 0:
                err = err_txt[-400:] if err_txt else "Unknown error"
                self.root.after(0, self._finish_count, False,
                                f"Fish counting failed:\n{err}", None)
            else:
                self.root.after(0, self._finish_count, True,
                                "Count complete!", out_path)

        except Exception as exc:
            self.root.after(0, self._finish_count, False, str(exc), None)

    def _finish_count(self, success, message, out_path=None):
        self.processing = False
        self.progress.stop()
        self.progress.pack_forget()
        self.run_btn.set_state(True)
        self.count_btn.set_state(True)
        self._live_badge.pack_forget()

        if success and out_path and os.path.exists(out_path):
            self._set_status("Count complete — playing output…", dot_color=GREEN)
            self.player.play(out_path, loop=True)
            messagebox.showinfo(
                "Count Complete",
                f"✔ Fish counting finished!\n\nOutput: {os.path.basename(out_path)}",
            )
        elif success:
            self._set_status("Fish count complete!", dot_color=GREEN)
            messagebox.showinfo("Done", message)
        else:
            self.player.stop()
            self._set_status("Error during fish counting.", dot_color=RED)
            messagebox.showerror("Counting Error",
                                 f"Failed to count fish.\n\n{message}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    root = tk.Tk()

    try:
        style = ttk.Style()
        style.theme_use("clam")
    except Exception:
        pass

    app = VideoProcessorApp(root)
    root.mainloop()
