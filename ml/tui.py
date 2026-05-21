#!/usr/bin/env python
"""Cancer Registry ML — Terminal UI

Run from the repository root:
    ml/.venv/Scripts/python.exe ml/tui.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import config
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Button, Footer, Header, Input, Label,
    RichLog, Select, Switch, TabbedContent, TabPane,
)

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
_ROOT    = Path(__file__).parent.parent
_PYTHON  = str(Path(__file__).parent / ".venv" / "Scripts" / "python.exe")
_SCRIPTS = Path(__file__).parent / "scripts"

CHECKPOINT_CONTRASTIVE = config.CHECKPOINT_CONTRASTIVE_DIR
CHECKPOINT_GROUP       = config.CHECKPOINT_GROUP_DIR
ANNOTATION_CSV         = config.ANNOTATION_CSV
LLM_ANNOTATION_CSV     = config.LLM_ANNOTATION_CSV
OUTPUT_PRODUCTION_DIR  = config.OUTPUT_PRODUCTION_DIR

_DEVICE_OPTIONS: list[tuple[str, str]] = [
    ("xpu", "xpu"), ("cuda", "cuda"), ("cpu", "cpu"), ("auto", "auto"),
]

_ANNOTATION_OPTIONS: list[tuple[str, str]] = [
    ("unified", ANNOTATION_CSV),
    ("llm",     LLM_ANNOTATION_CSV),
]

_TRAINING_MODES: list[tuple[str, str]] = [
    ("Group Classifier",        "train-groups"),
    ("Contrastive Fine-Tuning", "adapt-backbone"),
]


_MODE_SECTION = {
    "adapt-backbone":   "backbone",
    "train-groups":     "groups",
}


# ---------------------------------------------------------------------------
# Cold-start confirm modal
# ---------------------------------------------------------------------------
class ConfirmModal(ModalScreen[bool]):
    DEFAULT_CSS = """
    ConfirmModal { align: center middle; }
    #dialog {
        padding: 1 2; background: $surface; border: solid $accent;
        width: 60; height: auto;
    }
    #dialog Label { margin-bottom: 1; }
    #dialog Horizontal { align: center middle; }
    #dialog Button { margin: 0 1; }
    """

    def __init__(self, message: str) -> None:
        super().__init__()
        self._message = message

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label(self._message)
            with Horizontal():
                yield Button("Confirm", id="yes", variant="error")
                yield Button("Cancel",  id="no")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "yes")


# ---------------------------------------------------------------------------
# Annotate tab  (run_annotation.py)
# ---------------------------------------------------------------------------
class AnnotateView(VerticalScroll):
    def compose(self) -> ComposeResult:
        yield Label("Max rows  [dim](blank = all)[/]")
        yield Input(placeholder="e.g. 100", id="ann-maxrows")
        yield Label("LLM model")
        yield Input(placeholder="e.g. mistral", id="ann-llm-model")
        yield Label("LLM timeout (s)")
        yield Input(value="60", id="ann-llm-timeout")
        yield Label("List models  [dim](lists and exits)[/]")
        yield Switch(value=False, id="ann-list-models")
        yield Label("Compare models  [dim](runs all models on max-rows rows)[/]")
        yield Switch(value=False, id="ann-compare-models")
        yield Button("Run Annotation", id="btn-annotate-run", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "btn-annotate-run":
            return
        event.stop()
        cmd = [_PYTHON, str(_SCRIPTS / "run_annotation.py")]
        maxrows = self.query_one("#ann-maxrows", Input).value.strip()
        if maxrows:
            cmd += ["--max-rows", maxrows]
        model = self.query_one("#ann-llm-model", Input).value.strip()
        if model:
            cmd += ["--model", model]
        timeout = self.query_one("#ann-llm-timeout", Input).value.strip()
        if timeout:
            cmd += ["--llm-timeout", timeout]
        if self.query_one("#ann-list-models", Switch).value:
            cmd.append("--list-models")
        if self.query_one("#ann-compare-models", Switch).value:
            cmd.append("--compare-models")
        self.app.run_cmd(cmd)


# ---------------------------------------------------------------------------
# Training tab  (run_training.py — all modes)
# ---------------------------------------------------------------------------
class TrainingView(VerticalScroll):
    def compose(self) -> ComposeResult:
        yield Label("Mode")
        yield Select(_TRAINING_MODES, value="train-groups", id="tr-mode")

        # ── adapt-backbone ───────────────────────────────────────────────
        with Vertical(id="sec-backbone"):
            yield Label("Epochs  [dim]fine-tuning passes — 3 avoids overfitting the small labelled set[/]")
            yield Input(value="3", id="bb-epochs")
            yield Label("Batch size  [dim]in-batch negatives for InfoNCE loss — larger = harder loss[/]")
            yield Input(value="32", id="bb-batch")
            yield Label("LR  [dim]peak learning rate — 2e-5 is safe to avoid catastrophic forgetting[/]")
            yield Input(value="2e-5", id="bb-lr")
            yield Label("Temperature  [dim]InfoNCE sharpness — lower = harder loss; 0.07 default[/]")
            yield Input(value="0.07", id="bb-temp")
            yield Label("Max length  [dim]BERT token limit per report — 256 default[/]")
            yield Input(value="256", id="bb-maxlen")
            yield Label("Device")
            yield Select(_DEVICE_OPTIONS, value="xpu", id="bb-device")
            yield Label("Local only");      yield Switch(value=True,  id="bb-local-only")
            yield Label("Skip pair build"); yield Switch(value=False, id="bb-skip-pairs")
            yield Label("[dim]── Hard negatives (optional) ──[/]")
            yield Label("Hard-neg CSV  [dim](leave blank to skip)[/]")
            yield Input(placeholder="ml/output/training/contrastive/hard_neg_pairs.csv", id="bb-hard-csv")
            yield Label("Weight"); yield Input(value="0.5", id="bb-hard-weight")
            yield Label("Margin"); yield Input(value="0.3", id="bb-hard-margin")
            yield Button("Run Fine-tune", id="btn-backbone-run", variant="primary")

        # ── train-groups ─────────────────────────────────────────────────
        with Vertical(id="sec-groups"):
            yield Label("Epochs  [dim]one-shot training — 50 default; revisit at ~15k cases[/]")
            yield Input(value="50", id="gr-epochs")
            yield Label("LR  [dim]peak learning rate with cosine schedule — 5e-5 default[/]")
            yield Input(value="5e-5", id="gr-lr")
            yield Label("Annotation CSV")
            yield Select(_ANNOTATION_OPTIONS, value=ANNOTATION_CSV, id="gr-ann-csv")
            yield Label("Device")
            yield Select(_DEVICE_OPTIONS, value="xpu", id="gr-device")
            yield Label("Local only"); yield Switch(value=True, id="gr-local-only")
            yield Button("Run Train Groups", id="btn-groups-run", variant="primary")


    def on_mount(self) -> None:
        self._show_section("groups")

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "tr-mode":
            self._show_section(_MODE_SECTION.get(str(event.value), "groups"))

    def _show_section(self, active: str) -> None:
        for name in _MODE_SECTION.values():
            self.query_one(f"#sec-{name}").display = (name == active)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "btn-cold-start":
            event.stop()
            self.app.push_screen(
                ConfirmModal("Delete embedding cache?"),
                self._cold_start,
            )
        elif bid == "btn-backbone-run":
            event.stop(); self._run_backbone()
        elif bid == "btn-groups-run":
            event.stop(); self._run_groups()

    def _cold_start(self, confirmed: bool) -> None:
        if not confirmed:
            return
        log = self.app.query_one(RichLog)
        log.write("[yellow]── Cold start ──[/]")
        rel = config.EMBEDDING_CACHE_NPZ
        p = _ROOT / rel
        if p.exists():
            p.unlink()
            log.write(f"  [red]deleted[/]  {rel}")
        else:
            log.write(f"  [dim]missing[/]  {rel}")
        log.write("[green]── Done ──[/]")

    def _run_backbone(self) -> None:
        g = lambda wid: self.query_one(wid, Input).value.strip()
        cmd = [
            _PYTHON, str(_SCRIPTS / "run_training.py"),
            "--mode", "adapt-backbone",
            "--epochs",      g("#bb-epochs"),
            "--batch-size",  g("#bb-batch"),
            "--lr",          g("#bb-lr"),
            "--temperature", g("#bb-temp"),
            "--max-length",  g("#bb-maxlen"),
            "--device",      str(self.query_one("#bb-device", Select).value),
        ]
        if self.query_one("#bb-local-only", Switch).value:
            cmd.append("--local-only")
        if self.query_one("#bb-skip-pairs", Switch).value:
            cmd.append("--skip-pair-build")
        hard_csv = g("#bb-hard-csv")
        if hard_csv:
            cmd += ["--hard-neg-csv", hard_csv,
                    "--hard-neg-weight", g("#bb-hard-weight"),
                    "--hard-neg-margin", g("#bb-hard-margin")]
        self.app.run_cmd(cmd)

    def _run_groups(self) -> None:
        g = lambda wid: self.query_one(wid, Input).value.strip()
        cmd = [
            _PYTHON, str(_SCRIPTS / "run_training.py"),
            "--mode", "train-groups",
            "--epochs",        g("#gr-epochs"),
            "--lr",            g("#gr-lr"),
            "--annotation-csv", str(self.query_one("#gr-ann-csv", Select).value),
            "--device",        str(self.query_one("#gr-device", Select).value),
        ]
        if self.query_one("#gr-local-only", Switch).value:
            cmd.append("--local-only")
        self.app.run_cmd(cmd)


# ---------------------------------------------------------------------------
# Evaluate tab  (run_evaluation.py)
# ---------------------------------------------------------------------------
class EvaluateView(VerticalScroll):
    def compose(self) -> ComposeResult:
        yield Label("Label  [dim](optional)[/]")
        yield Input(placeholder="e.g. manual check", id="ev-label")
        yield Label("Annotation CSV")
        yield Select(_ANNOTATION_OPTIONS, value=ANNOTATION_CSV, id="ev-ann-csv")
        yield Label("Prediction CSV  [dim](blank = auto-detect)[/]")
        yield Input(placeholder="ml/output/production/.../petbert_predictions.csv", id="ev-pred-csv")
        yield Label("Out dir  [dim](blank = auto-detect)[/]")
        yield Input(placeholder="ml/output/evaluation/...", id="ev-out-dir")
        yield Button("Run Evaluate", id="btn-evaluate-run", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "btn-evaluate-run":
            return
        event.stop()
        g = lambda wid: self.query_one(wid, Input).value.strip()
        cmd = [
            _PYTHON, str(_SCRIPTS / "run_evaluation.py"),
            "--annotation-csv", str(self.query_one("#ev-ann-csv", Select).value),
        ]
        label = g("#ev-label")
        if label:
            cmd += ["--label", label]
        pred_csv = g("#ev-pred-csv")
        if pred_csv:
            cmd += ["--prediction-csv", pred_csv]
        out_dir = g("#ev-out-dir")
        if out_dir:
            cmd += ["--out-dir", out_dir]
        self.app.run_cmd(cmd)


# ---------------------------------------------------------------------------
# Production tab  (run_production.py)
# ---------------------------------------------------------------------------
class ProductionView(VerticalScroll):
    def compose(self) -> ComposeResult:
        yield Label("Group classifier threshold  [dim](0.85 is current production)[/]")
        yield Input(value="0.85", id="prod-group-threshold")
        yield Label("Label-presence threshold  [dim](Stage 3a — 0.5 default)[/]")
        yield Input(value="0.5", id="prod-lp-threshold")
        yield Label("Device")
        yield Select(_DEVICE_OPTIONS, value="xpu", id="prod-device")
        yield Label("Local only"); yield Switch(value=True, id="prod-local-only")
        yield Label("Max rows  [dim](blank = all)[/]")
        yield Input(placeholder="e.g. 100", id="prod-maxrows")
        yield Button("Run Production", id="btn-prod-run", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "btn-prod-run":
            return
        event.stop()

        device = str(self.query_one("#prod-device", Select).value)
        cmd = [
            _PYTHON, str(_SCRIPTS / "run_production.py"),
            "--device", device,
            "--group-classifier-threshold",
            self.query_one("#prod-group-threshold", Input).value.strip() or "0.85",
            "--label-presence-threshold",
            self.query_one("#prod-lp-threshold", Input).value.strip() or "0.5",
        ]
        if self.query_one("#prod-local-only", Switch).value:
            cmd.append("--local-only")
        maxrows = self.query_one("#prod-maxrows", Input).value.strip()
        if maxrows:
            cmd += ["--max-rows", maxrows]
        self.app.run_cmd(cmd)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
class MLTui(App):
    TITLE = "Cancer Registry ML"
    CSS = """
    TabbedContent { height: 1fr; }
    TabPane { padding: 1 2; overflow-y: auto; }

    Horizontal { height: auto; }
    Vertical   { height: auto; }

    Label  { margin-top: 1; }
    Input  { width: 50; }
    Switch { margin-top: 1; }
    Select { width: 50; }
    Button { margin-top: 1; margin-right: 1; }

    RichLog {
        height: 15;
        border: solid $accent;
        margin: 0 1 1 1;
    }
    """

    BINDINGS = [
        ("q",      "quit",      "Quit"),
        ("ctrl+k", "kill_proc", "Kill process"),
        ("ctrl+l", "clear_log", "Clear log"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._proc: asyncio.subprocess.Process | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent():
            with TabPane("Annotate",   id="tab-annotate"):   yield AnnotateView()
            with TabPane("Training",   id="tab-training"):   yield TrainingView()
            with TabPane("Evaluate",   id="tab-evaluate"):   yield EvaluateView()
            with TabPane("Production", id="tab-production"): yield ProductionView()
        yield RichLog(id="log", highlight=True, markup=True)
        yield Footer()

    def on_mount(self) -> None:
        self.query_one(RichLog).border_title = "LOG  (ctrl+k kill · ctrl+l clear)"

    def action_kill_proc(self) -> None:
        if self._proc and self._proc.returncode is None:
            self._proc.kill()
            self.query_one(RichLog).write("[yellow]─── Killed ───[/]")
            self._proc = None

    def action_clear_log(self) -> None:
        self.query_one(RichLog).clear()

    @work(exclusive=True)
    async def run_cmd(self, cmd: list[str]) -> None:
        log = self.query_one(RichLog)
        log.write(f"[bold cyan]$ {' '.join(cmd)}[/]")
        self._proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(_ROOT),
        )
        async for raw in self._proc.stdout:
            log.write(raw.decode(errors="replace").rstrip())
        rc = await self._proc.wait()
        self._proc = None
        log.write(f"[{'green' if rc == 0 else 'red'}]─── Exit {rc} ───[/]")


if __name__ == "__main__":
    MLTui().run()
