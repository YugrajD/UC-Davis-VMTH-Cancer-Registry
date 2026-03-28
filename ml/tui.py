#!/usr/bin/env python
"""Cancer Registry ML — Terminal UI

Run from the repository root:
    ml/.venv/Scripts/python.exe ml/tui.py
"""

from __future__ import annotations

import asyncio
import csv
from pathlib import Path

import config
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import (
    Button, Footer, Header, Input, Label,
    RichLog, Select, Static, Switch, TabbedContent, TabPane,
)

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
_ROOT    = Path(__file__).parent.parent
_PYTHON  = str(Path(__file__).parent / ".venv" / "Scripts" / "python.exe")
_SCRIPTS = Path(__file__).parent / "scripts"

CHECKPOINT_CONTRASTIVE = config.CHECKPOINT_CONTRASTIVE_DIR
CHECKPOINT_BINARY      = config.CHECKPOINT_BINARY_DIR
KEYWORD_ANNOTATION_CSV = config.KEYWORD_ANNOTATION_CSV
CO_NEG_BANK_CSV        = f"{config.OUTPUT_TRAINING_DIR}/binary/evaluation_co_bank.csv"
EVAL_HISTORY_CSV       = f"{config.OUTPUT_EVALUATION_DIR}/contrastive/evaluation_history.csv"

_COLD_START_FILES = [
    config.EMBEDDING_CACHE_NPZ,
    f"{config.OUTPUT_TRAINING_DIR}/contrastive/evaluation_co_bank.csv",
    f"{config.CHECKPOINT_CONTRASTIVE_DIR}/presence_classifier_current.pt",
]

_DEVICE_OPTIONS: list[tuple[str, str]] = [
    ("xpu", "xpu"), ("cuda", "cuda"), ("cpu", "cpu"), ("auto", "auto"),
]


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
# Status view
# ---------------------------------------------------------------------------
class StatusView(Static):
    def on_mount(self) -> None:
        self.update(self._content())

    def _content(self) -> str:
        lines: list[str] = []

        # Best checkpoint
        found = False
        for subdir, ckpt_dir in [("contrastive", CHECKPOINT_CONTRASTIVE),
                                   ("binary", CHECKPOINT_BINARY)]:
            p = _ROOT / f"{ckpt_dir}/presence_classifier_best.pt"
            if p.exists():
                lines.append(f"[bold green]Best checkpoint:[/] {ckpt_dir}/presence_classifier_best.pt [{subdir}]")
                found = True
                break
        if not found:
            lines.append("[bold red]No checkpoint found.[/]")

        # Last 5 rows of eval history
        hist = _ROOT / EVAL_HISTORY_CSV
        if hist.exists():
            lines.append("\n[bold]Recent evaluation history:[/]")
            with hist.open(newline="") as f:
                rows = list(csv.reader(f))
            if len(rows) > 1:
                header = rows[0]
                for row in rows[-5:]:
                    lines.append("  " + "  |  ".join(
                        f"[cyan]{h}[/]={v}" for h, v in zip(header, row)
                    ))
        else:
            lines.append(f"\n[dim]No eval history yet at {EVAL_HISTORY_CSV}[/]")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Annotate view
# ---------------------------------------------------------------------------
class AnnotateView(Static):
    def compose(self) -> ComposeResult:
        yield Label("Method")
        yield Select(
            [("keyword — fast rule-based", "keyword"), ("llm — Ollama/Claude", "llm")],
            value="keyword", id="ann-method",
        )
        yield Label("Ollama model  [dim](llm only)[/]")
        yield Input(placeholder="e.g. mistral", id="ann-llm-model")
        yield Label("LLM timeout (s)  [dim](llm only)[/]")
        yield Input(value="60", id="ann-llm-timeout")
        yield Label("Use Claude fallback  [dim](llm only)[/]")
        yield Switch(value=False, id="ann-use-claude")
        yield Button("Run Annotation", id="btn-annotate-run", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "btn-annotate-run":
            return
        event.stop()
        method = str(self.query_one("#ann-method", Select).value)
        cmd = [_PYTHON, str(_SCRIPTS / "run_annotation.py"), "--method", method]
        if method == "llm":
            model = self.query_one("#ann-llm-model", Input).value.strip()
            if model:
                cmd += ["--model", model]
            timeout = self.query_one("#ann-llm-timeout", Input).value.strip()
            if timeout:
                cmd += ["--llm-timeout", timeout]
            if self.query_one("#ann-use-claude", Switch).value:
                cmd.append("--use-claude")
        self.app.run_cmd(cmd)


# ---------------------------------------------------------------------------
# Backbone view
# ---------------------------------------------------------------------------
class BackboneView(Static):
    def compose(self) -> ComposeResult:
        yield Label("Epochs");      yield Input(value="3",    id="bb-epochs")
        yield Label("Batch size");  yield Input(value="32",   id="bb-batch")
        yield Label("LR");          yield Input(value="2e-5", id="bb-lr")
        yield Label("Temperature"); yield Input(value="0.07", id="bb-temp")
        yield Label("Max length");  yield Input(value="256",  id="bb-maxlen")
        yield Label("Device")
        yield Select(_DEVICE_OPTIONS, value="xpu", id="bb-device")
        yield Label("Local only");     yield Switch(value=True,  id="bb-local-only")
        yield Label("Skip pair build"); yield Switch(value=False, id="bb-skip-pairs")
        yield Label("[dim]── Hard negatives (optional) ──[/]")
        yield Label("Hard-neg CSV  [dim](leave blank to skip)[/]")
        yield Input(placeholder="ml/data/hard_neg_pairs.csv", id="bb-hard-csv")
        yield Label("Weight"); yield Input(value="0.5", id="bb-hard-weight")
        yield Label("Margin"); yield Input(value="0.3", id="bb-hard-margin")
        yield Button("Run Adapt Backbone", id="btn-backbone-run", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "btn-backbone-run":
            return
        event.stop()
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


# ---------------------------------------------------------------------------
# Classifier view  (most-used tab)
# ---------------------------------------------------------------------------
class ClassifierView(Static):
    def compose(self) -> ComposeResult:
        yield Label("Label");          yield Input(value="",                     placeholder="e.g. c21", id="cl-label")
        yield Label("Epochs");         yield Input(value="25",                   id="cl-epochs")
        yield Label("Hidden dim");     yield Input(value="512",                  id="cl-hidden")
        yield Label("CO neg/case");    yield Input(value="5",                    id="cl-co-neg")
        yield Label("FP neg/case");    yield Input(value="10",                   id="cl-fp-neg")
        yield Label("Recall weight");  yield Input(value="0.25",                 id="cl-recall")
        yield Label("Min similarity"); yield Input(value="0.05",                 id="cl-minsim")
        yield Label("Model path");     yield Input(value=CHECKPOINT_CONTRASTIVE, id="cl-model")
        yield Label("CO bank CSV");    yield Input(value=CO_NEG_BANK_CSV,        id="cl-co-bank")
        yield Label("Device")
        yield Select(_DEVICE_OPTIONS, value="xpu", id="cl-device")
        yield Label("Local only"); yield Switch(value=True, id="cl-local-only")
        with Horizontal():
            yield Button("Run Cycle",  id="btn-classifier-run", variant="primary")
            yield Button("Cold Start", id="btn-cold-start",     variant="error")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-classifier-run":
            event.stop()
            self._run()
        elif event.button.id == "btn-cold-start":
            event.stop()
            self.app.push_screen(
                ConfirmModal("Delete embedding cache + current checkpoint?"),
                self._cold_start,
            )

    def _run(self) -> None:
        g = lambda wid: self.query_one(wid, Input).value.strip()
        cmd = [
            _PYTHON, str(_SCRIPTS / "run_training.py"),
            "--mode", "train-classifier",
            "--model",             g("#cl-model"),
            "--epochs",            g("#cl-epochs"),
            "--hidden-dim",        g("#cl-hidden"),
            "--co-neg-per-case",   g("#cl-co-neg"),
            "--fp-neg-per-case",   g("#cl-fp-neg"),
            "--recall-weight",     g("#cl-recall"),
            "--embedding-min-sim", g("#cl-minsim"),
            "--device",            str(self.query_one("#cl-device", Select).value),
            "--co-neg-bank-csv",   g("#cl-co-bank"),
        ]
        label = g("#cl-label")
        if label:
            cmd += ["--label", label]
        if self.query_one("#cl-local-only", Switch).value:
            cmd.append("--local-only")
        self.app.run_cmd(cmd)

    def _cold_start(self, confirmed: bool) -> None:
        if not confirmed:
            return
        log = self.app.query_one(RichLog)
        log.write("[yellow]── Cold start ──[/]")
        for rel in _COLD_START_FILES:
            p = _ROOT / rel
            if p.exists():
                p.unlink()
                log.write(f"  [red]deleted[/]  {rel}")
            else:
                log.write(f"  [dim]missing[/]  {rel}")
        log.write("[green]── Done ──[/]")


# ---------------------------------------------------------------------------
# Evaluate view
# ---------------------------------------------------------------------------
class EvaluateView(Static):
    def compose(self) -> ComposeResult:
        yield Label("Label  [dim](optional)[/]")
        yield Input(placeholder="e.g. manual check", id="ev-label")
        yield Label("Annotation CSV")
        yield Input(value=KEYWORD_ANNOTATION_CSV, id="ev-ann-csv")
        yield Button("Run Evaluate", id="btn-evaluate-run", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "btn-evaluate-run":
            return
        event.stop()
        g = lambda wid: self.query_one(wid, Input).value.strip()
        cmd = [_PYTHON, str(_SCRIPTS / "run_evaluation.py"),
               "--annotation-csv", g("#ev-ann-csv")]
        label = g("#ev-label")
        if label:
            cmd += ["--label", label]
        self.app.run_cmd(cmd)


# ---------------------------------------------------------------------------
# Production view
# ---------------------------------------------------------------------------
class ProductionView(Static):
    def compose(self) -> ComposeResult:
        yield Label("Max rows  [dim](blank = all)[/]")
        yield Input(placeholder="e.g. 100", id="prod-maxrows")
        yield Button("Run Production", id="btn-prod-run", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "btn-prod-run":
            return
        event.stop()
        cmd = [_PYTHON, str(_SCRIPTS / "run_production.py")]
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
            with TabPane("Status",     id="tab-status"):     yield StatusView()
            with TabPane("Annotate",   id="tab-annotate"):   yield AnnotateView()
            with TabPane("Backbone",   id="tab-backbone"):   yield BackboneView()
            with TabPane("Classifier", id="tab-classifier"): yield ClassifierView()
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
