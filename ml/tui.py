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

CHECKPOINT_BINARY      = config.CHECKPOINT_BINARY_DIR
CHECKPOINT_CONTRASTIVE = config.CHECKPOINT_CONTRASTIVE_DIR
CHECKPOINT_GROUP       = config.CHECKPOINT_GROUP_DIR
KEYWORD_ANNOTATION_CSV = config.KEYWORD_ANNOTATION_CSV
LLM_ANNOTATION_CSV     = config.LLM_ANNOTATION_CSV
CO_NEG_BANK_CSV        = f"{config.OUTPUT_TRAINING_DIR}/binary/evaluation_co_bank.csv"
OUTPUT_PRODUCTION_DIR  = config.OUTPUT_PRODUCTION_DIR

_COLD_START_FILES = [
    config.EMBEDDING_CACHE_NPZ,
    f"{config.OUTPUT_TRAINING_DIR}/contrastive/evaluation_co_bank.csv",
    f"{config.CHECKPOINT_CONTRASTIVE_DIR}/presence_classifier_current.pt",
]

_DEVICE_OPTIONS: list[tuple[str, str]] = [
    ("xpu", "xpu"), ("cuda", "cuda"), ("cpu", "cpu"), ("auto", "auto"),
]

_ANNOTATION_OPTIONS: list[tuple[str, str]] = [
    ("keyword", KEYWORD_ANNOTATION_CSV),
    ("llm",     LLM_ANNOTATION_CSV),
]

_TRAINING_MODES: list[tuple[str, str]] = [
    ("train-classifier — label presence model",  "train-classifier"),
    ("adapt-backbone — fine-tune PetBERT",        "adapt-backbone"),
    ("train-groups — group classifier",           "train-groups"),
]


_MODE_SECTION = {
    "train-classifier": "classifier",
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
        yield Label("Method")
        yield Select(
            [("keyword — fast rule-based", "keyword"), ("llm — Ollama", "llm")],
            value="keyword", id="ann-method",
        )
        yield Label("Max rows  [dim](blank = all)[/]")
        yield Input(placeholder="e.g. 100", id="ann-maxrows")
        yield Label("Ollama model  [dim](llm only)[/]")
        yield Input(placeholder="e.g. mistral", id="ann-llm-model")
        yield Label("LLM timeout (s)  [dim](llm only)[/]")
        yield Input(value="60", id="ann-llm-timeout")
        yield Label("List models  [dim](llm only — lists and exits)[/]")
        yield Switch(value=False, id="ann-list-models")
        yield Label("Compare models  [dim](llm only — runs all models on max-rows rows)[/]")
        yield Switch(value=False, id="ann-compare-models")
        yield Button("Run Annotation", id="btn-annotate-run", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "btn-annotate-run":
            return
        event.stop()
        method = str(self.query_one("#ann-method", Select).value)
        cmd = [_PYTHON, str(_SCRIPTS / "run_annotation.py"), "--method", method]
        maxrows = self.query_one("#ann-maxrows", Input).value.strip()
        if maxrows:
            cmd += ["--max-rows", maxrows]
        if method == "llm":
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
        yield Select(_TRAINING_MODES, value="train-classifier", id="tr-mode")

        # ── train-classifier ─────────────────────────────────────────────
        with Vertical(id="sec-classifier"):
            yield Label("Label");          yield Input(value="", placeholder="e.g. c21", id="cl-label")
            yield Label("Epochs");         yield Input(value="25",   id="cl-epochs")
            yield Label("Hidden dim");     yield Input(value="512",  id="cl-hidden")
            yield Label("CO neg/case");    yield Input(value="5",    id="cl-co-neg")
            yield Label("FP neg/case");    yield Input(value="10",   id="cl-fp-neg")
            yield Label("Recall weight");  yield Input(value="0.25", id="cl-recall")
            yield Label("Min similarity"); yield Input(value="0.05", id="cl-minsim")
            yield Label("Model path");     yield Input(value=CHECKPOINT_CONTRASTIVE, id="cl-model")
            yield Label("CO bank CSV");    yield Input(value=CO_NEG_BANK_CSV, id="cl-co-bank")
            yield Label("Annotation CSV")
            yield Select(_ANNOTATION_OPTIONS, value=KEYWORD_ANNOTATION_CSV, id="cl-ann-csv")
            yield Label("Device")
            yield Select(_DEVICE_OPTIONS, value="xpu", id="cl-device")
            yield Label("Local only"); yield Switch(value=True, id="cl-local-only")
            with Horizontal():
                yield Button("Run Cycle",  id="btn-classifier-run", variant="primary")
                yield Button("Cold Start", id="btn-cold-start",     variant="error")

        # ── adapt-backbone ───────────────────────────────────────────────
        with Vertical(id="sec-backbone"):
            yield Label("Epochs");      yield Input(value="3",    id="bb-epochs")
            yield Label("Batch size");  yield Input(value="32",   id="bb-batch")
            yield Label("LR");          yield Input(value="2e-5", id="bb-lr")
            yield Label("Temperature"); yield Input(value="0.07", id="bb-temp")
            yield Label("Max length");  yield Input(value="256",  id="bb-maxlen")
            yield Label("Device")
            yield Select(_DEVICE_OPTIONS, value="xpu", id="bb-device")
            yield Label("Local only");      yield Switch(value=True,  id="bb-local-only")
            yield Label("Skip pair build"); yield Switch(value=False, id="bb-skip-pairs")
            yield Label("[dim]── Hard negatives (optional) ──[/]")
            yield Label("Hard-neg CSV  [dim](leave blank to skip)[/]")
            yield Input(placeholder="ml/data/hard_neg_pairs.csv", id="bb-hard-csv")
            yield Label("Weight"); yield Input(value="0.5", id="bb-hard-weight")
            yield Label("Margin"); yield Input(value="0.3", id="bb-hard-margin")
            yield Button("Run Fine-tune", id="btn-backbone-run", variant="primary")

        # ── train-groups ─────────────────────────────────────────────────
        with Vertical(id="sec-groups"):
            yield Label("Epochs"); yield Input(value="50",   id="gr-epochs")
            yield Label("LR");     yield Input(value="5e-5", id="gr-lr")
            yield Label("Annotation CSV")
            yield Select(_ANNOTATION_OPTIONS, value=KEYWORD_ANNOTATION_CSV, id="gr-ann-csv")
            yield Label("Device")
            yield Select(_DEVICE_OPTIONS, value="xpu", id="gr-device")
            yield Label("Local only"); yield Switch(value=True, id="gr-local-only")
            yield Button("Run Train Groups", id="btn-groups-run", variant="primary")


    def on_mount(self) -> None:
        self._show_section("classifier")

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id != "tr-mode":
            return
        self._show_section(_MODE_SECTION.get(str(event.value), "classifier"))

    def _show_section(self, active: str) -> None:
        for name in _MODE_SECTION.values():
            self.query_one(f"#sec-{name}").display = (name == active)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "btn-classifier-run":
            event.stop(); self._run_classifier()
        elif bid == "btn-cold-start":
            event.stop()
            self.app.push_screen(
                ConfirmModal("Delete embedding cache + current checkpoint?"),
                self._cold_start,
            )
        elif bid == "btn-backbone-run":
            event.stop(); self._run_backbone()
        elif bid == "btn-groups-run":
            event.stop(); self._run_groups()

    def _run_classifier(self) -> None:
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
            "--annotation-csv",    str(self.query_one("#cl-ann-csv", Select).value),
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
        yield Select(_ANNOTATION_OPTIONS, value=KEYWORD_ANNOTATION_CSV, id="ev-ann-csv")
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
_MODEL_OPTIONS: list[tuple[str, str]] = [
    ("Contrastive PetBERT  [dim](fine-tuned)[/]", "contrastive"),
    ("Default PetBERT",                            "petbert"),
]

_CLASSIFIER_OPTIONS: list[tuple[str, str]] = [
    ("Binary presence classifier", "binary"),
    ("Group classifier",           "group"),
]


class ProductionView(VerticalScroll):
    def compose(self) -> ComposeResult:
        yield Label("Model")
        yield Select(_MODEL_OPTIONS, value="contrastive", id="prod-model")
        yield Label("Classifier", id="prod-clf-label")
        yield Select(_CLASSIFIER_OPTIONS, value="binary", id="prod-classifier")
        yield Label("Group keyword  [dim](stage 2 behavior matching)[/]", id="prod-kw-label")
        yield Switch(value=True, id="prod-kw")
        yield Label("Device")
        yield Select(_DEVICE_OPTIONS, value="xpu", id="prod-device")
        yield Label("Local only"); yield Switch(value=True, id="prod-local-only")
        yield Label("Max rows  [dim](blank = all)[/]")
        yield Input(placeholder="e.g. 100", id="prod-maxrows")
        yield Button("Run Production", id="btn-prod-run", variant="primary")

    def on_mount(self) -> None:
        self._sync_visibility()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id in ("prod-model", "prod-classifier"):
            self._sync_visibility()

    def _sync_visibility(self) -> None:
        is_contrastive = self.query_one("#prod-model", Select).value == "contrastive"
        # Group classifier only exists for default PetBERT
        self.query_one("#prod-clf-label").display  = not is_contrastive
        self.query_one("#prod-classifier").display = not is_contrastive
        # Group keyword only applies to binary classifier
        clf_key   = str(self.query_one("#prod-classifier", Select).value)
        is_binary = is_contrastive or clf_key == "binary"
        self.query_one("#prod-kw-label").display = is_binary
        self.query_one("#prod-kw").display        = is_binary

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "btn-prod-run":
            return
        event.stop()

        model_key  = str(self.query_one("#prod-model",      Select).value)
        clf_key    = "binary" if model_key == "contrastive" else str(self.query_one("#prod-classifier", Select).value)
        device     = str(self.query_one("#prod-device",      Select).value)
        group_kw   = self.query_one("#prod-kw", Switch).value and clf_key == "binary"

        model_path = CHECKPOINT_CONTRASTIVE if model_key == "contrastive" else "SAVSNET/PetBERT"
        model_slug = "contrastive" if model_key == "contrastive" else "binary"

        cmd = [_PYTHON, str(_SCRIPTS / "run_production.py"), "--device", device,
               "--model", model_path]

        if clf_key == "group":
            cmd += [
                "--group-classifier", f"{CHECKPOINT_GROUP}/group_classifier_best.pt",
                "--out-dir", f"{OUTPUT_PRODUCTION_DIR}/{model_slug}_group",
            ]
        else:
            ckpt_dir = CHECKPOINT_CONTRASTIVE if model_key == "contrastive" else CHECKPOINT_BINARY
            cat_mode = "group-keyword" if group_kw else "default"
            out_slug  = f"{model_slug}_kw" if group_kw else model_slug
            cmd += [
                "--presence-classifier", f"{ckpt_dir}/presence_classifier_best.pt",
                "--categorization-mode", cat_mode,
                "--out-dir",             f"{OUTPUT_PRODUCTION_DIR}/{out_slug}",
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
