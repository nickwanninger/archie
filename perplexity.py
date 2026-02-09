from textual.app import App, ComposeResult
from textual.containers import Container, Vertical, Horizontal
from textual.widgets import Input, Header, Footer, Static, Label, RichLog
from textual.reactive import reactive
from textual.worker import Worker, WorkerState
from textual.message import Message
from textual import work
from textual.timer import Timer

import torch
import torch.nn.functional as F
import math
import sys
import io

# Import the model code
import mistral_model as mm


class PerplexityApp(App):
    CSS = """
    Screen {
        align: center middle;
    }
    
    #main-container {
        width: 100%;
        height: 100%;
        border: solid green;
        padding: 1 2;
    }

    #prompt-input {
        margin-bottom: 1;
    }

    .stat-box {
        background: $boost;
        padding: 1;
        margin-bottom: 1;
        border: tall $primary;
        height: auto;
    }

    #predictions-container {
        height: 1fr;
        border: solid blue;
        padding: 1;
        overflow-y: scroll;
    }

    .prediction-row {
        margin: 0 1; 
    }
    """

    BINDINGS = [("q", "quit", "Quit")]

    # Reactives
    perplexity = reactive(0.0)

    def __init__(self):
        super().__init__()
        self.debounce_timer = None
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.backends.mps.is_available():
            self.device = "mps"

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Label("Enter Prompt for Analysis:", classes="label"),
            Input(placeholder="Type here...", id="prompt-input"),
            Label("Stats", classes="label"),
            Static("Model Loading...", id="stats-view", classes="stat-box"),
            Label("Next Token Predictions (Top PPL*2)", classes="label"),
            RichLog(
                id="predictions-log",
                markup=True,
                highlight=True,
                wrap=True,
                auto_scroll=False,
            ),
            id="main-container",
        )
        yield Footer()

    def on_mount(self) -> None:
        self.load_model()

    @work(exclusive=True, thread=True)
    def load_model(self):
        try:
            # Capture stdout to avoid messing up the TUI during load
            # (Though textual might handle it, strictly speaking better to be safe)
            # Actually, we can let it print, it might show on the terminal startup or duplicate.
            # mm.load_model_from_checkpoint prints to console.

            model, tokenizer = mm.load_model_from_checkpoint(
                checkpoint_path=mm.CHECKPOINT_PATH, device=self.device
            )
            model.eval()
            self.model = model
            self.tokenizer = tokenizer
            self.call_from_thread(self.model_loaded)
        except Exception as e:
            self.call_from_thread(self.notify, f"Error loading: {e}", severity="error")

    def model_loaded(self):
        self.notify("Model loaded successfully!")
        self.query_one("#stats-view", Static).update("Model Ready. Type to analyze.")

    def on_input_changed(self, message: Input.Changed) -> None:
        if self.debounce_timer:
            self.debounce_timer.stop()

        # Only run analysis if model is loaded
        if self.model:
            self.debounce_timer = self.set_timer(
                0.2, lambda: self.analyze_text(message.value)
            )

    @work(exclusive=True, thread=True)
    def analyze_text(self, text: str):
        if not text:
            self.call_from_thread(self.update_stats, 0.0, [])
            return

        try:
            # Tokenize
            # Mistral tokenizer wrapper might vary, let's assume standard behavior
            # mm.get_tokenizer() returns tiktoken encoding
            tokens = self.tokenizer.encode(text)
            if not tokens:
                return

            input_tensor = torch.tensor(tokens).unsqueeze(0).to(self.device).long()

            with torch.no_grad():
                logits = self.model(input_tensor)

                # Calculate PPL
                ppl = 1.0
                if input_tensor.shape[1] > 1:
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = input_tensor[:, 1:].contiguous()
                    loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                    )
                    ppl = torch.exp(loss).item()
                else:
                    # Fallback for single token: use loss of predicting it from nothing?
                    # Or just 0.
                    pass

                # Next token predictions
                last_logits = logits[0, -1, :]
                probs = torch.softmax(last_logits, dim=-1)

                # Top K = PPL * 2
                k = max(5, int(ppl * 2))
                k = min(k, 50)  # Cap for performance/visuals

                top_probs, top_indices = torch.topk(probs, k)

                preds = []
                for p, idx in zip(top_probs, top_indices):
                    token = self.tokenizer.decode([idx.item()])
                    preds.append((token, p.item()))

                self.call_from_thread(self.update_stats, ppl, preds)

        except Exception as e:
            # self.call_from_thread(self.notify, f"Error: {e}", severity="error")
            pass

    def update_stats(self, ppl: float, preds: list):
        stats = self.query_one("#stats-view", Static)
        stats.update(f"Perplexity: {ppl:.4f}")

        log = self.query_one("#predictions-log", RichLog)
        log.clear()

        if not preds:
            log.write("No predictions.")
            return

        # Header for predictions
        log.write(f"[bold]Top {len(preds)} Predictions[/]")

        for token, prob in preds:
            # Escape textual markup
            safe_token = token.replace("[", "\\[").replace("]", "\\]")
            # Visualize bar
            bar_len = int(prob * 50)
            bar = "█" * bar_len
            log.write(f"{prob:6.2%} | [bold cyan]{safe_token}[/]  [dim]{bar}[/]")

        log.scroll_home(animate=False)


if __name__ == "__main__":
    app = PerplexityApp()
    app.run()
