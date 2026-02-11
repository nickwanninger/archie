from textual.app import App, ComposeResult
from textual.containers import Container, Vertical, VerticalScroll
from textual.widgets import Input, Header, Footer, Static, Label, RichLog
from textual.reactive import reactive
from textual.worker import Worker, WorkerState
from textual.message import Message
from textual import work
from rich.text import Text
from rich.panel import Panel

import torch
import torch.nn.functional as F
import math
import sys
import io

# Import the model code
import mistral_model as mm


def prob_to_style(prob):
    """
    Convert probability to a rich style.
    Low prob -> Red background
    High prob -> Green background
    """
    # Clamp prob just in case
    prob = max(0.0, min(1.0, prob))

    if prob < 0.5:
        # Red to Yellow
        # r=255, g goes 0->255
        ratio = prob * 2
        r = 255
        g = int(255 * ratio)
        b = 0
    else:
        # Yellow to Green
        # g=255, r goes 255->0
        ratio = (prob - 0.5) * 2
        r = int(255 * (1 - ratio))
        g = 255
        b = 0

    return f"black on #{r:02x}{g:02x}{b:02x}"


class InferenceApp(App):
    CSS = """
    Screen {
        layers: base;
    }
    
    #chat-container {
        height: 1fr;
        border: solid green;
        padding: 1;
        overflow-y: scroll;
        scrollbar-size: 1 1;
    }

    #prompt-input {
        dock: bottom;
        margin: 1 0;
    }

    .user-message {
        background: $primary-darken-2;
        padding: 1;
        margin: 1;
        border-left: wide $primary;
    }

    .assistant-message {
        background: $surface;
        padding: 1;
        margin: 1;
        border-left: wide $secondary;
    }
    """

    BINDINGS = [("q", "quit", "Quit")]

    def __init__(self):
        super().__init__()
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.backends.mps.is_available():
            self.device = "mps"

        self.current_response_text = None
        self.current_response_widget = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield Vertical(
            VerticalScroll(id="chat-container"),
            Input(placeholder="Type prompt and press Enter...", id="prompt-input"),
        )
        yield Footer()

    def on_mount(self) -> None:
        self.load_model()
        self.query_one("#chat-container").mount(
            Static("Model Loading... Please wait.", id="loading-msg")
        )

    @work(exclusive=True, thread=True)
    def load_model(self):
        try:
            model, tokenizer = mm.load_model_from_checkpoint(
                checkpoint_path=None, device=self.device
            )
            model.eval()
            self.model = model
            self.tokenizer = tokenizer
            self.call_from_thread(self.model_loaded)
        except Exception as e:
            self.call_from_thread(self.notify, f"Error loading: {e}", severity="error")

    def model_loaded(self):
        self.notify("Model loaded successfully!")
        try:
            self.query_one("#loading-msg").remove()
        except:
            pass
        self.query_one("#chat-container").mount(
            Static("Model Ready. Type to generate text.", classes="assistant-message")
        )

    async def on_input_submitted(self, message: Input.Submitted) -> None:
        text = message.value
        if not text:
            return

        if not self.model:
            self.notify("Model is not loaded yet.", severity="warning")
            return

        message.input.value = ""

        # Add user message
        chat = self.query_one("#chat-container")
        await chat.mount(Static(f"User: {text}", classes="user-message"))

        # Prepare assistant message placeholder
        self.current_response_text = Text()
        self.current_response_widget = Static(
            self.current_response_text, classes="assistant-message"
        )
        await chat.mount(self.current_response_widget)

        # Scroll to bottom
        chat.scroll_end(animate=True)

        # Start generation
        self.generate_response(text)

    @work(exclusive=True, thread=True)
    def generate_response(self, prompt: str):
        try:
            steps = 512  # Limit generation length
            for chunk in mm.stream_text(
                self.model, self.tokenizer, prompt, steps=steps, temperature=0.8
            ):
                if chunk["text"] == "<|endoftext|>":
                    break
                self.call_from_thread(self.update_response, chunk)

            self.call_from_thread(self.finalize_response)

        except Exception as e:
            self.call_from_thread(
                self.notify, f"Error generating: {e}", severity="error"
            )

    def update_response(self, chunk):
        token = chunk["text"]
        prob = chunk["prob"]

        if token == "<|endoftext|>":
            self.current_response_text.append(" [end]", style="bold dim")
        else:
            style = prob_to_style(prob)
            self.current_response_text.append(token, style=style)

        if self.current_response_widget:
            self.current_response_widget.update(self.current_response_text)
            self.query_one("#chat-container").scroll_end(animate=False)

    def finalize_response(self):
        self.current_response_text = None
        self.current_response_widget = None


if __name__ == "__main__":
    app = InferenceApp()
    app.run()
