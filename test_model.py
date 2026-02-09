import mistral_model as mistral_inference
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.console import Group
from rich.text import Text
import torchinfo
import torch.nn.functional as F  # Added as per instruction

# Import from our model definition
from mistral_model import (
    MistralLite,
    ModelArgs,
    CHECKPOINT_PATH,
    get_tokenizer,
    MODEL_ARGS,
)

console = Console()


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


console.print("[bold yellow]Loading model...[/]")
model, tokenizer = mistral_inference.load_model_from_checkpoint(
    checkpoint_path=mistral_inference.CHECKPOINT_PATH, device="cuda"
)
console.print("[bold green]Model loaded![/]")


torchinfo.summary(model)

console.print("[bold]Enter your prompt below. Type 'exit' or 'quit' to stop.[/]")


while True:
    try:
        prompt = console.input(">>> ")
        if prompt.lower() in ["exit", "quit"]:
            break

        steps = 1024

        generated_text = Text()
        generated_text.append(prompt, style="bold dim")

        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
        )
        task = progress.add_task("Generating...", total=steps)

        with Live(
            Group(Panel(generated_text, title="Output"), progress),
            console=console,
            refresh_per_second=10,
            vertical_overflow="visible",
        ):
            for chunk in mistral_inference.stream_text(
                model, tokenizer, prompt, steps=steps, temperature=0.8
            ):
                token = chunk["text"]
                prob = chunk["prob"]

                if token == "<|endoftext|>":
                    generated_text.append(" [model ended output]", style="bold dim")
                    break

                style = prob_to_style(prob)
                generated_text.append(token, style=style)
                progress.advance(task)
                progress.update(task, description=f"Generating... (prob: {prob:.2%})")
        console.print()

    except KeyboardInterrupt:
        console.print("\n[bold red]Interrupted[/]")
        break
    except EOFError:
        break
    except Exception as e:
        console.print(f"[bold red]{e}[/]")
        continue
