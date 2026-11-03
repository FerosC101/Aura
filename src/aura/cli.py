"""AURA command line.

    aura doctor              check the audio setup
    aura devices             list inputs and outputs
    aura live                process the microphone to the headphones
    aura demo psc_demo       run a scripted demo scene through the engine
    aura serve               start the API + UI server
    aura bench               run the end-to-end benchmark
"""

from __future__ import annotations

import time

import typer
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from .audio.devices import AUDIO_AVAILABLE, AUDIO_IMPORT_ERROR, list_devices
from .classification.registry import fallback_notes
from .config import EngineConfig
from .demo.scenes import all_scenes, get_scene
from .pipeline.engine import AuraEngine
from .types import ListeningMode

app = typer.Typer(
    add_completion=False,
    help="AURA - Hear what matters. Silence what doesn't.",
    no_args_is_help=True,
)
console = Console()

ACTION_STYLE = {
    "enhance": "bold green",
    "preserve": "green",
    "reduce": "yellow",
    "suppress": "red",
}


def _banner() -> Panel:
    return Panel(
        "[bold]AURA[/bold]  [dim]AI-Powered Selective Awareness[/dim]\n"
        "[dim]Hear what matters. Silence what doesn't.[/dim]",
        border_style="cyan",
    )


@app.command()
def devices() -> None:
    """List available audio devices."""
    if not AUDIO_AVAILABLE:
        console.print(f"[red]No audio backend:[/red] {AUDIO_IMPORT_ERROR}")
        raise typer.Exit(1)
    table = Table(title="Audio devices", header_style="bold cyan")
    table.add_column("#", justify="right")
    table.add_column("Name")
    table.add_column("In", justify="right")
    table.add_column("Out", justify="right")
    table.add_column("Rate", justify="right")
    table.add_column("Default")
    for d in list_devices():
        marks = " ".join(
            m for m, on in (("input", d.is_default_input), ("output", d.is_default_output)) if on
        )
        table.add_row(
            str(d.index), d.name, str(d.max_input_channels), str(d.max_output_channels),
            f"{d.default_samplerate:.0f}", marks or "",
        )
    console.print(table)


@app.command()
def doctor() -> None:
    """Check that everything needed for a live run is present."""
    console.print(_banner())
    ok = True

    if AUDIO_AVAILABLE:
        console.print("[green]OK[/green]   PortAudio backend available")
    else:
        console.print(f"[red]FAIL[/red] no audio backend: {AUDIO_IMPORT_ERROR}")
        ok = False

    ins = [d for d in list_devices() if d.can_input]
    outs = [d for d in list_devices() if d.can_output]
    console.print(f"[{'green' if ins else 'red'}]{'OK  ' if ins else 'FAIL'}[/] "
                  f"{len(ins)} input device(s)")
    console.print(f"[{'green' if outs else 'red'}]{'OK  ' if outs else 'FAIL'}[/] "
                  f"{len(outs)} output device(s)")
    ok = ok and bool(ins) and bool(outs)

    engine = AuraEngine(EngineConfig())
    for name, desc in engine.status()["backends"].items():
        if name == "fallbacks":
            for component, note in desc.items():
                console.print(f"[yellow]NOTE[/yellow] {component}: {note}")
            continue
        console.print(f"[green]OK[/green]   {name}: {desc.get('name')}")
    engine.close()

    if outs and any("speaker" in d.name.lower() for d in outs):
        console.print(
            "\n[yellow]Heads up:[/yellow] processing the microphone to loudspeakers "
            "will feed back. Use headphones for a live run."
        )
    console.print(f"\n{'[green]Ready.[/green]' if ok else '[red]Not ready.[/red]'}")
    raise typer.Exit(0 if ok else 1)


@app.command()
def live(
    mode: str = typer.Option("commute", help="focus | commute | walk | social | custom"),
    input_device: str = typer.Option(None, "--input", "-i", help="Input device index or name"),
    output_device: str = typer.Option(None, "--output", "-o", help="Output device index or name"),
    block: int = typer.Option(960, help="Block size in frames"),
    rate: int = typer.Option(48000, help="Sample rate"),
    seconds: float = typer.Option(0.0, help="Stop after N seconds (0 = until Ctrl-C)"),
    classifier: str = typer.Option("dsp", help="dsp | onnx (needs --classifier-model)"),
    classifier_model: str = typer.Option(
        None, help="Path to an ONNX model, e.g. YAMNet exported to .onnx"
    ),
) -> None:
    """Process the microphone to the headphones in real time."""
    _run_engine(
        EngineConfig(
            sample_rate=rate, block_size=block, mode=ListeningMode(mode),
            input_device=input_device, output_device=output_device,
            classifier_backend=classifier, classifier_model_path=classifier_model,
        ),
        signal=None, seconds=seconds,
        note="[yellow]Use headphones.[/yellow] Loudspeakers will feed back.",
    )


@app.command()
def demo(
    scene: str = typer.Argument("psc_demo", help="Scene key, or 'list'"),
    mode: str = typer.Option(None, help="Override the scene's listening mode"),
    seconds: float = typer.Option(0.0, help="Stop after N seconds"),
    silent: bool = typer.Option(False, help="Do not open an audio device"),
) -> None:
    """Run a scripted demo scene through the engine."""
    if scene == "list":
        table = Table(title="Demo scenes", header_style="bold cyan")
        table.add_column("Key")
        table.add_column("Title")
        table.add_column("Mode")
        table.add_column("Sources", justify="right")
        for s in all_scenes():
            table.add_row(s.key, s.title, s.mode.value, str(len(s.layers)))
        console.print(table)
        return

    try:
        sc = get_scene(scene)
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    cfg = EngineConfig(mode=ListeningMode(mode) if mode else sc.mode)
    console.print(f"[cyan]Rendering[/cyan] {sc.title}...")
    signal = sc.render(cfg.sample_rate)
    _run_engine(cfg, signal=signal, seconds=seconds or sc.duration * 2,
                note=sc.description, play=not silent)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8000),
    reload: bool = typer.Option(False),
) -> None:
    """Start the API and UI server."""
    import uvicorn

    console.print(_banner())
    console.print(f"[cyan]http://{host}:{port}[/cyan]   API docs at /docs\n")
    uvicorn.run(
        "aura.server.asgi:app" if reload else create_asgi_app(),
        host=host, port=port, reload=reload, log_level="info",
    )


@app.command()
def bench(
    scene: str = typer.Argument("psc_demo"),
) -> None:
    """Run the end-to-end benchmark for a scene."""
    import subprocess
    import sys
    from pathlib import Path

    script = Path(__file__).resolve().parents[2] / "scripts" / "bench_end_to_end.py"
    raise typer.Exit(subprocess.call([sys.executable, str(script), scene]))


def create_asgi_app():
    from .server.app import create_app

    return create_app()


# --------------------------------------------------------------------------
# Live rendering
# --------------------------------------------------------------------------


def _scene_table(engine: AuraEngine) -> Panel:
    data = engine.scene_dict()
    status = engine.metrics.snapshot()

    table = Table(box=None, header_style="bold cyan", expand=True)
    table.add_column("Sound", width=22)
    table.add_column("Conf", justify="right", width=6)
    table.add_column("Action", width=10)
    table.add_column("Gain", justify="right", width=8)
    table.add_column("Why", overflow="ellipsis")

    for card in data.get("sounds", []):
        style = ACTION_STYLE.get(card["action"], "white")
        label = card["label"]
        if card["safety_override"]:
            label = f"{label} [bold red](safety)[/bold red]"
        table.add_row(
            label,
            f"{card['confidence'] * 100:.0f}%",
            f"[{style}]{card['action']}[/{style}]",
            f"{card['gain_db']:+.1f} dB",
            f"[dim]{card['reason']}[/dim]",
        )
    if not data.get("sounds"):
        table.add_row("[dim]listening...[/dim]", "", "", "", "")

    lock = data.get("person_lock", {})
    lock_line = (
        f"[bold green]Person Lock: {lock.get('name')}[/bold green] "
        f"({lock.get('similarity', 0):.2f})"
        if lock.get("active") else "[dim]Person Lock: inactive[/dim]"
    )
    total = status.total
    header = (
        f"[bold]{data.get('mode', '').upper()}[/bold]   "
        f"{'[green]AURA ON[/green]' if data.get('enabled') else '[yellow]BYPASS[/yellow]'}   "
        f"{lock_line}\n"
        f"[dim]latency {total.get('mean_ms', 0):.1f} ms mean / {total.get('p95_ms', 0):.1f} ms p95"
        f"   load {status.processing_load * 100:.0f}%"
        f"   in {data.get('levels', {}).get('input_db', 0):.0f} dB"
        f"   out {data.get('levels', {}).get('output_db', 0):.0f} dB"
        f"   xruns {status.xruns}[/dim]"
    )
    if data.get("safety_override_active"):
        header += "\n[bold red]SAFETY OVERRIDE[/bold red]"
    return Panel(table, title=header, border_style="cyan")


def _run_engine(
    cfg: EngineConfig,
    *,
    signal,
    seconds: float,
    note: str = "",
    play: bool = True,
) -> None:
    console.print(_banner())
    if note:
        console.print(f"{note}\n")

    engine = AuraEngine(cfg)
    # A backend that could not be built falls back to the DSP default. Say so:
    # a silent downgrade looks exactly like a neural model running well.
    for component, reason in fallback_notes().items():
        console.print(f"[yellow]{component}:[/yellow] {reason}")
    if fallback_notes():
        console.print()

    try:
        engine.start(signal=signal)
    except Exception as exc:
        console.print(f"[red]Could not start audio:[/red] {exc}")
        engine.close()
        raise typer.Exit(1) from exc

    started = time.time()
    try:
        with Live(_scene_table(engine), console=console, refresh_per_second=8) as live_view:
            while True:
                time.sleep(0.12)
                live_view.update(_scene_table(engine))
                if seconds and (time.time() - started) >= seconds:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        engine.close()

    m = engine.metrics.snapshot()
    console.print(
        f"\n[dim]{m.frames} frames   "
        f"mean total latency {m.total.get('mean_ms', 0):.1f} ms   "
        f"p95 {m.total.get('p95_ms', 0):.1f} ms   "
        f"load {m.processing_load * 100:.0f}%   xruns {m.xruns}[/dim]"
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
