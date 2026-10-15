"""Command-line interface: ``bridge-lc --help``."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from bridge_lc.data.aoi import list_sites
from bridge_lc.legend import IGBP20, load_legend

app = typer.Typer(help="BRIDGE land classification toolkit", no_args_is_help=True)
console = Console()


@app.command()
def info() -> None:
    """Show the legend, its Noah-MP crosswalk and configured test-bed sites."""
    legend = load_legend()
    table = Table(title=f"Legend {legend.name} v{legend.version}")
    for col in ("id", "name", "level1", "IGBP", "Noah-MP ext"):
        table.add_column(col)
    for c in legend.classes:
        ext = f"[bold magenta]{c.noahmp_ext} (new)[/]" if c.noahmp_ext > 20 else str(c.noahmp_ext)
        table.add_row(str(c.id), c.name, c.level1, f"{c.igbp} {IGBP20[c.igbp]}", ext)
    console.print(table)

    sites = Table(title="Test-bed sites")
    for col in ("id", "name", "bbox"):
        sites.add_column(col)
    for s in list_sites():
        sites.add_row(s.id, s.name, str(s.bbox))
    console.print(sites)


@app.command("export-noahmp")
def export_noahmp(
    classification: Path = typer.Argument(..., help="Classified GeoTIFF (legend ids)"),
    out: Path = typer.Argument(..., help="Output GeoTIFF with Noah-MP codes"),
    scheme: str = typer.Option("noahmp_ext", help="igbp | noahmp_ext"),
) -> None:
    """Remap a classification raster to Noah-MP vegetation codes."""
    from bridge_lc.noahmp.export import remap_geotiff

    remap_geotiff(classification, out, scheme=scheme)
    console.print(f"[green]Wrote {out}[/] ({scheme})")


@app.command("mptable-stub")
def mptable_stub(out: Path = typer.Option(Path("mptable_bridge_ext.txt"))) -> None:
    """Write a template of new MPTABLE parameter columns for the extended Andean classes."""
    from bridge_lc.noahmp.crosswalk import mptable_stub as stub

    out.write_text(stub(load_legend()))
    console.print(f"[green]Wrote {out}[/]")


if __name__ == "__main__":
    app()
