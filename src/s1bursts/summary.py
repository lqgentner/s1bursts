"""Operation result and summary data classes for batch processing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.padding import Padding
from rich.table import Table

if TYPE_CHECKING:
    from collections.abc import Collection

MIN_S = 60


@dataclass
class OkResult[T]:
    """Successful or skipped operation result.

    Parameters
    ----------
    source
        The source burst or burst pair.
    result
        The resulting product.
    skipped
        Whether the operation was skipped (product already exists).
    """

    source: Any
    result: T
    skipped: bool = False


@dataclass
class ErrorResult:
    """Failed operation result.

    Parameters
    ----------
    source
        The source burst or burst pair.
    error
        The exception that caused the failure.
    """

    source: Any
    error: Exception


type OperationResult[T] = OkResult[T] | ErrorResult


@dataclass
class OperationSummary[T]:
    """Summary of batch operation results (processing, downloading, etc.).

    Parameters
    ----------
    n_total
        Total number of operations attempted.
    n_processed
        Number of successfully processed items.
    n_skipped
        Number of skipped operations (already existed).
    n_failed
        Number of failed operations.
    products
        The products successfully processed or already existed.
    failed_items
        List of operation results for failed items.
    processing_time
        Total operation time in seconds.
    """

    n_total: int
    n_processed: int
    n_skipped: int
    n_failed: int
    products: list[T]
    failed_items: list[ErrorResult]
    processing_time: float = 0.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate.

        The operation was successful if the item was processed or skipped (already existing).

        Returns
        -------
        float
            Success rate as a fraction (0.0 to 1.0).
        """
        if self.n_total == 0:
            return 1.0
        return (self.n_processed + self.n_skipped) / self.n_total

    @classmethod
    def from_results(
        cls,
        results: Collection[OkResult[T] | ErrorResult],
        start_time: float,
        end_time: float,
    ) -> OperationSummary[T]:
        """Create summary from a collection of operation results.

        Parameters
        ----------
        results
            Results from individual operations.
        start_time
            Unix timestamp at start of batch.
        end_time
            Unix timestamp at end of batch.

        Returns
        -------
        OperationSummary[T]
            Aggregated summary of results.
        """
        return cls(
            n_total=len(results),
            n_processed=sum(1 for r in results if isinstance(r, OkResult) and not r.skipped),
            n_failed=sum(1 for r in results if isinstance(r, ErrorResult)),
            n_skipped=sum(1 for r in results if isinstance(r, OkResult) and r.skipped),
            products=[r.result for r in results if isinstance(r, OkResult)],
            failed_items=[r for r in results if isinstance(r, ErrorResult)],
            processing_time=end_time - start_time,
        )

    def print_summary(self, console: Console | None = None) -> None:
        """Print formatted summary table.

        Parameters
        ----------
        console
            Rich console for output. Creates new one if not provided.
        """
        if console is None:
            console = Console(force_jupyter=False)

        table = Table(title="Operation Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", justify="right", style="magenta")

        table.add_row("Total", str(self.n_total))
        table.add_row("Processed", f"[green]{self.n_processed}[/green]")
        table.add_row("Skipped", f"[yellow]{self.n_skipped}[/yellow]")
        table.add_row("Failed", f"[red]{self.n_failed}[/red]")
        table.add_row("Success Rate", f"{self.success_rate * 100:.0f} %")
        table.add_row("Processing Time", f"{self.processing_time / MIN_S:.0f} min")

        console.print(table)

        if self.failed_items:
            console.print("Failed Items:", style="red bold")
            for result in self.failed_items:
                exc_name = type(result.error).__name__
                console.print(Padding(f"• {result.source}:", (0, 0, 0, 2)))
                console.print(
                    Padding(f"[bold red]{exc_name}:[/bold red] {result.error}", (0, 0, 0, 4)),
                )
