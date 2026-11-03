"""Operation result and summary data classes for batch processing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self

from rich.console import Console
from rich.markup import escape
from rich.padding import Padding
from rich.table import Table

from s1bursts.errors import OperationFailedError

if TYPE_CHECKING:
    from collections.abc import Collection

MIN_S = 60
# A failure group shows its error's first line up to this length, and this many
# of its sources; GPT messages can run to pages.
_MAX_MESSAGE_CHARS = 120
_EXAMPLES_PER_GROUP = 3


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

    def raise_on_failure(
        self,
        ignore: type[BaseException] | tuple[type[BaseException], ...] = (),
    ) -> Self:
        """Raise if any item failed, otherwise return the summary.

        Runs after the batch, when every product that could be made has been
        written, and raises one error listing all failures. Running the batch
        again skips the existing products and retries only the failed items,
        except bursts a ``BurstCatalog`` already found missing at the source,
        which the same catalog reports as failed again without a retry.
        Returning the summary lets the check sit inside a call chain, as in
        ``processor.create_coherences(pairs, crs=crs).raise_on_failure().products``.

        Parameters
        ----------
        ignore
            Error type, or tuple of types, whose failures are tolerated, such
            as ``BurstNotFoundError`` for bursts the source no longer serves.
            They do not raise, and stay in ``failed_items``.

        Returns
        -------
        Self
            This summary, unchanged.

        Raises
        ------
        OperationFailedError
            If any item failed with an error that is not ignored. Its message
            lists those items and their errors and counts the ignored ones; its
            ``summary`` is this summary, so ``err.summary.products`` still
            holds the products that were made.
        """
        ignored = ignore if isinstance(ignore, tuple) else (ignore,)
        if all(isinstance(r.error, ignored) for r in self.failed_items):
            return self
        raise OperationFailedError(self, ignore=ignored)

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
            for (exc_name, message), results in _group_failures(self.failed_items):
                text = f"{exc_name}: {message}" if message else exc_name
                # The multiplication sign reads as a count, which an x does not.
                count = f"{len(results)} ×"  # noqa: RUF001
                console.print(
                    Padding(f"• {count} [bold red]{escape(text)}[/bold red]", (0, 0, 0, 2)),
                )
                examples = ", ".join(str(r.source) for r in results[:_EXAMPLES_PER_GROUP])
                more = ", …" if len(results) > _EXAMPLES_PER_GROUP else ""
                console.print(Padding(escape(f"e.g. {examples}{more}"), (0, 0, 0, 4)))


def _group_failures(
    failed: list[ErrorResult],
) -> list[tuple[tuple[str, str], list[ErrorResult]]]:
    """Group failures by error type and first message line, most frequent first."""
    groups: dict[tuple[str, str], list[ErrorResult]] = {}
    for result in failed:
        lines = str(result.error).strip().splitlines()
        message = lines[0] if lines else ""
        if len(message) > _MAX_MESSAGE_CHARS:
            message = message[: _MAX_MESSAGE_CHARS - 1] + "…"
        groups.setdefault((type(result.error).__name__, message), []).append(result)
    # Stable, so groups of equal size keep the order they first failed in.
    return sorted(groups.items(), key=lambda group: -len(group[1]))
