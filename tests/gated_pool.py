"""A thread pool whose items block until its queue is cancelled, for testing batch interrupts.

A batch that stops early must cancel the items still queued. Asserting that
needs the workers to stay busy until the cancellation has happened, otherwise
a worker freed by the failing item may pick up the next one first. Here every
gated item blocks until the pool's ``shutdown(cancel_futures=True)`` has run,
so which items ran never depends on thread timing.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import threading
from typing import TYPE_CHECKING, override

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator
    from concurrent.futures import Future

# Bounds a wait that only times out if the code under test never cancels.
_TIMEOUT_S = 10


class Gate:
    """Hold batch items until the pool running them has cancelled its queue.

    Parameters
    ----------
    n_running
        Number of items that run at once, the pool's ``max_workers``.
    """

    def __init__(self, n_running: int) -> None:
        self.n_running = n_running
        self.started: list[object] = []
        self._lock = threading.Lock()
        self._opened = threading.Event()
        self._all_running = threading.Event()

    def enter(self, item: object) -> None:
        """Record that ``item`` started, then block until the queue is cancelled."""
        with self._lock:
            self.started.append(item)
            if len(self.started) >= self.n_running:
                self._all_running.set()
        if not self._opened.wait(timeout=_TIMEOUT_S):
            msg = "the batch never cancelled its queue"
            raise TimeoutError(msg)

    def executor(self, *, interrupt_at_submit: int | None = None) -> type[ThreadPoolExecutor]:
        """Return a ``ThreadPoolExecutor`` that opens this gate once it cancels its queue.

        With ``interrupt_at_submit``, submitting the item of that index raises
        ``KeyboardInterrupt``, as Ctrl-C while the queue is being filled.
        """
        gate = self

        class GatedExecutor(ThreadPoolExecutor):
            n_submitted = 0

            @override
            def submit[**P, T](
                self,
                fn: Callable[P, T],
                /,
                *args: P.args,
                **kwargs: P.kwargs,
            ) -> Future[T]:
                if self.n_submitted == interrupt_at_submit:
                    raise KeyboardInterrupt
                self.n_submitted += 1
                return super().submit(fn, *args, **kwargs)

            @override
            def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
                super().shutdown(wait=False, cancel_futures=cancel_futures)
                if cancel_futures:
                    gate._opened.set()
                if wait:
                    super().shutdown(wait=True)

        return GatedExecutor

    def interrupt[T](self, futures: Iterable[Future[T]]) -> Iterator[Future[T]]:
        """Stand in for ``as_completed``: raise ``KeyboardInterrupt`` once every worker is busy."""
        del futures
        if not self._all_running.wait(timeout=_TIMEOUT_S):
            msg = "the workers never all started"
            raise TimeoutError(msg)
        raise KeyboardInterrupt
        yield  # A generator, as as_completed is.  # pragma: no cover
