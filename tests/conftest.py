"""
Shared helpers for the test suites.

Every service is packaged as its own image with `COPY services/<name> .`, so its
modules are imported flat at runtime (`from routes.auth import router`, not
`from services.api.routes.auth import router`). The tests reproduce that layout by
putting the service directory first on `sys.path`.

Two services share module names (`main`, `routes`, `services`, `_config`, ...), so
the modules cached by a previous service are dropped before switching. That keeps a
single `pytest tests/` run honest; CI still runs one job per service because each
one needs a different dependency group.
"""

import sys
from pathlib import Path

ROOT: Path = Path(__file__).resolve().parent.parent

# Top-level module names that more than one service defines.
_SHARED_MODULE_NAMES: frozenset[str] = frozenset(
    {
        "_config",
        "_version",
        "db",
        "main",
        "metrics",
        "middlewares",
        "prom_metrics",
        "routes",
        "schemas",
        "services",
    }
)


def _unregister_metrics(module: object) -> None:
    """
    Release the Prometheus collectors a module registered when it was imported.

    Counters and histograms are module-level globals that register themselves in
    the process-wide registry. Re-importing the module would raise
    `DuplicateTimeseries`, so its collectors are dropped before it is discarded.
    """
    try:
        from prometheus_client import REGISTRY
        from prometheus_client.metrics import MetricWrapperBase
    except ImportError:  # a suite whose dependency group has no prometheus-client
        return

    for value in vars(module).values():
        if isinstance(value, MetricWrapperBase):
            try:
                REGISTRY.unregister(value)
            except KeyError:
                pass


def use_service(name: str) -> Path:
    """
    Make one service directory the flat import root for the modules that follow.

    Call this from a fixture, not at conftest import time: pytest imports every
    conftest during collection, so a module-level call would leave whichever
    service was collected last on top for the whole run.

    Args:
        name: Directory name under `services/`, e.g. "api" or "inference".

    Returns:
        Path: The service directory that was put on `sys.path`.
    """
    directory = ROOT / "services" / name

    for module_name in list(sys.modules):
        if module_name.split(".", 1)[0] in _SHARED_MODULE_NAMES:
            _unregister_metrics(sys.modules[module_name])
            del sys.modules[module_name]

    entry = str(directory)
    while entry in sys.path:
        sys.path.remove(entry)
    sys.path.insert(0, entry)
    return directory
