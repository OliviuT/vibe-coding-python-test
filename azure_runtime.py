"""Utilities for harmonising Azure App Service Python environments."""

from __future__ import annotations

import os
import sys
from pathlib import Path

AGENTS_PATH = "/agents/python"


def ensure_venv_site_packages_precedence() -> None:
    """Promote virtualenv site-packages ahead of Azure's helper tooling.

    Azure App Service prepends ``/agents/python`` to ``PYTHONPATH``. That
    directory ships a legacy ``typing_extensions`` module missing the
    ``Sentinel`` attribute required by modern ``pydantic``/``openai`` releases.
    We remove the helper path, promote the active virtual environment's
    ``site-packages`` entries (if any), then re-append ``/agents/python`` so the
    diagnostics tooling remains reachable without shadowing application
    dependencies.
    """

    agents_present = False
    while AGENTS_PATH in sys.path:
        sys.path.remove(AGENTS_PATH)
        agents_present = True

    site_packages_candidates: list[str] = []

    venv_root = os.environ.get("VIRTUAL_ENV")
    if venv_root:
        py_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
        venv_site = Path(venv_root) / "lib" / py_version / "site-packages"
        if venv_site.is_dir():
            site_packages_candidates.append(str(venv_site))

    if not site_packages_candidates:
        site_packages_candidates.extend(
            p
            for p in sys.path
            if "site-packages" in p and not p.startswith(AGENTS_PATH)
        )

    for candidate in reversed(site_packages_candidates):
        if candidate in sys.path:
            sys.path.remove(candidate)
        sys.path.insert(0, candidate)

    if agents_present:
        sys.path.append(AGENTS_PATH)
