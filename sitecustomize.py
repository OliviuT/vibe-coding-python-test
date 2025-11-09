"""Runtime adjustments for Azure App Service.

This project occasionally runs inside the Azure App Service container image
that injects ``/agents/python`` at the front of ``sys.path``. That directory
contains a legacy ``typing_extensions`` module which does not define
``Sentinel``. Modern versions of the ``openai`` and ``pydantic`` packages rely
on that symbol during import, so the process crashes before the application can
start.

Python automatically imports ``sitecustomize`` (when present on ``sys.path``)
right after the standard library ``site`` module. By adjusting ``sys.path``
here we ensure our virtual environment's ``site-packages`` directory comes
before ``/agents/python``. This restores the expected import order without
impacting local development environments.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _promote_venv_site_packages() -> None:
    """Ensure the active virtualenv's site-packages outranks `/agents/python`.

    Azure's Python images expose helper tooling in ``/agents/python`` and place
    it at the beginning of ``PYTHONPATH``. When combined with the ``openai``
    package (which depends on ``typing_extensions>=4.7``), imports fail because
    the helper module shadows the up-to-date copy bundled in the virtual
    environment. By relocating ``/agents/python`` to the end of ``sys.path`` we
    keep the helpers available while allowing modern dependencies to import.
    """

    venv_root = os.environ.get("VIRTUAL_ENV")
    if not venv_root:
        return

    py_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    site_packages = Path(venv_root) / "lib" / py_version / "site-packages"
    if not site_packages.is_dir():
        return

    site_packages_str = str(site_packages)

    if site_packages_str in sys.path:
        # Move the virtualenv path to the very front so it wins import lookups.
        sys.path.remove(site_packages_str)
    sys.path.insert(0, site_packages_str)

    agents_path = "/agents/python"
    if agents_path in sys.path:
        # Keep the Azure helpers available, but only after our dependencies.
        sys.path.remove(agents_path)
        sys.path.append(agents_path)


_promote_venv_site_packages()

