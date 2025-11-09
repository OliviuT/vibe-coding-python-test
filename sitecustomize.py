from __future__ import annotations
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

try:
    # Preferred: use the shared runtime helper.
    from azure_runtime import ensure_venv_site_packages_precedence
except Exception:
    # Fallback: inline implementation (mirrors main branch behavior).
    import os
    import sys
    from pathlib import Path

    def ensure_venv_site_packages_precedence() -> None:
        venv_root = os.environ.get("VIRTUAL_ENV")
        if not venv_root:
            return

        py_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
        site_packages = Path(venv_root) / "lib" / py_version / "site-packages"
        if not site_packages.is_dir():
            return

        site_packages_str = str(site_packages)
        if site_packages_str in sys.path:
            sys.path.remove(site_packages_str)
        sys.path.insert(0, site_packages_str)

        agents_path = "/agents/python"
        if agents_path in sys.path:
            sys.path.remove(agents_path)
            sys.path.append(agents_path)

# Run adjustment at import time.
ensure_venv_site_packages_precedence()
