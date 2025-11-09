"""Runtime adjustments for Azure App Service."""

from __future__ import annotations

from azure_runtime import ensure_venv_site_packages_precedence

ensure_venv_site_packages_precedence()
