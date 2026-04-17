"""Programmatic uvicorn launcher.

Railway passes start commands via exec form, which skips the shell and
therefore does not expand ``${PORT}``. Running uvicorn through Python
avoids that: we read PORT from os.environ directly.
"""
from __future__ import annotations

import os

import uvicorn


def main() -> None:
    port = int(os.environ.get("PORT", "8080"))
    host = os.environ.get("HOST", "0.0.0.0")
    workers = int(os.environ.get("WEB_CONCURRENCY", "1"))
    log_level = os.environ.get("LOG_LEVEL", "info").lower()

    uvicorn.run(
        "webapp.main:app",
        host=host,
        port=port,
        workers=workers,
        log_level=log_level,
        proxy_headers=True,
        forwarded_allow_ips="*",
    )


if __name__ == "__main__":
    main()
