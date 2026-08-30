"""快取與設定檔位置。"""

from __future__ import annotations

import os

APP_NAME = "KTISV"


def app_data_dir() -> str:
    override = os.environ.get("KTISV_DATA_DIR")
    if override:
        base = override
    elif os.name == "nt":
        base = os.path.join(os.environ.get("LOCALAPPDATA",
                                           os.path.expanduser("~")), APP_NAME)
    else:
        base = os.path.join(os.path.expanduser("~"), ".local", "share", APP_NAME.lower())
    os.makedirs(base, exist_ok=True)
    return base


def cache_dir(*parts: str) -> str:
    path = os.path.join(app_data_dir(), "cache", *parts)
    os.makedirs(path, exist_ok=True)
    return path


def settings_path() -> str:
    return os.path.join(app_data_dir(), "settings.json")
