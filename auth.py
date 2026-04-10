"""
auth.py
───────
用戶認證模組（輕量化 JSON 存儲）

功能：
  - 用戶註冊（帳號 + SHA-256 密碼雜湊）
  - 用戶登入驗證
  - 用戶設定持久化（每個帳號各自獨立）

存儲格式（users.json）：
  {
    "<username>": {
      "password_hash": "<sha256>",
      "settings": {
        "available_cash":    500000,
        "current_holdings":  {"2330": {"cost": 850.0, "shares": 1000}},
        "target_pool":       ["2330", "2317", ...],
        "ga_config":         {...},
        "mc_config":         {...},
        "top_n":             3,
        "short_term_mode":   true
      }
    }
  }
"""

import json
import hashlib
from pathlib import Path

USERS_FILE = Path(__file__).parent / "users.json"


# ── 底層讀寫 ──────────────────────────────────────────────────

def _load_users() -> dict:
    if USERS_FILE.exists():
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_users(users: dict) -> None:
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)


def _hash(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


# ── 公開 API ──────────────────────────────────────────────────

def register_user(username: str, password: str) -> tuple[bool, str]:
    """
    註冊新用戶。

    Returns
    -------
    (success: bool, message: str)
    """
    username = username.strip()
    if len(username) < 2:
        return False, "帳號長度至少 2 個字元。"
    if len(password) < 6:
        return False, "密碼長度至少 6 個字元。"

    users = _load_users()
    if username in users:
        return False, "此帳號已存在，請選擇其他名稱。"

    users[username] = {
        "password_hash": _hash(password),
        "settings": {},
    }
    _save_users(users)
    return True, "註冊成功！請使用新帳號登入。"


def verify_user(username: str, password: str) -> tuple[bool, str]:
    """
    驗證用戶登入。

    Returns
    -------
    (success: bool, message: str)
    """
    users = _load_users()
    if username not in users:
        return False, "帳號不存在。"
    if users[username]["password_hash"] != _hash(password):
        return False, "密碼錯誤，請重試。"
    return True, "登入成功！"


def load_user_settings(username: str) -> dict:
    """讀取用戶上次儲存的設定；若無則返回空字典。"""
    users = _load_users()
    return users.get(username, {}).get("settings", {})


def save_user_settings(username: str, settings: dict) -> None:
    """
    覆寫儲存用戶設定。

    settings 中不應含不可序列化的物件（pandas DataFrame 等）。
    """
    users = _load_users()
    if username not in users:
        return
    users[username]["settings"] = settings
    _save_users(users)
