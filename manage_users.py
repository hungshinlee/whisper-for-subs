#!/usr/bin/env python3
"""
使用者帳號管理工具

用法：
  新增 / 更新使用者：  python manage_users.py add <username> <password>
  刪除使用者：        python manage_users.py delete <username>
  列出所有使用者：    python manage_users.py list
  驗證密碼：         python manage_users.py verify <username> <password>

.users 檔案格式（每行一筆）：
  username:pbkdf2:sha256:<salt>:<hash>   ← 雜湊密碼（建議）
  username:<plaintext>                   ← 明文密碼（向下相容，不建議）
  # 開頭為註解，空行忽略
"""

import hashlib
import os
import secrets
import sys
from pathlib import Path

USERS_FILE = Path(__file__).parent / ".users"
ITERATIONS = 260_000   # OWASP 2023 推薦值
ALGORITHM  = "sha256"


# ── 雜湊 / 驗證 ────────────────────────────────────────────────

def hash_password(password: str) -> str:
    """回傳 'pbkdf2:sha256:<salt>:<hex_hash>' 格式的字串。"""
    salt = secrets.token_hex(16)
    key  = hashlib.pbkdf2_hmac(ALGORITHM, password.encode(), salt.encode(), ITERATIONS)
    return f"pbkdf2:sha256:{salt}:{key.hex()}"


def verify_password(stored: str, provided: str) -> bool:
    """
    驗證密碼。支援：
    - pbkdf2:sha256:<salt>:<hash>  （雜湊格式）
    - 其他字串                     （明文格式，向下相容）
    """
    if stored.startswith("pbkdf2:sha256:"):
        try:
            _, _, salt, stored_hash = stored.split(":", 3)
            key = hashlib.pbkdf2_hmac(ALGORITHM, provided.encode(), salt.encode(), ITERATIONS)
            return secrets.compare_digest(key.hex(), stored_hash)
        except Exception:
            return False
    # 明文 fallback（向下相容）
    return secrets.compare_digest(stored, provided)


# ── 讀寫 .users ────────────────────────────────────────────────

def _read_lines() -> list[str]:
    if not USERS_FILE.exists():
        return []
    return USERS_FILE.read_text(encoding="utf-8").splitlines()


def _write_lines(lines: list[str]) -> None:
    USERS_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_users() -> dict[str, str]:
    """回傳 {username: stored_credential} dict。"""
    users: dict[str, str] = {}
    for line in _read_lines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        username, credential = line.split(":", 1)
        users[username.strip()] = credential.strip()
    return users


def add_user(username: str, password: str) -> None:
    username = username.strip()
    hashed   = hash_password(password)
    lines    = _read_lines()

    # 若帳號已存在則更新，否則新增
    updated = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#") or ":" not in stripped:
            continue
        u, _ = stripped.split(":", 1)
        if u.strip() == username:
            lines[i] = f"{username}:{hashed}"
            updated  = True
            break

    if not updated:
        lines.append(f"{username}:{hashed}")

    _write_lines(lines)
    action = "更新" if updated else "新增"
    print(f"✅ {action}使用者：{username}")


def delete_user(username: str) -> None:
    username = username.strip()
    lines    = _read_lines()
    new_lines = []
    found = False
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("#") and ":" in stripped:
            u, _ = stripped.split(":", 1)
            if u.strip() == username:
                found = True
                continue
        new_lines.append(line)

    if not found:
        print(f"❌ 找不到使用者：{username}")
        sys.exit(1)

    _write_lines(new_lines)
    print(f"✅ 已刪除使用者：{username}")


def list_users() -> None:
    users = load_users()
    if not users:
        print("（無使用者）")
        return
    for username, cred in users.items():
        fmt = "雜湊" if cred.startswith("pbkdf2:") else "⚠️  明文"
        print(f"  {username}  [{fmt}]")


def verify_user(username: str, password: str) -> None:
    users = load_users()
    if username not in users:
        print(f"❌ 找不到使用者：{username}")
        sys.exit(1)
    if verify_password(users[username], password):
        print(f"✅ 驗證成功：{username}")
    else:
        print(f"❌ 密碼錯誤：{username}")
        sys.exit(1)


# ── CLI ───────────────────────────────────────────────────────

def main() -> None:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(0)

    cmd = args[0].lower()

    if cmd == "add" and len(args) == 3:
        add_user(args[1], args[2])
    elif cmd == "delete" and len(args) == 2:
        delete_user(args[1])
    elif cmd == "list":
        list_users()
    elif cmd == "verify" and len(args) == 3:
        verify_user(args[1], args[2])
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
