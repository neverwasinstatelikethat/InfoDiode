"""Сервис аутентификации: JWT-токены, хеширование паролей, управление пользователями."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import bcrypt
import jwt

from app.config import settings

logger = logging.getLogger(__name__)

# Файл хранения пользователей (JSON для оффлайн-работы)
_USERS_FILE = Path("data/users.json")


def _ensure_users_file() -> None:
    """Создаёт файл пользователей, если не существует."""
    _USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not _USERS_FILE.exists():
        _USERS_FILE.write_text("{}", encoding="utf-8")


def _load_users() -> dict[str, dict]:
    """Загружает пользователей из файла."""
    _ensure_users_file()
    try:
        data = _USERS_FILE.read_text(encoding="utf-8")
        return json.loads(data) if data else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _save_users(users: dict[str, dict]) -> None:
    """Сохраняет пользователей в файл."""
    _ensure_users_file()
    _USERS_FILE.write_text(json.dumps(users, ensure_ascii=False, indent=2), encoding="utf-8")


def hash_password(password: str) -> str:
    """Хеширует пароль с помощью bcrypt."""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Проверяет пароль против хеша."""
    return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))


def create_access_token(data: dict, expires_delta: int | None = None) -> str:
    """Создаёт JWT-токен доступа.

    Args:
        data: Данные для кодирования в токен (обычно {"sub": username}).
        expires_delta: Время жизни токена в секундах. Если None — из настроек.

    Returns:
        Закодированный JWT-токен.
    """
    to_encode = data.copy()
    expire = time.time() + (expires_delta or settings.jwt_expire_minutes * 60)
    to_encode["exp"] = expire
    return jwt.encode(to_encode, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> dict | None:
    """Декодирует и валидирует JWT-токен.

    Args:
        token: JWT-токен.

    Returns:
        Данные токена или None, если токен невалиден.
    """
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        return payload
    except jwt.ExpiredSignatureError:
        logger.warning("Токен истёк")
        return None
    except jwt.InvalidTokenError:
        logger.warning("Невалидный токен")
        return None


def register_user(username: str, email: str, password: str, full_name: str = "") -> dict | None:
    """Регистрирует нового пользователя.

    Args:
        username: Логин пользователя.
        email: Email адрес.
        password: Пароль в открытом виде.
        full_name: Полное имя (ФИО).

    Returns:
        Данные пользователя или None, если пользователь уже существует.
    """
    users = _load_users()
    if username in users:
        return None

    users[username] = {
        "username": username,
        "email": email,
        "full_name": full_name or username,
        "hashed_password": hash_password(password),
        "email_recipients": [email],  # Список email для отправки XML
        "default_recipient": email,
        "theme": "dark",
        "created_at": time.time(),
    }
    _save_users(users)
    logger.info("Зарегистрирован пользователь: %s", username)
    return users[username]


def authenticate_user(username: str, password: str) -> dict | None:
    """Аутентифицирует пользователя по логину и паролю.

    Args:
        username: Логин.
        password: Пароль.

    Returns:
        Данные пользователя или None при неудаче.
    """
    users = _load_users()
    user = users.get(username)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return user


def get_user(username: str) -> dict | None:
    """Возвращает данные пользователя по логину."""
    return _load_users().get(username)


def update_user_profile(username: str, full_name: str | None = None, email: str | None = None) -> dict | None:
    """Обновляет профиль пользователя.

    Args:
        username: Логин.
        full_name: Новое ФИО.
        email: Новый email.

    Returns:
        Обновлённые данные пользователя или None.
    """
    users = _load_users()
    user = users.get(username)
    if not user:
        return None

    if full_name is not None:
        user["full_name"] = full_name
    if email is not None:
        user["email"] = email
        if email not in user.get("email_recipients", []):
            user.setdefault("email_recipients", []).append(email)

    users[username] = user
    _save_users(users)
    return user


def change_password(username: str, old_password: str, new_password: str) -> bool:
    """Меняет пароль пользователя.

    Args:
        username: Логин.
        old_password: Старый пароль.
        new_password: Новый пароль.

    Returns:
        True, если пароль успешно изменён.
    """
    users = _load_users()
    user = users.get(username)
    if not user:
        return False
    if not verify_password(old_password, user["hashed_password"]):
        return False

    user["hashed_password"] = hash_password(new_password)
    users[username] = user
    _save_users(users)
    logger.info("Пароль изменён для пользователя: %s", username)
    return True


def update_email_settings(
    username: str,
    email_recipients: list[str] | None = None,
    default_recipient: str | None = None,
) -> dict | None:
    """Обновляет настройки email пользователя.

    Args:
        username: Логин.
        email_recipients: Список email-адресов для отправки XML.
        default_recipient: Email по умолчанию.

    Returns:
        Обновлённые данные пользователя или None.
    """
    users = _load_users()
    user = users.get(username)
    if not user:
        return None

    if email_recipients is not None:
        user["email_recipients"] = email_recipients
    if default_recipient is not None:
        user["default_recipient"] = default_recipient

    users[username] = user
    _save_users(users)
    return user


def create_default_admin() -> None:
    """Создаёт администратора по умолчанию, если пользователей нет."""
    users = _load_users()
    if not users:
        register_user(
            username="admin",
            email="admin@infodiode.local",
            password="admin",
            full_name="Администратор",
        )
        logger.info("Создан пользователь по умолчанию: admin/admin")
