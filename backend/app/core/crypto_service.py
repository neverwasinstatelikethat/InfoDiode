"""Сервис шифрования GPG."""

from __future__ import annotations

import logging
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)


def encrypt_xml(xml_content: str, output_path: str | Path | None = None) -> bytes:
    """Шифрует XML-контент с помощью GPG.

    Args:
        xml_content: Строка XML.
        output_path: Путь для сохранения (если None, возвращает байты).

    Returns:
        Зашифрованные байты.
    """
    import gnupg

    gpg = gnupg.GPG(gnupghome=settings.gpg_home)

    # GPG encrypt expects bytes or ASCII-safe string
    # XML content with Cyrillic must be properly encoded before encryption
    xml_bytes = xml_content.encode("utf-8") if isinstance(xml_content, str) else xml_content
    
    encrypted = gpg.encrypt(
        xml_bytes,
        recipients=[settings.gpg_recipient],
        always_trust=True,
    )

    if not encrypted.ok:
        logger.error("Ошибка шифрования GPG: %s", encrypted.status)
        raise RuntimeError(f"GPG encryption failed: {encrypted.status}")

    # encrypted.data is already bytes (GPG armored output is ASCII)
    encrypted_bytes = encrypted.data if encrypted.data else str(encrypted).encode("ascii", errors="replace")

    if output_path is not None:
        Path(output_path).write_bytes(encrypted_bytes)
        logger.info("Зашифрованный XML сохранён: %s", output_path)

    return encrypted_bytes


def decrypt_xml(encrypted_bytes: bytes) -> str:
    """Расшифровывает XML-контент.

    Args:
        encrypted_bytes: Зашифрованные байты.

    Returns:
        Расшифрованная строка XML.
    """
    import gnupg

    gpg = gnupg.GPG(gnupghome=settings.gpg_home)

    # Decrypt expects bytes
    decrypted = gpg.decrypt(encrypted_bytes)

    if not decrypted.ok:
        raise RuntimeError(f"GPG decryption failed: {decrypted.status}")

    # decrypted.data is bytes, decode as UTF-8 (XML with Cyrillic)
    return decrypted.data.decode("utf-8") if decrypted.data else str(decrypted)
