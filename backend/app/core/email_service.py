"""Сервис отправки email через локальный SMTP (Mailpit)."""

from __future__ import annotations

import logging
import smtplib
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)


def send_xml_email(
    encrypted_data: bytes,
    filename: str = "infodiode_data.xml.gpg",
    subject: str = "InfoDiode: Encrypted SCADA Data",
) -> bool:
    """Отправляет зашифрованный XML по email через локальный SMTP.

    Args:
        encrypted_data: Зашифрованные данные.
        filename: Имя вложения.
        subject: Тема письма.

    Returns:
        True если отправка успешна.
    """
    try:
        msg = MIMEMultipart()
        msg["From"] = settings.smtp_from
        msg["To"] = settings.smtp_from  # отправляем себе
        msg["Subject"] = subject

        # Текст письма
        body = "InfoDiode: зашифрованные данные SCADA мнемосхемы"
        msg.attach(MIMEText(body, "plain", "utf-8"))

        # Вложение
        from email import encoders

        attachment = MIMEBase("application", "pgp-encrypted")
        attachment.set_payload(encrypted_data)
        encoders.encode_base64(attachment)
        attachment.add_header("Content-Disposition", f"attachment; filename={filename}")
        msg.attach(attachment)

        # Отправка
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port) as server:
            server.sendmail(settings.smtp_from, [settings.smtp_from], msg.as_string())

        logger.info("Email отправлен: %s (%d байт)", filename, len(encrypted_data))
        return True

    except Exception as e:
        logger.error("Ошибка отправки email: %s", e)
        return False
