"""Генератор XML в формате <sheme>.

Формат ОБЯЗАТЕЛЬНО должен совпадать с примером:
<sheme id="UUID">
<parameters timestamp = "HH:MM:SS.mmm">
    <param id="1">VALUE</param>
</parameters>
</sheme>

Внимание: <sheme> (не scheme), пробелы вокруг = в timestamp.
"""

from __future__ import annotations

import uuid
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom.minidom import parseString
from xml.sax.saxutils import escape

from app.models.schemas import SnapshotData


def generate_xml(
    snapshots: list[SnapshotData],
    scheme_id: str | None = None,
) -> str:
    """Генерирует XML из списка снимков данных.

    Args:
        snapshots: Список снимков (каждый 500мс).
        scheme_id: UUID мнемосхемы (если None, генерируется).

    Returns:
        Строка XML в точном формате <sheme>.
    """
    if scheme_id is None:
        scheme_id = str(uuid.uuid4())

    # Ручная генерация для точного формата
    lines = [f'<sheme id="{scheme_id}">']

    for snapshot in snapshots:
        lines.append(f'<parameters timestamp = "{snapshot.timestamp}">')
        for param_id in sorted(snapshot.params.keys()):
            value = snapshot.params[param_id]
            # Экранирование спецсимволов XML для безопасности
            escaped_value = escape(str(value))
            # Получаем метаданные если есть
            meta = snapshot.param_metadata.get(param_id)
            if meta and (meta.short_name or meta.full_name or meta.unit):
                # XML с атрибутами name/desc/unit
                name_attr = f' name="{escape(meta.short_name)}"' if meta.short_name else ''
                desc_attr = f' desc="{escape(meta.full_name)}"' if meta.full_name else ''
                unit_attr = f' unit="{escape(meta.unit)}"' if meta.unit else ''
                lines.append(f'    <param id="{param_id}"{name_attr}{desc_attr}{unit_attr}>{escaped_value}</param>')
            else:
                # Стандартный формат без атрибутов
                lines.append(f'    <param id="{param_id}">{escaped_value}</param>')
        lines.append("</parameters>")

    lines.append("</sheme>")

    return "\n".join(lines)


def validate_xml_format(xml_string: str) -> list[str]:
    """Проверяет соответствие XML формату примера.

    Args:
        xml_string: Строка XML.

    Returns:
        Список ошибок (пустой = всё корректно).
    """
    errors: list[str] = []

    # Проверка корневого элемента
    if not xml_string.strip().startswith("<sheme"):
        errors.append("Корневой элемент должен быть <sheme> (не <scheme>)")

    if 'id="' not in xml_string[:50]:
        errors.append("Отсутствует атрибут id в <sheme>")

    # Проверка формата timestamp
    if 'timestamp = "' not in xml_string:
        errors.append('Timestamp должен быть в формате: timestamp = "HH:MM:SS.mmm" (с пробелами вокруг =)')

    # Проверка элементов param
    if '<param id="' not in xml_string:
        errors.append("Отсутствуют элементы <param>")

    # Проверка закрывающего тега
    if not xml_string.strip().endswith("</sheme>"):
        errors.append("Отсутствует закрывающий тег </sheme>")

    return errors


def create_snapshot(
    timestamp: str | float,
    param_values: dict[int, str],
    param_metadata: dict[int, tuple[str, str, str]] | None = None,
) -> SnapshotData:
    """Создаёт данные снимка для XML.

    Args:
        timestamp: Таймстемп в формате HH:MM:SS.mmm (строка) или в секундах (float/int).
        param_values: Словарь param_id -> значение (строка).
        param_metadata: Опциональный словарь param_id -> (short_name, full_name, unit).

    Returns:
        Данные снимка.
    """
    from app.models.schemas import ParamMetadata
    from app.utils.xml_utils import format_timestamp

    # Конвертируем float/int (миллисекунды от frame extractor) в строку HH:MM:SS.mmm
    if isinstance(timestamp, (int, float)):
        timestamp = format_timestamp(timestamp / 1000.0)

    metadata_dict: dict[int, ParamMetadata] = {}
    if param_metadata:
        for pid, (short_name, full_name, unit) in param_metadata.items():
            metadata_dict[pid] = ParamMetadata(
                short_name=short_name,
                full_name=full_name,
                unit=unit,
            )

    return SnapshotData(timestamp=timestamp, params=param_values, param_metadata=metadata_dict)
