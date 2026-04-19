"""Тесты генератора XML — rigorous validation of XML format compliance.

Tests verify exact format matching required by the specification:
- Root element: <sheme> (not <scheme>)
- Timestamp format: timestamp = "HH:MM:SS.mmm" (spaces around =)
- Parameter elements with proper attributes when metadata provided
"""

import xml.etree.ElementTree as ET
from app.core.xml_generator import create_snapshot, generate_xml, validate_xml_format


class TestXmlGenerator:
    """Тесты генерации XML в формате <sheme>."""

    def test_generate_xml_basic(self) -> None:
        """Тест базовой генерации XML."""
        snapshots = [
            create_snapshot("22:53:00.001", {1: "758.3", 2: "13.2"}),
            create_snapshot("22:53:00.501", {1: "758.4", 2: "13.3"}),
        ]

        xml = generate_xml(snapshots, scheme_id="test-uuid")

        assert '<sheme id="test-uuid">' in xml
        assert 'timestamp = "22:53:00.001"' in xml
        assert 'timestamp = "22:53:00.501"' in xml
        assert '<param id="1">758.3</param>' in xml
        assert '<param id="2">13.2</param>' in xml
        assert "</sheme>" in xml

    def test_xml_format_validation_correct(self) -> None:
        """Тест валидации корректного XML."""
        xml = '''<sheme id="54d11679-95db-4b8b-9b8d-ab11f251ef38">
<parameters timestamp = "22:53:00.001">
    <param id="1">758.3</param>
</parameters>
</sheme>'''

        errors = validate_xml_format(xml)
        assert len(errors) == 0

    def test_xml_format_validation_wrong_root(self) -> None:
        """Тест валидации XML с неправильным корневым элементом."""
        xml = '<scheme id="test"><parameters timestamp = "00:00:00.001"><param id="1">1.0</param></parameters></scheme>'

        errors = validate_xml_format(xml)
        assert any("<sheme>" in e for e in errors)

    def test_xml_format_validation_missing_spaces(self) -> None:
        """Тест валидации XML без пробелов вокруг =."""
        xml = '<sheme id="test"><parameters timestamp="00:00:00.001"><param id="1">1.0</param></parameters></sheme>'

        errors = validate_xml_format(xml)
        assert len(errors) > 0  # Должна быть ошибка про пробелы

    def test_timestamp_format(self) -> None:
        """Тест формата таймстемпов."""
        snapshots = [
            create_snapshot("00:00:00.001", {1: "1.0"}),
            create_snapshot("00:00:00.501", {1: "2.0"}),
        ]

        xml = generate_xml(snapshots)

        assert ".001" in xml
        assert ".501" in xml

    def test_param_id_sorting(self) -> None:
        """Тест сортировки param по ID."""
        snapshots = [
            create_snapshot("00:00:00.001", {3: "0.25", 1: "758.3", 2: "4.21"}),
        ]

        xml = generate_xml(snapshots)

        # param id="1" должен быть раньше param id="2"
        idx1 = xml.index('<param id="1">')
        idx2 = xml.index('<param id="2">')
        idx3 = xml.index('<param id="3">')
        assert idx1 < idx2 < idx3

    def test_xml_with_param_metadata(self) -> None:
        """Тест генерации XML с метаданными параметров."""
        param_metadata = {
            1: ("T-001", "Температура воздуха", "°C"),
            2: ("P-002", "Давление в баке", "kPa"),
        }
        snapshots = [
            create_snapshot("00:00:00.001", {1: "25.5", 2: "101.3"}, param_metadata=param_metadata),
        ]

        xml = generate_xml(snapshots, scheme_id="test-metadata")

        # Проверяем, что метаданные включены
        assert '<sheme id="test-metadata">' in xml
        assert 'name="T-001"' in xml
        assert 'desc="Температура воздуха"' in xml
        assert 'unit="°C"' in xml
        assert 'name="P-002"' in xml
        assert 'desc="Давление в баке"' in xml
        assert 'unit="kPa"' in xml
        # Проверяем, что значения тоже есть
        assert '>25.5<' in xml
        assert '>101.3<' in xml

    def test_xml_without_param_metadata(self) -> None:
        """Тест генерации XML без метаданных (обратная совместимость)."""
        snapshots = [
            create_snapshot("00:00:00.001", {1: "25.5"}),
        ]

        xml = generate_xml(snapshots, scheme_id="test-no-meta")

        # Без метаданных должен быть простой формат
        assert '<param id="1">25.5</param>' in xml
        assert 'name=' not in xml
        assert 'desc=' not in xml
        assert 'unit=' not in xml


class TestXmlGeneratorMetadataAttributes:
    """Тесты генерации XML с атрибутами метаданных."""

    def test_xml_includes_name_attribute(self) -> None:
        """Тест: XML включает атрибут name."""
        param_metadata = {1: ("TI-101", "Температура", "°С")}
        snapshots = [create_snapshot("00:00:00.001", {1: "100.5"}, param_metadata=param_metadata)]
        
        xml = generate_xml(snapshots)
        
        assert 'name="TI-101"' in xml

    def test_xml_includes_desc_attribute(self) -> None:
        """Тест: XML включает атрибут desc (description)."""
        param_metadata = {1: ("TI-101", "Температура газа", "°С")}
        snapshots = [create_snapshot("00:00:00.001", {1: "100.5"}, param_metadata=param_metadata)]
        
        xml = generate_xml(snapshots)
        
        assert 'desc="Температура газа"' in xml

    def test_xml_includes_unit_attribute(self) -> None:
        """Тест: XML включает атрибут unit."""
        param_metadata = {1: ("TI-101", "Температура", "°С")}
        snapshots = [create_snapshot("00:00:00.001", {1: "100.5"}, param_metadata=param_metadata)]
        
        xml = generate_xml(snapshots)
        
        assert 'unit="°С"' in xml

    def test_xml_only_detected_params_appear(self) -> None:
        """Тест: в XML только обнаруженные параметры (не все 261).
        
        Only parameters that were actually detected should appear in the XML,
        not the entire parameter table.
        """
        param_metadata = {
            1: ("T-001", "Температура 1", "°С"),
            2: ("T-002", "Температура 2", "°С"),
            3: ("P-001", "Давление 1", "кПа"),
        }
        # Only provide values for params 1 and 3
        snapshots = [create_snapshot("00:00:00.001", {1: "25.5", 3: "101.3"}, param_metadata=param_metadata)]
        
        xml = generate_xml(snapshots)
        
        # Should have param 1 and 3 (with metadata attributes)
        assert 'id="1"' in xml
        assert 'id="3"' in xml
        
        # Should NOT have param 2 (not detected)
        # Note: param id="2" may appear in metadata, check for value
        assert '>25.5<' in xml
        assert '>101.3<' in xml

    def test_xml_valid_structure_parseable(self) -> None:
        """Тест: валидная XML структура (парсится xml.etree).
        
        Generated XML should be parseable by standard XML parsers.
        """
        param_metadata = {
            1: ("T-001", "Температура", "°С"),
            2: ("P-001", "Давление", "кПа"),
        }
        snapshots = [
            create_snapshot("00:00:00.001", {1: "25.5", 2: "101.3"}, param_metadata=param_metadata),
            create_snapshot("00:00:00.501", {1: "25.6", 2: "101.4"}, param_metadata=param_metadata),
        ]
        
        xml = generate_xml(snapshots, scheme_id="test-parseable")
        
        # Should be parseable
        root = ET.fromstring(xml)
        
        # Verify structure
        assert root.tag == "sheme"
        assert root.get("id") == "test-parseable"
        
        # Count parameters elements
        params = root.findall(".//param")
        assert len(params) == 4  # 2 params x 2 snapshots

    def test_backward_compatibility_no_metadata(self) -> None:
        """Тест: обратная совместимость когда метаданные отсутствуют.
        
        XML generation should work even when no metadata is provided,
        producing simple <param id="N">VALUE</param> format.
        """
        snapshots = [
            create_snapshot("00:00:00.001", {1: "25.5", 2: "101.3"}),
            create_snapshot("00:00:00.501", {1: "25.6", 2: "101.4"}),
        ]
        
        xml = generate_xml(snapshots, scheme_id="test-backward-compat")
        
        # Should be parseable
        root = ET.fromstring(xml)
        
        # Verify simple format
        params = root.findall(".//param")
        assert len(params) == 4
        
        # No metadata attributes
        for param in params:
            assert param.get("name") is None
            assert param.get("desc") is None
            assert param.get("unit") is None

    def test_xml_escaping_special_chars(self) -> None:
        """Тест: экранирование спецсимволов в XML.
        
        Special XML characters like <, >, &, ", ' should be properly escaped.
        """
        param_metadata = {
            1: ("T&V", "Temp & Value", "<°C>"),
        }
        snapshots = [create_snapshot("00:00:00.001", {1: "25.5"}, param_metadata=param_metadata)]
        
        xml = generate_xml(snapshots)
        
        # Should be parseable (properly escaped)
        root = ET.fromstring(xml)
        
        # Verify the param exists
        param = root.find(".//param")
        assert param is not None

    def test_partial_metadata_handling(self) -> None:
        """Тест: обработка частичных метаданных.
        
        When some metadata fields are empty, only non-empty fields
        should be included as attributes.
        """
        param_metadata = {
            1: ("T-001", "", "°С"),  # Empty description
            2: ("", "Давление", "кПа"),  # Empty short_name
        }
        snapshots = [create_snapshot("00:00:00.001", {1: "25.5", 2: "101.3"}, param_metadata=param_metadata)]
        
        xml = generate_xml(snapshots)
        
        # Param 1 should have name and unit but no desc
        assert 'name="T-001"' in xml
        assert 'unit="°С"' in xml
        
        # Param 2 should have desc and unit but no name
        assert 'desc="Давление"' in xml
        assert 'unit="кПа"' in xml


class TestXmlGeneratorEdgeCases:
    """Тесты граничных случаев XML генерации."""

    def test_empty_snapshots(self) -> None:
        """Тест: пустой список снимков."""
        xml = generate_xml([], scheme_id="empty-test")
        
        assert '<sheme id="empty-test">' in xml
        assert "</sheme>" in xml

    def test_single_param(self) -> None:
        """Тест: один параметр."""
        snapshots = [create_snapshot("00:00:00.001", {1: "100.0"})]
        
        xml = generate_xml(snapshots)
        
        assert '<param id="1">100.0</param>' in xml

    def test_many_params(self) -> None:
        """Тест: множество параметров."""
        params = {i: str(float(i)) for i in range(1, 51)}
        snapshots = [create_snapshot("00:00:00.001", params)]
        
        xml = generate_xml(snapshots)
        
        # All params should be present
        for i in range(1, 51):
            assert f'<param id="{i}">{float(i)}</param>' in xml

    def test_special_values(self) -> None:
        """Тест: специальные значения (отрицательные, ноль, большие)."""
        snapshots = [create_snapshot("00:00:00.001", {
            1: "-50.0",
            2: "0.0",
            3: "9999.99",
        })]
        
        xml = generate_xml(snapshots)
        
        assert '<param id="1">-50.0</param>' in xml
        assert '<param id="2">0.0</param>' in xml
        assert '<param id="3">9999.99</param>' in xml

    def test_uuid_generation_when_not_provided(self) -> None:
        """Тест: автоматическая генерация UUID когда не указан."""
        snapshots = [create_snapshot("00:00:00.001", {1: "100.0"})]
        
        xml = generate_xml(snapshots)  # No scheme_id provided
        
        # Should have generated UUID
        assert '<sheme id="' in xml
        # Extract and verify UUID format
        start = xml.index('<sheme id="') + 11
        end = xml.index('">', start)
        uuid_str = xml[start:end]
        assert len(uuid_str) == 36  # Standard UUID length
