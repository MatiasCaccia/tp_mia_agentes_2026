"""Tests para document_parser (herramienta libre M1).

Cubre las funcionalidades principales:
  - Modo índice (block_index=-1)
  - Lectura de bloque específico
  - Ventana de fuente con bloques vecinos (source_window_radius)
  - Ventana deslizante de generación (recent_output)
  - Coerción de tipos (strings en vez de ints)
  - Manejo de errores (archivo inexistente, bloque fuera de rango)
  - Archivo vacío
  - Fingerprint estable
  - Include/exclude del índice de referencia
"""

from __future__ import annotations

import os
import textwrap

import pytest

from student_framework.tools.document_parser import document_parser


@pytest.fixture()
def sample_file(tmp_path):
    """Crea un archivo de 120 líneas para testear."""
    content = ""
    for i in range(1, 121):
        content += f"Línea {i}: contenido de ejemplo para testing del document parser.\n"
    path = tmp_path / "sample.txt"
    path.write_text(content, encoding="utf-8")
    return str(path)


@pytest.fixture()
def short_file(tmp_path):
    """Archivo corto que cabe en un solo bloque."""
    content = "Primera línea\nSegunda línea\nTercera línea\n"
    path = tmp_path / "short.txt"
    path.write_text(content, encoding="utf-8")
    return str(path)


@pytest.fixture()
def empty_file(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("", encoding="utf-8")
    return str(path)


# --- Modo índice (block_index=-1) ---

class TestReferenceIndex:
    def test_index_returns_document_info(self, sample_file):
        result = document_parser(path=sample_file, block_index=-1, lines_per_block=50)
        assert "REFERENCIA FIJA" in result
        assert "120 líneas" in result
        assert "3 bloques" in result
        assert "Huella:" in result

    def test_index_shows_all_blocks(self, sample_file):
        result = document_parser(path=sample_file, block_index=-1, lines_per_block=50)
        assert "[Bloque 0]" in result
        assert "[Bloque 1]" in result
        assert "[Bloque 2]" in result

    def test_index_includes_protocol(self, sample_file):
        result = document_parser(path=sample_file, block_index=-1)
        assert "PROTOCOLO SUGERIDO" in result

    def test_index_with_custom_lines_per_block(self, sample_file):
        result = document_parser(path=sample_file, block_index=-1, lines_per_block=30)
        assert "4 bloques" in result

    def test_index_single_block_file(self, short_file):
        result = document_parser(path=short_file, block_index=-1, lines_per_block=50)
        assert "3 líneas" in result
        assert "1 bloques" in result


# --- Lectura de bloques específicos ---

class TestBlockReading:
    def test_read_first_block(self, sample_file):
        result = document_parser(path=sample_file, block_index=0, lines_per_block=50)
        assert "BLOQUE OBJETIVO" in result
        assert "Línea 1:" in result
        assert "Línea 50:" in result

    def test_read_second_block(self, sample_file):
        result = document_parser(path=sample_file, block_index=1, lines_per_block=50)
        assert "Línea 51:" in result
        assert "Línea 100:" in result

    def test_read_last_block(self, sample_file):
        result = document_parser(path=sample_file, block_index=2, lines_per_block=50)
        assert "Línea 101:" in result
        assert "Línea 120:" in result

    def test_block_includes_source_window_header(self, sample_file):
        result = document_parser(path=sample_file, block_index=0, lines_per_block=50)
        assert "VENTANA LOCAL DE FUENTE" in result
        assert "Bloque objetivo: 0" in result

    def test_block_includes_generation_window(self, sample_file):
        result = document_parser(path=sample_file, block_index=0, lines_per_block=50)
        assert "VENTANA DESLIZANTE DE GENERACIÓN" in result

    def test_block_includes_instructions(self, sample_file):
        result = document_parser(path=sample_file, block_index=0, lines_per_block=50)
        assert "INSTRUCCIONES DE USO" in result


# --- Ventana de fuente (source_window_radius) ---

class TestSourceWindow:
    def test_radius_0_only_target(self, sample_file):
        result = document_parser(
            path=sample_file, block_index=1, lines_per_block=50, source_window_radius=0,
        )
        assert "Bloque 1" in result
        assert "BLOQUE OBJETIVO" in result
        assert "Línea 51:" in result

    def test_radius_1_includes_neighbors(self, sample_file):
        result = document_parser(
            path=sample_file, block_index=1, lines_per_block=50, source_window_radius=1,
        )
        assert "Bloque 0/" in result
        assert "BLOQUE OBJETIVO" in result
        assert "Bloque 2/" in result

    def test_radius_at_start_no_negative_blocks(self, sample_file):
        result = document_parser(
            path=sample_file, block_index=0, lines_per_block=50, source_window_radius=1,
        )
        assert "Bloque 0" in result
        assert "Bloque 1/" in result
        # No debería haber bloque -1
        assert "Bloque -1" not in result

    def test_radius_at_end_no_overflow(self, sample_file):
        result = document_parser(
            path=sample_file, block_index=2, lines_per_block=50, source_window_radius=1,
        )
        assert "Bloque 1/" in result
        assert "Bloque 2" in result
        assert "Bloque 3" not in result


# --- Ventana deslizante de generación (recent_output) ---

class TestRecentGenerationWindow:
    def test_no_recent_output(self, sample_file):
        result = document_parser(path=sample_file, block_index=0, recent_output="")
        assert "Sin memoria reciente" in result

    def test_short_recent_output_preserved(self, sample_file):
        recent = "Este es el output reciente del agente."
        result = document_parser(path=sample_file, block_index=0, recent_output=recent)
        assert recent in result

    def test_long_recent_output_trimmed(self, sample_file):
        recent = "X" * 5000
        result = document_parser(
            path=sample_file, block_index=0, recent_output=recent, recent_window_chars=500,
        )
        assert "Se omitieron" in result
        assert "4500" in result  # 5000 - 500 omitidos


# --- Include/exclude del índice de referencia ---

class TestIncludeReferenceIndex:
    def test_include_reference_true(self, sample_file):
        result = document_parser(
            path=sample_file, block_index=0, include_reference_index=True,
        )
        assert "Índice compacto" in result
        assert "[Bloque 0]" in result

    def test_include_reference_false(self, sample_file):
        result = document_parser(
            path=sample_file, block_index=0, include_reference_index=False,
        )
        assert "Índice compacto" not in result
        assert "[Bloque 0]" not in result
        # Pero sí tiene el header con total de líneas
        assert "120 líneas" in result


# --- Coerción de tipos ---

class TestTypeCoercion:
    def test_block_index_as_string(self, sample_file):
        result = document_parser(path=sample_file, block_index="0", lines_per_block=50)  # type: ignore[arg-type]
        assert "Línea 1:" in result

    def test_block_index_invalid_string_defaults_to_index(self, sample_file):
        result = document_parser(path=sample_file, block_index="abc")  # type: ignore[arg-type]
        assert "REFERENCIA FIJA" in result  # -1 default -> modo índice

    def test_lines_per_block_as_string(self, sample_file):
        result = document_parser(path=sample_file, block_index=-1, lines_per_block="30")  # type: ignore[arg-type]
        assert "4 bloques" in result

    def test_include_reference_as_string(self, sample_file):
        result = document_parser(
            path=sample_file, block_index=0, include_reference_index="false",  # type: ignore[arg-type]
        )
        assert "Índice compacto" not in result

    def test_source_window_radius_as_string(self, sample_file):
        result = document_parser(
            path=sample_file, block_index=1, lines_per_block=50, source_window_radius="1",  # type: ignore[arg-type]
        )
        assert "Bloque 0/" in result
        assert "Bloque 2/" in result


# --- Manejo de errores ---

class TestErrors:
    def test_file_not_found(self):
        result = document_parser(path="/ruta/que/no/existe.txt")
        assert "Error" in result
        assert "no se encontró" in result

    def test_block_index_out_of_range(self, sample_file):
        result = document_parser(path=sample_file, block_index=999, lines_per_block=50)
        assert "Error" in result
        assert "fuera de rango" in result

    def test_negative_block_index_beyond_minus_one(self, sample_file):
        result = document_parser(path=sample_file, block_index=-5, lines_per_block=50)
        assert "Error" in result
        assert "fuera de rango" in result

    def test_empty_file(self, empty_file):
        result = document_parser(path=empty_file)
        assert "vacío" in result.lower()


# --- Fingerprint ---

class TestFingerprint:
    def test_fingerprint_is_stable(self, sample_file):
        r1 = document_parser(path=sample_file, block_index=-1)
        r2 = document_parser(path=sample_file, block_index=-1)
        # Extraer la huella de ambas salidas
        for line in r1.splitlines():
            if "Huella:" in line:
                hash1 = line.split("Huella:")[1].strip()
                break
        for line in r2.splitlines():
            if "Huella:" in line:
                hash2 = line.split("Huella:")[1].strip()
                break
        assert hash1 == hash2

    def test_fingerprint_changes_with_content(self, tmp_path):
        path = tmp_path / "changing.txt"
        path.write_text("version 1\n", encoding="utf-8")
        r1 = document_parser(path=str(path), block_index=-1)

        path.write_text("version 2\n", encoding="utf-8")
        r2 = document_parser(path=str(path), block_index=-1)

        hash1 = [l for l in r1.splitlines() if "Huella:" in l][0]
        hash2 = [l for l in r2.splitlines() if "Huella:" in l][0]
        assert hash1 != hash2
