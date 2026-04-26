#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import argostranslate.package
import argostranslate.translate
import srt


def ensure_package(from_code: str, to_code: str) -> None:
    installed_languages = argostranslate.translate.get_installed_languages()
    from_lang = next((lang for lang in installed_languages if lang.code == from_code), None)
    to_lang = next((lang for lang in installed_languages if lang.code == to_code), None)

    if from_lang is not None and to_lang is not None:
        try:
            from_lang.get_translation(to_lang)
            return
        except Exception:
            pass

    argostranslate.package.update_package_index()
    available_packages = argostranslate.package.get_available_packages()
    package = next(
        (pkg for pkg in available_packages if pkg.from_code == from_code and pkg.to_code == to_code),
        None,
    )
    if package is None:
        raise SystemExit(f"No existe paquete offline para {from_code}->{to_code}")

    download_path = package.download()
    argostranslate.package.install_from_path(download_path)


def translate_text(text: str, from_code: str, to_code: str) -> str:
    placeholder = " __COPILOT_LINEBREAK__ "
    normalized = text.replace("\r\n", "\n").replace("\n", placeholder).strip()
    if not normalized:
        return text
    translated = argostranslate.translate.translate(normalized, from_code, to_code)
    return translated.replace(placeholder, "\n").strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Traduce un SRT preservando timecodes.")
    parser.add_argument("input_srt")
    parser.add_argument("output_srt")
    parser.add_argument("from_lang")
    parser.add_argument("to_lang")
    args = parser.parse_args()

    input_path = Path(args.input_srt)
    output_path = Path(args.output_srt)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ensure_package(args.from_lang, args.to_lang)

    subtitles = list(srt.parse(input_path.read_text(encoding="utf-8-sig")))
    translated = []
    for subtitle in subtitles:
        translated.append(
            srt.Subtitle(
                index=subtitle.index,
                start=subtitle.start,
                end=subtitle.end,
                content=translate_text(subtitle.content, args.from_lang, args.to_lang),
                proprietary=subtitle.proprietary,
            )
        )

    output_path.write_text(srt.compose(translated), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
