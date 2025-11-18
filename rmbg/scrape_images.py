#!/usr/bin/env python3
"""Download all images referenced by a local HTML file or web page."""
from __future__ import annotations

import argparse
import base64
import sys
from html.parser import HTMLParser
from mimetypes import guess_extension
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlparse

import requests


class _ImgSrcCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.sources: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "img":
            return
        attr_map = {k.lower(): v for k, v in attrs if v is not None}
        src = attr_map.get("src")
        if src:
            self.sources.append(src)
        srcset = attr_map.get("srcset")
        if srcset:
            for candidate in srcset.split(","):
                part = candidate.strip().split(" ")[0]
                if part:
                    self.sources.append(part)


def _read_html(source: str) -> tuple[str, str]:
    parsed = urlparse(source)
    if parsed.scheme in {"http", "https"}:
        resp = requests.get(source, timeout=15)
        resp.raise_for_status()
        return resp.text, source
    if parsed.scheme == "file":
        path = Path(parsed.path)
        html_text = path.read_text(encoding="utf-8")
        return html_text, path.as_uri()
    path = Path(source)
    if path.exists():
        html_text = path.read_text(encoding="utf-8")
        return html_text, path.resolve().as_uri()
    if parsed.scheme:
        raise ValueError(f"Unsupported scheme: {parsed.scheme}")
    raise FileNotFoundError(f"Could not locate HTML source: {source}")


def _resolve_sources(raw_sources: Iterable[str], base_url: str) -> list[str]:
    resolved: list[str] = []
    seen: set[str] = set()
    for raw in raw_sources:
        cleaned = raw.strip()
        if not cleaned:
            continue
        absolute = urljoin(base_url, cleaned)
        if absolute in seen:
            continue
        seen.add(absolute)
        resolved.append(absolute)
    return resolved


def _ensure_destination(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    if not dest.is_dir():
        raise NotADirectoryError(dest)


def _unique_filename(dest: Path, preferred: str) -> Path:
    sanitized = preferred or "image"
    candidate = dest / sanitized
    if candidate.suffix == "":
        candidate = candidate.with_suffix(".bin")
    counter = 1
    current = candidate
    while current.exists():
        stem = candidate.stem
        current = candidate.with_name(f"{stem}_{counter}{candidate.suffix}")
        counter += 1
    return current


def _filename_for_url(url: str, response: requests.Response, dest: Path, index: int) -> Path:
    parsed = urlparse(url)
    raw_name = Path(parsed.path).name
    if not raw_name:
        content_type = response.headers.get("content-type", "").split(";")[0]
        ext = guess_extension(content_type) or ".bin"
        raw_name = f"image_{index}{ext}"
    return _unique_filename(dest, raw_name)


def _download_http(url: str, dest: Path, session: requests.Session, index: int, timeout: float) -> Path | None:
    try:
        response = session.get(url, timeout=timeout, stream=True)
        response.raise_for_status()
    except requests.RequestException as error:
        print(f"!! Failed to download {url}: {error}", file=sys.stderr)
        return None
    target = _filename_for_url(url, response, dest, index)
    with target.open("wb") as fh:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                fh.write(chunk)
    return target


def _download_data_uri(url: str, dest: Path, index: int) -> Path | None:
    try:
        header, data = url.split(",", 1)
    except ValueError:
        print(f"!! Could not parse data URI #{index}", file=sys.stderr)
        return None
    mediatype = header.split(";")[0][5:] if header.startswith("data:") else ""
    ext = guess_extension(mediatype or "application/octet-stream") or ".bin"
    target = _unique_filename(dest, f"inline_{index}{ext}")
    if ";base64" in header:
        payload = base64.b64decode(data)
    else:
        payload = data.encode("utf-8")
    target.write_bytes(payload)
    return target


def download_images(source: str, dest: Path, timeout: float, max_images: int | None, dry_run: bool) -> list[Path]:
    html_text, base_url = _read_html(source)
    parser = _ImgSrcCollector()
    parser.feed(html_text)
    image_urls = _resolve_sources(parser.sources, base_url)
    if max_images is not None:
        image_urls = image_urls[:max_images]
    if dry_run:
        for url in image_urls:
            print(url)
        return []

    _ensure_destination(dest)
    saved: list[Path] = []
    session = requests.Session()
    for index, url in enumerate(image_urls, start=1):
        if url.startswith("data:"):
            saved_path = _download_data_uri(url, dest, index)
        elif urlparse(url).scheme in {"http", "https"}:
            saved_path = _download_http(url, dest, session, index, timeout)
        elif urlparse(url).scheme == "file":
            local_path = Path(urlparse(url).path)
            if not local_path.exists():
                print(f"!! Local file not found: {local_path}", file=sys.stderr)
                continue
            target = _unique_filename(dest, local_path.name)
            target.write_bytes(local_path.read_bytes())
            saved_path = target
        else:
            print(f"!! Unsupported image URL: {url}", file=sys.stderr)
            continue
        if saved_path is not None:
            saved.append(saved_path)
            print(saved_path)
    return saved


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download images from a web page or local HTML file.")
    parser.add_argument("source", help="URL, file path, or file:// URI of the HTML page")
    parser.add_argument("--dest", default="downloaded_images", help="Directory to store downloaded images")
    parser.add_argument("--timeout", type=float, default=15.0, help="HTTP timeout in seconds")
    parser.add_argument("--max-images", type=int, help="Limit the number of images downloaded")
    parser.add_argument("--dry-run", action="store_true", help="List discovered image URLs without downloading")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    dest = Path(args.dest)
    try:
        download_images(args.source, dest, timeout=args.timeout, max_images=args.max_images, dry_run=args.dry_run)
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
