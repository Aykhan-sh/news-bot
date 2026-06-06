from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

_env = Environment(
    loader=FileSystemLoader(str(_PROMPTS_DIR)),
    autoescape=select_autoescape(default=False),
    trim_blocks=True,
    lstrip_blocks=True,
    undefined=StrictUndefined,
)


def render(template_name: str, **context: Any) -> str:
    tmpl = _env.get_template(template_name)
    return tmpl.render(**context)


def render_string(source: str, **context: Any) -> str:
    return _env.from_string(source).render(**context)
