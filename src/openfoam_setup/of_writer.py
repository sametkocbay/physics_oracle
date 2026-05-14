"""Render nested dicts as OpenFOAM dictionary files.

The writer is intentionally minimal — it only supports what configs/openfoam.yaml
needs:

  * key/value entries (rendered as ``key   value;``)
  * sub-dicts (rendered as ``name\\n{\\n ... \\n}``)
  * lists (rendered as ``key   (a b c);``)
  * bare-key entries — a ``None`` value renders as ``key;`` (used in
    ``fluxRequired`` and ``cache``)
  * inline-style sub-dicts via the marker key ``__inline_children__: true``
    on the parent — used inside ``boundaryField`` blocks so each patch is
    rendered as ``inlet { type fixedValue; value uniform (...); }``

The output is semantically equivalent to OpenFOAM's dict parser; whitespace
differs slightly from what the previous hand-written templates produced.
"""
from __future__ import annotations

from io import StringIO


_INLINE_MARKER = "__inline_children__"
_INDENT = "    "
_VALUE_COL = 16


def _render_scalar(v) -> str:
    if v is True:
        return "true"
    if v is False:
        return "false"
    return str(v)


def _render_list(items) -> str:
    parts = [_render_scalar(x) if not isinstance(x, list)
             else _render_list(x) for x in items]
    return "(" + " ".join(parts) + ")"


def _is_simple(value) -> bool:
    """A value is 'simple' if it can be rendered on a single line."""
    if value is None:
        return True
    if isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, list):
        return all(_is_simple(x) for x in value)
    return False


def _render_value(v) -> str:
    if isinstance(v, list):
        return _render_list(v)
    return _render_scalar(v)


def _entry_line(key: str, value, indent: str) -> str:
    if value is None:
        return f"{indent}{key};"
    pad = max(1, _VALUE_COL - len(key))
    return f"{indent}{key}{' ' * pad}{_render_value(value)};"


def _inline_block(d: dict) -> str:
    parts = []
    for k, v in d.items():
        if k == _INLINE_MARKER:
            continue
        if v is None:
            parts.append(f"{k};")
        else:
            parts.append(f"{k} {_render_value(v)};")
    return "{ " + " ".join(parts) + " }"


def _render_dict_body(body: dict, depth: int, out: StringIO) -> None:
    """Render the body of a dict (without the surrounding `{}`).

    Sub-dicts render as `name\\n{\\n ...\\n}` blocks by default.  A parent dict
    can opt every sub-dict into the inline form by setting
    `__inline_children__: true` — used in 0/ boundaryField sections.
    """
    indent = _INDENT * depth
    inline_children = bool(body.get(_INLINE_MARKER, False))
    first = True
    for key, value in body.items():
        if key == _INLINE_MARKER:
            continue
        if isinstance(value, dict):
            if inline_children:
                pad = max(1, _VALUE_COL - len(key))
                out.write(f"{indent}{key}{' ' * pad}{_inline_block(value)}\n")
            else:
                if not first:
                    out.write("\n")
                out.write(f"{indent}{key}\n{indent}{{\n")
                _render_dict_body(value, depth + 1, out)
                out.write(f"{indent}}}\n")
        else:
            out.write(_entry_line(key, value, indent) + "\n")
        first = False


FOAM_HEADER = """/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\\      /  F ield         | OpenFOAM dataset case (auto-generated)
   \\\\    /   O peration     |
    \\\\  /    A nd           |
     \\\\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    format      ascii;
    class       {cls};
    object      {obj};
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
"""


def render_foam_dict(cls: str, obj: str, body: dict) -> str:
    """Render a complete OpenFOAM dict file (header + body)."""
    out = StringIO()
    out.write(FOAM_HEADER.format(cls=cls, obj=obj))
    out.write("\n")
    _render_dict_body(body, depth=0, out=out)
    return out.getvalue()
