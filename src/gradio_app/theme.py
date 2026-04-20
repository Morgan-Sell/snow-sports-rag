from __future__ import annotations

from typing import Any

__all__ = ["make_alpine_theme"]

__doc__ = """Alpine-lodge colour palette wrapped as a Gradio theme.

The theme is intentionally restrained: glacier blue primary, pine green
accent, and a slate-ice neutral track. Typography and radius are nudged
toward a crisp/premium feel.
"""


def make_alpine_theme() -> Any:
    """Return a :class:`gradio.themes.Base` styled for the alpine look.

    The import of :mod:`gradio` is deferred so that importing this module
    (for typing, tests, or tooling) does not require the optional UI
    dependency to be installed.

    Returns
    -------
    gradio.themes.Base
        Theme instance ready to pass to ``gr.Blocks(theme=...)``.
    """
    import gradio as gr

    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.sky,
        secondary_hue=gr.themes.colors.emerald,
        neutral_hue=gr.themes.colors.slate,
        spacing_size=gr.themes.sizes.spacing_md,
        text_size=gr.themes.sizes.text_md,
        font=(
            gr.themes.GoogleFont("Inter"),
            "system-ui",
            "-apple-system",
            "Segoe UI",
            "Roboto",
            "sans-serif",
        ),
        font_mono=(
            gr.themes.GoogleFont("JetBrains Mono"),
            "ui-monospace",
            "SFMono-Regular",
            "Menlo",
            "monospace",
        ),
    ).set(
        body_background_fill="#F7FAFC",
        body_background_fill_dark="#0B1220",
        body_text_color="#1E293B",
        body_text_color_dark="#E2E8F0",
        background_fill_primary="#FFFFFF",
        background_fill_primary_dark="#111827",
        background_fill_secondary="#EEF4F8",
        background_fill_secondary_dark="#0F172A",
        border_color_primary="#D8E2EC",
        border_color_primary_dark="#1F2937",
        block_shadow=(
            "0 1px 2px rgba(15, 23, 42, 0.04), 0 6px 24px rgba(15, 23, 42, 0.06)"
        ),
        block_radius="14px",
        block_border_width="1px",
        button_primary_background_fill="#1E6FA9",
        button_primary_background_fill_hover="#175A8A",
        button_primary_text_color="#FFFFFF",
        button_secondary_background_fill="#E2ECF3",
        button_secondary_background_fill_hover="#CFDDE7",
        button_secondary_text_color="#1E293B",
        input_background_fill="#FFFFFF",
        input_border_color="#D8E2EC",
        input_border_color_focus="#1E6FA9",
        input_radius="10px",
    )
    return theme
