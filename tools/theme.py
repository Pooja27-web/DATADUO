"""Midnight Graphite theme and reusable presentation helpers for DATADUO."""

import html

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from tools import palette as p

PALETTE = p.COLORWAY  # Backwards-compatible alias for older imports.
MONO = "'IBM Plex Mono', 'SF Mono', 'Menlo', monospace"


def apply_theme(bg_uri: str | None = None) -> None:
    """Apply the shared Streamlit theme; safe to call once near app startup."""
    bg_layer = (
        f"background-image:linear-gradient(rgba(11,15,23,.90),rgba(11,15,23,.96)),url('{bg_uri}');"
        if bg_uri else "background-image:linear-gradient(135deg,#0B0F17,#101827 55%,#0B0F17);"
    )
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600;700&family=Sora:wght@400;500;600;700&display=swap');
        :root{{--bg:{p.BG};--surface:{p.SURFACE};--border:{p.LINE};--text:{p.TEXT};
              --muted:{p.MUTED};--accent:{p.ACCENT};--platinum:{p.PLATINUM};--teal:{p.TEAL};}}
        .stApp{{background-color:var(--bg);{bg_layer}background-repeat:no-repeat;background-position:center right;background-size:cover;background-attachment:fixed;color:var(--text);font-family:'Inter',sans-serif;}}
        [data-testid="stDecoration"]{{display:none;}}
        [data-testid="stHeader"]{{background:transparent;}}
        .block-container{{max-width:1440px;padding-top:1.8rem;padding-bottom:3rem;}}
        h1,h2,h3{{font-family:'Sora',sans-serif!important;color:var(--text);letter-spacing:-.025em;}}
        h1{{font-size:clamp(1.8rem,3vw,2.6rem)!important;font-weight:600!important;}}
        h2{{font-size:1.4rem!important;font-weight:500!important;}}
        h3{{font-size:1.08rem!important;font-weight:500!important;}}
        p,label,li,[data-testid="stMarkdownContainer"]{{font-family:'Inter',sans-serif;}}
        code,pre,[data-testid="stMetricValue"]{{font-family:{MONO}!important;}}
        [data-testid="stSidebar"]{{background:#101722;border-right:1px solid var(--border);}}
        [data-testid="stSidebar"]>div:first-child{{padding-top:1.3rem;}}
        [data-testid="stMetric"],[data-testid="stExpander"],[data-testid="stForm"]{{background:var(--surface);border:1px solid var(--border);border-radius:9px;}}
        [data-testid="stMetric"]{{padding:1rem 1.1rem;transition:border-color .18s ease,transform .18s ease;}}
        [data-testid="stMetric"]:hover{{border-color:var(--accent);transform:translateY(-2px);}}
        [data-testid="stMetricLabel"]{{color:var(--muted);font-size:.72rem;letter-spacing:.08em;text-transform:uppercase;}}
        [data-testid="stMetricValue"]{{color:var(--text);font-size:1.55rem;}}
        .stButton>button,.stDownloadButton>button{{border:1px solid var(--border);border-radius:7px;background:var(--surface);color:var(--text);font-weight:500;transition:all .18s ease;}}
        .stButton>button:hover,.stDownloadButton>button:hover{{border-color:var(--accent);background:#222D3E;color:#fff;}}
        .stButton>button:focus-visible,.stDownloadButton>button:focus-visible{{outline:2px solid var(--platinum);outline-offset:2px;}}
        .stTextInput input,.stNumberInput input,.stTextArea textarea,[data-baseweb="select"]>div{{background:#101722;color:var(--text);border-color:var(--border);border-radius:7px;}}
        .stTabs [data-baseweb="tab-list"]{{gap:.8rem;border-bottom:1px solid var(--border);}}
        .stTabs [data-baseweb="tab"]{{background:transparent;color:var(--muted);padding:.7rem .25rem;}}
        .stTabs [aria-selected="true"]{{color:var(--accent)!important;border-bottom:2px solid var(--accent);}}
        [data-testid="stPlotlyChart"]{{background:rgba(20,27,38,.72);border:1px solid var(--border);border-radius:9px;padding:5px;}}
        [data-testid="stDataFrame"]{{border:1px solid var(--border);border-radius:9px;overflow:hidden;}}
        [data-testid="stAlert"]{{border-radius:8px;}}
        a{{color:var(--platinum)!important;}}
        .mast{{display:flex;align-items:center;gap:14px;padding:2px 0 14px;border-bottom:1px solid var(--border);margin-bottom:22px;}}
        .mast .brand{{font-family:'Sora',sans-serif;font-size:20px;color:#fff;letter-spacing:.04em;}}
        .mast .sub{{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-top:2px;}}
        .headline{{font-family:'Sora',sans-serif;font-size:clamp(24px,3vw,34px);line-height:1.2;font-weight:600;color:#fff;margin:0 0 10px;max-width:820px;}}
        .lede{{font-size:14px;line-height:1.65;color:#B8C3D2;max-width:760px;margin:0 0 22px;}}
        .chart-title{{margin:14px 0 6px;font-size:12px;font-weight:600;letter-spacing:.07em;text-transform:uppercase;color:var(--text);}}
        .chart-note{{margin:6px 0 20px;padding:0 0 0 10px;border-left:2px solid var(--accent);color:#B8C3D2;font-size:12.5px;line-height:1.55;}}
        .guide{{border:1px solid var(--border);background:rgba(20,27,38,.8);padding:12px 16px;margin:0 0 18px;border-radius:9px;}}
        .guide h4{{margin:0 0 6px;font-size:12px;text-transform:uppercase;letter-spacing:.07em;color:var(--accent);}}
        .guide ul{{margin:0;padding-left:18px;}}.guide li{{font-size:12.5px;color:#B8C3D2;line-height:1.55;}}
        @media(prefers-reduced-motion:reduce){{*,*::before,*::after{{transition-duration:.01ms!important;animation-duration:.01ms!important;}}}}
        @media(max-width:768px){{.block-container{{padding:1rem 1rem 2rem;}}[data-testid="stMetricValue"]{{font-size:1.25rem;}}}}
        </style>
        """,
        unsafe_allow_html=True,
    )


def inject_css(bg_uri: str = "") -> None:
    """Compatibility entry point used by app.py; inject the shared theme."""
    apply_theme(bg_uri or None)


def register_plotly_theme() -> None:
    """Register the same palette, typography and dark surfaces for Plotly."""
    axis = dict(gridcolor="rgba(154,168,186,0.15)", zerolinecolor="rgba(154,168,186,0.25)", color=p.MUTED)
    pio.templates["dataduo"] = go.layout.Template(
        layout=go.Layout(
            colorway=p.COLORWAY,
            colorscale=dict(sequential=p.SCALE, sequentialminus=p.SCALE, diverging=p.DIVERGING),
            font=dict(family=MONO, size=12, color=p.TEXT),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=axis, yaxis=axis,
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=p.TEXT)),
            margin=dict(l=40, r=20, t=38, b=42),
        )
    )
    pio.templates.default = "dataduo"


def masthead(brand: str, subtitle: str, headline: str, lede: str) -> None:
    mark = (
        "<svg width='40' height='40' viewBox='0 0 40 40' aria-hidden='true'>"
        f"<rect x='2' y='2' width='36' height='36' rx='9' fill='none' stroke='{p.LINE}' stroke-width='1.5'/>"
        f"<rect x='10' y='22' width='4' height='8' fill='{p.ACCENT}'/>"
        f"<rect x='18' y='16' width='4' height='14' fill='{p.TEAL}'/>"
        f"<rect x='26' y='10' width='4' height='20' fill='{p.PLATINUM}'/></svg>"
    )
    st.markdown(
        f"<div class='mast'>{mark}<div><div class='brand'>{html.escape(brand)}</div>"
        f"<div class='sub'>{html.escape(subtitle)}</div></div></div>"
        f"<h1 class='headline'>{html.escape(headline)}</h1><p class='lede'>{html.escape(lede)}</p>",
        unsafe_allow_html=True,
    )


def chart_title(text: str) -> None:
    st.markdown(f"<div class='chart-title'>{html.escape(str(text))}</div>", unsafe_allow_html=True)


def caption(text: str) -> None:
    """Render the explanation directly beneath a chart."""
    st.markdown(f"<div class='chart-note'>{html.escape(str(text))}</div>", unsafe_allow_html=True)


def guide(title: str, items) -> None:
    lis = "".join(f"<li>{html.escape(str(item))}</li>" for item in items)
    st.markdown(f"<div class='guide'><h4>{html.escape(title)}</h4><ul>{lis}</ul></div>", unsafe_allow_html=True)
