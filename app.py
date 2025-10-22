import streamlit as st
import streamlit.components.v1 as components
import os
import time
import pandas as pd
import altair as alt
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

from src.pdf_loader import extract_text_from_pdfs
from src.vectorstore import split_documents, create_vector_store
from src.rag_chain import (
    build_rag_chain,
    GROQ_MODEL_OPTIONS,
    GROQ_MODEL_LABELS,
    GROQ_DEFAULT_MODEL,
    OPENAI_DEFAULT_MODEL,
)

load_dotenv()

st.set_page_config(
    page_title="DocuMind — PDF Intelligence",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

CRIMSON     = "#8B2626"
CRIMSON_DARK= "#6B1D1D"
CRIMSON_LGT = "#A13434"
YELLOW      = "#F1E5A1"
ORANGE      = "#EF6905"
ORANGE_DARK = "#C25302"
GREEN       = "#486C2F"

BG          = "#FAF9F6"
BG2         = "#FFFFFF"
BG3         = "#F2EFE9"
BORDER      = "rgba(0, 0, 0, 0.06)"
BORDER2     = "rgba(0, 0, 0, 0.12)"
T1          = "#1A1A1A"
T2          = "#333333"
T3          = "#666666"
T4          = "#999999"

BROWN       = "#5C4033"

STYLESHEET = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,500;0,600;0,700;1,400;1,600&family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, sans-serif;
    background-color: {BG};
    color: {T1};
    scroll-behavior: smooth;
}}

.stApp {{ background: {BG}; min-height: 100vh; }}

.block-container {{
    padding: 0 !important;
    max-width: 100% !important;
}}

header[data-testid="stHeader"] {{
    background: transparent !important;
    z-index: 10000 !important;
}}
header[data-testid="stHeader"] [data-testid="stToolbar"] {{ display: none !important; }}
header[data-testid="stHeader"] .stAppDeployButton {{ display: none !important; }}
#MainMenu, footer, .stDeployButton {{ display: none !important; visibility: hidden !important; }}

section[data-testid="stSidebar"] {{
    background: {BG2};
    border-right: 1px solid {BORDER2};
    width: 280px !important;
}}
section[data-testid="stSidebar"] > div {{
    padding: 1.5rem 1.25rem;
}}
section[data-testid="stSidebar"][aria-expanded="false"] {{
    display: none !important;
}}

/* Model radio picker */
[data-testid="stSidebar"] .stRadio > div {{
    display: flex;
    flex-direction: column;
    gap: 3px;
}}
[data-testid="stSidebar"] .stRadio label {{
    display: flex;
    align-items: center;
    padding: 0.45rem 0.65rem;
    border-radius: 7px;
    border: 1px solid rgba(0,0,0,0.08);
    cursor: pointer;
    transition: background 0.15s, border-color 0.15s;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.78rem !important;
    color: {T3} !important;
    background: transparent;
    font-weight: 400 !important;
}}
[data-testid="stSidebar"] .stRadio label:has(input:checked) {{
    background: rgba(139,38,38,0.05) !important;
    border-color: {CRIMSON} !important;
    color: {T1} !important;
    font-weight: 600 !important;
}}
[data-testid="stSidebar"] .stRadio label:hover {{
    background: rgba(0,0,0,0.03);
    border-color: rgba(0,0,0,0.15);
}}
[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p {{
    font-family: 'Inter', sans-serif !important;
    font-size: 0.78rem !important;
}}
/* Replace default radio circle with a styled dot */
[data-testid="stSidebar"] .stRadio input[type="radio"] {{
    accent-color: {CRIMSON};
    width: 13px;
    height: 13px;
    margin-right: 2px;
}}

.stTabs [data-baseweb="tab-list"] {{
    background: transparent;
    border-bottom: 1px solid {BORDER2};
    gap: 0;
    padding: 0 0.5rem;
}}
.stTabs [data-baseweb="tab"] {{
    font-family: 'Inter', sans-serif;
    font-size: 0.77rem;
    font-weight: 600;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: {T3};
    padding: 0.8rem 1.5rem;
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    border-radius: 0;
    transition: color 0.2s;
}}
.stTabs [aria-selected="true"] {{
    color: {T1} !important;
    border-bottom: 2px solid {CRIMSON} !important;
    background: transparent !important;
}}

.stButton > button {{
    font-family: 'Inter', sans-serif;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    border-radius: 6px;
    padding: 0.52rem 1rem;
    transition: background 0.15s ease, border-color 0.15s ease, color 0.15s ease;
    border: 1px solid {BORDER2};
    background: {BG2};
    color: {T2} !important;
    cursor: pointer;
    width: 100%;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}}
.stButton > button:hover {{
    border-color: rgba(139,38,38,0.35);
    color: {CRIMSON} !important;
    background: rgba(139,38,38,0.04);
    box-shadow: none;
}}
.stButton > button[kind="primary"],
.stButton > button[kind="primary"] p,
.stButton > button[kind="primary"] span,
.stButton > button[kind="primary"] div {{
    background: {CRIMSON} !important;
    border-color: {CRIMSON} !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    letter-spacing: 0.04em;
    box-shadow: 0 1px 3px rgba(139,38,38,0.25) !important;
    transition: background 0.15s ease !important;
}}
.stButton > button[kind="primary"]:hover,
.stButton > button[kind="primary"]:hover p,
.stButton > button[kind="primary"]:hover span {{
    background: {CRIMSON_DARK} !important;
    border-color: {CRIMSON_DARK} !important;
    color: #ffffff !important;
    box-shadow: 0 2px 6px rgba(139,38,38,0.3) !important;
}}

.stTextInput label, .stSelectbox label, .stFileUploader label {{
    font-family: 'Inter', sans-serif;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: {T3} !important;
}}
.stTextInput > div > div > input {{
    background: {BG2} !important;
    border: 1px solid {BORDER2} !important;
    border-radius: 7px !important;
    color: {T1} !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.88rem !important;
    padding: 0.65rem 0.9rem !important;
    transition: border-color 0.2s;
}}
.stTextInput > div > div > input:focus {{
    border-color: {CRIMSON} !important;
    box-shadow: 0 0 0 3px rgba(139,38,38,0.1) !important;
    outline: none !important;
}}
.stTextInput > div > div > input::placeholder {{
    color: {T4} !important;
}}

.stSelectbox > div > div,
.stSelectbox [data-baseweb="select"] > div {{
    background: {BG2} !important;
    border: 1px solid {BORDER2} !important;
    border-radius: 7px !important;
    color: {T1} !important;
}}
[data-baseweb="popover"] ul {{
    background: {BG3} !important;
    border: 1px solid {BORDER2} !important;
}}
[data-baseweb="popover"] li {{
    color: {T2} !important;
    font-size: 0.84rem !important;
    font-family: 'Inter', sans-serif !important;
}}
[data-baseweb="popover"] li:hover {{
    background: rgba(139,38,38,0.1) !important;
    color: {T1} !important;
}}

.stFileUploader > div,
.stFileUploader section,
.stFileUploader [data-testid="stFileUploaderDropzone"] {{
    background: {BG2} !important;
    border: 1px dashed {BORDER2} !important;
    border-radius: 8px !important;
    transition: border-color 0.2s;
}}
.stFileUploader > div:hover,
.stFileUploader [data-testid="stFileUploaderDropzone"]:hover {{
    border-color: {CRIMSON} !important;
}}
.stFileUploader label,
.stFileUploader [data-testid="stFileUploaderDropzone"] span,
.stFileUploader small {{
    color: {T3} !important;
    font-size: 0.78rem !important;
}}
.stFileUploader [data-testid="stFileUploaderDropzoneInput"] + div button {{
    background: {BG3} !important;
    border: 1px solid {BORDER2} !important;
    color: {T2} !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.78rem !important;
    border-radius: 5px !important;
}}

.stSpinner > div {{ border-top-color: {CRIMSON} !important; }}

[data-testid="stChatMessage"] {{
    background: transparent;
    padding: 0.5rem 0;
    border: none;
}}
[data-testid="stChatMessageContent"] {{
    font-size: 0.91rem;
    line-height: 1.78;
    color: {T1};
    font-family: 'Inter', sans-serif;
}}

.stChatInput > div {{
    background: {BG} !important;
    border: 1px solid {BORDER2} !important;
    border-radius: 10px !important;
    transition: border-color 0.2s;
}}
.stChatInput > div:focus-within {{
    border-color: {CRIMSON} !important;
    box-shadow: 0 0 0 3px rgba(139,38,38,0.08) !important;
}}
.stChatInput textarea, .stChatInput textarea:focus {{
    background: {BG} !important;
    color: {T1} !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
    caret-color: {CRIMSON} !important;
}}
[data-testid="stChatInputContainer"] {{
    background: {BG} !important;
}}
[data-testid="stChatInputContainer"] > div {{
    background: {BG} !important;
    border: 1px solid {BORDER2} !important;
    border-radius: 10px !important;
}}

/* ── Chat Input Bar ── */
section[data-testid="stBottom"] {{
    background: {BG} !important;
    border-top: 1px solid {BORDER} !important;
    padding: 0.75rem 1rem !important;
}}
section[data-testid="stBottom"] > div {{
    background: {BG} !important;
}}
[data-testid="stChatInputContainer"] {{
    background: {BG2} !important;
    border: 1.5px solid {BORDER2} !important;
    border-radius: 14px !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06) !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
    padding: 0.25rem 0.5rem !important;
}}
[data-testid="stChatInputContainer"]:focus-within {{
    border-color: {CRIMSON} !important;
    box-shadow: 0 2px 16px rgba(139,38,38,0.1) !important;
}}
[data-testid="stChatInputContainer"] > div {{
    background: {BG2} !important;
    border: none !important;
    border-radius: 14px !important;
}}
.stChatInput > div {{
    background: {BG2} !important;
    border: none !important;
    border-radius: 14px !important;
}}
.stChatInput > div:focus-within {{
    border: none !important;
    box-shadow: none !important;
}}
.stChatInput textarea, .stChatInput textarea:focus {{
    background: {BG2} !important;
    color: {T1} !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.92rem !important;
    caret-color: {CRIMSON} !important;
    border: none !important;
    outline: none !important;
}}
.stChatInput textarea::placeholder {{
    color: {T4} !important;
    font-style: italic !important;
}}
/* Send button */
[data-testid="stChatInputSubmitButton"] button {{
    background: {CRIMSON} !important;
    border: none !important;
    border-radius: 9px !important;
    width: 36px !important;
    height: 36px !important;
    color: #fff !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    transition: background 0.18s !important;
    cursor: pointer !important;
}}
[data-testid="stChatInputSubmitButton"] button:hover {{
    background: {CRIMSON_DARK} !important;
}}
[data-testid="stChatInputSubmitButton"] button svg {{
    fill: #fff !important;
    stroke: #fff !important;
    width: 18px !important;
    height: 18px !important;
}}

.stExpander {{
    background: {BG2} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 8px !important;
}}
.stExpander summary, .stExpander p {{
    color: {T3} !important;
    font-size: 0.81rem !important;
}}

.stDataFrame {{
    font-size: 0.8rem;
    border: 1px solid {BORDER} !important;
    border-radius: 8px !important;
}}
.stDataFrame th {{
    background: {BG2} !important;
    color: {T3} !important;
    font-weight: 600 !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.07em !important;
    text-transform: uppercase !important;
}}
.stDataFrame td {{ color: {T2} !important; border-color: {BORDER} !important; }}

div[data-testid="stMarkdownContainer"] p {{
    color: {T2};
    font-size: 0.88rem;
    line-height: 1.65;
}}

.stWarning, .stInfo, .stError, .stSuccess {{ font-size: 0.84rem; }}

/* ======================== NAVBAR ======================== */
.dm-nav {{
    position: fixed;
    top: 0; left: 0; right: 0;
    height: 64px;
    background: rgba(255, 255, 255, 0.85);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-bottom: 1px solid {BORDER2};
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 3rem;
    z-index: 9999;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.02);
}}
.dm-nav-left {{
    display: flex;
    align-items: center;
    gap: 2.5rem;
}}
.dm-logo {{
    font-family: 'Lora', Georgia, serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: {T1};
    letter-spacing: -0.01em;
    text-decoration: none;
    cursor: default;
}}
.dm-logo span {{ color: {CRIMSON}; }}
.dm-nav-links {{
    display: flex;
    align-items: center;
    gap: 0;
}}
.dm-nav-link, .dm-nav-link:link, .dm-nav-link:visited {{
    font-family: 'Inter', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: {T3} !important;
    padding: 0.4rem 0.85rem;
    border-radius: 5px;
    cursor: pointer;
    transition: color 0.15s;
    text-decoration: none !important;
}}
.dm-nav-link:hover, .dm-nav-link:active {{ color: {T1} !important; text-decoration: none !important; }}
.dm-nav-right {{
    display: flex;
    align-items: center;
    gap: 0.8rem;
}}
.dm-nav-btn {{
    font-family: 'Inter', sans-serif;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    color: #ffffff;
    background: {ORANGE};
    border: none;
    border-radius: 7px;
    padding: 0.48rem 1.2rem;
    cursor: pointer;
    transition: all 0.16s ease;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08);
}}
.dm-nav-btn:hover {{ background: {ORANGE_DARK}; box-shadow: 0 4px 10px rgba(0, 0, 0, 0.12); }}
.dm-nav-pill {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: {CRIMSON};
    background: rgba(139,38,38,0.1);
    border: 1px solid rgba(139,38,38,0.25);
    border-radius: 100px;
    padding: 0.22rem 0.7rem;
}}

/* ======================== LANDING ======================== */
.dm-page {{ padding-top: 64px; }}

.dm-hero {{
    position: relative;
    padding: 9rem 3rem 7rem;
    max-width: 1100px;
    margin: 0 auto;
    overflow: hidden;
}}
.dm-hero-accent-line {{
    width: 56px;
    height: 3px;
    background: linear-gradient(90deg, {CRIMSON}, {ORANGE});
    border-radius: 2px;
    margin-bottom: 2rem;
}}
.dm-hero-kicker {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: {CRIMSON};
    margin-bottom: 1.4rem;
    display: block;
}}
.dm-hero-title {{
    font-family: 'Lora', Georgia, serif;
    font-size: 5rem;
    font-weight: 700;
    line-height: 1.1;
    color: {T1};
    letter-spacing: -0.04em;
    margin-bottom: 1.8rem;
    max-width: 760px;
}}
.dm-hero-title em {{
    font-style: italic;
    color: {CRIMSON};
    background: none;
    position: relative;
    display: inline-block;
}}
.dm-hero-title em::after {{
    content: '';
    position: absolute;
    left: 0; right: 0; bottom: 2px;
    height: 3px;
    background: linear-gradient(90deg, {CRIMSON}, {ORANGE});
    border-radius: 2px;
    opacity: 0.45;
}}
.dm-hero-body {{
    font-size: 1.1rem;
    color: {T2};
    line-height: 1.75;
    max-width: 560px;
    margin: 0 0 2.5rem 0;
}}
.dm-hero-cta-row {{
    display: flex;
    align-items: center;
    gap: 1.25rem;
    flex-wrap: wrap;
    margin-bottom: 4rem;
}}
.dm-hero-stat {{
    display: flex;
    align-items: flex-end;
    gap: 0.4rem;
}}
.dm-hero-stat-num {{
    font-family: 'Lora', serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: {T1};
    line-height: 1;
}}
.dm-hero-stat-label {{
    font-size: 0.72rem;
    color: {T4};
    font-family: 'Inter', sans-serif;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding-bottom: 0.3rem;
}}
.dm-hero-stats {{
    display: flex;
    gap: 2.5rem;
    padding-top: 2.5rem;
    border-top: 1px solid {BORDER};
}}
.dm-hero-tags {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-bottom: 2.5rem;
}}
.dm-tag {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.63rem;
    font-weight: 500;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: {T3};
    background: {BG2};
    border: 1px solid {BORDER2};
    border-radius: 100px;
    padding: 0.24rem 0.75rem;
}}

.dm-section {{
    max-width: 1160px;
    margin: 0 auto;
    padding: 4.5rem 2rem;
}}
.dm-section-label {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: {T4};
    margin-bottom: 2rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid {BORDER};
}}
.dm-section-title {{
    font-family: 'Lora', Georgia, serif;
    font-size: 2rem;
    font-weight: 700;
    color: {T1};
    margin-bottom: 0.6rem;
    line-height: 1.22;
}}
.dm-section-sub {{
    font-size: 0.95rem;
    color: {T2};
    max-width: 520px;
    line-height: 1.7;
    margin-bottom: 3rem;
}}

.dm-divider {{
    border: none;
    border-top: 1px solid {BORDER};
    margin: 0;
}}

/* Steps */
.dm-steps {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1px;
    background: {BORDER};
    border: 1px solid {BORDER};
    border-radius: 12px;
    overflow: hidden;
}}
.dm-step {{
    background: {BG2};
    padding: 2.25rem 1.75rem;
    position: relative;
}}
.dm-step-num {{
    font-family: 'Lora', Georgia, serif;
    font-size: 2.5rem;
    font-weight: 700;
    color: {BORDER2};
    line-height: 1;
    margin-bottom: 1.25rem;
}}
.dm-step-title {{
    font-family: 'Inter', sans-serif;
    font-size: 0.95rem;
    font-weight: 700;
    color: {T1};
    margin-bottom: 0.55rem;
    letter-spacing: -0.01em;
}}
.dm-step-body {{
    font-size: 0.82rem;
    color: {T3};
    line-height: 1.65;
}}

/* Features */
.dm-features {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1px;
    background: {BORDER};
    border: 1px solid {BORDER};
    border-radius: 12px;
    overflow: hidden;
}}
.dm-feat {{
    background: {BG2};
    padding: 2.25rem 1.85rem;
    transition: background 0.18s;
}}
.dm-feat:hover {{ background: {BG3}; }}
.dm-feat-num {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    color: {ORANGE};
    margin-bottom: 1rem;
    display: block;
}}
.dm-feat-title {{
    font-family: 'Inter', sans-serif;
    font-size: 0.98rem;
    font-weight: 700;
    color: {T1};
    margin-bottom: 0.6rem;
    letter-spacing: -0.01em;
    line-height: 1.3;
}}
.dm-feat-body {{
    font-size: 0.81rem;
    color: {T3};
    line-height: 1.65;
}}
.dm-feat-body code {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: {CRIMSON};
    background: rgba(139,38,38,0.1);
    padding: 0.05rem 0.35rem;
    border-radius: 3px;
}}

/* Tech */
.dm-tech {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1px;
    background: {BORDER};
    border: 1px solid {BORDER};
    border-radius: 10px;
    overflow: hidden;
}}
.dm-tech-cell {{
    background: {BG2};
    padding: 1.35rem 1.4rem;
}}
.dm-tech-name {{
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    font-size: 1.1rem;
    color: {BROWN};
    margin-bottom: 0.4rem;
}}
.dm-tech-desc {{
    font-size: 0.95rem;
    color: {BROWN};
    line-height: 1.5;
    opacity: 0.85;
}}

/* ======================== FOOTER ======================== */
.dm-footer {{
    background: {BG2};
    border-top: 1px solid {BORDER};
    padding: 3.5rem 2rem 2.25rem;
}}
.dm-footer-inner {{
    max-width: 1160px;
    margin: 0 auto;
    display: grid;
    grid-template-columns: 2fr 1fr 1fr;
    gap: 3rem;
    margin-bottom: 2.5rem;
}}
.dm-footer-brand {{
    font-family: 'Lora', Georgia, serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: {T1};
    margin-bottom: 0.65rem;
}}
.dm-footer-brand span {{ color: {CRIMSON}; }}
.dm-footer-tagline {{
    font-size: 0.81rem;
    color: {T4};
    line-height: 1.65;
    max-width: 280px;
    margin-bottom: 1.25rem;
}}
.dm-footer-badge {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.63rem;
    font-weight: 600;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: {CRIMSON};
    background: rgba(139,38,38,0.08);
    border: 1px solid rgba(139,38,38,0.2);
    border-radius: 4px;
    padding: 0.22rem 0.6rem;
    display: inline-block;
}}
.dm-footer-col-title {{
    font-family: 'Inter', sans-serif;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {BROWN};
    margin-bottom: 1rem;
}}
.dm-footer-item {{
    font-size: 0.8rem;
    color: {BROWN};
    opacity: 0.85;
    margin-bottom: 0.45rem;
    line-height: 1.5;
}}
.dm-footer-item code {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 600;
    color: {BROWN};
    background: rgba(92, 64, 51, 0.08);
    padding: 0.03rem 0.3rem;
    border-radius: 3px;
}}
.dm-footer-bottom {{
    max-width: 1160px;
    margin: 0 auto;
    padding-top: 1.75rem;
    border-top: 1px solid {BORDER};
    display: flex;
    align-items: center;
    justify-content: space-between;
}}
.dm-footer-copy {{
    font-size: 0.75rem;
    color: {T4};
    font-family: 'Inter', sans-serif;
}}
.dm-footer-stack {{
    display: flex;
    gap: 1rem;
    align-items: center;
}}
.dm-footer-stack-item {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.63rem;
    color: {T4};
    letter-spacing: 0.06em;
}}

/* ======================== AUTH ======================== */
.dm-auth-page {{
    padding-top: 62px;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    background: {BG};
}}
.dm-auth-body {{
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 4rem 1.5rem;
}}
.dm-auth-card-head {{
    margin-bottom: 2rem;
}}
.dm-auth-title {{
    font-family: 'Lora', Georgia, serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: {T1};
    margin-bottom: 0.4rem;
    line-height: 1.25;
}}
.dm-auth-sub {{
    font-size: 0.84rem;
    color: {T3};
    line-height: 1.6;
}}
.dm-auth-hint {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: {T4};
    line-height: 1.75;
    padding-top: 1.25rem;
    border-top: 1px solid {BORDER};
}}
.dm-auth-hint b {{ color: {CRIMSON}; font-weight: 600; }}

/* ======================== WORKSPACE NAV ======================== */
.dm-ws-nav {{
    position: fixed;
    top: 0; left: 0; right: 0;
    height: 58px;
    background: rgba(250, 249, 246, 0.95);
    backdrop-filter: blur(12px);
    border-bottom: 1px solid {BORDER2};
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 1.5rem;
    z-index: 9999;
}}
.dm-ws-left {{
    display: flex;
    align-items: center;
    gap: 0.75rem;
}}
.dm-ws-logo {{
    font-family: 'Lora', Georgia, serif;
    font-size: 1rem;
    font-weight: 700;
    color: {T1};
}}
.dm-ws-logo span {{ color: {CRIMSON}; }}
.dm-ws-badge {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: {T4};
    background: {BG2};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 0.18rem 0.55rem;
}}
.dm-ws-right {{
    display: flex;
    align-items: center;
    gap: 1rem;
}}
.dm-ws-status {{
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.64rem;
    font-weight: 600;
    letter-spacing: 0.09em;
    text-transform: uppercase;
}}
.dm-ws-status.on {{ color: {CRIMSON}; }}
.dm-ws-status.off {{ color: {T4}; }}
.dm-ws-user {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.66rem;
    color: {T3};
    background: {BG2};
    border: 1px solid {BORDER};
    border-radius: 5px;
    padding: 0.25rem 0.7rem;
}}

.dm-workspace-wrap {{ padding-top: 58px; }}

/* Sidebar labels */
.sb-lbl {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.61rem;
    font-weight: 600;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    color: {T4};
    padding-bottom: 0.6rem;
    display: block;
}}
.sb-rule {{
    border: none;
    border-top: 1px solid {BORDER};
    margin: 1.2rem 0;
}}
.dm-model-badge {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.63rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: {YELLOW};
    background: rgba(233,196,106,0.08);
    border: 1px solid rgba(233,196,106,0.18);
    border-radius: 4px;
    padding: 0.2rem 0.5rem;
    display: inline-block;
    margin-top: 0.4rem;
}}

/* Chat */
.dm-empty-card {{
    background: {BG2};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 2.5rem 2rem;
}}
.dm-empty-title {{
    font-family: 'Inter', sans-serif;
    font-size: 0.95rem;
    font-weight: 700;
    color: {T1};
    margin-bottom: 0.4rem;
    letter-spacing: -0.01em;
}}
.dm-empty-body {{
    font-size: 0.83rem;
    color: {T3};
    line-height: 1.6;
}}

/* Citations */
.dm-cite {{
    background: {BG2};
    border-left: 2px solid {CRIMSON};
    border-radius: 0 7px 7px 0;
    padding: 0.85rem 1.15rem;
    margin-top: 0.6rem;
}}
.dm-cite-meta {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.63rem;
    font-weight: 600;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: {CRIMSON};
    margin-bottom: 0.4rem;
}}
.dm-cite-text {{
    font-size: 0.8rem;
    color: {T3};
    line-height: 1.6;
}}

/* Analytics */
.dm-metric-row {{
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 1px;
    background: {BORDER};
    border: 1px solid {BORDER};
    border-radius: 10px;
    overflow: hidden;
    margin-bottom: 2rem;
}}
.dm-metric {{
    background: {BG2};
    padding: 1.5rem 1.25rem;
    text-align: center;
}}
.dm-metric-num {{
    font-family: 'Lora', Georgia, serif;
    font-size: 2rem;
    font-weight: 700;
    color: {T1};
    line-height: 1;
    margin-bottom: 0.4rem;
}}
.dm-metric-lbl {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    text-transform: uppercase;
    letter-spacing: 0.11em;
    color: {T4};
}}
.dm-chart-lbl {{
    font-family: 'Inter', sans-serif;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: {T4};
    padding: 1.75rem 0 0.75rem;
    border-bottom: 1px solid {BORDER};
    margin-bottom: 0.75rem;
}}
</style>
"""

st.markdown(STYLESHEET, unsafe_allow_html=True)


def init_state():
    defaults = {
        "view": "landing",
        "authenticated": False,
        "user_email": "",
        "api_key": os.getenv("GROQ_API_KEY", os.getenv("OPENAI_API_KEY", "")),
        "selected_model": None,
        "chat_history": [],
        "rag_chain": None,
        "chunk_records": [],
        "doc_count": 0,
        "index_stats": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    if (
        st.session_state.get("selected_model")
        and st.session_state["selected_model"] not in GROQ_MODEL_OPTIONS
    ):
        st.session_state["selected_model"] = GROQ_DEFAULT_MODEL
        st.session_state["rag_chain"] = None

init_state()


FOOTER_HTML = f"""
<div class="dm-footer">
    <div class="dm-footer-inner">
        <div>
            <div class="dm-footer-brand">Docu<span>Mind</span></div>
            <div class="dm-footer-tagline">
                A local-first PDF intelligence tool. Upload documents, index them on your machine, and ask questions in plain language with full source attribution.
            </div>
            <span class="dm-footer-badge">Open Source</span>
        </div>
        <div>
            <div class="dm-footer-col-title">Features</div>
            <div class="dm-footer-item">Local Embeddings</div>
            <div class="dm-footer-item">Source Citations</div>
            <div class="dm-footer-item">Multi-PDF Search</div>
            <div class="dm-footer-item">Chunk Analytics</div>
            <div class="dm-footer-item">Conversation Memory</div>
        </div>
        <div>
            <div class="dm-footer-col-title">Stack</div>
            <div class="dm-footer-item"><code>LangChain</code> RAG orchestration</div>
            <div class="dm-footer-item"><code>FAISS</code> vector store</div>
            <div class="dm-footer-item"><code>HuggingFace</code> embeddings</div>
            <div class="dm-footer-item"><code>Groq</code> / <code>OpenAI</code> LLM</div>
            <div class="dm-footer-item"><code>Streamlit</code> UI framework</div>
        </div>
    </div>
    <div class="dm-footer-bottom">
        <div class="dm-footer-copy">DocuMind &mdash; PDF Document Intelligence</div>
        <div class="dm-footer-stack">
            <span class="dm-footer-stack-item">LangChain</span>
            <span class="dm-footer-stack-item">FAISS</span>
            <span class="dm-footer-stack-item">Streamlit</span>
            <span class="dm-footer-stack-item">Groq / OpenAI</span>
        </div>
    </div>
</div>
"""


if st.session_state.view == "landing":
    # ── JavaScript scroll injector (works via iframe → window.parent) ──
    components.html("""
    <script>
    (function() {
        function scrollTo(id) {
            var el = window.parent.document.getElementById(id);
            if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
        function attachHandlers() {
            var links = window.parent.document.querySelectorAll('[data-scroll-target]');
            if (!links.length) { setTimeout(attachHandlers, 200); return; }
            links.forEach(function(link) {
                link.addEventListener('click', function(e) {
                    e.preventDefault();
                    scrollTo(this.getAttribute('data-scroll-target'));
                });
            });
        }
        attachHandlers();
    })();
    </script>
    """, height=0)

    st.markdown(f"""
    <nav class="dm-nav">
        <div class="dm-nav-left">
            <span class="dm-logo">Docu<span>Mind</span></span>
            <div class="dm-nav-links">
                <a data-scroll-target="features" href="#" class="dm-nav-link">Features</a>
                <a data-scroll-target="how-it-works" href="#" class="dm-nav-link">How it works</a>
                <a data-scroll-target="stack" href="#" class="dm-nav-link">Stack</a>
            </div>
        </div>
        <div class="dm-nav-right">
            <span class="dm-nav-pill">v2.0</span>
        </div>
    </nav>
    <div class="dm-page">
    """, unsafe_allow_html=True)

    # ── Hero ──
    st.markdown(f"""
    <div class="dm-hero">
        <div class="dm-hero-accent-line"></div>
        <span class="dm-hero-kicker">PDF Document Intelligence</span>
        <div class="dm-hero-title">
            Your documents,<br><em>finally answerable.</em>
        </div>
        <div class="dm-hero-body">
            DocuMind transforms static PDFs into a searchable knowledge base.
            Ask any question in plain English and get precise answers,
            every one cited to the exact source page.
        </div>
        <div class="dm-hero-tags">
            <span class="dm-tag">&#x2714;&nbsp; Local Embeddings</span>
            <span class="dm-tag">&#x2714;&nbsp; Source Citations</span>
            <span class="dm-tag">&#x2714;&nbsp; Multi-PDF</span>
            <span class="dm-tag">&#x2714;&nbsp; Groq &amp; OpenAI</span>
            <span class="dm-tag">&#x2714;&nbsp; Conversation Memory</span>
            <span class="dm-tag">&#x2714;&nbsp; Chunk Analytics</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    _, c, _ = st.columns([2.5, 1, 2.5])
    with c:
        if st.button("Open Workspace →", type="primary", use_container_width=True):
            st.session_state.view = "auth"
            st.rerun()

    st.markdown("""
    <div style="max-width:1100px;margin:0 auto;padding:2.5rem 3rem 0;">
        <div style="display:flex;gap:3.5rem;padding-top:2.5rem;border-top:1px solid rgba(0,0,0,0.06);">
            <div>
                <div style="font-family:'Lora',serif;font-size:2.4rem;font-weight:700;color:#1A1A1A;line-height:1;">100%</div>
                <div style="font-size:0.7rem;color:#999;font-family:'Inter',sans-serif;letter-spacing:0.08em;text-transform:uppercase;margin-top:0.3rem;">Local Embeddings</div>
            </div>
            <div>
                <div style="font-family:'Lora',serif;font-size:2.4rem;font-weight:700;color:#1A1A1A;line-height:1;">0</div>
                <div style="font-size:0.7rem;color:#999;font-family:'Inter',sans-serif;letter-spacing:0.08em;text-transform:uppercase;margin-top:0.3rem;">Data Sent to Cloud</div>
            </div>
            <div>
                <div style="font-family:'Lora',serif;font-size:2.4rem;font-weight:700;color:#1A1A1A;line-height:1;">6+</div>
                <div style="font-size:0.7rem;color:#999;font-family:'Inter',sans-serif;letter-spacing:0.08em;text-transform:uppercase;margin-top:0.3rem;">LLM Models</div>
            </div>
            <div>
                <div style="font-family:'Lora',serif;font-size:2.4rem;font-weight:700;color:#1A1A1A;line-height:1;">&infin;</div>
                <div style="font-size:0.7rem;color:#999;font-family:'Inter',sans-serif;letter-spacing:0.08em;text-transform:uppercase;margin-top:0.3rem;">PDFs Supported</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="dm-divider" style="margin-top:3rem;">', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="dm-section" id="how-it-works">
        <div class="dm-section-label">How it works</div>
        <div class="dm-section-title">Four steps from PDF to answer.</div>
        <div class="dm-section-sub">
            The entire pipeline runs on your machine. No document content ever leaves your device &mdash; only inference calls go to the LLM.
        </div>
        <div class="dm-steps">
            <div class="dm-step">
                <div class="dm-step-num">1</div>
                <div class="dm-step-title">Upload PDFs</div>
                <div class="dm-step-body">Drag one or several PDF files into the sidebar. Any file size, any layout &mdash; scanned or digital.</div>
            </div>
            <div class="dm-step">
                <div class="dm-step-num">2</div>
                <div class="dm-step-title">Index locally</div>
                <div class="dm-step-body">Documents are split into chunks, embedded on your hardware using all-MiniLM-L6-v2, and stored in a local FAISS index. No data leaves your machine.</div>
            </div>
            <div class="dm-step">
                <div class="dm-step-num">3</div>
                <div class="dm-step-title">Ask a question</div>
                <div class="dm-step-body">Type any natural-language question. Follow-up questions are automatically rewritten using conversation history so context is never lost.</div>
            </div>
            <div class="dm-step">
                <div class="dm-step-num">4</div>
                <div class="dm-step-title">Get a cited answer</div>
                <div class="dm-step-body">The LLM answers from retrieved chunks only. Every response shows the source document and page number it was drawn from.</div>
            </div>
        </div>
    </div>

    <hr class="dm-divider">

    <div class="dm-section" id="features">
        <div class="dm-section-label">Features</div>
        <div class="dm-section-title">Built for precision, not guessing.</div>
        <div class="dm-section-sub">
            Every design decision optimizes for accuracy and traceability. Hallucinations are structurally impossible because the LLM can only answer from retrieved text.
        </div>
        <div class="dm-features">
            <div class="dm-feat">
                <span class="dm-feat-num">01</span>
                <div class="dm-feat-title">Local Embeddings, Zero API Cost</div>
                <div class="dm-feat-body">Text is embedded on your own hardware using <code>all-MiniLM-L6-v2</code>. No embedding tokens are billed, and no document content is sent to any external service.</div>
            </div>
            <div class="dm-feat">
                <span class="dm-feat-num">02</span>
                <div class="dm-feat-title">Conversation-Aware Retrieval</div>
                <div class="dm-feat-body">Follow-up questions are automatically rewritten as self-contained search queries using the full conversation history. You never need to repeat context.</div>
            </div>
            <div class="dm-feat">
                <span class="dm-feat-num">03</span>
                <div class="dm-feat-title">Exact Source Attribution</div>
                <div class="dm-feat-body">Every answer expands to show the exact document, page number, and text passage the model used. Nothing is fabricated outside the indexed corpus.</div>
            </div>
            <div class="dm-feat">
                <span class="dm-feat-num">04</span>
                <div class="dm-feat-title">Groq and OpenAI Support</div>
                <div class="dm-feat-body">Paste any Groq key (<code>gsk_</code>) or OpenAI key (<code>sk-</code>). The app detects the provider automatically and surfaces the right model selector.</div>
            </div>
            <div class="dm-feat">
                <span class="dm-feat-num">05</span>
                <div class="dm-feat-title">Multi-PDF Cross-Search</div>
                <div class="dm-feat-body">Upload several PDFs at once. Retrieval searches across all indexed documents simultaneously and attributes each chunk to its source file.</div>
            </div>
            <div class="dm-feat">
                <span class="dm-feat-num">06</span>
                <div class="dm-feat-title">Chunk Analytics Dashboard</div>
                <div class="dm-feat-body">After indexing, inspect chunk length distributions, page-level density heatmaps, sequence scatter plots, and per-document coverage with interactive Altair charts.</div>
            </div>
        </div>
    </div>

    <hr class="dm-divider">

    <div class="dm-section" id="stack">
        <div class="dm-section-label">Technology Stack</div>
        <div class="dm-section-title">Production-grade components.</div>
        <div class="dm-section-sub">
            Each library was chosen for reliability and long-term maintenance, not novelty.
        </div>
        <div class="dm-tech">
            <div class="dm-tech-cell">
                <div class="dm-tech-name">LangChain</div>
                <div class="dm-tech-desc">RAG chain orchestration, prompt templates, and message history management</div>
            </div>
            <div class="dm-tech-cell">
                <div class="dm-tech-name">FAISS</div>
                <div class="dm-tech-desc">Facebook AI Similarity Search &mdash; local vector index with no external database required</div>
            </div>
            <div class="dm-tech-cell">
                <div class="dm-tech-name">HuggingFace</div>
                <div class="dm-tech-desc">all-MiniLM-L6-v2 sentence transformer for CPU-efficient local embeddings</div>
            </div>
            <div class="dm-tech-cell">
                <div class="dm-tech-name">Groq / OpenAI</div>
                <div class="dm-tech-desc">LLM inference via user-supplied API key &mdash; Groq for speed, OpenAI for coverage</div>
            </div>
        </div>
    </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(FOOTER_HTML, unsafe_allow_html=True)


elif st.session_state.view == "auth":
    st.markdown(f"""
    <nav class="dm-nav">
        <div class="dm-nav-left">
            <span class="dm-logo">Docu<span>Mind</span></span>
        </div>
        <div class="dm-nav-right">
            <span class="dm-nav-pill">Sign In</span>
        </div>
    </nav>
    <div class="dm-page">
    """, unsafe_allow_html=True)

    st.markdown("<div style='height: 3rem'></div>", unsafe_allow_html=True)

    _, col, _ = st.columns([1, 1.25, 1])
    with col:
        st.markdown(f"""
        <div class="dm-auth-card-head">
            <div class="dm-auth-title">Sign in to DocuMind</div>
            <div class="dm-auth-sub">
                Enter your email and API key to access the workspace.
                Your key is used only for LLM inference and is never stored anywhere.
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.container(border=True):
            email_val = st.text_input(
                "Email address",
                value=st.session_state.user_email or "",
                placeholder="you@domain.com",
            )
            api_val = st.text_input(
                "API Key",
                type="password",
                value=st.session_state.api_key or "",
                placeholder="gsk_... or sk-...",
            )
            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
            c1, c2 = st.columns([1, 1.4])
            with c1:
                if st.button("Back", use_container_width=True):
                    st.session_state.view = "landing"
                    st.rerun()
            with c2:
                if st.button("Sign In", type="primary", use_container_width=True):
                    if not email_val.strip():
                        st.error("Email is required.")
                    elif not api_val.strip():
                        st.error("API key is required.")
                    else:
                        k = api_val.strip()
                        if not (k.startswith("gsk_") or k.startswith("sk-")):
                            st.error("Key must begin with gsk_ (Groq) or sk- (OpenAI).")
                        else:
                            st.session_state.api_key = k
                            st.session_state.user_email = email_val.strip()
                            st.session_state.authenticated = True
                            st.session_state.selected_model = (
                                GROQ_DEFAULT_MODEL if k.startswith("gsk_")
                                else OPENAI_DEFAULT_MODEL
                            )
                            st.session_state.view = "workspace"
                            st.rerun()

        st.markdown(f"""
        <div class="dm-auth-hint" style="margin-top:1.25rem;">
            Groq key format: <b>gsk_xxxxxxxxxxxx</b><br>
            OpenAI key format: <b>sk-xxxxxxxxxxxx</b><br>
            Groq keys unlock model switching. OpenAI keys default to gpt-4o-mini.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


elif st.session_state.view == "workspace":
    if not st.session_state.authenticated:
        st.session_state.view = "auth"
        st.rerun()

    is_groq   = st.session_state.api_key.strip().startswith("gsk_")
    active    = st.session_state.rag_chain is not None
    dot_cls   = "on" if active else "off"
    dot_char  = "&#9679;" if active else "&#9675;"
    status_lbl = "Index Ready" if active else "No Index"

    st.markdown(f"""
    <nav class="dm-ws-nav">
        <div class="dm-ws-left">
            <span class="dm-ws-logo">Docu<span>Mind</span></span>
            <span class="dm-ws-badge">Workspace</span>
        </div>
        <div class="dm-ws-right">
            <span class="dm-ws-status {dot_cls}">{dot_char}&nbsp;{status_lbl}</span>
            <span class="dm-ws-user">{st.session_state.user_email}</span>
        </div>
    </nav>
    <div class="dm-workspace-wrap">
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown('<span class="sb-lbl">Documents</span>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        st.markdown('<hr class="sb-rule"><span class="sb-lbl">Model</span>', unsafe_allow_html=True)

        if is_groq:
            current_model  = st.session_state.selected_model or GROQ_DEFAULT_MODEL
            display_labels = [GROQ_MODEL_LABELS.get(m, m) for m in GROQ_MODEL_OPTIONS]
            current_idx    = (
                GROQ_MODEL_OPTIONS.index(current_model)
                if current_model in GROQ_MODEL_OPTIONS else 0
            )

            chosen_label = st.radio(
                "model_radio",
                options=display_labels,
                index=current_idx,
                label_visibility="collapsed",
                key="model_radio_widget",
            )
            chosen_model = GROQ_MODEL_OPTIONS[display_labels.index(chosen_label)]

            if st.session_state.selected_model != chosen_model:
                st.session_state.selected_model = chosen_model
                if st.session_state.get("vectorstore"):
                    st.session_state.rag_chain = build_rag_chain(
                        st.session_state.vectorstore,
                        st.session_state.api_key,
                        model_override=chosen_model,
                    )
                st.rerun()

            st.markdown('<div class="dm-model-badge">Groq Inference</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div style="font-size:0.82rem;color:{T2};padding:0.2rem 0;font-family:Inter,sans-serif;">'
                f'{OPENAI_DEFAULT_MODEL}</div>'
                f'<div class="dm-model-badge" style="color:{CRIMSON};background:rgba(139,38,38,0.08);border-color:rgba(139,38,38,0.2);">OpenAI Inference</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:0.65rem'></div>", unsafe_allow_html=True)

        if st.button("Index Documents", type="primary", use_container_width=True):
            if not uploaded_files:
                st.warning("Upload at least one PDF first.")
            else:
                with st.spinner("Indexing..."):
                    try:
                        t0   = time.time()
                        docs = extract_text_from_pdfs(uploaded_files)
                        if not docs:
                            st.error("No readable text found in the uploaded files.")
                        else:
                            chunks      = split_documents(docs)
                            vectorstore = create_vector_store(chunks, st.session_state.api_key)
                            
                            st.session_state.vectorstore = vectorstore
                            st.session_state.rag_chain = build_rag_chain(
                                vectorstore,
                                st.session_state.api_key,
                                model_override=st.session_state.selected_model,
                            )
                            records = [
                                {
                                    "ID":       f"C{i:04d}",
                                    "Sequence": i,
                                    "Document": os.path.basename(
                                        c.metadata.get("source", "document.pdf")
                                    ).split("_")[0] if len(
                                        os.path.basename(c.metadata.get("source", ""))
                                    ) > 40 else os.path.basename(c.metadata.get("source", "document.pdf")),
                                    "Page":     c.metadata.get("page", 1),
                                    "Length":   len(c.page_content),
                                    "Preview":  c.page_content[:120] + "...",
                                }
                                for i, c in enumerate(chunks, 1)
                            ]
                            elapsed = time.time() - t0
                            st.session_state.chunk_records = records
                            st.session_state.doc_count     = len(uploaded_files)
                            st.session_state.index_stats   = {
                                "chunks":  len(chunks),
                                "docs":    len(uploaded_files),
                                "pages":   len(docs),
                                "elapsed": elapsed,
                            }
                            st.success(f"{len(chunks)} chunks indexed in {elapsed:.1f}s")
                            st.rerun()
                    except Exception as exc:
                        err = str(exc)
                        if "401" in err or "invalid_api_key" in err:
                            st.error("Invalid API Key. The key you entered was rejected. Sign out and try again with a valid key.")
                        elif "model_decommissioned" in err or "decommissioned" in err:
                            st.error("The selected model has been decommissioned. Choose a different model in the sidebar dropdown.")
                        elif "model_not_found" in err or ("404" in err and "model" in err):
                            st.error("Model not found or not available on your Groq plan. Select Gemma 2 9B from the model dropdown and try again.")
                        else:
                            st.error(f"Indexing failed: {exc}")

        st.markdown('<hr class="sb-rule"><span class="sb-lbl">Session</span>', unsafe_allow_html=True)

        if st.button("Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

        if st.button("Reset Index", use_container_width=True):
            st.session_state.rag_chain     = None
            st.session_state.vectorstore   = None
            st.session_state.chunk_records = []
            st.session_state.doc_count     = 0
            st.session_state.index_stats   = {}
            st.rerun()

        st.markdown('<hr class="sb-rule">', unsafe_allow_html=True)

        if st.button("Sign Out", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

    tab_chat, tab_analytics = st.tabs(["Conversation", "Analytics"])

    with tab_chat:
        if not st.session_state.chat_history:
            if st.session_state.rag_chain:
                st.markdown(f"""
                <div class="dm-empty-card">
                    <div class="dm-empty-title">Index ready</div>
                    <div class="dm-empty-body">Your documents are indexed. Ask anything in the input below.</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="dm-empty-card">
                    <div class="dm-empty-title">No documents indexed yet</div>
                    <div class="dm-empty-body">Upload one or more PDF files in the sidebar, select a model, then click Index Documents to begin.</div>
                </div>
                """, unsafe_allow_html=True)

        for msg in st.session_state.chat_history:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            with st.chat_message(role):
                st.markdown(msg.content)

        query = st.chat_input("Ask something about your documents...")
        if query:
            if not st.session_state.rag_chain:
                st.info("Index your documents first using the sidebar.")
            else:
                st.session_state.chat_history.append(HumanMessage(content=query))
                with st.chat_message("user"):
                    st.markdown(query)

                with st.chat_message("assistant"):
                    with st.spinner(""):
                        try:
                            result = st.session_state.rag_chain.invoke({
                                "input": query,
                                "chat_history": st.session_state.chat_history[:-1],
                            })
                            answer = result["answer"]
                            # Strip all reasoning/thinking blocks from models like Qwen3
                            import re
                            answer = re.sub(r"<think>.*?</think>\s*", "", answer, flags=re.DOTALL)
                            answer = re.sub(r"<draft>.*?</draft>\s*", "", answer, flags=re.DOTALL)
                            # Strip any orphaned opening/closing tags
                            answer = re.sub(r"</?think>|</?draft>", "", answer)
                            answer = answer.strip()
                            
                            st.markdown(answer)

                            ctx = result.get("context", [])
                            if ctx:
                                with st.expander("Sources"):
                                    for i, doc in enumerate(ctx, 1):
                                        src = doc.metadata.get("source", "Document")
                                        pg  = doc.metadata.get("page", "N/A")
                                        st.markdown(f"""
                                        <div class="dm-cite">
                                            <div class="dm-cite-meta">Ref {i:02d} &mdash; {src} &mdash; Page {pg}</div>
                                            <div class="dm-cite-text">{doc.page_content[:400]}...</div>
                                        </div>
                                        """, unsafe_allow_html=True)

                            st.session_state.chat_history.append(AIMessage(content=answer))
                        except Exception as exc:
                            err = str(exc)
                            if "401" in err or "invalid_api_key" in err:
                                st.error("Invalid API Key. Click Sign Out in the sidebar and sign in again with a valid key.")
                            elif "model_decommissioned" in err or "decommissioned" in err:
                                st.error("The selected model has been decommissioned. Choose a different model in the sidebar.")
                            elif "model_not_found" in err or ("404" in err and "model" in err):
                                st.error("Model not found or not available on your Groq plan. Select a different model in the sidebar (Gemma 2 9B is recommended).")
                            elif "429" in err or "rate_limit" in err:
                                st.error("Rate limit reached. Please wait a moment and try again.")
                            elif "context_length" in err or "context window" in err:
                                st.error("Your question plus the retrieved context exceeded the model token limit. Try a shorter question.")
                            else:
                                st.error(f"Query failed: {exc}")

    with tab_analytics:
        if not st.session_state.chunk_records:
            st.markdown(f"""
            <div class="dm-empty-card">
                <div class="dm-empty-title">No index data yet</div>
                <div class="dm-empty-body">Upload one or more PDFs in the sidebar and click Index Documents. The analytics dashboard will appear here with chunk distribution charts, page density maps, and a full chunk inspector table.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            df    = pd.DataFrame(st.session_state.chunk_records)
            stats = st.session_state.index_stats
            elapsed_s = stats.get('elapsed', 0)
            elapsed_lbl = f"{elapsed_s:.1f}s" if elapsed_s < 60 else f"{elapsed_s/60:.1f}m"

            st.markdown(f"""
            <div class="dm-metric-row">
                <div class="dm-metric">
                    <div class="dm-metric-num">{stats.get('chunks', len(df))}</div>
                    <div class="dm-metric-lbl">Total Chunks</div>
                </div>
                <div class="dm-metric">
                    <div class="dm-metric-num">{stats.get('docs', st.session_state.doc_count)}</div>
                    <div class="dm-metric-lbl">Documents</div>
                </div>
                <div class="dm-metric">
                    <div class="dm-metric-num">{int(df['Length'].mean()) if not df.empty else 0}</div>
                    <div class="dm-metric-lbl">Avg Chunk (chars)</div>
                </div>
                <div class="dm-metric">
                    <div class="dm-metric-num">{df['Page'].nunique()}</div>
                    <div class="dm-metric-lbl">Unique Pages</div>
                </div>
                <div class="dm-metric">
                    <div class="dm-metric-num">{elapsed_lbl}</div>
                    <div class="dm-metric-lbl">Index Time</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            doc_agg = df.groupby("Document").agg(
                Chunks=("ID", "count"),
                Pages=("Page", "nunique"),
                Avg_Length=("Length", "mean"),
                Total_Chars=("Length", "sum"),
            ).reset_index()
            doc_agg["Avg_Length"] = doc_agg["Avg_Length"].round(0).astype(int)

            chart_cfg = dict(
                background=BG2,
                axis=alt.AxisConfig(
                    labelColor=T2, titleColor=T3, gridColor="rgba(0,0,0,0.05)",
                    domainColor=BORDER2, tickColor=BORDER2,
                    labelFont="Inter", titleFont="Inter",
                    labelFontSize=11, titleFontSize=11,
                ),
                legend=alt.LegendConfig(
                    labelColor=T2, titleColor=T2,
                    labelFont="Inter", titleFont="Inter",
                    labelFontSize=11, titleFontSize=11,
                ),
                title=alt.TitleConfig(color=T1, font="Inter", fontSize=12, fontWeight=600),
                view=alt.ViewConfig(strokeWidth=0),
                padding={"left": 12, "right": 12, "top": 16, "bottom": 12},
            )
            palette = [CRIMSON, ORANGE, "#486C2F", CRIMSON_DARK, "#6B4F3A", "#C77DFF", "#E8C4A0"]

            st.markdown('<div class="dm-chart-lbl">Chunks per Document</div>', unsafe_allow_html=True)
            st.caption("How many text chunks each PDF contributed to the index. Higher chunk counts indicate longer or denser documents.")
            bar = (
                alt.Chart(doc_agg)
                .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    x=alt.X("Document:N", sort="-y", title=None,
                            axis=alt.Axis(labelAngle=-25, labelLimit=180)),
                    y=alt.Y("Chunks:Q", title="Chunk count"),
                    color=alt.Color("Document:N",
                                    scale=alt.Scale(range=palette), legend=None),
                    tooltip=[
                        alt.Tooltip("Document:N", title="Document"),
                        alt.Tooltip("Chunks:Q", title="Chunks"),
                        alt.Tooltip("Pages:Q", title="Pages"),
                        alt.Tooltip("Avg_Length:Q", title="Avg Chars/Chunk"),
                        alt.Tooltip("Total_Chars:Q", title="Total Characters"),
                    ],
                )
                .properties(height=260)
                .configure(**chart_cfg)
            )
            st.altair_chart(bar, use_container_width=True)

            st.markdown('<div class="dm-chart-lbl">Chunk Length Distribution</div>', unsafe_allow_html=True)
            st.caption("Distribution of character lengths across all chunks. Ideally chunks cluster between 400–800 characters — short chunks miss context, long chunks dilute relevance.")
            c1, c2 = st.columns(2)

            with c1:
                hist = (
                    alt.Chart(df)
                    .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3, opacity=0.85)
                    .encode(
                        x=alt.X("Length:Q", bin=alt.Bin(maxbins=28), title="Character Length"),
                        y=alt.Y("count():Q", title="Chunk Count"),
                        color=alt.value(CRIMSON),
                        tooltip=[
                            alt.Tooltip("Length:Q", bin=True, title="Length range"),
                            alt.Tooltip("count():Q", title="Count"),
                        ],
                    )
                    .properties(height=260, title="Length Histogram")
                    .configure(**chart_cfg)
                )
                st.altair_chart(hist, use_container_width=True)

            with c2:
                df["LenBucket"] = pd.cut(
                    df["Length"],
                    bins=[0, 200, 400, 600, 800, 1200, 99999],
                    labels=["< 200", "200–400", "400–600", "600–800", "800–1200", "> 1200"],
                )
                bucket_agg = df.groupby("LenBucket", observed=True).size().reset_index(name="Count")
                donut = (
                    alt.Chart(bucket_agg)
                    .mark_arc(innerRadius=52, outerRadius=94, cornerRadius=3)
                    .encode(
                        theta=alt.Theta("Count:Q"),
                        color=alt.Color(
                            "LenBucket:N",
                            scale=alt.Scale(range=palette),
                            legend=alt.Legend(title="Length Range", orient="right"),
                        ),
                        tooltip=[
                            alt.Tooltip("LenBucket:N", title="Range"),
                            alt.Tooltip("Count:Q", title="Chunks"),
                        ],
                    )
                    .properties(height=260, title="Chunk Size Breakdown")
                    .configure(**chart_cfg)
                )
                st.altair_chart(donut, use_container_width=True)

            st.markdown('<div class="dm-chart-lbl">Sequence vs. Chunk Length</div>', unsafe_allow_html=True)
            st.caption("Each point is one chunk plotted by its position in the document (x) and its character length (y). Gaps or abrupt length drops can indicate page breaks, headings, or sparsely-written sections.")
            scatter = (
                alt.Chart(df)
                .mark_circle(opacity=0.72, size=52)
                .encode(
                    x=alt.X("Sequence:Q", title="Chunk Position in Document",
                            scale=alt.Scale(zero=False)),
                    y=alt.Y("Length:Q", title="Character Length",
                            scale=alt.Scale(zero=False)),
                    color=alt.Color(
                        "Document:N",
                        scale=alt.Scale(range=palette),
                        legend=alt.Legend(
                            title="Document", orient="bottom",
                            direction="horizontal",
                        ),
                    ),
                    tooltip=[
                        alt.Tooltip("ID:N", title="Chunk ID"),
                        alt.Tooltip("Document:N", title="Document"),
                        alt.Tooltip("Page:Q", title="Page"),
                        alt.Tooltip("Length:Q", title="Characters"),
                        alt.Tooltip("Preview:N", title="Preview"),
                    ],
                )
                .properties(height=280, title="Chunk Position vs. Length")
                .configure(**chart_cfg)
            )
            st.altair_chart(scatter, use_container_width=True)

            st.markdown('<div class="dm-chart-lbl">Page-Level Chunk Density</div>', unsafe_allow_html=True)
            st.caption("A heatmap showing how many chunks were extracted from each page of each document. Dark cells indicate content-heavy pages; light cells indicate sparse or image-heavy pages.")
            page_df = df.groupby(["Document", "Page"]).agg(Chunks=("ID", "count")).reset_index()
            heatmap = (
                alt.Chart(page_df)
                .mark_rect(cornerRadius=2)
                .encode(
                    x=alt.X("Page:O", title="Page Number"),
                    y=alt.Y("Document:N", title=None),
                    color=alt.Color(
                        "Chunks:Q",
                        scale=alt.Scale(range=[BG3, CRIMSON],
                                        domain=[0, page_df["Chunks"].max()]),
                        legend=alt.Legend(title="Chunks / Page"),
                    ),
                    tooltip=[
                        alt.Tooltip("Document:N", title="Document"),
                        alt.Tooltip("Page:O", title="Page"),
                        alt.Tooltip("Chunks:Q", title="Chunks on page"),
                    ],
                )
                .properties(
                    height=max(100, 56 * df["Document"].nunique()),
                    title="Chunk Density by Page",
                )
                .configure(**chart_cfg)
            )
            st.altair_chart(heatmap, use_container_width=True)

            st.markdown('<div class="dm-chart-lbl">Chunk Inspector</div>', unsafe_allow_html=True)
            st.caption("Full table of every indexed chunk. Use this to verify text extraction quality and spot chunks that are too short (truncated) or too long (missed a split).")
            st.dataframe(
                df[["ID", "Document", "Page", "Length", "Preview"]].rename(
                    columns={"Length": "Chars", "Preview": "Text Preview"}
                ),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "ID":          st.column_config.TextColumn("ID", width="small"),
                    "Document":    st.column_config.TextColumn("Document", width="medium"),
                    "Page":        st.column_config.NumberColumn("Page", width="small"),
                    "Chars":       st.column_config.NumberColumn("Chars", width="small", help="Number of characters in this chunk"),
                    "Text Preview":st.column_config.TextColumn("Text Preview", width="large"),
                },
            )

    st.markdown("</div>", unsafe_allow_html=True)
