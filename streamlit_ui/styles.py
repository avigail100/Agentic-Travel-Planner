"""Styling for the Streamlit Bond travel planner UI."""

import streamlit as st


def apply_app_styles() -> None:
    """Apply page config, theme variables, and CSS overrides."""

    st.set_page_config(
        page_title="Bond Travel Planner",
        page_icon="🤖",
        layout="wide",
    )

    if "ui_theme" not in st.session_state:
        st.session_state.ui_theme = "Light"

    st.markdown(
        """
    <style>
    :root {
        --navy: #071a3d;
        --navy2: #0b2d66;
        --blue: #2563eb;
        --blue2: #1d4ed8;
        --soft-blue: #eaf2ff;
        --border: #dbeafe;
        --text: #0f172a;
        --muted: #64748b;
        --card: rgba(255, 255, 255, 0.72);
        --radius-xl: 28px;
        --radius-lg: 22px;
    }

    .stApp {
        background: var(--app-bg);
        color: var(--text);
    }

    header[data-testid="stHeader"],
    div[data-testid="stToolbar"],
    #MainMenu,
    footer {
        display: none;
    }

    .block-container {
        padding: 0.35rem 0.65rem 0.45rem;
        max-width: 100%;
    }

    /* Main glass panels */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--card);
        border: 1px solid var(--glass-border);
        border-radius: 20px;
        padding: 0.65rem;
        min-height: auto;
        box-shadow: 0 12px 34px rgba(15, 23, 42, 0.10);
        color: var(--text);
        backdrop-filter: blur(20px);
    }

    div[data-testid="stVerticalBlockBorderWrapper"] > div {
        border: none;
    }

    /* Text colors */
    .stMarkdown,
    .stMarkdown p,
    .stMarkdown h1,
    .stMarkdown h2,
    .stMarkdown h3,
    .stCaptionContainer,
    label,
    div[data-testid="stRadio"] label,
    div[data-testid="stTextInput"] label,
    div[data-testid="stTextArea"] label,
    div[data-testid="stSelectbox"] label,
    div[data-testid="stExpander"] summary {
        color: var(--text) !important;
    }

    .stMarkdown h3 {
        font-size: 16px !important;
        margin: 0.65rem 0 0.35rem !important;
    }

    .stMarkdown p {
        margin-bottom: 0.35rem !important;
    }

    div[data-testid="stRadio"] p,
    div[data-testid="stTextInput"] p,
    div[data-testid="stTextArea"] p,
    div[data-testid="stSelectbox"] p,
    div[data-testid="stExpander"] p {
        color: var(--text) !important;
    }

    /* Inputs */
    input,
    textarea,
    div[data-testid="stTextInput"] input,
    div[data-testid="stTextArea"] textarea,
    div[data-testid="stSelectbox"] div[data-baseweb="select"],
    div[data-testid="stChatInput"] textarea {
        color: var(--text) !important;
        background: var(--input-bg) !important;
        border: 1px solid var(--border) !important;
        border-radius: 16px !important;
        caret-color: var(--text) !important;
    }

    input::placeholder,
    textarea::placeholder {
        color: #94a3b8 !important;
        opacity: 1 !important;
    }

    /* Buttons */
    .stButton button,
    button[kind],
    div[data-testid="stFormSubmitButton"] button {
        background: var(--button-bg) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
        border-radius: 15px !important;
        min-height: 32px;
        padding: 0.25rem 0.55rem !important;
        font-size: 13px !important;
        font-weight: 650 !important;
        transition: all 0.18s ease;
    }

    section[data-testid="stSidebar"] .stButton button,
    div[data-testid="stVerticalBlockBorderWrapper"] .stButton button {
        min-height: 30px;
    }

    .stButton button *,
    button[kind] *,
    div[data-testid="stFormSubmitButton"] button * {
        color: inherit !important;
    }

    .stButton button:hover,
    button[kind]:hover,
    div[data-testid="stFormSubmitButton"] button:hover {
        background: var(--button-hover) !important;
        color: var(--navy2) !important;
        border-color: #bfdbfe !important;
        transform: translateY(-1px);
    }

    .stButton button:disabled,
    button[kind]:disabled {
        background: #f1f5f9 !important;
        color: #94a3b8 !important;
        border-color: #e2e8f0 !important;
        transform: none;
    }

    div[data-testid="stBaseButton-primary"] button,
    button[data-testid="stBaseButton-primary"],
    [data-testid="stBaseButton-primary"],
    .stButton button[kind="primary"],
    button[kind="primary"] {
        background: linear-gradient(135deg, var(--blue), var(--blue2)) !important;
        color: #ffffff !important;
        border-color: transparent !important;
        box-shadow: 0 10px 22px rgba(37, 99, 235, 0.24) !important;
    }

    button[data-testid="stBaseButton-primary"] *,
    [data-testid="stBaseButton-primary"] *,
    button[kind="primary"] * {
        color: #ffffff !important;
    }

    /* Hero */
    .hero {
        position: relative;
        overflow: hidden;
        text-align: center;
        padding: 8px 12px 7px;
        margin: 0 0 6px;
        border-radius: 16px;
        background:
            radial-gradient(circle at 20% 0%, rgba(96, 165, 250, 0.22), transparent 32%),
            radial-gradient(circle at 80% 25%, rgba(37, 99, 235, 0.16), transparent 28%),
            var(--hero-bg);
        border: 1px solid var(--glass-border);
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.07);
    }

    .hero h1 {
        margin: 0 0 3px 0;
        font-size: 22px;
        line-height: 1.05;
        letter-spacing: 0;
        color: var(--text);
    }

    .hero p {
        margin: 0;
        color: var(--muted);
        font-size: 11px;
    }

    .sidebar-brand {
        padding: 4px 2px 8px;
    }

    .sidebar-brand h1 {
        margin: 0;
        font-size: 22px;
        letter-spacing: 0;
    }

    .sidebar-brand p {
        margin: 4px 0 0;
        color: var(--muted);
    }

    .side-item,
    .info-card,
    .travel-card,
    .hitl-card {
        background: var(--side-bg);
        border: 1px solid var(--glass-border);
        border-radius: 14px;
        padding: 9px;
        margin: 6px 0;
        color: var(--text);
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
        backdrop-filter: blur(18px);
    }

    .travel-card {
        min-height: 168px;
        transition: all 0.18s ease;
    }

    .travel-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 22px 45px rgba(15, 23, 42, 0.11);
    }

    .travel-card h3,
    .info-card h3,
    .hitl-card h3 {
        margin-top: 0;
        margin-bottom: 10px;
        color: var(--card-title);
        font-size: 18px;
    }

    .card-section-title {
        margin: 20px 0 4px;
        font-size: 20px;
        font-weight: 800;
        color: var(--text);
    }

    .small {
        color: var(--muted);
        font-size: 13px;
    }

    /* Chat */
    .msg-row {
        display: flex;
        gap: 9px;
        align-items: flex-start;
        margin: 7px 0 10px;
    }

    .msg-row.user {
        flex-direction: row-reverse;
    }

    .avatar {
        width: 34px;
        height: 34px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        flex: 0 0 34px;
        font-size: 18px;
        background: var(--avatar-bg);
        border: 1px solid var(--glass-border);
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.08);
    }

    .chat-user,
    .chat-bot {
        padding: 10px 13px;
        line-height: 1.42;
        word-break: break-word;
    }

    .chat-user {
        background: linear-gradient(135deg, var(--blue), var(--blue2));
        color: #ffffff;
        border-radius: 20px 20px 5px 20px;
        max-width: 75%;
        box-shadow: 0 14px 30px rgba(37, 99, 235, 0.24);
    }

    .chat-bot {
        background: var(--message-bg);
        color: var(--text);
        border: 1px solid var(--glass-border);
        border-radius: 20px 20px 20px 5px;
        max-width: 82%;
        box-shadow: 0 12px 26px rgba(15, 23, 42, 0.06);
    }

    .chat-user *, .chat-user {
        color: #ffffff !important;
    }

    .empty-state {
        text-align: center;
        padding: 26px 18px;
        border-radius: 18px;
        background: var(--message-bg);
        border: 1px dashed var(--border);
        color: var(--text);
    }

    .empty-state h2 {
        margin: 0 0 8px;
    }

    .empty-state p {
        margin: 0;
        color: var(--muted);
    }

    .typing {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: var(--message-bg);
        border: 1px solid var(--glass-border);
        border-radius: 999px;
        padding: 9px 14px;
        color: var(--muted);
        box-shadow: 0 10px 26px rgba(15, 23, 42, 0.06);
    }

    .typing-step {
        margin: -8px 0 16px 54px;
        padding: 8px 0 8px 13px;
        border-left: 3px solid var(--blue);
        color: var(--muted);
        font-size: 13px;
        line-height: 1.45;
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:has(.hitl-unified) {
        width: min(900px, calc(82% - 54px)) !important;
        margin: -2px 0 16px 54px !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 8px 24px 24px 24px !important;
        background:
            radial-gradient(circle at 8% 0%, rgba(96, 165, 250, 0.18), transparent 30%),
            linear-gradient(180deg, var(--side-bg), var(--card)) !important;
        box-shadow: 0 22px 58px rgba(15, 23, 42, 0.12) !important;
        overflow: hidden !important;
    }

    .hitl-unified {
        padding: 18px 22px 15px;
        background: linear-gradient(90deg, rgba(37,99,235,0.12), transparent);
        border-radius: 18px;
        border-bottom: 1px solid var(--glass-border);
        margin-bottom: 14px;
    }

    .hitl-unified-row {
        display: flex;
        align-items: center;
        gap: 13px;
    }

    .hitl-unified-body {
        padding: 16px 22px 20px;
    }

    .hitl-unified-note {
        color: var(--text);
        background: var(--message-bg);
        border: 1px solid var(--glass-border);
        border-radius: 16px;
        padding: 13px 14px;
        margin-bottom: 13px;
    }

    .hitl-unified-label {
        font-size: 12px;
        font-weight: 850;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin: 4px 0 8px;
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:has(.hitl-unified) div[data-testid="stCheckbox"] {
        background: var(--message-bg) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 13px !important;
        padding: 7px 10px !important;
        margin: 6px 0 !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:has(.hitl-unified) div[data-testid="stCheckbox"] label {
        align-items: flex-start !important;
    }

    .typing::after {
        content: "";
        width: 26px;
        text-align: left;
        animation: dots 1.3s infinite;
    }

    @keyframes dots {
        0% { content: ""; }
        33% { content: "."; }
        66% { content: ".."; }
        100% { content: "..."; }
    }

    /* Dialog / HITL */
    div[data-testid="stDialog"] > div {
        background: var(--card) !important;
        color: var(--text) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 24px !important;
        box-shadow: 0 34px 90px rgba(0, 0, 0, 0.34) !important;
        backdrop-filter: blur(26px) !important;
        padding: 0.35rem !important;
    }

    div[data-testid="stDialog"] h2,
    div[data-testid="stDialog"] h3,
    div[data-testid="stDialog"] p,
    div[data-testid="stDialog"] label,
    div[data-testid="stDialog"] span,
    div[data-testid="stDialog"] div {
        color: var(--text) !important;
    }


    /* Strong theme override for Streamlit modal. Streamlit's native dialog can keep
       a dark surface even when the page is light, so we theme every dialog layer. */
    div[data-testid="stDialog"],
    div[data-testid="stDialog"] section,
    div[data-testid="stDialog"] div[role="dialog"],
    div[role="dialog"],
    div[aria-modal="true"] {
        background: transparent !important;
        color: var(--text) !important;
    }

    div[data-testid="stDialog"] > div,
    div[role="dialog"] > div,
    div[aria-modal="true"] > div {
        background: var(--card) !important;
        color: var(--text) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 24px !important;
        box-shadow: 0 34px 90px rgba(0, 0, 0, 0.24) !important;
        backdrop-filter: blur(26px) !important;
    }

    /* Dialog title and close button */
    div[data-testid="stDialog"] h2,
    div[data-testid="stDialog"] h3,
    div[role="dialog"] h2,
    div[role="dialog"] h3 {
        color: var(--text) !important;
    }

    div[data-testid="stDialog"] button[aria-label="Close"],
    div[role="dialog"] button[aria-label="Close"] {
        color: var(--text) !important;
        background: var(--button-bg) !important;
        border-radius: 999px !important;
        border: 1px solid var(--glass-border) !important;
    }

    /* HITL form controls */
    div[data-testid="stDialog"] input,
    div[data-testid="stDialog"] textarea,
    div[data-testid="stDialog"] div[data-baseweb="select"] {
        background: var(--input-bg) !important;
        color: var(--text) !important;
        border: 1px solid var(--glass-border) !important;
    }

    div[data-testid="stDialog"] input:focus,
    div[data-testid="stDialog"] textarea:focus {
        border-color: var(--blue) !important;
        box-shadow: 0 0 0 2px rgba(37,99,235,0.18) !important;
        outline: none !important;
    }

    .hitl-shell {
        border-radius: 22px;
        overflow: hidden;
        background:
            radial-gradient(circle at 10% 0%, rgba(96, 165, 250, 0.20), transparent 32%),
            radial-gradient(circle at 92% 15%, rgba(37, 99, 235, 0.14), transparent 34%),
            var(--hitl-bg);
        border: 1px solid var(--glass-border);
        box-shadow: 0 18px 44px rgba(15, 23, 42, 0.14);
    }

    .hitl-top {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 18px 20px 14px;
        border-bottom: 1px solid var(--glass-border);
    }

    .hitl-icon {
        width: 42px;
        height: 42px;
        border-radius: 15px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, var(--blue), var(--blue2));
        color: #ffffff !important;
        font-size: 20px;
        box-shadow: 0 14px 30px rgba(37, 99, 235, 0.24);
        flex: 0 0 42px;
    }

    .hitl-title {
        font-weight: 850;
        font-size: 19px;
        line-height: 1.2;
    }

    .hitl-subtitle {
        color: var(--muted) !important;
        font-size: 13px;
        margin-top: 3px;
    }

    .hitl-body {
        padding: 18px 20px 20px;
    }

    .hitl-question {
        background: var(--message-bg);
        border: 1px solid var(--glass-border);
        border-radius: 18px;
        padding: 14px 16px;
        margin-bottom: 14px;
        font-size: 15px;
        line-height: 1.45;
    }

    .hitl-meta {
        background: var(--message-bg);
        border: 1px solid var(--glass-border);
        border-radius: 16px;
        padding: 10px 12px;
        margin: 8px 0;
        color: var(--muted) !important;
        font-size: 13px;
    }

    .hitl-actions .stButton button {
        min-height: 42px !important;
        border-radius: 14px !important;
        font-size: 14px !important;
        font-weight: 800 !important;
    }

    .hitl-danger button {
        border-color: rgba(239, 68, 68, 0.35) !important;
    }

    .hitl-card {
        border: 1px solid var(--glass-border);
        background: var(--hitl-bg);
        text-align: left;
        padding: 18px;
    }

    hr {
        border: none;
        border-top: 1px solid var(--border);
        margin: 18px 0;
    }

    div[data-testid="stChatInput"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }

    div[data-testid="stChatInput"] > div {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }

    div[data-testid="stChatInput"] > div,
    div[data-testid="stChatInput"] [data-baseweb="textarea"],
    div[data-testid="stChatInput"] [data-baseweb="base-input"],
    div[data-testid="stChatInput"] [data-baseweb="input"],
    div[data-testid="stChatInput"] [data-testid="stChatInputTextArea"],
    div[data-testid="stChatInput"] textarea {
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
    }

    div[data-testid="stChatInput"] * {
        border-color: transparent !important;
    }

    @media (max-width: 1100px) {
        .hero h1 { font-size: 30px; }
        .chat-user, .chat-bot { max-width: 90%; }
    }


    /* Compact Qt-style top header */
    .app-shell-header {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 0 8px;
        margin: 0;
        height: 42px;
        border-radius: 14px;
        background:
            radial-gradient(circle at 12% 0%, rgba(96,165,250,0.26), transparent 30%),
            linear-gradient(90deg, rgba(15,23,42,0.97), rgba(30,58,138,0.94), rgba(37,99,235,0.88));
        border: 1px solid rgba(147,197,253,0.35);
        box-shadow: 0 10px 28px rgba(15,23,42,0.15);
        color: #ffffff;
    }

    .app-title-wrap {
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .app-title {
        font-size: 17px;
        font-weight: 850;
        line-height: 1;
        letter-spacing: -0.02em;
        color: #ffffff;
    }

    .app-subtitle {
        display: none;
    }

    .header-settings-note {
        font-size: 12px;
        color: var(--muted);
        margin-bottom: 6px;
    }

    .header-label {
        font-size: 11px;
        color: var(--muted);
        margin-bottom: -2px;
    }

    .header-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 9px;
        border-radius: 999px;
        background: rgba(255,255,255,0.13);
        border: 1px solid rgba(255,255,255,0.18);
        color: #ffffff;
        font-size: 12px;
        font-weight: 700;
    }

    /* Make topbar controls compact */
    .topbar-control .stButton button,
    .topbar-control button[kind],
    .topbar-menu-button .stButton button {
        min-height: 30px !important;
        height: 30px !important;
        padding: 0.1rem 0.4rem !important;
        border-radius: 11px !important;
        font-size: 12px !important;
    }

    .topbar-menu-button .stButton button {
        font-size: 17px !important;
        width: 34px !important;
        padding: 0 !important;
    }

    /* Left drawer */
    .drawer-panel {
        animation: drawerIn 0.18s ease-out;
    }

    @keyframes drawerIn {
        from { opacity: 0; transform: translateX(-10px); }
        to { opacity: 1; transform: translateX(0); }
    }

    .cards-strip {
        margin-top: 12px;
        padding-top: 10px;
        border-top: 1px solid var(--border);
    }

    .chat-shell-title {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin: 2px 0 8px;
    }

    .chat-status-dot {
        width: 9px;
        height: 9px;
        border-radius: 50%;
        background: #22c55e;
        box-shadow: 0 0 0 5px rgba(34,197,94,0.12);
    }

    </style>
    """,
        unsafe_allow_html=True,
    )

    theme = st.session_state.get("ui_theme", "Light")
    theme_vars = {
        "Light": {
            "app_bg": "linear-gradient(135deg, #eef6ff 0%, #f8fbff 45%, #ffffff 100%)",
            "card": "rgba(255, 255, 255, 0.74)",
            "message_bg": "rgba(255, 255, 255, 0.78)",
            "hitl_bg": "rgba(255, 247, 237, 0.90)",
            "text": "#0f172a",
            "muted": "#64748b",
            "border": "#dbeafe",
            "glass_border": "rgba(191, 219, 254, 0.82)",
            "input_bg": "rgba(255, 255, 255, 0.88)",
            "side_bg": "rgba(248, 251, 255, 0.76)",
            "button_bg": "rgba(255, 255, 255, 0.86)",
            "button_hover": "#eff6ff",
            "hero_bg": "rgba(255, 255, 255, 0.55)",
            "avatar_bg": "rgba(255, 255, 255, 0.72)",
            "card_title": "#0b2a5b",
        },
        "Dark": {
            "app_bg": "linear-gradient(135deg, #06132d 0%, #0f1f3d 58%, #111827 100%)",
            "card": "rgba(15, 23, 42, 0.78)",
            "message_bg": "rgba(17, 24, 39, 0.82)",
            "hitl_bg": "rgba(42, 29, 20, 0.92)",
            "text": "#e5eefc",
            "muted": "#94a3b8",
            "border": "#26456f",
            "glass_border": "rgba(96, 165, 250, 0.28)",
            "input_bg": "rgba(15, 23, 42, 0.92)",
            "side_bg": "rgba(19, 33, 58, 0.78)",
            "button_bg": "rgba(15, 23, 42, 0.86)",
            "button_hover": "#1e3a5f",
            "hero_bg": "rgba(15, 23, 42, 0.52)",
            "avatar_bg": "rgba(30, 41, 59, 0.86)",
            "card_title": "#dbeafe",
        },
    }[theme if theme in {"Light", "Dark"} else "Light"]

    st.markdown(
        f"""
    <style>
    :root {{
        --app-bg: {theme_vars["app_bg"]};
        --card: {theme_vars["card"]};
        --message-bg: {theme_vars["message_bg"]};
        --hitl-bg: {theme_vars["hitl_bg"]};
        --text: {theme_vars["text"]};
        --muted: {theme_vars["muted"]};
        --border: {theme_vars["border"]};
        --glass-border: {theme_vars["glass_border"]};
        --input-bg: {theme_vars["input_bg"]};
        --side-bg: {theme_vars["side_bg"]};
        --button-bg: {theme_vars["button_bg"]};
        --button-hover: {theme_vars["button_hover"]};
        --hero-bg: {theme_vars["hero_bg"]};
        --avatar-bg: {theme_vars["avatar_bg"]};
        --card-title: {theme_vars["card_title"]};
    }}

    input,
    textarea,
    div[data-testid="stTextInput"] input,
    div[data-testid="stChatInput"] textarea {{
        background: var(--input-bg) !important;
    }}

    .side-item,
    .info-card,
    .travel-card {{
        background: var(--side-bg);
    }}

    .hitl-card {{
        background: var(--hitl-bg);
    }}
    </style>
    """,
        unsafe_allow_html=True,
    )



    # ---------------------------------------------------------------------
    # Extra polished app styling
    # ---------------------------------------------------------------------
    st.markdown(
        """
    <style>
    /* Cleaner app spacing */
    .block-container {
        padding-top: 0.25rem !important;
    }

    /* Make every bordered panel feel more like a desktop app panel */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 18px !important;
        box-shadow: 0 18px 45px rgba(15, 23, 42, 0.10) !important;
    }

    /* Header: smaller, flatter, more Qt-like */
    .app-shell-header {
        height: 36px !important;
        border-radius: 12px !important;
        padding: 0 12px !important;
        background: linear-gradient(90deg, #0f172a 0%, #1e3a8a 58%, #2563eb 100%) !important;
    }

    .app-title {
        font-size: 15px !important;
        font-weight: 850 !important;
    }

    .header-pill {
        min-height: 30px;
        justify-content: center;
        border-radius: 11px !important;
        font-size: 11px !important;
        background: var(--side-bg) !important;
        border: 1px solid var(--glass-border) !important;
        color: var(--text) !important;
    }

    /* Drawer: more like ChatGPT/Claude side nav */
    .sidebar-brand {
        padding: 8px 4px 12px !important;
        border-bottom: 1px solid var(--border);
        margin-bottom: 8px;
    }

    .sidebar-brand h1 {
        font-size: 20px !important;
    }

    .side-item {
        padding: 10px 11px !important;
        border-radius: 14px !important;
    }

    /* Recent-chat buttons look like nav rows, not form buttons */
    .drawer-panel + div .stButton button,
    button[data-testid="baseButton-secondary"] {
        text-align: left !important;
    }

    /* Chat area: softer and roomier */
    .chat-shell-title {
        padding: 4px 4px 10px;
        border-bottom: 1px solid var(--border);
        margin-bottom: 10px !important;
    }

    .msg-row {
        margin: 10px 0 14px !important;
    }

    .avatar {
        width: 32px !important;
        height: 32px !important;
        flex-basis: 32px !important;
        font-size: 16px !important;
    }

    .chat-user,
    .chat-bot {
        padding: 11px 14px !important;
        font-size: 14px !important;
        line-height: 1.48 !important;
    }

    .chat-user {
        border-radius: 19px 19px 6px 19px !important;
        max-width: 72% !important;
    }

    .chat-bot {
        border-radius: 19px 19px 19px 6px !important;
        max-width: 78% !important;
    }

    /* Empty state more elegant */
    .empty-state {
        max-width: 640px;
        margin: 72px auto 0;
        padding: 34px 28px !important;
        border-radius: 24px !important;
        border: 1px solid var(--glass-border) !important;
        box-shadow: 0 18px 46px rgba(15, 23, 42, 0.06);
    }

    .empty-state h2 {
        font-size: 24px;
        margin-bottom: 8px !important;
    }

    /* Better input field */
    div[data-testid="stChatInput"] textarea {
        min-height: 46px !important;
        border-radius: 999px !important;
        padding-left: 18px !important;
        padding-right: 18px !important;
    }

    /* HITL modal: force light/dark variables and remove black native surface */
    div[data-testid="stDialog"] > div,
    div[role="dialog"] > div,
    div[aria-modal="true"] > div {
        background: var(--card) !important;
        color: var(--text) !important;
    }

    .hitl-shell {
        background: var(--side-bg) !important;
    }

    .hitl-question,
    .hitl-meta {
        background: var(--message-bg) !important;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )


    # ---------------------------------------------------------------------
    # Final UI polish override: chat input + HITL
    # ---------------------------------------------------------------------
    st.markdown(
        """
    <style>
    /* Chat input: remove Streamlit's dark native wrapper completely */
    section[data-testid="stBottom"],
    div[data-testid="stBottomBlockContainer"],
    div[data-testid="stChatInput"],
    div[data-testid="stChatInput"] > div,
    div[data-testid="stChatInput"] form {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding-left: 0 !important;
        padding-right: 0 !important;
    }

    div[data-testid="stChatInput"] div {
        box-shadow: none !important;
    }

    div[data-testid="stChatInput"] [data-baseweb="textarea"] {
        background: var(--input-bg) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 999px !important;
        overflow: hidden !important;
        box-shadow: 0 12px 34px rgba(15, 23, 42, 0.08) !important;
    }

    div[data-testid="stChatInput"] textarea {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: var(--text) !important;
        min-height: 46px !important;
        padding: 13px 18px !important;
    }

    div[data-testid="stChatInput"] textarea:focus {
        outline: none !important;
        box-shadow: none !important;
    }

    div[data-testid="stChatInput"] button {
        background: linear-gradient(135deg, var(--blue), var(--blue2)) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 14px !important;
        width: 38px !important;
        height: 38px !important;
        margin-right: 4px !important;
        box-shadow: 0 10px 22px rgba(37, 99, 235, 0.20) !important;
    }

    div[data-testid="stChatInput"] button:disabled {
        background: rgba(148, 163, 184, 0.28) !important;
        color: rgba(255,255,255,0.72) !important;
        box-shadow: none !important;
    }

    /* HITL modal: compact, bright in Light mode, deep but readable in Dark mode */
    div[data-testid="stDialog"],
    div[data-testid="stDialog"] section,
    div[data-testid="stDialog"] div[role="dialog"],
    div[role="dialog"],
    div[aria-modal="true"] {
        background: rgba(15, 23, 42, 0.30) !important;
        color: var(--text) !important;
    }

    div[data-testid="stDialog"] > div,
    div[role="dialog"] > div,
    div[aria-modal="true"] > div {
        background:
            radial-gradient(circle at 0% 0%, rgba(96,165,250,0.14), transparent 32%),
            radial-gradient(circle at 100% 10%, rgba(37,99,235,0.10), transparent 30%),
            var(--card) !important;
        color: var(--text) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 26px !important;
        padding: 0.55rem !important;
        box-shadow: 0 34px 90px rgba(15, 23, 42, 0.22) !important;
        backdrop-filter: blur(28px) !important;
    }

    div[data-testid="stDialog"] h2 {
        font-size: 17px !important;
        font-weight: 850 !important;
        color: var(--text) !important;
        opacity: 0.86;
        margin-bottom: 0.7rem !important;
    }

    div[data-testid="stDialog"] button[aria-label="Close"] {
        background: var(--button-bg) !important;
        color: var(--text) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 999px !important;
        width: 32px !important;
        height: 32px !important;
    }

    .hitl-shell {
        border-radius: 24px !important;
        overflow: hidden !important;
        background:
            radial-gradient(circle at 8% 0%, rgba(96, 165, 250, 0.20), transparent 30%),
            linear-gradient(180deg, var(--side-bg), var(--card)) !important;
        border: 1px solid var(--glass-border) !important;
        box-shadow: 0 20px 54px rgba(15, 23, 42, 0.12) !important;
    }

    .hitl-top {
        padding: 18px 20px 15px !important;
        background: linear-gradient(90deg, rgba(37,99,235,0.10), transparent) !important;
        border-bottom: 1px solid var(--glass-border) !important;
    }

    .hitl-icon {
        width: 44px !important;
        height: 44px !important;
        border-radius: 16px !important;
        background: linear-gradient(135deg, var(--blue), var(--blue2)) !important;
        box-shadow: 0 14px 30px rgba(37,99,235,0.25) !important;
    }

    .hitl-title {
        font-size: 18px !important;
        font-weight: 900 !important;
        color: var(--text) !important;
    }

    .hitl-subtitle {
        font-size: 13px !important;
        color: var(--muted) !important;
    }

    .hitl-body {
        padding: 18px 20px 20px !important;
    }

    .hitl-question {
        background: var(--message-bg) !important;
        color: var(--text) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 18px !important;
        padding: 15px 16px !important;
        margin-bottom: 14px !important;
        font-size: 15px !important;
        box-shadow: 0 10px 26px rgba(15,23,42,0.05) !important;
    }

    .hitl-meta {
        background: var(--message-bg) !important;
        color: var(--text) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 16px !important;
        padding: 11px 13px !important;
        margin: 8px 0 !important;
    }

    .hitl-options-title {
        font-size: 12px;
        font-weight: 850;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin: 6px 0 8px;
    }

    .hitl-divider {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 14px 0 10px;
        color: var(--muted);
        font-size: 12px;
    }

    .hitl-divider::before,
    .hitl-divider::after {
        content: "";
        height: 1px;
        flex: 1;
        background: var(--glass-border);
    }

    .hitl-actions {
        margin-top: 14px;
    }

    .hitl-actions .stButton button,
    div[data-testid="stDialog"] .stButton button {
        min-height: 42px !important;
        border-radius: 15px !important;
        font-weight: 850 !important;
        font-size: 13px !important;
    }

    div[data-testid="stDialog"] input,
    div[data-testid="stDialog"] textarea,
    div[data-testid="stDialog"] div[data-baseweb="select"] {
        background: var(--input-bg) !important;
        color: var(--text) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 15px !important;
    }

    div[data-testid="stDialog"] input:focus,
    div[data-testid="stDialog"] textarea:focus {
        border-color: var(--blue) !important;
        box-shadow: 0 0 0 3px rgba(37,99,235,0.16) !important;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )



    # ---------------------------------------------------------------------
    # Clean final layout override
    # ---------------------------------------------------------------------
    st.markdown(
        """
    <style>
    /* Disable Streamlit native chat input completely if any old instance remains. */
    div[data-testid="stChatInput"],
    section[data-testid="stBottom"],
    div[data-testid="stBottomBlockContainer"] {
        display: none !important;
    }

    /* Page layout: compact header, drawer, chat panel aligned to the top. */
    .block-container {
        padding: 0.12rem 0.8rem 0.7rem !important;
    }

    /* The top header panel should not look like a huge card. */
    .block-container > div:first-child div[data-testid="stVerticalBlockBorderWrapper"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
    }

    .app-shell-header {
        height: 32px !important;
        border-radius: 13px !important;
        padding: 0 12px !important;
        background: linear-gradient(90deg, #0f172a 0%, #1e3a8a 56%, #2563eb 100%) !important;
        box-shadow: 0 8px 20px rgba(37,99,235,0.14) !important;
    }

    .app-title {
        font-size: 15px !important;
        font-weight: 900 !important;
    }

    .header-pill {
        height: 28px !important;
        min-height: 28px !important;
        border-radius: 999px !important;
        background: var(--message-bg) !important;
        color: var(--text) !important;
        border: 1px solid var(--glass-border) !important;
        box-shadow: 0 8px 22px rgba(15,23,42,0.05) !important;
    }

    .topbar-theme-button .stButton button {
        height: 28px !important;
        min-height: 28px !important;
        border-radius: 999px !important;
        padding: 0 0.55rem !important;
        font-size: 13px !important;
    }

    .topbar-menu-button .stButton button,
    button[kind="secondary"] {
        box-shadow: 0 8px 22px rgba(15,23,42,0.04) !important;
    }

    /* Main panels: remove the giant empty-card feeling. */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        min-height: auto !important;
    }

    /* Sidebar as a clean drawer. */
    .sidebar-brand {
        padding: 7px 4px 12px !important;
        margin-bottom: 9px !important;
        border-bottom: 1px solid var(--border) !important;
    }
    .sidebar-brand h1 {
        font-size: 20px !important;
    }
    .side-item,
    .info-card {
        border-radius: 16px !important;
        padding: 13px !important;
    }

    /* Chat card fills the useful space and keeps content aligned. */
    .chat-shell-title {
        padding: 5px 4px 11px !important;
        margin-bottom: 8px !important;
        border-bottom: 1px solid var(--border) !important;
    }

    .chat-status-dot {
        width: 10px !important;
        height: 10px !important;
    }

    /* Better message proportions. */
    .chat-user {
        max-width: 68% !important;
        border-radius: 19px 19px 7px 19px !important;
    }
    .chat-bot {
        max-width: 74% !important;
        border-radius: 19px 19px 19px 7px !important;
    }
    .msg-row {
        margin: 11px 0 15px !important;
    }

    /* Empty state should sit in the middle of the chat panel, not create a huge blank page. */
    .empty-state {
        max-width: 620px !important;
        margin: 34px auto 0 !important;
        border-radius: 24px !important;
        border: 1px solid var(--glass-border) !important;
        background: var(--message-bg) !important;
        box-shadow: 0 18px 44px rgba(15,23,42,0.06) !important;
    }

    /* Custom in-panel input: no black bar, no fixed bottom, clean Qt-like composer. */
    .inline-chat-input-shell {
        margin-top: 10px;
        padding: 8px;
        border-radius: 22px;
        background: var(--message-bg);
        border: 1px solid var(--glass-border);
        box-shadow: 0 18px 46px rgba(15,23,42,0.08);
    }

    .inline-chat-input-shell div[data-testid="stForm"] {
        border: none !important;
        background: transparent !important;
        padding: 0 !important;
        box-shadow: none !important;
    }

    .inline-chat-input-shell input {
        min-height: 44px !important;
        border-radius: 999px !important;
        background: var(--input-bg) !important;
        border: 1px solid transparent !important;
        color: var(--text) !important;
        padding: 0 16px !important;
        box-shadow: none !important;
    }

    .inline-chat-input-shell input:focus {
        border-color: var(--blue) !important;
        box-shadow: 0 0 0 3px rgba(37,99,235,0.14) !important;
    }

    .inline-chat-input-shell .stButton button,
    .inline-chat-input-shell div[data-testid="stFormSubmitButton"] button {
        height: 44px !important;
        min-height: 44px !important;
        border-radius: 16px !important;
        background: linear-gradient(135deg, var(--blue), var(--blue2)) !important;
        color: #ffffff !important;
        border: none !important;
        box-shadow: 0 12px 26px rgba(37,99,235,0.22) !important;
        font-size: 18px !important;
        font-weight: 900 !important;
    }

    .inline-chat-input-shell .stButton button:disabled,
    .inline-chat-input-shell div[data-testid="stFormSubmitButton"] button:disabled {
        background: rgba(148,163,184,0.25) !important;
        color: rgba(255,255,255,0.8) !important;
        box-shadow: none !important;
    }

    /* HITL: cleaner centered modal that follows light/dark theme. */
    div[data-testid="stDialog"],
    div[role="dialog"],
    div[aria-modal="true"] {
        background: rgba(15, 23, 42, 0.22) !important;
    }

    div[data-testid="stDialog"] > div,
    div[role="dialog"] > div,
    div[aria-modal="true"] > div {
        background:
            radial-gradient(circle at 0% 0%, rgba(96,165,250,0.16), transparent 34%),
            var(--card) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 28px !important;
        box-shadow: 0 34px 92px rgba(15,23,42,0.24) !important;
        color: var(--text) !important;
        padding: 0.65rem !important;
    }

    .hitl-shell {
        border-radius: 24px !important;
        background: linear-gradient(180deg, var(--side-bg), var(--card)) !important;
        border: 1px solid var(--glass-border) !important;
        box-shadow: 0 22px 58px rgba(15,23,42,0.12) !important;
    }

    .hitl-top {
        padding: 18px 22px 15px !important;
        background: linear-gradient(90deg, rgba(37,99,235,0.12), transparent) !important;
    }

    .hitl-icon {
        width: 46px !important;
        height: 46px !important;
        border-radius: 17px !important;
    }

    .hitl-title {
        font-size: 19px !important;
        font-weight: 900 !important;
    }

    .hitl-body {
        padding: 18px 22px 22px !important;
    }

    .hitl-question,
    .hitl-meta {
        background: var(--message-bg) !important;
        color: var(--text) !important;
        border: 1px solid var(--glass-border) !important;
        box-shadow: 0 10px 26px rgba(15,23,42,0.05) !important;
    }

    .hitl-shell {
        max-width: 880px !important;
        margin: 8px auto 10px !important;
    }

    .hitl-shell + div[data-testid="stHorizontalBlock"],
    .hitl-actions {
        max-width: 880px !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }

    .hitl-actions .stButton button,
    div[data-testid="stDialog"] .stButton button {
        min-height: 44px !important;
        border-radius: 16px !important;
        font-weight: 850 !important;
    }

    .hitl-shell div[data-testid="stCheckbox"] {
        background: var(--message-bg);
        border: 1px solid var(--glass-border);
        border-radius: 13px;
        padding: 7px 10px;
        margin: 6px 0;
    }

    .hitl-shell div[data-testid="stCheckbox"] label {
        align-items: flex-start !important;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )


    # ---------------------------------------------------------------------
    # Visual refresh layer
    # ---------------------------------------------------------------------
    st.markdown(
        """
    <style>
    /* A final, intentional design layer. This keeps the app chat-first while
       adding warmer travel-app contrast and better responsive behavior. */
    :root {
        --accent-teal: #0891b2;
        --accent-coral: #0ea5e9;
        --accent-green: #16a34a;
        --shadow-soft: 0 18px 48px rgba(15, 23, 42, 0.10);
        --shadow-hover: 0 24px 58px rgba(15, 23, 42, 0.14);
    }

    .stApp {
        background:
            linear-gradient(135deg, rgba(224, 242, 254, 0.88) 0%, rgba(248, 250, 252, 0.94) 38%, rgba(255, 247, 237, 0.84) 100%),
            var(--app-bg) !important;
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    [data-testid="stAppViewContainer"] > .main {
        background:
            radial-gradient(circle at 8% 8%, rgba(8, 145, 178, 0.14), transparent 30%),
            radial-gradient(circle at 92% 10%, rgba(249, 115, 22, 0.13), transparent 28%),
            radial-gradient(circle at 82% 88%, rgba(22, 163, 74, 0.08), transparent 28%);
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background:
            linear-gradient(180deg, rgba(255,255,255,0.82), rgba(248,250,252,0.72)),
            var(--card) !important;
        border-color: rgba(186, 230, 253, 0.74) !important;
        border-radius: 18px !important;
        box-shadow: var(--shadow-soft) !important;
    }

    .app-shell-header {
        background:
            linear-gradient(90deg, rgba(7,26,61,0.98), rgba(14,116,144,0.94) 48%, rgba(37,99,235,0.90) 100%) !important;
        border-color: rgba(186, 230, 253, 0.38) !important;
    }

    .app-title::before {
        content: "✈️";
        display: inline-grid;
        place-items: center;
        width: 25px;
        height: 25px;
        margin-right: 8px;
        border-radius: 9px;
        background: rgba(255,255,255,0.16);
    }

    .header-pill {
        font-weight: 800 !important;
    }

    .sidebar-brand h1 {
        color: var(--text) !important;
        letter-spacing: 0;
    }

    .sidebar-brand p {
        font-size: 12px;
        font-weight: 700;
    }

    .side-item,
    .info-card,
    .travel-card {
        background:
            linear-gradient(180deg, rgba(255,255,255,0.68), rgba(248,250,252,0.74)),
            var(--side-bg) !important;
        border-color: rgba(186, 230, 253, 0.70) !important;
    }

    .travel-card {
        min-height: 154px !important;
        border-radius: 16px !important;
    }

    .travel-card:hover {
        box-shadow: var(--shadow-hover) !important;
        border-color: rgba(125, 211, 252, 0.86) !important;
    }

    .chat-shell-title {
        align-items: center;
    }

    .chat-shell-title b {
        font-size: 17px;
        letter-spacing: 0;
    }

    .chat-status-dot {
        background: var(--accent-green) !important;
    }

    .avatar {
        background:
            linear-gradient(180deg, rgba(255,255,255,0.86), rgba(240,249,255,0.78)),
            var(--avatar-bg) !important;
    }

    .chat-bot {
        background:
            linear-gradient(180deg, rgba(255,255,255,0.90), rgba(248,250,252,0.82)),
            var(--message-bg) !important;
    }

    .chat-user {
        background: linear-gradient(135deg, var(--accent-teal), var(--blue)) !important;
    }

    .empty-state {
        text-align: left !important;
        padding: 30px !important;
        background:
            linear-gradient(135deg, rgba(255,255,255,0.88), rgba(240,249,255,0.76) 58%, rgba(255,247,237,0.78)),
            var(--message-bg) !important;
    }

    .empty-state h2 {
        font-size: 27px !important;
        line-height: 1.14;
        color: var(--text);
    }

    .empty-state p {
        max-width: 560px;
        line-height: 1.55;
    }

    .prompt-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 10px;
        margin-top: 20px;
    }

    .prompt-chip {
        min-height: 76px;
        padding: 12px;
        border-radius: 16px;
        background: rgba(255,255,255,0.62);
        border: 1px solid rgba(186, 230, 253, 0.78);
        color: var(--text);
        box-shadow: 0 10px 26px rgba(15,23,42,0.05);
    }

    .prompt-chip b {
        display: block;
        margin-bottom: 4px;
        font-size: 13px;
    }

    .prompt-chip span {
        color: var(--muted);
        font-size: 12px;
        line-height: 1.35;
    }

    .inline-chat-input-shell {
        background:
            linear-gradient(180deg, rgba(255,255,255,0.78), rgba(240,249,255,0.68)),
            var(--message-bg) !important;
        border-color: rgba(125, 211, 252, 0.72) !important;
    }

    .inline-chat-input-shell:empty {
        display: none !important;
    }

    .inline-chat-input-shell input {
        font-size: 14px !important;
    }

    .inline-chat-input-shell div[data-testid="stFormSubmitButton"] button {
        background: linear-gradient(135deg, var(--accent-teal), var(--blue)) !important;
    }

    .card-section-title {
        font-size: 16px !important;
        margin: 18px 0 8px !important;
    }

    div[data-testid="stExpander"] {
        border: 1px solid var(--glass-border) !important;
        border-radius: 16px !important;
        overflow: hidden;
    }

    @media (max-width: 900px) {
        .block-container {
            padding: 0.45rem 0.55rem 0.7rem !important;
        }

        .app-title {
            font-size: 13px !important;
        }

        .header-pill {
            font-size: 0 !important;
            width: 34px;
        }

        .header-pill::after {
            content: "●";
            font-size: 15px;
            color: var(--accent-green);
        }

        .chat-user,
        .chat-bot {
            max-width: 88% !important;
        }

        .prompt-grid {
            grid-template-columns: 1fr;
        }
    }

    /* Dark theme receives the same layout, with darker surfaces instead of pale glass. */
    @media (prefers-color-scheme: dark) {
        .stApp {
            background: var(--app-bg) !important;
        }
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    if theme == "Dark":
        st.markdown(
            """
    <style>
    /* Dark mode readability: the refresh layer above intentionally adds pale
       surfaces for light mode, so dark mode gets explicit darker replacements. */
    .stApp {
        background:
            radial-gradient(circle at 8% 8%, rgba(8,145,178,0.16), transparent 30%),
            radial-gradient(circle at 92% 12%, rgba(249,115,22,0.10), transparent 28%),
            linear-gradient(135deg, #050b18 0%, #0b1628 54%, #111827 100%) !important;
    }

    [data-testid="stAppViewContainer"] > .main {
        background: transparent !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"],
    .side-item,
    .info-card,
    .travel-card,
    .chat-bot,
    .empty-state,
    .prompt-chip,
    .avatar,
    .hitl-shell,
    .hitl-question,
    .hitl-meta {
        background:
            linear-gradient(180deg, rgba(15, 23, 42, 0.94), rgba(17, 24, 39, 0.88)),
            var(--card) !important;
        border-color: rgba(56, 189, 248, 0.24) !important;
        color: var(--text) !important;
    }

    .chat-bot *,
    .empty-state *,
    .prompt-chip *,
    .side-item *,
    .info-card *,
    .travel-card *,
    .hitl-shell *,
    .hitl-question *,
    .hitl-meta * {
        color: inherit !important;
    }

    .small,
    .empty-state p,
    .prompt-chip span,
    .sidebar-brand p,
    .hitl-subtitle {
        color: #a9b7cb !important;
    }

    .travel-card h3,
    .info-card h3,
    .hitl-card h3,
    .card-section-title,
    .sidebar-brand h1,
    .chat-shell-title b {
        color: #f8fafc !important;
    }

    .header-pill {
        background: rgba(15, 23, 42, 0.86) !important;
        color: #f8fafc !important;
    }

    input,
    textarea,
    div[data-testid="stTextInput"] input,
    div[data-testid="stTextArea"] textarea,
    div[data-testid="stSelectbox"] div[data-baseweb="select"] {
        background: rgba(2, 6, 23, 0.82) !important;
        color: #f8fafc !important;
        border-color: rgba(56, 189, 248, 0.28) !important;
    }

    input::placeholder,
    textarea::placeholder {
        color: #94a3b8 !important;
    }

    div[data-testid="stForm"] {
        background:
            linear-gradient(180deg, rgba(15, 23, 42, 0.90), rgba(17, 24, 39, 0.82)) !important;
        border: 1px solid rgba(56, 189, 248, 0.24) !important;
        box-shadow: 0 18px 46px rgba(0, 0, 0, 0.26) !important;
    }
    </style>
    """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
    <style>
    /* Composer cleanup: style Streamlit's real form instead of a raw HTML wrapper.
       This removes the orphan decorative line that appeared above the input. */
    .inline-chat-input-shell {
        display: none !important;
    }

    div[data-testid="stForm"]:has(input[aria-label="Message"]) {
        border-radius: 22px !important;
        padding: 8px !important;
        margin-top: 10px !important;
        background: var(--message-bg) !important;
        border: 1px solid var(--glass-border) !important;
        box-shadow: 0 18px 46px rgba(15,23,42,0.08) !important;
    }

    div[data-testid="stForm"]:has(input[aria-label="Message"])::before,
    div[data-testid="stForm"]:has(input[aria-label="Message"])::after,
    div[data-testid="stForm"]:has(input[aria-label="Message"]) form::before,
    div[data-testid="stForm"]:has(input[aria-label="Message"]) form::after {
        display: none !important;
        content: none !important;
        border: 0 !important;
    }

    div[data-testid="stForm"]:has(input[aria-label="Message"]) input {
        min-height: 44px !important;
        border-radius: 999px !important;
        padding: 0 16px !important;
    }

    div[data-testid="stForm"]:has(input[aria-label="Message"]) div[data-testid="stFormSubmitButton"] button {
        height: 44px !important;
        min-height: 44px !important;
        border-radius: 16px !important;
        background: linear-gradient(135deg, var(--accent-teal), var(--blue)) !important;
        color: #ffffff !important;
        border: none !important;
        box-shadow: 0 12px 26px rgba(37,99,235,0.22) !important;
        font-size: 18px !important;
        font-weight: 900 !important;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <style>
    /* Remove Streamlit's reserved top gutter. In newer Streamlit versions the
       visible .block-container padding is not the only thing creating top space. */
    html,
    body,
    #root,
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewContainer"] > .main,
    [data-testid="stMain"],
    section.main {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }

    header[data-testid="stHeader"],
    div[data-testid="stToolbar"],
    div[data-testid="stDecoration"],
    div[data-testid="stStatusWidget"] {
        height: 0 !important;
        min-height: 0 !important;
        max-height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        display: none !important;
    }

    [data-testid="stMainBlockContainer"],
    .main .block-container,
    .block-container {
        padding: 0.45rem 0.5rem 0.45rem !important;
        margin-top: 0 !important;
    }

    /* Collapse the invisible Streamlit wrappers created by CSS-only st.markdown
       calls. These are the main source of the big blank band above the toolbar. */
    .element-container:has(style),
    .stMarkdown:has(style),
    div[data-testid="stMarkdownContainer"]:has(style) {
        display: none !important;
        height: 0 !important;
        min-height: 0 !important;
        max-height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        overflow: hidden !important;
    }

    .block-container > div:first-child,
    .block-container > div:first-child > div:first-child,
    .block-container > div:first-child div[data-testid="stVerticalBlock"] {
        margin-top: 0.12rem !important;
        padding-top: 0 !important;
    }

    .element-container:has(.topbar-menu-button),
    .element-container:has(.app-shell-header),
    .element-container:has(.header-pill),
    .element-container:has(.topbar-theme-button) {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }

    .block-container > div,
    .block-container div[data-testid="stVerticalBlock"] {
        gap: 0.28rem !important;
    }

    .block-container div[data-testid="stHorizontalBlock"] {
        gap: 0.48rem !important;
    }

    .sidebar-brand {
        padding: 2px 3px 8px !important;
        margin-bottom: 6px !important;
    }

    .sidebar-brand h1 {
        font-size: 17px !important;
    }

    .sidebar-brand p,
    .small {
        font-size: 11px !important;
    }

    .side-item,
    .info-card {
        padding: 9px 11px !important;
        margin: 5px 0 !important;
    }

    .stMarkdown h3 {
        font-size: 14px !important;
        margin: 0.45rem 0 0.25rem !important;
    }

    .chat-shell-title {
        padding: 2px 4px 7px !important;
        margin-bottom: 3px !important;
    }

    .right-rail-title {
        padding: 2px 4px 8px;
        margin-bottom: 6px;
        border-bottom: 1px solid var(--border);
    }

    .right-rail-title b {
        font-size: 15px;
    }

    .right-empty-card {
        margin-top: 8px !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:has(.right-rail-title) .travel-card {
        min-height: auto !important;
        padding: 10px 11px !important;
        margin: 7px 0 !important;
        border-radius: 14px !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:has(.right-rail-title) .travel-card h3 {
        font-size: 14px !important;
        line-height: 1.25 !important;
        margin-bottom: 7px !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:has(.right-rail-title) .card-section-title {
        font-size: 13px !important;
        margin: 12px 0 4px !important;
    }

    .empty-state {
        margin: 14px auto 0 !important;
        padding: 22px !important;
    }

    .prompt-grid {
        margin-top: 13px !important;
    }

    div[data-testid="stForm"]:has(input[aria-label="Message"]) {
        margin-top: 6px !important;
        padding: 6px !important;
    }

    /* Let the two main work areas occupy the page instead of ending mid-screen. */
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.sidebar-brand),
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.chat-shell-title),
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.right-rail-title) {
        min-height: calc(100vh - 74px) !important;
    }

    .chat-bot p,
    .chat-user p {
        margin: 0.2rem 0 0.55rem !important;
    }

    .chat-bot p:last-child,
    .chat-user p:last-child {
        margin-bottom: 0 !important;
    }

    .chat-bot strong {
        color: var(--card-title) !important;
        font-weight: 850;
    }

    .chat-bot code,
    .chat-user code {
        padding: 2px 6px;
        border-radius: 7px;
        background: rgba(15, 23, 42, 0.08);
        color: inherit;
        font-size: 0.92em;
    }

    .chat-bot ul,
    .chat-bot ol {
        margin: 0.35rem 0 0.85rem 1.25rem !important;
        padding: 0 !important;
        list-style-position: outside;
    }

    .chat-bot li {
        margin: 0.28rem 0 !important;
        padding: 0 !important;
        border-radius: 0;
        background: transparent;
        border: none;
    }

    .chat-bot li::marker {
        color: var(--blue);
        font-size: 0.95em;
    }

    .answer-heading {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        margin: 0.25rem 0 0.45rem;
        padding: 4px 9px;
        border-radius: 999px;
        background: rgba(8, 145, 178, 0.10);
        border: 1px solid rgba(8, 145, 178, 0.16);
        color: var(--card-title);
        font-size: 13px;
        font-weight: 850;
    }

    .answer-title {
        display: block;
        margin: 0 0 0.7rem;
        padding: 12px 14px;
        border-radius: 15px;
        background:
            linear-gradient(135deg, rgba(8,145,178,0.14), rgba(37,99,235,0.08)),
            var(--message-bg);
        border: 1px solid rgba(8,145,178,0.18);
        color: var(--card-title);
        font-size: 16px;
        font-weight: 900;
    }

    .answer-subheading {
        display: flex;
        align-items: center;
        gap: 8px;
        margin: 0.82rem 0 0.42rem;
        padding-top: 0.72rem;
        border-top: 1px solid var(--border);
        color: var(--card-title);
        font-size: 14px;
        font-weight: 900;
    }

    .answer-subheading:first-child {
        border-top: none;
        padding-top: 0;
    }

    .answer-section-icon {
        width: 22px;
        height: 22px;
        border-radius: 8px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: rgba(37, 99, 235, 0.10);
        color: var(--blue);
        font-size: 13px;
        flex: 0 0 22px;
    }

    .answer-row {
        display: block;
        margin: 0.28rem 0;
        padding: 0;
        border-radius: 0;
        background: transparent;
        border: none;
    }

    .answer-label {
        color: var(--card-title);
        font-weight: 850;
    }

    .answer-option-line {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 2px 0;
    }

    .answer-option-tab {
        list-style: none;
        margin: 0.45rem 0 0.65rem -0.6rem;
    }

    .answer-option-tab details {
        border-left: 3px solid rgba(37, 99, 235, 0.24);
        padding-left: 0.75rem;
    }

    .answer-option-tab summary {
        cursor: pointer;
        display: inline-flex;
        align-items: center;
        gap: 7px;
        padding: 6px 10px;
        border-radius: 999px;
        background: rgba(37, 99, 235, 0.08);
        border: 1px solid rgba(37, 99, 235, 0.14);
        color: var(--card-title);
        font-weight: 900;
    }

    .answer-option-tab summary::marker,
    .answer-option-tab summary::-webkit-details-marker {
        display: none;
    }

    .answer-option-tab summary::before {
        content: "▸";
        color: var(--blue);
        font-size: 11px;
    }

    .answer-option-tab details[open] summary::before {
        content: "▾";
    }

    .answer-option-detail-list {
        margin: 0.42rem 0 0.2rem 1.35rem !important;
    }

    .answer-option-detail-list li {
        margin: 0.22rem 0 !important;
    }

    .answer-stars {
        color: #f59e0b;
        letter-spacing: 1px;
        white-space: nowrap;
    }

    .answer-muted {
        color: var(--muted);
        font-size: 13px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )


    st.markdown(
        """
    <style>
    /* Parser fix: the assistant answer should read like clean text, not as many
       tiny expandable pills. These rules keep bullets compact and readable. */
    .answer-option-tab,
    .answer-option-tab details,
    .answer-option-tab summary {
        list-style: initial !important;
        display: initial !important;
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        margin: 0 !important;
    }

    .answer-option-tab summary::before {
        content: none !important;
    }

    .chat-bot ul,
    .chat-bot ol {
        margin: 0.35rem 0 0.8rem 1.25rem !important;
        padding-left: 0.8rem !important;
    }

    .chat-bot li {
        margin: 0.24rem 0 !important;
        line-height: 1.42 !important;
    }

    .answer-row {
        display: block !important;
        margin: 0.22rem 0 !important;
    }

    .answer-subheading {
        margin-top: 0.9rem !important;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )


    st.markdown(
        """
    <style>
    /* Final answer rendering: use normal Markdown output, not custom parser tabs. */
    .chat-bot h1,
    .chat-bot h2,
    .chat-bot h3,
    .chat-bot h4 {
        color: var(--card-title) !important;
        margin: 0.85rem 0 0.45rem !important;
        line-height: 1.25 !important;
    }
    .chat-bot h1 {
        font-size: 20px !important;
        padding: 12px 14px;
        border-radius: 16px;
        background: linear-gradient(135deg, rgba(8,145,178,0.12), rgba(37,99,235,0.08));
        border: 1px solid rgba(37,99,235,0.12);
    }
    .chat-bot h2 { font-size: 17px !important; }
    .chat-bot h3 { font-size: 15px !important; }
    .chat-bot p {
        margin: 0.35rem 0 0.65rem !important;
        line-height: 1.55 !important;
    }
    .chat-bot ul,
    .chat-bot ol {
        margin: 0.25rem 0 0.8rem 1.25rem !important;
        padding-left: 1rem !important;
    }
    .chat-bot li {
        margin: 0.25rem 0 !important;
        line-height: 1.48 !important;
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
    }
    .chat-bot li::marker { color: var(--blue); }
    .chat-bot hr {
        border: none !important;
        border-top: 1px solid var(--border) !important;
        margin: 1rem 0 !important;
    }
    .chat-bot table {
        width: 100%;
        border-collapse: collapse;
        margin: 0.65rem 0 0.9rem;
        font-size: 13px;
    }
    .chat-bot th,
    .chat-bot td {
        border: 1px solid var(--border);
        padding: 7px 9px;
        text-align: left;
    }
    .chat-bot th {
        background: rgba(37,99,235,0.08);
        color: var(--card-title);
    }
    .answer-option-tab,
    .answer-option-tab details,
    .answer-option-tab summary,
    .answer-heading,
    .answer-subheading,
    .answer-row,
    .answer-option-line {
        all: unset !important;
    }
    .answer-stars {
        color: #f59e0b !important;
        letter-spacing: 1px;
        white-space: nowrap;
        font-weight: 900;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )


    st.markdown(
        """
    <style>
    /* Clean styled Markdown parser output */
    .travel-answer {
        font-size: 14px;
        line-height: 1.56;
    }

    .travel-answer-title {
        font-size: 21px !important;
        margin: 0 0 0.85rem !important;
        padding: 13px 15px !important;
        border-radius: 17px !important;
        background:
            linear-gradient(135deg, rgba(8,145,178,0.14), rgba(37,99,235,0.08)),
            var(--message-bg) !important;
        border: 1px solid rgba(37,99,235,0.14) !important;
        color: var(--card-title) !important;
    }

    .travel-answer-section {
        font-size: 17px !important;
        margin: 1rem 0 0.45rem !important;
        padding-top: 0.85rem !important;
        border-top: 1px solid var(--border) !important;
        color: var(--card-title) !important;
    }

    .travel-answer-subsection {
        font-size: 15px !important;
        margin: 0.75rem 0 0.35rem !important;
        color: var(--card-title) !important;
    }


    /* Styled parser section headings: emoji labels inside answer text */
    .travel-answer-subsection {
        display: flex !important;
        align-items: center !important;
        gap: 7px !important;
        margin: 0.9rem 0 0.45rem !important;
        padding: 7px 10px !important;
        border-radius: 999px !important;
        background: rgba(37, 99, 235, 0.08) !important;
        border: 1px solid rgba(37, 99, 235, 0.14) !important;
        color: var(--card-title) !important;
        width: fit-content !important;
    }
    .travel-answer-subsection + ul {
        margin-top: 0.25rem !important;
    }

    .travel-answer p {
        margin: 0.35rem 0 0.65rem !important;
    }

    .travel-answer ul,
    .travel-answer ol {
        margin: 0.3rem 0 0.85rem 1.3rem !important;
        padding-left: 0.8rem !important;
    }

    .travel-answer li {
        margin: 0.24rem 0 !important;
        line-height: 1.48 !important;
    }

    .travel-answer li::marker {
        color: var(--blue);
    }

    .travel-answer strong {
        color: var(--card-title) !important;
        font-weight: 850 !important;
    }

    .travel-answer table {
        width: 100%;
        border-collapse: collapse;
        margin: 0.7rem 0 0.9rem;
        font-size: 13px;
        overflow: hidden;
        border-radius: 14px;
    }

    .travel-answer th,
    .travel-answer td {
        border: 1px solid var(--border);
        padding: 7px 9px;
        text-align: left;
    }

    .travel-answer th {
        background: rgba(37,99,235,0.08);
        color: var(--card-title);
    }

    .answer-stars {
        color: #f59e0b !important;
        letter-spacing: 1px;
        white-space: nowrap;
        font-weight: 900;
    }

    .hitl-answer-preview {
        margin-top: 10px;
        padding: 10px 12px;
        border-radius: 15px;
        background: var(--message-bg);
        border: 1px solid var(--glass-border);
        color: var(--muted);
        font-size: 13px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )
