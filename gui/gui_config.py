from pathlib import Path
import plan_and_execute_agent as agent_backend

APP_DIR = Path(__file__).resolve().parent
ROBOT_IMAGE = APP_DIR / "bond_robot.png"

SIF_DESCRIPTIONS = {
    "1": "LOW - approve every plan before execution",
    "2": "MEDIUM - guided with budget/alternatives approval",
    "3": "HIGH - autonomous",
}


LIGHT_THEME = {
    "app_bg": "#F7F9FF",
    "card": "#FFFFFF",
    "card2": "#F4F8FF",
    "text": "#101A3D",
    "muted": "#647299",
    "border": "#DDE7FA",
    "accent": "#3B82F6",
    "accent_hover": "#2563EB",
    "agent_header": "#EAF3FF",
    "agent_border": "#BBD7FF",
    "user_header": "#F0F4FF",
    "user_border": "#D9E2F7",
    "input_bg": "#FFFFFF",
    "bubble_agent": "#F8FBFF",
    "bubble_user": "#F2F7FF",
}

DARK_THEME = {
    "app_bg": "#0B1020",
    "card": "#111827",
    "card2": "#121A2B",
    "text": "#F5F7FB",
    "muted": "#A8B3CF",
    "border": "#263653",
    "accent": "#4DA3FF",
    "accent_hover": "#2F8EF5",
    "agent_header": "#10233B",
    "agent_border": "#326FB4",
    "user_header": "#1B2436",
    "user_border": "#344663",
    "input_bg": "#0F172A",
    "bubble_agent": "#182235",
    "bubble_user": "#1B2740",
}


def set_session_sif(thread_id: str, sif: str) -> dict:
    if hasattr(agent_backend, "set_session_sif"):
        return agent_backend.set_session_sif(thread_id, sif)

    if sif not in ("1", "2", "3"):
        raise ValueError("SIF must be 1, 2, or 3")

    max_replan = getattr(agent_backend, "MAX_REPLAN_CYCLES", 6)
    max_exec = getattr(agent_backend, "MAX_EXECUTOR_STEPS", 4)
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": max_replan * max_exec * 4,
    }

    state = agent_backend.graph.get_state(config)
    prefs = dict((state.values or {}).get("user_preferences") or {})
    prefs["sif"] = sif
    agent_backend.graph.update_state(config, {"user_preferences": prefs})
    return prefs
