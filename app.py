import uuid
from typing import Any, Dict, List, Tuple
import sqlite3

import streamlit as st
from streamlit_ui.cards import render_cards, render_saved_panel
from streamlit_ui.formatting import (
    as_float,
    esc,
    markdown_to_html,
    message_to_html,
)
from streamlit_ui.persistence import (
    apply_trip_cards_update,
    bootstrap_user_conversations,
    init_ui_tables,
    is_new_trip_request,
    load_conversation_cards_from_db,
    load_conversation_messages_from_db,
    load_user_conversations_from_db,
    make_chat_title,
    save_conversation_to_db,
)
from streamlit_ui.styles import apply_app_styles
from langgraph.types import Command
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from plan_and_execute_agent import (
    graph,
    _extract_budget,
    MAX_REPLAN_CYCLES,
    MAX_EXECUTOR_STEPS,
    PROGRESS_MAP,
    build_structured_data_from_messages,
    set_session_sif,
)

# ---------------------------------------------------------------------
# App constants
# ---------------------------------------------------------------------

DEFAULT_SESSION_ID = "demo_session"


# ---------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------

apply_app_styles()

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def default_config(thread_id: str) -> Dict[str, Any]:
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": MAX_REPLAN_CYCLES * MAX_EXECUTOR_STEPS * 8,
    }


def init_graph_state_if_needed(thread_id: str) -> Tuple[dict, str, list]:
    """Load existing graph memory or initialize it. Returns prefs, chat_history, approved_hosts."""
    config = default_config(thread_id)

    try:
        existing = graph.get_state(config)
        if not existing or not existing.values:
            graph.update_state(
                config,
                {
                    "input": "",
                    "plan": [],
                    "past_steps": [],
                    "response": "",
                    "executor_steps": 0,
                    "replan_count": 0,
                    "total_budget": 0.0,
                    "calculated_total": 0.0,
                    "over_budget": False,
                    "user_preferences": {},
                    "chat_history": "",
                    "reflection_memory": [],
                    "critic_passed": False,
                    "critic_count": 0,
                    "messages": [],
                    "approved_hosts": [],
                },
            )
            return {}, "", []

        values = existing.values or {}
        prefs = values.get("user_preferences") or {}
        chat_history = values.get("chat_history") or ""
        approved_hosts = values.get("approved_hosts") or []
        return prefs, chat_history, approved_hosts

    except Exception:
        return {}, "", []


def build_initial_state(user_input: str, thread_id: str, is_resume: bool = False) -> Dict[str, Any]:
    config = default_config(thread_id)
    try:
        existing = graph.get_state(config)
        sv = existing.values or {} if existing else {}
    except Exception:
        sv = {}

    saved_prefs = sv.get("user_preferences") or {}
    saved_chat_history = sv.get("chat_history") or ""
    saved_hosts = sv.get("approved_hosts") or []
    has_crashed = bool(sv.get("crashed", False))

    if is_resume and has_crashed:
        return {
            "input":            sv.get("input", ""),
            "plan":             sv.get("plan", []),
            "past_steps":       sv.get("past_steps", []),
            "response":         "",
            "executor_steps":   sv.get("executor_steps", 0),
            "replan_count":     sv.get("replan_count", 0),
            "total_budget":     sv.get("total_budget", 0.0),
            "calculated_total": sv.get("calculated_total", 0.0),
            "over_budget":      sv.get("over_budget", False),
            "user_preferences": saved_prefs,
            "chat_history":     saved_chat_history,
            "reflection_memory": sv.get("reflection_memory", []),
            "critic_passed":    False,
            "critic_count":     sv.get("critic_count", 0),
            "messages":         [],        
            "approved_hosts":   list(saved_hosts),
            "crashed":          True,      
            "executed_tool_calls": sv.get("executed_tool_calls", []),
            "security_flag":    False,
        }

    return {
        "input": user_input,
        "plan": [],
        "past_steps": [],
        "response": "",
        "executor_steps": 0,
        "replan_count": 0,
        "total_budget": _extract_budget(user_input),
        "calculated_total": 0.0,
        "over_budget": False,
        "user_preferences": saved_prefs,
        "chat_history": saved_chat_history,
        "reflection_memory": [],
        "critic_passed": False,
        "critic_count": 0,
        "messages": [],
        "approved_hosts": list(saved_hosts),
        "security_flag": False,
    }


def get_existing_message_count(thread_id: str) -> int:
    try:
        state = graph.get_state(default_config(thread_id))
        return len((state.values or {}).get("messages") or [])
    except Exception:
        return 0


def get_final_values(thread_id: str) -> Dict[str, Any]:
    try:
        state = graph.get_state(default_config(thread_id))
        return state.values or {}
    except Exception:
        return {}


def is_greeting_or_too_short(message: str) -> bool:
    clean = (message or "").strip().lower()
    return (
        len(clean) < 4
        or clean in {
            "hi", "hey", "hello", "h", "yo",
            "שלום", "היי", "הי", "הלו",
        }
    )


def greeting_response() -> str:
    return (
        "Hi, I'm Bond 🤖✈️\n\n"
        "Tell me what trip you want to plan. For example:\n\n"
        "- Plan a 5-day trip to Paris\n"
        "- Find flights and hotels in Tokyo\n"
        "- Suggest a warm beach destination in April"
    )


# ---------------------------------------------------------------------
# HITL handling
# ---------------------------------------------------------------------

def extract_interrupt_payload(chunk: dict) -> dict:
    intr = chunk["__interrupt__"]
    if isinstance(intr, (list, tuple)):
        return intr[0].value
    return intr.value


def hitl_prompt_message(payload: dict) -> str:
    kind = (payload or {}).get("type", "approval")

    if kind == "user_question":
        question = payload.get("question") or "I need one more detail before I continue."
        options = payload.get("options") or []
        if options:
            choices = "\n".join(f"- {option}" for option in options)
            return f"{question}\n\nQuick choices:\n{choices}"
        return str(question)

    if kind == "web_host_approval":
        query = payload.get("query", "")
        category = payload.get("category", "")
        parts = ["I need your approval before using web search."]
        if query:
            parts.append(f"**Query:** {query}")
        if category:
            parts.append(f"**Category:** {category}")
        return "\n\n".join(parts)

    if kind == "sif_plan_approval":
        plan = payload.get("plan") or []
        if plan:
            steps = "\n".join(f"{idx}. {step}" for idx, step in enumerate(plan, 1))
            return f"I made a plan and need your approval before I execute it.\n\n{steps}"
        return "I made a plan and need your approval before I execute it."

    if kind == "sif_alternatives_offer":
        return "I found initial options. Would you like me to continue automatically, or narrow the search with a preference?"

    if kind == "sif_budget_breach":
        budget = as_float(payload.get("budget"), 0.0)
        total_cost = as_float(payload.get("total_cost"), 0.0)
        overage = as_float(payload.get("overage"), max(0.0, total_cost - budget))
        return (
            f"The estimated cost is **${total_cost:,.2f}**, which is "
            f"**${overage:,.2f}** over your **${budget:,.2f}** budget. "
            "Do you want to approve it or set a new budget?"
        )

    return "I paused and need your decision before I continue."


def user_friendly_run_error(error: Exception) -> str:
    message = str(error)
    if "GRAPH_RECURSION_LIMIT" in message or "Recursion limit" in message:
        return (
            "I got stuck while refining this trip and stopped before looping too long.\n\n"
            "Please ask me for the final result again, or narrow the request a bit "
            "(for example: dates, budget, flights only, hotels only, or a specific destination)."
        )
    return f"I hit a problem while planning this trip: {message}"


def run_until_pause_or_done(stream_input: Any, thread_id: str) -> None:
    """Run the graph until it finishes or hits a real interrupt.
    This is the Streamlit equivalent of what worked in Qt:
    - no fake answer
    - save pending interrupt
    - wait for a user button / text
    - resume with Command(resume=...)
    """
    config = default_config(thread_id)
    final_response = ""

    status_box = st.empty()

    try:
        for chunk in graph.stream(stream_input, config, stream_mode="updates"):
            if not chunk:
                continue

            if "__interrupt__" in chunk:
                payload = extract_interrupt_payload(chunk)
                st.session_state.chat_messages.append(
                    {"role": "assistant", "content": hitl_prompt_message(payload)}
                )
                save_current_chat_session()
                st.session_state.pending_interrupt = payload
                st.session_state.pending_thread_id = thread_id
                status_box.markdown('<div class="typing-step">Bond is waiting for your confirmation</div>', unsafe_allow_html=True)
                return

            for node_name, node_update in chunk.items():
                if node_name in PROGRESS_MAP:
                    log = PROGRESS_MAP[node_name]
                    st.session_state.logs.append(log)
                    status_box.markdown(f'<div class="typing-step">{esc(log)}</div>', unsafe_allow_html=True)

                if isinstance(node_update, dict) and node_update.get("response"):
                    final_response = node_update["response"]

        # Finished without interruption
        values = get_final_values(thread_id)
        if not final_response:
            final_response = values.get("response", "")

        if not final_response:
            final_response = "Bond finished processing, but no final response was generated."

        # -----------------------------------------------------------------------------
        # UI FIX: Clean up the step text to natural language when showing the crash queue
        # -----------------------------------------------------------------------------
        if values.get("crashed"):
            remaining = values.get("plan", [])
            done_count = len(values.get("past_steps", []))
            
            addon = "\n\n---\n**⚠️ Run Interrupted (API limit or error)**\n\n"
            addon += f"**Completed steps:** {done_count}\n\n"
            
            if remaining:
                addon += f"**Remaining tasks to execute:**\n"
                
                import re
                def _clean_step_for_ui(step_text: str) -> str:
                    # Remove generic tool prefixes (fetch_, search_, plan_, etc.)
                    clean = re.sub(r'\b(?:fetch|find|calculate|search|plan|lookup|convert|save)_', '', step_text, flags=re.IGNORECASE)
                    # Replace underscores with spaces
                    clean = clean.replace('_', ' ')
                    # Clean up pythonic syntax like kwargs and quotes (e.g. city="Rome" -> city Rome)
                    clean = re.sub(r'[=")(]', ' ', clean)
                    # Collapse multiple spaces into one
                    clean = re.sub(r'\s+', ' ', clean).strip()
                    # Capitalize first letter only
                    if clean:
                        clean = clean[0].upper() + clean[1:]
                    return clean

                for i, s in enumerate(remaining, 1):
                    addon += f"{i}. {_clean_step_for_ui(s)}\n"
            # else:
                # addon += "Generating final summary...\n"
                
            addon += "\n*Type **'c'** or **'continue'** to resume exactly from where I stopped.*"
            final_response = final_response.strip() + addon
        # -----------------------------------------------------------------------------

        message_count = st.session_state.get("existing_message_count", 0)
        messages = (values.get("messages", []) or [])[message_count:]

        try:
            structured_data = build_structured_data_from_messages(
                messages,
                final_response,
                st.session_state.get("last_user_message", ""),
            )
        except Exception:
            structured_data = {}

        st.session_state.chat_messages.append(
            {"role": "assistant", "content": final_response}
        )
        if structured_data:
            updated_cards = apply_trip_cards_update(
                st.session_state.cards,
                structured_data,
                st.session_state.get("last_user_message", ""),
                force_replace=st.session_state.pop("cards_force_replace", False),
            )
            st.session_state.cards = updated_cards
            st.session_state.conversation_cards[thread_id] = updated_cards
        save_current_chat_session()
        st.session_state.pending_interrupt = None
        st.session_state.pending_thread_id = None
        status_box.empty()

    except Exception as e:
        st.session_state.chat_messages.append(
            {"role": "assistant", "content": user_friendly_run_error(e)}
        )
        save_current_chat_session()
        st.session_state.pending_interrupt = None
        st.session_state.pending_thread_id = None
        status_box.empty()


def make_resume_value(payload: dict, action: str, answer_text: str = "", selected_option: str = "") -> Any:
    kind = (payload or {}).get("type")

    if kind == "user_question":
        if action == "answer":
            if selected_option:
                return selected_option
            return answer_text.strip()
        if action == "cancel":
            return ""
        return answer_text.strip() or selected_option or ""

    if kind == "web_host_approval":
        if action == "approve":
            return {"action": "approve"}
        if action == "edit":
            return {"action": "edit", "category": selected_option}
        return {"action": "cancel"}

    if kind == "sif_plan_approval":
        if action == "approve":
            return {"action": "approve"}
        if action in {"edit", "selected"}:
            steps = [line.strip() for line in answer_text.splitlines() if line.strip()]
            return {"action": "edit", "plan": steps} if steps else {"action": "approve"}
        return {"action": "cancel"}

    if kind == "sif_alternatives_offer":
        preference = answer_text.strip()
        if action == "narrow" and preference:
            return {"action": "narrow", "preference": preference}
        return {"action": "continue"}

    if kind == "sif_budget_breach":
        if action == "approve":
            return {"action": "approve"}
        if action == "new_budget":
            amount = as_float(answer_text, as_float((payload or {}).get("budget"), 0.0))
            return {"action": "new_budget", "amount": amount}
        return {"action": "cancel"}

    # SIF gates from sif.py usually expect an action dict.
    if action == "approve":
        return {"action": "approve"}
    if action == "answer":
        return {"action": "approve", "answer": answer_text.strip()}
    return {"action": "cancel"}


def hitl_display_text(payload: dict, action: str, answer_text: str = "", selected_option: str = "") -> str:
    kind = (payload or {}).get("type", "approval")

    if kind == "user_question":
        if action == "cancel":
            return "Cancelled"
        return (answer_text.strip() or selected_option or "Skipped").strip()
    if kind == "web_host_approval":
        if action == "approve":
            return "Approved web search"
        if action == "edit":
            return f"Use category: {selected_option}" if selected_option else "Use selected category"
        return "Cancelled web search"
    if kind == "sif_plan_approval":
        if action == "approve":
            return "Approved all plan steps"
        if action == "selected":
            count = len([line for line in answer_text.splitlines() if line.strip()])
            return f"Run selected plan steps ({count})"
        if action == "edit":
            return "Use my edited plan"
        return "Cancelled the plan"
    if kind == "sif_alternatives_offer":
        if action == "narrow" and answer_text.strip():
            return answer_text.strip()
        return "Continue automatically"
    if kind == "sif_budget_breach":
        if action == "approve":
            return "Approved over-budget result"
        if action == "new_budget":
            return f"New budget: {answer_text.strip()}"
        return "Cancelled the search"
    if action == "approve":
        return "Approved"
    if action == "answer" and answer_text.strip():
        return answer_text.strip()
    return "Cancelled"


def queue_hitl_resume(
    payload: dict,
    thread_id: str,
    action: str,
    answer_text: str = "",
    selected_option: str = "",
    rerun: bool = True,
) -> None:
    """Close the HITL dialog immediately, show the user's choice in chat,
    and resume the graph on the next rerun so the UI feels responsive.
    """
    resume_value = make_resume_value(payload, action, answer_text, selected_option)
    display_text = hitl_display_text(payload, action, answer_text, selected_option)

    st.session_state.chat_messages.append({"role": "user", "content": display_text})
    save_current_chat_session()

    st.session_state.pending_interrupt = None
    st.session_state.pending_thread_id = None
    st.session_state.pending_hitl_resume = {
        "thread_id": thread_id,
        "resume_value": resume_value,
    }
    st.session_state.hitl_submit_locked = True
    # First rerun should only close the dialog and show the user's choice + typing.
    # The graph resumes on the following rerun.
    st.session_state.pending_hitl_resume_ready = False
    st.session_state.waiting_for_response = True
    if rerun:
        st.rerun()


def queue_hitl_resume_callback(
    payload: dict,
    thread_id: str,
    action: str,
    answer_text: str = "",
    selected_option: str = "",
) -> None:
    queue_hitl_resume(
        payload,
        thread_id,
        action,
        answer_text=answer_text,
        selected_option=selected_option,
        rerun=False,
    )


def queue_hitl_resume_from_state(
    payload: dict,
    thread_id: str,
    action: str,
    answer_key: str = "",
    selected_option: str = "",
) -> None:
    answer_text = str(st.session_state.get(answer_key, "")) if answer_key else ""
    if answer_key and not answer_text.strip():
        return
    queue_hitl_resume_callback(payload, thread_id, action, answer_text, selected_option)


def queue_selected_plan_resume(
    payload: dict,
    thread_id: str,
    plan: List[str],
    plan_key_suffix: str,
) -> None:
    selected_steps = [
        str(step)
        for idx, step in enumerate(plan, 1)
        if st.session_state.get(f"hitl_plan_step_{plan_key_suffix}_{idx}", True)
    ]
    queue_hitl_resume_callback(payload, thread_id, "selected", "\n".join(selected_steps))


def render_hitl_section_title(title: str, subtitle: str = "") -> None:
    subtitle_html = f'<div class="hitl-section-subtitle">{esc(subtitle)}</div>' if subtitle else ""
    st.markdown(
        f'<div class="hitl-section-title">{esc(title)}{subtitle_html}</div>',
        unsafe_allow_html=True,
    )


def render_hitl_actions_start() -> None:
    st.markdown('<div class="hitl-actions-marker" aria-hidden="true"></div>', unsafe_allow_html=True)


def render_hitl_unified_header(title: str, subtitle: str, icon: str = "✨") -> None:
    st.markdown(
        f"""
<div class="hitl-unified">
  <div class="hitl-unified-accent"></div>
  <div class="hitl-unified-row">
    <div class="hitl-icon">{icon}</div>
    <div class="hitl-unified-copy">
      <div class="hitl-title-row">
        <div class="hitl-title">{esc(title)}</div>
        <span class="hitl-badge">Awaiting approval</span>
      </div>
      <div class="hitl-subtitle">{esc(subtitle)}</div>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_hitl_note(text: str) -> None:
    st.markdown(
        f'<div class="hitl-unified-note"><span class="hitl-note-icon" aria-hidden="true">i</span>'
        f'<span class="hitl-note-copy">{markdown_to_html(text)}</span></div>',
        unsafe_allow_html=True,
    )


_DETAIL_ICONS = {
    "Query": "🔍",
    "Category": "🏷️",
    "Unknown hosts": "🌐",
}


def render_hitl_details(items: List[tuple[str, str]]) -> None:
    rows: List[str] = []
    for label, value in items:
        if not value:
            continue
        icon = _DETAIL_ICONS.get(label, "•")
        if label == "Category":
            value_html = f'<span class="hitl-chip">{esc(value)}</span>'
        elif label == "Unknown hosts":
            hosts = [host.strip() for host in value.split(",") if host.strip()]
            chips = "".join(f'<span class="hitl-host-chip">{esc(host)}</span>' for host in hosts)
            value_html = f'<div class="hitl-host-list">{chips}</div>'
        elif label == "Query":
            value_html = f'<span class="hitl-details-value hitl-query">{esc(value)}</span>'
        else:
            value_html = f'<span class="hitl-details-value">{esc(value)}</span>'
        rows.append(
            f'<div class="hitl-details-row">'
            f'<span class="hitl-details-icon" aria-hidden="true">{icon}</span>'
            f'<div class="hitl-details-main">'
            f'<span class="hitl-details-label">{esc(label)}</span>'
            f'{value_html}'
            f"</div></div>"
        )
    if rows:
        st.markdown(
            f'<div class="hitl-details-wrap">'
            f'<div class="hitl-section-title hitl-section-title--inline">Request details</div>'
            f'<div class="hitl-details">{"".join(rows)}</div></div>',
            unsafe_allow_html=True,
        )


def render_hitl_panel() -> None:
    payload = st.session_state.get("pending_interrupt")
    if not payload:
        st.markdown(
            """
<div class="info-card">
  <h3>HITL status</h3>
  <div class="small">No pending approval right now.</div>
</div>
""",
            unsafe_allow_html=True,
        )
        return

    kind = payload.get("type", "unknown")
    thread_id = st.session_state.get("pending_thread_id") or active_thread_id()

    if kind == "user_question":
        options = payload.get("options") or []

        with st.container(border=True):
            render_hitl_unified_header(
                "Bond needs your input",
                "Choose an option or type a short answer. Cancel will resume Bond with a safe skip.",
                "💬",
            )
            render_hitl_note("Bond paused and is waiting for your answer before continuing.")

            selected = ""
            if options:
                st.markdown('<div class="hitl-options-title">Quick choices</div>', unsafe_allow_html=True)
                option_cols = st.columns(min(len(options), 3))
                for idx, option in enumerate(options):
                    with option_cols[idx % min(len(options), 3)]:
                        st.button(
                            str(option),
                            key=f"hitl_quick_option_{idx}",
                            use_container_width=True,
                            disabled=app_is_busy(),
                            on_click=queue_hitl_resume_callback,
                            args=(payload, thread_id, "answer", "", str(option)),
                        )
                st.markdown('<div class="hitl-divider"><span>or type your own answer</span></div>', unsafe_allow_html=True)

            if not options:
                render_hitl_section_title("Your answer", "Type a short reply for Bond to continue.")
            answer = st.text_input(
                "Your answer",
                key="hitl_answer",
                placeholder="Type a short answer...",
                label_visibility="collapsed",
            )

            render_hitl_actions_start()
            col_a, col_b = st.columns([1.35, 0.85])
            with col_a:
                st.button(
                    "Send answer",
                    use_container_width=True,
                    type="primary",
                    disabled=app_is_busy(),
                    on_click=queue_hitl_resume_from_state,
                    args=(payload, thread_id, "answer", "hitl_answer", selected),
                )
            with col_b:
                st.button(
                    "Cancel",
                    use_container_width=True,
                    disabled=app_is_busy(),
                    on_click=queue_hitl_resume_callback,
                    args=(payload, thread_id, "cancel"),
                )
        return

    if kind == "web_host_approval":
        unknown_hosts = payload.get("unknown_hosts") or []
        alternatives = payload.get("alternatives") or []

        with st.container(border=True):
            render_hitl_unified_header(
                "Approve web search",
                "Bond wants to search external sources before continuing.",
                "🌐",
            )
            render_hitl_note("Approve this search so Bond can continue planning with updated information.")
            render_hitl_details(
                [
                    ("Query", str(payload.get("query", ""))),
                    ("Category", str(payload.get("category", ""))),
                    ("Unknown hosts", ", ".join(unknown_hosts)),
                ]
            )

            selected_category = ""
            if alternatives:
                render_hitl_section_title(
                    "Override category",
                    "Pick a different search category if the current one is not right.",
                )
                selected_category = st.selectbox(
                    "Alternative category",
                    alternatives,
                    key="hitl_category",
                    label_visibility="collapsed",
                )

            render_hitl_actions_start()
            col_a, col_b, col_c = st.columns([1.15, 1.15, 0.8])
            with col_a:
                st.button("Approve", use_container_width=True, type="primary", disabled=app_is_busy(), on_click=queue_hitl_resume_callback, args=(payload, thread_id, "approve"))
            with col_b:
                if alternatives:
                    st.button("Use category", use_container_width=True, disabled=app_is_busy(), on_click=queue_hitl_resume_callback, args=(payload, thread_id, "edit", "", selected_category))
            with col_c:
                st.button("Cancel", use_container_width=True, disabled=app_is_busy(), on_click=queue_hitl_resume_callback, args=(payload, thread_id, "cancel"))
        return

    if kind == "sif_plan_approval":
        plan = payload.get("plan") or []
        plan_key_suffix = str(abs(hash((thread_id, tuple(str(step) for step in plan)))))

        with st.container(border=True):
            render_hitl_unified_header(
                "Approve the plan",
                "Choose which steps Bond should execute, or edit the plan directly.",
                "🧭",
            )
            render_hitl_note("Bond created a plan and needs your approval before execution.")

            selected_steps: List[str] = []
            if plan:
                st.markdown('<div class="hitl-unified-label">Planned steps</div>', unsafe_allow_html=True)
                for idx, step in enumerate(plan, 1):
                    checked = st.checkbox(
                        str(step),
                        value=True,
                        key=f"hitl_plan_step_{plan_key_suffix}_{idx}",
                        disabled=app_is_busy(),
                    )
                    if checked:
                        selected_steps.append(str(step))

            selected_plan_text = "\n".join(selected_steps)
            edit_key = f"hitl_plan_edit_{plan_key_suffix}"

            render_hitl_section_title("Revise plan", "Edit steps below before approving.")
            revised_plan = st.text_area(
                "Optional revised plan",
                value=selected_plan_text or "\n".join(str(step) for step in plan),
                key=edit_key,
                help="One step per line. You can edit after selecting steps.",
                label_visibility="collapsed",
            )

            render_hitl_actions_start()
            col_a, col_b, col_c, col_d = st.columns([1.0, 1.05, 1.0, 0.8])
            with col_a:
                st.button("Approve all", use_container_width=True, type="primary", disabled=app_is_busy(), on_click=queue_hitl_resume_callback, args=(payload, thread_id, "approve"))
            with col_b:
                st.button("Run selected", use_container_width=True, disabled=app_is_busy() or not selected_steps, on_click=queue_selected_plan_resume, args=(payload, thread_id, [str(step) for step in plan], plan_key_suffix))
            with col_c:
                st.button("Use edited", use_container_width=True, disabled=app_is_busy(), on_click=queue_hitl_resume_from_state, args=(payload, thread_id, "edit", edit_key))
            with col_d:
                st.button("Cancel", use_container_width=True, disabled=app_is_busy(), on_click=queue_hitl_resume_callback, args=(payload, thread_id, "cancel"))
        return

    if kind == "sif_alternatives_offer":
        with st.container(border=True):
            render_hitl_unified_header(
                "Continue or narrow search",
                "Bond found initial options. You can guide the next pass.",
                "🎯",
            )
            render_hitl_note("Continue automatically, or add a preference to narrow the results.")

            past_steps = payload.get("past_steps") or []
            if past_steps:
                with st.expander("What Bond checked so far", expanded=False):
                    for step in past_steps:
                        st.markdown(f"- {esc(step)}")

            render_hitl_section_title("Refine results", "Add a preference to narrow what Bond searches next.")
            preference = st.text_input(
                "Optional preference",
                placeholder="Europe only, under $800, beach resort, near public transport...",
                key="hitl_narrow_preference",
                label_visibility="collapsed",
            )

            render_hitl_actions_start()
            col_a, col_b = st.columns(2)
            with col_a:
                st.button("Continue", use_container_width=True, type="primary", disabled=app_is_busy(), on_click=queue_hitl_resume_callback, args=(payload, thread_id, "continue"))
            with col_b:
                st.button("Narrow search", use_container_width=True, disabled=app_is_busy(), on_click=queue_hitl_resume_from_state, args=(payload, thread_id, "narrow", "hitl_narrow_preference"))
        return

    if kind == "sif_budget_breach":
        budget = as_float(payload.get("budget"), 0.0)
        total_cost = as_float(payload.get("total_cost"), 0.0)
        overage = as_float(payload.get("overage"), max(0.0, total_cost - budget))

        with st.container(border=True):
            render_hitl_unified_header(
                "Budget approval needed",
                "The estimated cost is above the budget you gave Bond.",
                "💰",
            )
            render_hitl_note(
                f"Estimated cost is ${total_cost:,.2f}, which is ${overage:,.2f} over your ${budget:,.2f} budget."
            )

            render_hitl_section_title("Adjust budget", "Set a new limit if you want Bond to keep searching.")
            new_budget = st.text_input(
                "New budget",
                placeholder="2000",
                key="hitl_new_budget",
                label_visibility="collapsed",
            )

            render_hitl_actions_start()
            col_a, col_b, col_c = st.columns([1.15, 1.15, 0.8])
            with col_a:
                st.button("Approve", use_container_width=True, type="primary", disabled=app_is_busy(), on_click=queue_hitl_resume_callback, args=(payload, thread_id, "approve"))
            with col_b:
                st.button("Set budget", use_container_width=True, disabled=app_is_busy(), on_click=queue_hitl_resume_from_state, args=(payload, thread_id, "new_budget", "hitl_new_budget"))
            with col_c:
                st.button("Cancel", use_container_width=True, disabled=app_is_busy(), on_click=queue_hitl_resume_callback, args=(payload, thread_id, "cancel"))
        return

    # Generic SIF or unknown interrupt.
    with st.container(border=True):
        render_hitl_unified_header(
            "Approval needed",
            "Bond paused and needs your decision before continuing.",
            "✅",
        )
        render_hitl_note("Review the request below and choose how to continue.")
        st.json(payload)

        render_hitl_actions_start()
        col_a, col_b = st.columns(2)
        with col_a:
            st.button("Approve", use_container_width=True, type="primary", disabled=app_is_busy(), on_click=queue_hitl_resume_callback, args=(payload, thread_id, "approve"))
        with col_b:
            st.button("Cancel", use_container_width=True, disabled=app_is_busy(), on_click=queue_hitl_resume_callback, args=(payload, thread_id, "cancel"))

init_ui_tables()


def active_thread_id() -> str:
    return (
        st.session_state.get("conversation_id")
        or st.session_state.get("session_id")
        or DEFAULT_SESSION_ID
    )


def new_conversation_id(user_id: str) -> str:
    safe_user = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in user_id)
    safe_user = safe_user.strip("_") or DEFAULT_SESSION_ID
    return f"{safe_user}_chat_{uuid.uuid4().hex[:8]}"


def save_current_chat_session() -> None:
    user_id = st.session_state.get("session_id", DEFAULT_SESSION_ID)
    conversation_id = active_thread_id()
    messages = list(st.session_state.chat_messages)
    cards = dict(st.session_state.get("cards") or {})
    st.session_state.conversation_chats[conversation_id] = messages
    st.session_state.conversation_cards[conversation_id] = cards

    # Move to top ONLY when saving a new message
    user_chats = st.session_state.recent_chats_by_user.setdefault(user_id, [])
    if conversation_id in user_chats:
        user_chats.remove(conversation_id)
    user_chats.insert(0, conversation_id)
    st.session_state.recent_chats_by_user[user_id] = user_chats[:12]

    save_conversation_to_db(user_id, conversation_id, messages, cards)


def load_conversation(conversation_id: str) -> None:
    conversation_id = (conversation_id or "").strip() or new_conversation_id(st.session_state.session_id)
    # Notice: Removed save_current_chat_session() from here to prevent reordering on click
    st.session_state.conversation_id = conversation_id
    messages = st.session_state.conversation_chats.get(conversation_id)
    if messages is None:
        messages = load_conversation_messages_from_db(conversation_id)
    st.session_state.chat_messages = list(messages or [])
    st.session_state.conversation_chats[conversation_id] = list(st.session_state.chat_messages)
    cards = st.session_state.conversation_cards.get(conversation_id)
    if cards is None:
        cards = load_conversation_cards_from_db(conversation_id)
        st.session_state.conversation_cards[conversation_id] = cards
    st.session_state.cards = dict(cards or {})
    st.session_state.logs = []
    st.session_state.pending_interrupt = None
    st.session_state.pending_thread_id = None
    st.session_state.pending_hitl_resume = None
    st.session_state.pending_hitl_resume_ready = False
    st.session_state.hitl_submit_locked = False
    st.session_state.pending_regular_question = None
    st.session_state.pending_regular_question_ready = False
    st.session_state.waiting_for_response = False


def switch_user_session(user_id: str) -> None:
    user_id = (user_id or "").strip() or DEFAULT_SESSION_ID
    # Notice: Removed save_current_chat_session() from here
    st.session_state.session_id = user_id

    conversation_chats, recent_ids, conversation_id, conversation_cards = bootstrap_user_conversations(user_id)
    st.session_state.conversation_chats.update(conversation_chats)
    st.session_state.conversation_cards.update(conversation_cards)
    st.session_state.recent_chats_by_user[user_id] = recent_ids

    st.session_state.conversation_id = conversation_id
    st.session_state.chat_messages = list(st.session_state.conversation_chats.get(conversation_id, []))
    st.session_state.cards = dict(st.session_state.conversation_cards.get(conversation_id, {}))
    st.session_state.logs = []
    st.session_state.pending_interrupt = None
    st.session_state.pending_thread_id = None
    st.session_state.pending_hitl_resume = None
    st.session_state.pending_hitl_resume_ready = False
    st.session_state.hitl_submit_locked = False
    st.session_state.pending_regular_question = None
    st.session_state.pending_regular_question_ready = False
    st.session_state.waiting_for_response = False

def conversation_preview(conversation_id: str) -> str:
    messages = (
        st.session_state.chat_messages
        if conversation_id == active_thread_id()
        else st.session_state.conversation_chats.get(conversation_id, [])
    )
    if not messages:
        messages = load_conversation_messages_from_db(conversation_id)
        if messages:
            st.session_state.conversation_chats[conversation_id] = messages
    return make_chat_title(messages)


def render_chat_message(msg: Dict[str, str]) -> None:
    is_user = msg.get("role") == "user"
    row_class = "msg-row user" if is_user else "msg-row bot"
    bubble_class = "chat-user" if is_user else "chat-bot answer"
    avatar = "🧑" if is_user else "🤖"
    content = message_to_html(msg)
    st.markdown(
        f"""
<div class="{row_class}">
  <div class="avatar">{avatar}</div>
  <div class="{bubble_class}">{content}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def app_is_busy() -> bool:
    """True while a regular user question is waiting/running.
    Used to disable controls and show an immediate typing state.
    """
    return bool(
        st.session_state.get("waiting_for_response")
        or st.session_state.get("pending_regular_question")
        or st.session_state.get("pending_hitl_resume")
        or st.session_state.get("hitl_submit_locked")
    )


def render_typing_message(text: str = "Bond is thinking") -> None:
    st.markdown(
        f"""
<div class="msg-row bot">
  <div class="avatar">🤖</div>
  <div class="typing">{esc(text)}</div>
</div>
""",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------

if "session_id" not in st.session_state:
    st.session_state.session_id = DEFAULT_SESSION_ID

if "conversation_chats" not in st.session_state or "recent_chats_by_user" not in st.session_state:
    boot_chats, boot_recent, boot_active, boot_cards = bootstrap_user_conversations(st.session_state.session_id)
    st.session_state.conversation_chats = boot_chats
    st.session_state.recent_chats_by_user = {st.session_state.session_id: boot_recent}
    st.session_state.conversation_id = boot_active
    st.session_state.conversation_cards = boot_cards

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = st.session_state.recent_chats_by_user.get(
        st.session_state.session_id,
        [st.session_state.session_id],
    )[0]

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = list(
        st.session_state.conversation_chats.get(st.session_state.conversation_id, [])
    )

if "logs" not in st.session_state:
    st.session_state.logs = []

if "cards" not in st.session_state:
    st.session_state.cards = {}

if "conversation_cards" not in st.session_state:
    st.session_state.conversation_cards = {}

if st.session_state.conversation_id not in st.session_state.conversation_cards:
    st.session_state.conversation_cards[st.session_state.conversation_id] = load_conversation_cards_from_db(
        st.session_state.conversation_id
    )

if not st.session_state.cards:
    st.session_state.cards = dict(
        st.session_state.conversation_cards.get(st.session_state.conversation_id, {})
    )

if "pending_interrupt" not in st.session_state:
    st.session_state.pending_interrupt = None

if "pending_thread_id" not in st.session_state:
    st.session_state.pending_thread_id = None

if st.session_state.conversation_id not in st.session_state.conversation_chats:
    db_messages = load_conversation_messages_from_db(st.session_state.conversation_id)
    st.session_state.conversation_chats[st.session_state.conversation_id] = list(db_messages)

if not st.session_state.recent_chats_by_user.get(st.session_state.session_id):
    st.session_state.recent_chats_by_user[st.session_state.session_id] = [st.session_state.conversation_id]

if "last_user_message" not in st.session_state:
    st.session_state.last_user_message = ""

if "cards_force_replace" not in st.session_state:
    st.session_state.cards_force_replace = False

if "existing_message_count" not in st.session_state:
    st.session_state.existing_message_count = 0

if "sif_level" not in st.session_state:
    st.session_state.sif_level = 3

if "menu_open" not in st.session_state:
    st.session_state.menu_open = True

if "waiting_for_response" not in st.session_state:
    st.session_state.waiting_for_response = False

if "pending_regular_question" not in st.session_state:
    st.session_state.pending_regular_question = None

if "pending_hitl_resume" not in st.session_state:
    st.session_state.pending_hitl_resume = None

if "pending_hitl_resume_ready" not in st.session_state:
    st.session_state.pending_hitl_resume_ready = True

if "hitl_submit_locked" not in st.session_state:
    st.session_state.hitl_submit_locked = False

if "pending_regular_question_ready" not in st.session_state:
    st.session_state.pending_regular_question_ready = False


# ---------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------

# Top application header: compact toolbar.
# Theme is a direct toolbar action; settings keeps session and autonomy only.
header_container = st.container(border=False)
with header_container:
    menu_col, title_col, status_col, theme_col, settings_col = st.columns(
        [0.24, 3.7, 0.55, 0.34, 0.30],
        gap="small",
        vertical_alignment="center",
    )

    with menu_col:
        st.markdown('<div class="topbar-menu-button">', unsafe_allow_html=True)
        if st.button(
            "☰",
            key="toggle_menu",
            help="Open / close left menu",
            use_container_width=True,
            disabled=app_is_busy(),
        ):
            st.session_state.menu_open = not st.session_state.menu_open
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with title_col:
        st.markdown(
            """
<div class="app-shell-header">
  <div class="app-title-wrap">
    <div class="app-title">Bond AI Travel Planner ✈️</div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

    with status_col:
        status_label = "🟡 Thinking" if app_is_busy() else "🟢 Ready"
        st.markdown(f'<div class="header-pill">{status_label}</div>', unsafe_allow_html=True)

    with theme_col:
        next_theme = "Dark" if st.session_state.ui_theme == "Light" else "Light"
        theme_icon = "🌙" if next_theme == "Dark" else "☀️"
        st.markdown('<div class="topbar-theme-button">', unsafe_allow_html=True)
        if st.button(
            theme_icon,
            key="theme_toggle_topbar",
            help=f"Switch to {next_theme} mode",
            use_container_width=True,
            disabled=app_is_busy(),
        ):
            st.session_state.ui_theme = next_theme
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with settings_col:
        with st.popover("⚙", use_container_width=True):
            st.markdown('<div class="header-settings-note"><b>Settings</b></div>', unsafe_allow_html=True)

            with st.form("header_session_form", clear_on_submit=False):
                typed_session_id = st.text_input(
                    "Session",
                    value=st.session_state.session_id,
                    placeholder="session",
                )
                if st.form_submit_button("Use session", use_container_width=True, disabled=app_is_busy()):
                    switch_user_session(typed_session_id)
                    st.rerun()

            st.session_state.sif_level = st.selectbox(
                "Autonomy / SIF",
                options=[1, 2, 3],
                index=max(0, int(st.session_state.sif_level) - 1),
                format_func=lambda value: {
                    1: "Ask",
                    2: "Balanced",
                    3: "Auto",
                }[value],
                help="Ask = approve plans, Balanced = occasional checkpoints, Auto = only required approvals.",
            )

try:
    set_session_sif(active_thread_id(), str(int(st.session_state.sif_level)))
except Exception:
    pass

# Main content: left drawer is optional; trip cards live in a right rail.
deferred_work_slot = None

if st.session_state.menu_open:
    left_col, main_col, right_col = st.columns([0.86, 3.72, 1.08], gap="medium")
else:
    main_col, right_col = st.columns([4.25, 1.15], gap="medium")

if st.session_state.menu_open:
    with left_col:
        with st.container(border=True, key="left_sidebar_shell"):
            st.markdown(
                """
<div class="drawer-panel">
  <div class="sidebar-brand">
    <h1>🤖 Bond</h1>
    <p>AI Travel Agent</p>
  </div>
</div>
""",
                unsafe_allow_html=True,
            )

            if st.button("＋ New chat", use_container_width=True, type="primary", disabled=app_is_busy()):
                user_id = st.session_state.session_id
                conversation_id = new_conversation_id(user_id)
                st.session_state.conversation_chats[conversation_id] = []
                
                # Force the new chat to the very top of the list
                user_chats = st.session_state.recent_chats_by_user.setdefault(user_id, [])
                user_chats.insert(0, conversation_id)
                
                st.session_state.conversation_cards[conversation_id] = {}
                st.session_state.cards_force_replace = False
                save_conversation_to_db(user_id, conversation_id, [], {})
                load_conversation(conversation_id)
                st.rerun()

            st.markdown(
                f"""
<div class="side-item">
  <b>Current session</b><br>
  <span class="small">{esc(st.session_state.session_id)}</span>
</div>
""",
                unsafe_allow_html=True,
            )

            current_user_chats = st.session_state.recent_chats_by_user.get(
                st.session_state.session_id,
                [active_thread_id()],
            )
            db_recent = load_user_conversations_from_db(st.session_state.session_id, limit=12)
            if db_recent:
                current_user_chats = [row["conversation_id"] for row in db_recent]
                st.session_state.recent_chats_by_user[st.session_state.session_id] = current_user_chats
                for row in db_recent:
                    st.session_state.conversation_chats.setdefault(row["conversation_id"], row.get("messages") or [])

            st.markdown("### Recent chats")
            # Let the page grow naturally so saved preferences are never clipped.
            chat_history_container = st.container(border=False, key="recent_chats_scroll")
            
            with chat_history_container:
                # Loop through all chats without the [:8] limit so the user can scroll through the entire history
                for conversation_id in current_user_chats:
                    col_chat, col_del = st.columns([0.84, 0.16], gap="small", vertical_alignment="center")
                    label = "●" if conversation_id == active_thread_id() else "○"
                    preview = conversation_preview(conversation_id)
                    
                    with col_chat:
                        if st.button(
                            f"{label} {preview}",
                            key=f"conversation_{conversation_id}",
                            use_container_width=True,
                            disabled=app_is_busy(),
                        ):
                            if conversation_id != active_thread_id():
                                load_conversation(conversation_id)
                                st.rerun()
                    
                    with col_del:
                        if st.button(
                            "🗑️",
                            key=f"delete_conv_{conversation_id}",
                            use_container_width=True,
                            disabled=app_is_busy(),
                            help="Delete this chat permanently",
                        ):
                            with sqlite3.connect("checkpoints.db") as conn:
                                conn.execute("DELETE FROM ui_conversations WHERE conversation_id = ?", (conversation_id,))
                                try:
                                    conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (conversation_id,))
                                except sqlite3.OperationalError:
                                    pass
                                try:
                                    conn.execute("DELETE FROM checkpoint_blobs WHERE thread_id = ?", (conversation_id,))
                                except sqlite3.OperationalError:
                                    pass
                                try:
                                    conn.execute("DELETE FROM checkpoint_writes WHERE thread_id = ?", (conversation_id,))
                                except sqlite3.OperationalError:
                                    pass
                            
                            user_id = st.session_state.session_id
                            if conversation_id in st.session_state.recent_chats_by_user.get(user_id, []):
                                st.session_state.recent_chats_by_user[user_id].remove(conversation_id)
                            if conversation_id in st.session_state.conversation_chats:
                                del st.session_state.conversation_chats[conversation_id]
                            if conversation_id in st.session_state.conversation_cards:
                                del st.session_state.conversation_cards[conversation_id]
                                
                            if conversation_id == active_thread_id():
                                remaining_chats = st.session_state.recent_chats_by_user.get(user_id, [])
                                if remaining_chats:
                                    load_conversation(remaining_chats[0])
                                else:
                                    new_id = new_conversation_id(user_id)
                                    st.session_state.conversation_chats[new_id] = []
                                    st.session_state.recent_chats_by_user[user_id] = [new_id]
                                    load_conversation(new_id)
                            st.rerun()

            render_saved_panel(active_thread_id())

with main_col:
    with st.container(border=True, key="chat_shell"):
        st.markdown(
            """
<div class="chat-shell-title">
  <div><b>Chat</b> <span class="small">Ask Bond to plan or refine your trip</span></div>
  <div class="chat-status-dot"></div>
</div>
""",
            unsafe_allow_html=True,
        )

        chat_area = st.container(height=700 if not st.session_state.menu_open else 680, border=False, key="chat_scroll_area",)
        with chat_area:
            if not st.session_state.chat_messages:
                st.markdown(
                    """
<div class="empty-state">
  <h2>Hi, I'm Bond 🤖✈️</h2>
  <p>Tell me what trip you want to plan. I can help with flights, hotels, activities, transport, visa, currency and warnings.</p>
  <div class="prompt-grid">
    <div class="prompt-chip">
      <b>City break</b>
      <span>Plan 4 days in Rome with food, museums and a relaxed budget.</span>
    </div>
    <div class="prompt-chip">
      <b>Beach escape</b>
      <span>Find a warm April destination with hotels near the water.</span>
    </div>
    <div class="prompt-chip">
      <b>Smart compare</b>
      <span>Compare Tokyo and Seoul for price, weather and transport.</span>
    </div>
  </div>
</div>
""",
                    unsafe_allow_html=True,
                )
            else:
                for msg in st.session_state.chat_messages:
                    render_chat_message(msg)

            if st.session_state.get("waiting_for_response"):
                render_typing_message("Bond is thinking")
                deferred_work_slot = st.empty()

            if (
                st.session_state.get("pending_interrupt")
                and not st.session_state.get("pending_hitl_resume")
                and not st.session_state.get("hitl_submit_locked")
            ):
                render_hitl_panel()

        # Custom in-panel chat input.
        # Using st.chat_input makes Streamlit pin the input to the browser bottom,
        # which created the black floating bar and broke the Qt-like layout.
        with st.form("inline_chat_input_form", clear_on_submit=True):
            input_col, send_col = st.columns([12, 0.7], gap="small", vertical_alignment="center")
            with input_col:
                user_message = st.text_input(
                    "Message",
                    placeholder="Tell Bond what trip you want to plan...",
                    label_visibility="collapsed",
                    disabled=app_is_busy() or bool(st.session_state.get("pending_interrupt")),
                    key="inline_chat_message",
                )
            with send_col:
                submitted = st.form_submit_button(
                    "↑",
                    use_container_width=True,
                    disabled=app_is_busy() or bool(st.session_state.get("pending_interrupt")),
                )

        if submitted and user_message and user_message.strip():
            thread_id = active_thread_id()
            clean_message = user_message.strip()

            st.session_state.chat_messages.append({"role": "user", "content": clean_message})
            st.session_state.last_user_message = clean_message
            st.session_state.logs = []
            if is_new_trip_request(clean_message):
                st.session_state.cards = {}
                st.session_state.conversation_cards[thread_id] = {}
                st.session_state.cards_force_replace = True
            else:
                st.session_state.cards_force_replace = False
            save_current_chat_session()

            RESUME_WORDS = {"continue", "resume", "c"}
            is_resume_attempt = clean_message.lower() in RESUME_WORDS
            
            has_crashed = False
            try:
                existing_state = graph.get_state(default_config(thread_id))
                if existing_state and existing_state.values:
                    has_crashed = bool(existing_state.values.get("crashed", False))
            except Exception:
                pass

            if is_resume_attempt and has_crashed:
                st.session_state.existing_message_count = get_existing_message_count(thread_id)
                st.session_state.pending_regular_question = {
                    "thread_id": thread_id,
                    "message": clean_message,
                    "is_resume": True,
                    "existing_message_count": st.session_state.existing_message_count,
                }
                st.session_state.pending_regular_question_ready = False
                st.session_state.waiting_for_response = True
                save_current_chat_session()
                st.rerun()

            if is_greeting_or_too_short(clean_message):
                st.session_state.chat_messages.append(
                    {"role": "assistant", "content": greeting_response()}
                )
                save_current_chat_session()
                st.rerun()

            try:
                # Save SIF to graph memory before starting the run.
                set_session_sif(thread_id, str(int(st.session_state.sif_level)))
            except Exception:
                pass

            # Do not run the graph in the same rerun.
            # First rerun immediately shows the user's message and disables input.
            st.session_state.existing_message_count = get_existing_message_count(thread_id)
            st.session_state.pending_regular_question = {
                "thread_id": thread_id,
                "message": clean_message,
                "is_resume": False,
                "existing_message_count": st.session_state.existing_message_count,
            }
            # First rerun should only show the user message + typing.
            # The graph runs on the following rerun.
            st.session_state.pending_regular_question_ready = False
            st.session_state.waiting_for_response = True
            save_current_chat_session()
            st.rerun()

with right_col:
    with st.container(border=True, key="trip_cards_shell"):
        st.markdown(
            """
<div class="right-rail-title">
  <div><b>Trip cards</b><span class="small"> Live itinerary details</span></div>
</div>
""",
            unsafe_allow_html=True,
        )
        right_cards = st.container(border=False, key="right_rail_scroll")
        with right_cards:
            if st.session_state.get("cards"):
                render_cards(st.session_state.cards, columns=1, detail_columns=1)
            else:
                st.markdown(
                    """
<div class="info-card right-empty-card">
  <h3>Trip board</h3>
  <div class="small">Flights, hotels, activities, transport, warnings and estimated costs will appear here after Bond finds structured details.</div>
</div>
""",
                    unsafe_allow_html=True,
                )


# ---------------------------------------------------------------------
# Deferred agent processing
# ---------------------------------------------------------------------
# The first fragment pass marks the work as ready and lets the browser paint
# the closed HITL panel / typing state. The timed pass then runs the graph.
@st.fragment(run_every=0.5)
def process_deferred_agent_work() -> None:
    if st.session_state.get("pending_hitl_resume"):
        if not st.session_state.get("pending_hitl_resume_ready"):
            st.session_state.pending_hitl_resume_ready = True
            return

        pending_hitl = st.session_state.pending_hitl_resume
        st.session_state.pending_hitl_resume = None
        st.session_state.pending_hitl_resume_ready = False
        st.session_state.hitl_submit_locked = False
        st.session_state.waiting_for_response = True

        run_until_pause_or_done(
            Command(resume=pending_hitl["resume_value"]),
            pending_hitl["thread_id"],
        )

        st.session_state.waiting_for_response = False
        st.session_state.hitl_submit_locked = False
        save_current_chat_session()
        st.rerun(scope="app")

    if st.session_state.get("pending_regular_question"):
        if not st.session_state.get("pending_regular_question_ready"):
            st.session_state.pending_regular_question_ready = True
            return

        pending = st.session_state.pending_regular_question
        st.session_state.pending_regular_question = None
        st.session_state.pending_regular_question_ready = False
        st.session_state.waiting_for_response = True
        st.session_state.existing_message_count = pending.get(
            "existing_message_count",
            get_existing_message_count(pending["thread_id"]),
        )

        is_resume = pending.get("is_resume", False)
        initial_state = build_initial_state(pending["message"], pending["thread_id"], is_resume=is_resume)
        run_until_pause_or_done(initial_state, pending["thread_id"])
        st.session_state.waiting_for_response = False
        save_current_chat_session()
        st.rerun(scope="app")


if st.session_state.get("pending_hitl_resume") or st.session_state.get("pending_regular_question"):
    if deferred_work_slot is not None:
        with deferred_work_slot:
            process_deferred_agent_work()
    else:
        process_deferred_agent_work()
