"""
sif.py  —  Supervised-Independence Factor (SIF) for the Plan-and-Execute Travel Agent
================================================================================

The SIF controls how much autonomy the agent has vs. how often it pauses
for human confirmation.  Three levels are defined:

  SIF 1 — LOW autonomy
      • After planning (and after every replan that produces a *new* plan),
        the agent pauses and shows the plan to the user before executing.
        The user can approve, edit the plan text, or cancel the run.

  SIF 2 — MEDIUM autonomy
      • The plan gate from SIF-1 is NOT active.
      • Gate A — Alternatives offer: after the first execution cycle the
        agent offers the user a chance to narrow the search by preference,
        continent, or price tier before continuing.
      • Gate B — Over-budget confirmation: if the replanner detects a
        budget breach it pauses and asks whether to keep searching outside
        the budget, set a new budget, or cancel.

  SIF 3 — HIGH autonomy  (default)
      • No SIF-specific interrupts.  Only the existing web-host approval
        gate is active.

Architecture notes
------------------
• SIF is stored inside `user_preferences["sif"]` — the same SqliteSaver-
  backed dict that already persists user preferences across sessions.
  No new state key is required.
• All new nodes are pure functions; they read `state` and return a delta.
• Two new interrupt types are introduced:
    "sif_plan_approval"      — SIF 1 plan/replan gate
    "sif_alternatives_offer" — SIF 2 alternatives gate (Gate A)
    "sif_budget_breach"      — SIF 2 over-budget gate  (Gate B)
  The existing `_resume_value_for` dispatcher in the main file routes
  these to the handlers defined at the bottom of this module.
• Graph wiring changes are minimal and confined to one block in the main
  file (see the "GRAPH WIRING" section at the bottom of this docstring).

GRAPH WIRING SUMMARY
--------------------
Replace the current static edge:

    builder.add_edge("planner", "execute")

with:

    builder.add_conditional_edges(
        "planner", route_after_planner,
        {"sif_plan_gate": "sif_plan_gate", "execute": "execute"},
    )
    builder.add_node("sif_plan_gate", sif_plan_gate_node)
    builder.add_conditional_edges(
        "sif_plan_gate", route_after_plan_gate,
        {"execute": "execute", END: END},
    )

Add the SIF-2 gates after `after_tools_node` and inside `replan_node`
(see inline comments in those sections below).

    builder.add_node("sif_alternatives_gate", sif_alternatives_gate_node)
    builder.add_conditional_edges(
        "sif_alternatives_gate", route_after_alternatives_gate,
        {"execute": "execute", "replan": "replan"},
    )

The budget-breach gate is injected inside replan_node itself (see the
helper `maybe_sif2_budget_interrupt` called from replan_node).
"""

from __future__ import annotations

from langgraph.graph import END
from langgraph.types import interrupt

# ---------------------------------------------------------------------------
# SIF helpers
# ---------------------------------------------------------------------------

def get_sif(state: dict) -> int:
    """Return the current SIF level (1, 2, or 3).  Defaults to 3."""
    raw = (state.get("user_preferences") or {}).get("sif", "3")
    try:
        level = int(raw)
    except (TypeError, ValueError):
        level = 3
    return max(1, min(3, level))   # clamp to [1, 2, 3]


# ---------------------------------------------------------------------------
# SIF-1 — Plan Gate
# ---------------------------------------------------------------------------

def route_after_planner(state: dict) -> str:
    """Routing function called right after plan_node.
    SIF 1 → always go through the plan-approval gate.
    SIF 2/3 → skip directly to execute."""
    if get_sif(state) == 1:
        return "sif_plan_gate"
    return "execute"


def sif_plan_gate_node(state: dict) -> dict:
    """SIF-1 gate: show the freshly generated (or revised) plan to the user
    and wait for approval, edit, or cancellation.

    Interrupt payload type: "sif_plan_approval"
    Resume value expected:
        {"action": "approve"}
        {"action": "edit",   "plan": ["step 1", "step 2", ...]}
        {"action": "cancel"}
    """
    plan = state.get("plan", [])

    decision = interrupt({
        "type": "sif_plan_approval",
        "plan": plan,
    })

    action = (decision or {}).get("action", "cancel")

    if action == "approve":
        # Nothing changes — the plan proceeds as-is.
        return {}

    if action == "edit":
        new_steps = (decision or {}).get("plan", plan)
        if isinstance(new_steps, str):
            # Accept a newline-separated string as a convenience.
            new_steps = [s.strip() for s in new_steps.splitlines() if s.strip()]
        print(f"  ✏️   Plan edited by user ({len(new_steps)} steps).")
        return {"plan": new_steps}

    # action == "cancel"
    print("  🚫  Run cancelled by user at plan-approval gate.")
    return {
        "plan": [],
        "response": "Trip planning cancelled at your request.",
    }


def route_after_plan_gate(state: dict) -> str:
    """After the plan gate: go to execute unless the user cancelled."""
    if state.get("response") == "Trip planning cancelled at your request.":
        return END
    return "execute"


# ---------------------------------------------------------------------------
# SIF-2 — Gate A: Alternatives offer (after first execution cycle)
# ---------------------------------------------------------------------------

# We only want to offer alternatives ONCE per request (not after every step).
# We track this with a sentinel in past_steps.
_ALTERNATIVES_OFFERED_MARKER = "__sif2_alternatives_offered__"


def _alternatives_already_offered(state: dict) -> bool:
    return any(
        _ALTERNATIVES_OFFERED_MARKER in s
        for s in (state.get("past_steps") or [])
    )


def should_offer_alternatives(state: dict) -> bool:
    """True when SIF is 2 AND we have at least one completed step AND we
    haven't already offered alternatives this request."""
    if get_sif(state) != 2:
        return False
    if _alternatives_already_offered(state):
        return False
    # Offer after the first real execution cycle (past_steps is non-empty).
    return bool(state.get("past_steps"))


def route_after_tools_sif(state: dict) -> str:
    """Extended routing used instead of `route_after_tools` when SIF-2 is
    active.  Adds the alternatives gate before the first replan.

    Possible return values:
        "execute"              — more steps remain, keep going
        "sif_alternatives_gate" — SIF-2: offer alternatives before replanning
        "replan"               — no more steps, go straight to replan
        "no_match"             — destination not found
    """
    from plan_and_execute_agent import _no_match_detected   # avoid circular at module level
    if _no_match_detected(state["messages"]):
        return "no_match"
    if state.get("plan"):
        return "execute"
    # Plan exhausted — decide whether to offer alternatives.
    if should_offer_alternatives(state):
        return "sif_alternatives_gate"
    return "replan"


def sif_alternatives_gate_node(state: dict) -> dict:
    """SIF-2 Gate A: before the replanner synthesises results, offer the
    user a chance to narrow down alternatives.

    Interrupt payload type: "sif_alternatives_offer"
    Resume value expected:
        {"action": "continue"}                  — proceed with what we have
        {"action": "narrow", "preference": str} — inject a preference hint
    """
    decision = interrupt({
        "type":    "sif_alternatives_offer",
        "past_steps": state.get("past_steps", []),
        "input":   state.get("input", ""),
    })

    action = (decision or {}).get("action", "continue")

    # Record the marker so we don't offer again on the next replan cycle.
    marker = _ALTERNATIVES_OFFERED_MARKER
    new_past = list(state.get("past_steps", [])) + [marker]

    if action == "narrow":
        preference = (decision or {}).get("preference", "").strip()
        if preference:
            hint = f"USER NARROWING PREFERENCE: {preference}"
            new_past.append(hint)
            print(f"  🎯  User narrowed search: {preference}")

    return {"past_steps": new_past}


def route_after_alternatives_gate(state: dict) -> str:
    """Always go to replan after the alternatives gate — the replanner will
    pick up any injected preference hint from past_steps."""
    return "replan"


# ---------------------------------------------------------------------------
# SIF-2 — Gate B: Over-budget confirmation (called from inside replan_node)
# ---------------------------------------------------------------------------

def maybe_sif2_budget_interrupt(state: dict, budget: float, total_cost: float) -> dict | None:
    """Call this from replan_node BEFORE issuing a FinalResponse when
    `over_budget` is True.

    Returns a state-delta dict if the gate fires (the replanner should merge
    it and return early), or None if the gate should not fire.

    Interrupt payload type: "sif_budget_breach"
    Resume value expected:
        {"action": "approve"}                   — search outside budget
        {"action": "new_budget", "amount": float} — set a new budget limit
        {"action": "cancel"}                    — abort the search
    """
    if get_sif(state) != 2:
        return None
    if not (budget > 0 and total_cost > budget):
        return None

    decision = interrupt({
        "type":        "sif_budget_breach",
        "budget":      budget,
        "total_cost":  total_cost,
        "overage":     round(total_cost - budget, 2),
    })

    action = (decision or {}).get("action", "cancel")

    if action == "approve":
        print("  ✅  User approved searching outside budget.")
        # Signal to the replanner that budget constraint is lifted.
        return {"total_budget": 0.0, "over_budget": False}

    if action == "new_budget":
        new_amount = float((decision or {}).get("amount", budget))
        print(f"  💰  User set new budget: ${new_amount:.2f}")
        return {"total_budget": new_amount, "over_budget": total_cost > new_amount}

    # action == "cancel"
    print("  🚫  User cancelled after budget breach.")
    return {
        "plan": [],
        "response": f"Search cancelled. Estimated cost (${total_cost:.2f}) exceeded your budget (${budget:.2f}).",
    }


# ---------------------------------------------------------------------------
# Terminal UI handlers (register these in _resume_value_for in the main file)
# ---------------------------------------------------------------------------

def _prompt_plan_approval(payload: dict) -> dict:
    """Interactive terminal prompt for the SIF-1 plan gate."""
    plan = payload.get("plan", [])

    print("\n" + "═" * 62)
    print("📋  SIF-1: PLAN APPROVAL REQUIRED")
    print("═" * 62)
    print("  The agent proposes the following plan:\n")
    for i, step in enumerate(plan, 1):
        print(f"    {i}. {step}")
    print("-" * 62)
    print("  [a] Approve  — execute this plan as-is")
    print("  [e] Edit     — paste a revised plan (one step per line, blank line to finish)")
    print("  [c] Cancel   — abort this request")
    print("═" * 62)

    while True:
        choice = input("  Your choice [a/e/c]: ").strip().lower()
        if choice in ("a", "approve", ""):
            print("  ✅  Plan approved.\n")
            return {"action": "approve"}
        if choice in ("c", "cancel"):
            print("  🚫  Request cancelled.\n")
            return {"action": "cancel"}
        if choice in ("e", "edit"):
            print("  Enter your revised plan (type the step text, or just the step number to keep it). Enter a blank line when done:")
            lines = []
            while True:
                line = input("    > ").strip()
                if not line:
                    break
                
                # Check if the input is a valid step number from the original plan
                if line.isdigit() and 1 <= int(line) <= len(plan):
                    lines.append(plan[int(line) - 1])
                else:
                    lines.append(line)
                    
            if lines:
                print(f"  ✏️   Plan updated ({len(lines)} steps).\n")
                return {"action": "edit", "plan": lines}
            print("  (Empty plan — keeping original.)")
            return {"action": "approve"}
            print("  (Empty plan — keeping original.)")
            return {"action": "approve"}
        print("  Please enter 'a', 'e', or 'c'.")


def _prompt_alternatives_offer(payload: dict) -> dict:
    """Interactive terminal prompt for the SIF-2 alternatives gate."""
    print("\n" + "═" * 62)
    print("🔍  SIF-2: NARROW YOUR SEARCH?")
    print("═" * 62)
    print("  The agent has finished its first search round.")
    print("  You can guide the next search, or let the agent decide.\n")
    print("  [1] Continue — let the agent proceed automatically")
    print("  [2] Narrow   — specify a preference (continent / price / amenity)")
    print("═" * 62)

    choice = input("  Your choice [1/2]: ").strip()
    if choice == "2":
        pref = input("  Enter your preference (e.g. 'Europe only', 'under $800', 'beach resort'): ").strip()
        if pref:
            print(f"  🎯  Narrowing to: {pref}\n")
            return {"action": "narrow", "preference": pref}
    print("  ▶️   Continuing automatically.\n")
    return {"action": "continue"}


def _prompt_budget_breach(payload: dict) -> dict:
    """Interactive terminal prompt for the SIF-2 over-budget gate."""
    budget     = payload.get("budget", 0)
    total_cost = payload.get("total_cost", 0)
    overage    = payload.get("overage", 0)

    print("\n" + "═" * 62)
    print("💸  IF-2: BUDGET BREACH DETECTED")
    print("═" * 62)
    print(f"  Your budget : ${budget:.2f}")
    print(f"  Found total : ${total_cost:.2f}  (+${overage:.2f} over budget)")
    print("-" * 62)
    print("  [1] Approve      — continue searching outside my budget")
    print("  [2] New budget   — set a higher budget and re-search")
    print("  [3] Cancel       — abort this request")
    print("═" * 62)

    while True:
        choice = input("  Your choice [1/2/3]: ").strip()
        if choice == "1":
            print("  ✅  Approved — searching outside budget.\n")
            return {"action": "approve"}
        if choice == "2":
            raw = input("  Enter new budget (e.g. 2000): ").strip().replace("$", "").replace(",", "")
            try:
                amount = float(raw)
                print(f"  💰  New budget set: ${amount:.2f}\n")
                return {"action": "new_budget", "amount": amount}
            except ValueError:
                print("  Invalid amount — try again.")
                continue
        if choice == "3":
            print("  🚫  Request cancelled.\n")
            return {"action": "cancel"}
        print("  Please enter 1, 2, or 3.")


# Public dispatch table — merge this into _resume_value_for in the main file.
SIF_INTERRUPT_HANDLERS = {
    "sif_plan_approval":     _prompt_plan_approval,
    "sif_alternatives_offer": _prompt_alternatives_offer,
    "sif_budget_breach":     _prompt_budget_breach,
}
