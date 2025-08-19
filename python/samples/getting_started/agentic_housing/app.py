import asyncio
import os
import secrets
import importlib.util
from dataclasses import asdict
from typing import Dict, Any

from flask import Flask, render_template, request, jsonify, session as flask_session
from semantic_kernel import Kernel
from semantic_kernel.contents import ChatHistory
from semantic_kernel.functions import KernelArguments

import importlib.util, sys, os

# ---- Dynamic import of your existing SK script (works even with hyphen in filename)
AGENT_FILE = os.environ.get("AGENT_FILE", "AgenticHousing-SemanticKernel.py")

_spec = importlib.util.spec_from_file_location("agentic_core", AGENT_FILE)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Cannot load agent module from {AGENT_FILE}")
agentic_core = importlib.util.module_from_spec(_spec)
sys.modules["agentic_core"] = agentic_core
_spec.loader.exec_module(agentic_core)  # type: ignore

# Shortcuts to functions/classes we call from your module
Borrower = agentic_core.Borrower
EligibilityOutcome = agentic_core.EligibilityOutcome
Session = agentic_core.Session
Phase = agentic_core.Phase
parse_message_into_borrower = agentic_core.parse_message_into_borrower
compute_eligibility = agentic_core.compute_eligibility
current_listings = agentic_core.current_listings
AREA_INFO = agentic_core.AREA_INFO
Config = agentic_core.Config
build_prompt_config = agentic_core.build_prompt_config

# ---- Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", secrets.token_hex(16))

# Store per-browser session state in-memory (OK for demo)
_sessions: Dict[str, Dict[str, Any]] = {}


class WebAgent:
    def __init__(self) -> None:
        cfg = Config()
        self.kernel = Kernel()
        self.service_id = cfg.add_service_to_kernel(self.kernel)
        prompt_cfg = build_prompt_config(cfg.service, self.service_id)
        self.agent_func = self.kernel.add_function(
            function_name="reply",
            plugin_name="agent",
            prompt_template_config=prompt_cfg,
        )
        self.chat = ChatHistory()
        self.sess = Session()

    @staticmethod
    def _history_str(chat: ChatHistory, last_n: int = 6) -> str:
        msgs = chat.messages[-last_n:]
        parts = []
        for m in msgs:
            role = getattr(m, "role", "")
            content = getattr(m, "content", "")
            parts.append(f"{role.capitalize()}: {content}")
        return "\n".join(parts)

    def _build_facts(self) -> Dict[str, Any]:
        facts = self.sess.facts()
        # listings snapshot (used by the LLM prompt too)
        if self.sess.phase in (Phase.LIST, Phase.AREA):
            lst = current_listings(self.sess)
            facts["current_listings"] = [
                {
                    "i": i + 1,
                    "id": r["id"],
                    "title": r["title"],
                    "price": r["price"],
                    "type": r["type"],
                    "suburb": r["suburb"],
                }
                for i, r in enumerate(lst)
            ]
        # area snapshot
        if self.sess.phase == Phase.AREA and self.sess.picked_listing is not None:
            listing = next(
                (
                    r
                    for r in agentic_core.PROPERTIES
                    if r["id"] == self.sess.picked_listing
                ),
                None,
            )
            if listing:
                facts["area_info"] = AREA_INFO.get(listing["suburb"], {})
        return facts

    def _handle_commands(self, text: str) -> str | None:
        t = text.strip().lower()
        if t in {"/help"}:
            return (
                "Commands: /help, /reset, /next (advance phase), /list (back to listings), "
                "/pick <#> (select listing)"
            )
        if t == "/reset":
            self.sess = Session()
            self.chat = ChatHistory()
            return "Session reset."
        if t == "/next":
            if self.sess.phase == Phase.ELIG:
                self.sess.phase = Phase.LIST
                return "Moving to listings."
            elif self.sess.phase == Phase.LIST:
                if self.sess.picked_listing is not None:
                    self.sess.phase = Phase.AREA
                    return "Showing area details."
                return "Pick a listing first with /pick <#>."
            elif self.sess.phase == Phase.AREA:
                self.sess.phase = Phase.DONE
                return "All done."
        if t == "/list":
            self.sess.phase = Phase.LIST
            return "Back to listings."
        if t.startswith("/pick"):
            parts = t.split()
            if len(parts) == 2 and parts[1].isdigit():
                idx = int(parts[1])
                rows = current_listings(self.sess)
                if 1 <= idx <= len(rows):
                    self.sess.picked_listing = rows[idx - 1]["id"]
                    self.sess.phase = Phase.AREA
                    return f"Selected listing #{idx}."
                return "Invalid pick index."
            return "Usage: /pick <#>"
        return None

    def _maybe_compute_eligibility(self):
        if self.sess.phase != Phase.ELIG:
            return
        need = []
        b = self.sess.borrower
        if b.gross_annual_income is None:
            need.append("income")
        if b.monthly_debts is None:
            need.append("debts")
        if b.monthly_expenses is None:
            need.append("expenses")
        if b.savings is None:
            need.append("savings")
        if b.target_price is None:
            need.append("target price")
        if b.location_hint is None:
            need.append("location")
        if len(need) <= 2:
            self.sess.last_outcome = compute_eligibility(self.sess.borrower)

    def _phase_color(self) -> str:
        return {
            Phase.ELIG: "eligibility",
            Phase.LIST: "listings",
            Phase.AREA: "area",
            Phase.DONE: "done",
        }.get(self.sess.phase, "eligibility")

    async def send_async(self, text: str) -> Dict[str, Any]:
        # Commands first
        cmd_reply = self._handle_commands(text)
        if cmd_reply is not None:
            # still record in history for context
            self.chat.add_user_message(text)
            self.chat.add_assistant_message(cmd_reply)
            return {
                "reply": cmd_reply,
                "phase": self.sess.phase,
                "phaseClass": self._phase_color(),
                "facts": self._build_facts(),
            }

        # Update borrower from free text
        self.sess.borrower, updates = parse_message_into_borrower(
            text, self.sess.borrower
        )
        if "preferred_type" in updates:
            self.sess.preferred_type = updates.get("preferred_type")

        self._maybe_compute_eligibility()

        # chat history
        self.chat.add_user_message(text)
        hist = self._history_str(self.chat)

        # Build facts and call SK
        facts = self._build_facts()
        result = await self.kernel.invoke(
            self.agent_func,
            KernelArguments(
                phase=self.sess.phase,
                facts=__import__("json").dumps(facts, ensure_ascii=False),
                history=hist,
                user_input=text,
            ),
        )
        reply_text = str(result)
        self.chat.add_assistant_message(reply_text)

        # Simple auto-advance: if user asked to see listings and eligible
        if (
            self.sess.phase == Phase.ELIG
            and any(k in text.lower() for k in ["listings", "houses", "show houses"])
            and self.sess.last_outcome
        ):
            self.sess.phase = Phase.LIST

        return {
            "reply": reply_text,
            "phase": self.sess.phase,
            "phaseClass": self._phase_color(),
            "facts": facts,
        }


def get_webagent() -> WebAgent:
    sid = flask_session.get("sid")
    if not sid:
        sid = secrets.token_hex(8)
        flask_session["sid"] = sid
    if sid not in _sessions:
        _sessions[sid] = {"agent": WebAgent()}
    return _sessions[sid]["agent"]


@app.route("/")
def index():
    return render_template("index.html")


@app.post("/api/message")
def api_message():
    data = request.get_json(force=True)
    user_text = (data or {}).get("message", "").strip()
    if not user_text:
        return jsonify({"error": "empty"}), 400

    agent = get_webagent()
    # Run the async SK call synchronously for Flask
    payload = asyncio.run(agent.send_async(user_text))
    return jsonify(payload)


if __name__ == "__main__":
    app.run(debug=True)
