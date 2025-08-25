"""
Agentic Housing Demo (Semantic Kernel, Python)

Three cooperative agents with a simple router:
  1) EligibilityAgent – collects borrower facts and computes synthetic borrowing capacity.
  2) ListingsAgent – searches a synthetic property dataset by price, type, and location.
  3) AreaAgent – shows synthetic local info (schools, PTV, restaurants) for a chosen listing.

Orchestration pattern:
  • A finite-state loop (phase: eligibility → listings → area → done) + an LLM for NLG.
  • Deterministic business logic lives in Python; the LLM makes responses friendly/interactive.

Requirements
-----------
Python 3.10+
python -m pip install "semantic-kernel<1.23"


.env (in the same folder):
  GLOBAL_LLM_SERVICE="OpenAI"  # or AzureOpenAI
  # If OpenAI
  OPENAI_API_KEY="sk-..."
  OPENAI_CHAT_MODEL_ID="gpt-5"  # or gpt-4o, etc.

Run
---
python AgenticHousing-SemanticKernel.py

Type freely (e.g., "I earn 110k, debts 450/mo, savings 40k; aiming for 800k in Melbourne").
Use /help to see commands. Use /next to move to the next agent when ready.
"""

from __future__ import annotations
import asyncio
import json
import math
import os
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from semantic_kernel import Kernel
from semantic_kernel.contents import ChatHistory
from semantic_kernel.functions import KernelArguments
from semantic_kernel.prompt_template import PromptTemplateConfig
from semantic_kernel.prompt_template.input_variable import InputVariable
from semantic_kernel.connectors.ai.open_ai import (
    AzureChatCompletion,
    OpenAIChatCompletion,
    AzureChatPromptExecutionSettings,
    OpenAIChatPromptExecutionSettings,
)

# ----------------------------
# Utilities & Config
# ----------------------------


class Config:
    def __init__(self) -> None:
        load_dotenv()
        self.service = os.getenv("GLOBAL_LLM_SERVICE", "AzureOpenAI").strip()
        # OpenAI
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.openai_model = os.getenv("OPENAI_CHAT_MODEL_ID", "gpt-5")
        # Azure OpenAI
        self.az_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.az_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.az_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
        self.az_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")

    def add_service_to_kernel(self, kernel: Kernel) -> str:
        service_id = "default"
        if self.service.lower() == "openai":
            kernel.add_service(OpenAIChatCompletion(service_id=service_id))
        else:
            kernel.add_service(AzureChatCompletion(service_id=service_id))
        return service_id


# ----------------------------
# Domain Models & Synthetic Data
# ----------------------------


@dataclass
class Borrower:
    first_name: Optional[str] = None  # NEW
    last_name: Optional[str] = None  # NEW
    gross_annual_income: Optional[float] = None
    monthly_debts: Optional[float] = None
    monthly_expenses: Optional[float] = None
    dependents: int = 0
    savings: Optional[float] = None
    target_price: Optional[float] = None
    loan_term_years: int = 30
    rate_assessment: float = 0.069
    serviceability_buffer: float = 0.03
    location_hint: Optional[str] = None


@dataclass
class EligibilityOutcome:
    max_loan: float
    max_monthly_repay: float
    lvr: Optional[float]
    requires_lmi: bool
    eligible: bool
    notes: List[str] = field(default_factory=list)


# Synthetic property dataset (toy examples)
PROPERTIES: List[Dict[str, Any]] = []

# Synthetic area info
AREA_INFO: Dict[str, Dict[str, List[str]]] = {}

# ----------------------------
# Deterministic Engines
# ----------------------------


def resolve_suburb_key(name: Optional[str]) -> Optional[str]:
    t = (name or "").strip().lower()
    if not t:
        return None
    aliases = {"glen waverley": "Glen Waverley"}
    if t in aliases:
        return aliases[t]
    for k in {(r.get("suburb") or "").strip() for r in PROPERTIES}:
        if k.lower() == t:
            return k
    for k in AREA_INFO.keys():
        if k.lower() == t:
            return k
    return None


def pmt(pr: float, n_months: int, rate_monthly: float) -> float:
    """Standard annuity payment formula."""
    if rate_monthly == 0:
        return pr / max(n_months, 1)
    return (
        pr
        * (rate_monthly * (1 + rate_monthly) ** n_months)
        / ((1 + rate_monthly) ** n_months - 1)
    )


def compute_eligibility(b: Borrower) -> EligibilityOutcome:
    notes: List[str] = []

    # Defaults for missing items
    income = (b.gross_annual_income or 0.0) / 12.0
    debts = b.monthly_debts or 0.0
    expenses = b.monthly_expenses if b.monthly_expenses is not None else 2500.0

    # Simple synthetic rule-of-thumb for max repayment: 30% of gross monthly income - debts - dependent buffer
    dependent_buffer = 300.0 * max(b.dependents, 0)
    max_monthly_repay = max(
        0.0, 0.30 * income - debts - dependent_buffer - 0.10 * expenses
    )

    assessed_rate = (
        b.rate_assessment + b.serviceability_buffer
    )  # e.g., 9.9% if 6.9% + 3%
    r = assessed_rate / 12.0
    n = b.loan_term_years * 12

    # Invert PMT to get principal capacity: binary search
    def capacity_from_payment(pmt_target: float) -> float:
        lo, hi = 0.0, 2_000_000.0
        for _ in range(60):
            mid = (lo + hi) / 2
            m = pmt(mid, n, r)
            if m > pmt_target:
                hi = mid
            else:
                lo = mid
        return lo

    max_loan = capacity_from_payment(max_monthly_repay)

    lvr = None
    requires_lmi = False
    eligible = max_loan > 0

    if b.target_price and b.savings is not None:
        deposit = b.savings
        needed_loan = max(0.0, b.target_price - deposit)
        if b.target_price > 0:
            lvr = needed_loan / b.target_price
            if lvr > 0.90:
                requires_lmi = True
                notes.append("Deposit below 10% (LVR > 90%). Synthetic rule flags LMI.")
        if needed_loan > max_loan:
            eligible = False
            notes.append("Target price exceeds synthetic borrowing capacity.")

    if max_monthly_repay <= 0:
        eligible = False
        notes.append("Insufficient monthly surplus under synthetic rules.")

    return EligibilityOutcome(
        max_loan=round(max_loan, 2),
        max_monthly_repay=round(max_monthly_repay, 2),
        lvr=round(lvr, 3) if lvr is not None else None,
        requires_lmi=requires_lmi,
        eligible=eligible,
        notes=notes,
    )


def filter_properties(
    budget_max: Optional[float] = None,
    city: Optional[str] = None,  # kept for signature compat, but ignored
    suburb: Optional[str] = None,
    ptype: Optional[str] = None,
) -> List[Dict[str, Any]]:
    rows = PROPERTIES

    # coerce price → float in case CE returns strings/decimals
    def _price(r):
        try:
            return float(r.get("price", 0) or 0)
        except Exception:
            return 0.0

    if budget_max is not None:
        rows = [r for r in rows if _price(r) <= float(budget_max)]

    # ⬇️ Only filter by suburb and type
    if suburb:
        s = suburb.strip().lower()
        rows = [r for r in rows if (r.get("suburb") or "").strip().lower() == s]

    if ptype:
        t = ptype.strip().lower()
        rows = [r for r in rows if (r.get("type") or "").strip().lower() == t]

    return sorted(rows, key=lambda r: _price(r))[:8]


# ----------------------------
# Simple NLU to extract facts from free text
# ----------------------------

_NUM = r"([0-9]+(?:\.[0-9]+)?)"

NAME_FULL_PAT = re.compile(
    r"(?i)\b(?:my name is|i am|i'm)\s+([A-Za-z][A-Za-z\-']+)\s+([A-Za-z][A-Za-z\-']+)\b"
)
FIRSTNAME_PAT = re.compile(r"(?i)\bfirst\s*name\s*[:=\s]*([A-Za-z][A-Za-z\-']+)\b")
LASTNAME_PAT = re.compile(r"(?i)\blast\s*name\s*[:=\s]*([A-Za-z][A-Za-z\-']+)\b")

INCOME_PAT = re.compile(rf"(?i)(?:earn|income|salary)\s*[:=\s]*\$?{_NUM}\s*(k|,?000)?")
DEBT_PAT = re.compile(
    rf"(?i)(?:debt|debts|repay(?:ment)?s?)\s*[:=\s]*\$?{_NUM}\s*/?\s*(mo|month|monthly)?"
)
SAV_PAT = re.compile(rf"(?i)(?:savings|deposit|cash)\s*[:=\s]*\$?{_NUM}\s*(k|,?000)?")
PRICE_PAT = re.compile(rf"(?i)(?:target|budget|price)\s*[:=\s]*\$?{_NUM}\s*(k|,?000)?")
DEPS_PAT = re.compile(rf"(?i)(?:dependents?|kids?)\s*[:=\s]*{_NUM}")
EXP_PAT = re.compile(
    rf"(?i)(?:expenses|spend)\s*[:=\s]*\$?{_NUM}\s*/?\s*(mo|month|monthly)?"
)
TYPE_PAT = re.compile(
    r"(?i)\b(house|houses|apartment|apartments|unit|units|townhouse|townhouses)\b"
)
SUBURB_PAT = re.compile(
    r"(?i)(?:in|at|around)\s+(?!income\b)([A-Za-z][A-Za-z\s\-']{1,40})(?:,\s*VIC|\s*VIC)?\b"
)

CITY_PAT = re.compile(r"(?i)\bMelbourne\b")


def k_to_number(val: str, suff: Optional[str]) -> float:
    num = float(val)
    if suff and ("k" in suff.lower() or "000" in suff):
        num *= 1000.0
    return num


def parse_message_into_borrower(
    msg: str, b: Borrower
) -> Tuple[Borrower, Dict[str, Any]]:
    updates: Dict[str, Any] = {}
    if m := INCOME_PAT.search(msg):
        updates["gross_annual_income"] = k_to_number(m.group(1), m.group(2))
    if m := DEBT_PAT.search(msg):
        amt = float(m.group(1))
        # assume provided monthly if unit omitted
        updates["monthly_debts"] = amt
    if m := EXP_PAT.search(msg):
        amt = float(m.group(1))
        updates["monthly_expenses"] = amt
    if m := SAV_PAT.search(msg):
        updates["savings"] = k_to_number(m.group(1), m.group(2))
    if m := PRICE_PAT.search(msg):
        updates["target_price"] = k_to_number(m.group(1), m.group(2))
    if m := DEPS_PAT.search(msg):
        updates["dependents"] = int(float(m.group(1)))

    if m := TYPE_PAT.search(msg):
        t = m.group(1).lower().rstrip("s")
        updates["preferred_type"] = "apartment" if t in ["apartment", "unit"] else t

    if CITY_PAT.search(msg):
        updates["location_hint"] = "Melbourne"

    if m := SUBURB_PAT.search(msg):
        raw = m.group(1).strip()
        canon = resolve_suburb_key(raw)
        if canon:
            updates["location_hint"] = canon

    if m := NAME_FULL_PAT.search(msg):
        updates["first_name"] = m.group(1).strip().title()
        updates["last_name"] = m.group(2).strip().title()
    else:
        if m := FIRSTNAME_PAT.search(msg):
            updates["first_name"] = m.group(1).strip().title()
        if m := LASTNAME_PAT.search(msg):
            updates["last_name"] = m.group(1).strip().title()

    for k, v in updates.items():
        if hasattr(b, k):
            setattr(b, k, v)
    return b, updates


# ----------------------------
# LLM Prompt (single function, role-aware)
# ----------------------------

AGENT_PROMPT = """
You are part of a 3-agent housing assistant. Be concise, friendly, and action-oriented.

PHASE: {{$phase}}

SYSTEM FACTS (JSON):
{{$facts}}

RECENT MESSAGES:
{{$history}}

USER:
{{$user_input}}

GUIDELINES:
- For Eligibility phase: ask only for missing critical fields (first name, last name, income, debts, expenses, dependents, savings, target price, location). When sufficient, briefly summarize capacity and offer "/next" to see listings.
- For Listings phase: **use ONLY the items provided in SYSTEM FACTS under `current_listings`. Never invent properties.** If `current_listings` is empty or missing, say "No matches in my dataset for the current filters" and suggest narrowing/widening filters (e.g., suburb, type, or budget). Present 3–5 options from `current_listings` as a numbered list with price, type, and suburb. Offer "/pick <#>" for area details.
- For Area phase: If the user asks for specific categories (e.g., only PTV or only restaurants), show **only** those categories present in `area_info` / `requested_categories`. Do not include categories they did not ask for. If none were specified, give a concise mix across schools, PTV, and restaurants. If a requested category has no data, say so briefly.
- Always end with one clear next action the user can take (e.g., "/next", "/pick 2", or provide a missing field).

REPLY:
"""


def build_prompt_config(service: str, service_id: str):
    if service.lower() == "openai":
        exec_settings = OpenAIChatPromptExecutionSettings(
            service_id=service_id,
            ai_model_id=os.getenv("OPENAI_CHAT_MODEL_ID", "gpt-5"),
        )
    else:
        exec_settings = AzureChatPromptExecutionSettings(
            service_id=service_id,
            ai_model_id=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt-35-turbo"),
            max_completion_tokens=800,
            temperature=0.2,
        )

    return PromptTemplateConfig(
        template=AGENT_PROMPT,
        name="agent_reply",
        template_format="semantic-kernel",
        input_variables=[
            InputVariable(name="phase", description="current phase", is_required=True),
            InputVariable(name="facts", description="json facts", is_required=True),
            InputVariable(name="history", description="chat history", is_required=True),
            InputVariable(
                name="user_input", description="user message", is_required=True
            ),
        ],
        execution_settings=exec_settings,
    )


# ----------------------------
# Orchestrator / Router
# ----------------------------


class Phase:
    ELIG = "eligibility"
    LIST = "listings"
    AREA = "area"
    DONE = "done"


@dataclass
class Session:
    borrower: Borrower = field(default_factory=Borrower)
    phase: str = Phase.ELIG
    picked_listing: Optional[int] = None
    # Derived fields
    last_outcome: Optional[EligibilityOutcome] = None
    preferred_type: Optional[str] = None

    def facts(self) -> Dict[str, Any]:
        base = asdict(self.borrower)
        base["preferred_type"] = self.preferred_type
        base["phase"] = self.phase
        base["picked_listing"] = self.picked_listing
        if self.last_outcome:
            base["eligibility_outcome"] = asdict(self.last_outcome)
        return base


HELP = (
    "Commands: /help, /reset, /next (advance phase), /list (back to listings), "
    "/pick <#> (select listing), /quit"
)


def human_readable_history(chat: ChatHistory, last_n: int = 6) -> str:
    # Render a short history string for the LLM
    msgs = chat.messages[-last_n:]
    chunks = []
    for m in msgs:
        role = getattr(m, "role", "")
        content = getattr(m, "content", "")
        chunks.append(f"{role.capitalize()}: {content}")
    return "\n".join(chunks)


async def run_cli() -> None:
    cfg = Config()
    kernel = Kernel()
    service_id = cfg.add_service_to_kernel(kernel)

    # Add a single role-aware function
    prompt_cfg = build_prompt_config(cfg.service, service_id)
    agent_func = kernel.add_function(
        function_name="reply",
        plugin_name="agent",
        prompt_template_config=prompt_cfg,
    )

    chat = ChatHistory()
    sess = Session()

    print("Controls:", HELP)
    print('Tip: "I earn 110k, debts 450/mo, savings 40k — budget 800k in Brunswick"')

    while True:
        user = input("you       > ").strip()
        if not user:
            continue
        if user.lower() in {"/quit", "/q", ":q"}:
            print("bye!")
            break
        if user.lower() == "/help":
            print(HELP)
            continue
        if user.lower() == "/reset":
            sess = Session()
            chat = ChatHistory()
            print("Session reset.")
            continue
        if user.lower() == "/next":
            if sess.phase == Phase.ELIG:
                sess.phase = Phase.LIST
            elif sess.phase == Phase.LIST:
                if sess.picked_listing is not None:
                    sess.phase = Phase.AREA
                else:
                    print("Pick a listing first with /pick <#>.")
            elif sess.phase == Phase.AREA:
                sess.phase = Phase.DONE
            else:
                print("Already at DONE.")
            # fallthrough to respond
        if user.lower() == "/list":
            sess.phase = Phase.LIST
        if user.lower().startswith("/pick"):
            parts = user.split()
            if len(parts) == 2 and parts[1].isdigit():
                idx = int(parts[1])
                results = current_listings(sess)
                if 1 <= idx <= len(results):
                    sess.picked_listing = results[idx - 1]["id"]
                    sess.phase = Phase.AREA
                else:
                    print("Invalid pick index.")
                    continue
            else:
                print("Usage: /pick <#>")
                continue

        # Update session from free text
        before = asdict(sess.borrower)
        sess.borrower, updates = parse_message_into_borrower(user, sess.borrower)
        if "preferred_type" in updates:
            sess.preferred_type = updates.get("preferred_type")

        # Compute eligibility when enough info
        need = []
        if sess.phase == Phase.ELIG:
            if sess.borrower.gross_annual_income is None:
                need.append("income")
            if sess.borrower.monthly_debts is None:
                need.append("debts")
            if sess.borrower.monthly_expenses is None:
                need.append("expenses")
            if sess.borrower.savings is None:
                need.append("savings")
            if sess.borrower.target_price is None:
                need.append("target price")
            if sess.borrower.location_hint is None:
                need.append("location")

            if len(need) <= 2:  # heuristic: compute early when mostly filled
                sess.last_outcome = compute_eligibility(sess.borrower)

        # If in listings, compute current filtered results
        listing_snippet = []
        if sess.phase in (Phase.LIST, Phase.AREA):
            max_budget = sess.borrower.target_price or (
                sess.last_outcome.max_loan if sess.last_outcome else None
            )
            city, suburb = (None, None)
            if sess.borrower.location_hint:
                # crude: treat known suburbs; else treat as city
                lh = sess.borrower.location_hint
                suburb = lh if lh in AREA_INFO else None
                city = "Melbourne"
            results = filter_properties(
                max_budget, city=city, suburb=suburb, ptype=sess.preferred_type
            )
            listing_snippet = [
                {
                    "i": i + 1,
                    "id": r["id"],
                    "title": r["title"],
                    "price": r["price"],
                    "type": r["type"],
                    "suburb": r["suburb"],
                }
                for i, r in enumerate(results)
            ]

        # If in area and have pick, fetch area info
        area_snippet = {}
        if sess.phase == Phase.AREA and sess.picked_listing is not None:
            chosen = next(
                (r for r in PROPERTIES if r["id"] == sess.picked_listing), None
            )
            if chosen:
                area_snippet = AREA_INFO.get(chosen["suburb"], {})

        facts = sess.facts()
        if listing_snippet:
            facts["current_listings"] = listing_snippet
        if area_snippet:
            facts["area_info"] = area_snippet

        # Compose short history for the LLM
        chat.add_user_message(user)
        hist = human_readable_history(chat)

        # Ask the agent LLM to reply
        reply = await kernel.invoke(
            agent_func,
            KernelArguments(
                phase=sess.phase,
                facts=json.dumps(facts, ensure_ascii=False),
                history=hist,
                user_input=user,
            ),
        )
        reply_text = str(reply)
        print(f"assistant > {reply_text}")
        chat.add_assistant_message(reply_text)

        # Auto-advance heuristic: if user asked to see listings and we're eligible
        if (
            sess.phase == Phase.ELIG
            and any(k in user.lower() for k in ["listings", "houses", "show houses"])
            and sess.last_outcome
        ):
            sess.phase = Phase.LIST


def current_listings(sess: Session) -> List[Dict[str, Any]]:
    max_budget = sess.borrower.target_price or (
        sess.last_outcome.max_loan if sess.last_outcome else None
    )

    suburb = None
    if sess.borrower.location_hint:
        suburb = resolve_suburb_key(sess.borrower.location_hint)

    # ⬇️ city=None so it won't filter out CE rows with blank city
    return filter_properties(
        max_budget, city=None, suburb=suburb, ptype=sess.preferred_type
    )


if __name__ == "__main__":
    try:
        asyncio.run(run_cli())
    except KeyboardInterrupt:
        print("\nInterrupted.")
