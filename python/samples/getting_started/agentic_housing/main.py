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

    # 1) Normalize inputs (monthly basis)
    monthly_income = float(b.gross_annual_income or 0.0) / 12.0
    debts = float(b.monthly_debts or 0.0)
    expenses = float(b.monthly_expenses if b.monthly_expenses is not None else 2500.0)
    dependents = max(int(b.dependents or 0), 0)

    # 2) Repayment capacity (conservative but reasonable)
    #    - DSR cap: at most 35% of gross monthly income
    #    - Surplus cap: income - expenses - debts - dependent buffer
    dependent_buffer = 300.0 * dependents
    dsr_cap = 0.35 * monthly_income
    surplus_cap = monthly_income - expenses - debts - dependent_buffer
    max_monthly_repay = max(0.0, min(dsr_cap, surplus_cap))

    # Add helpful sanity notes (doesn't change the math)
    if expenses > monthly_income * 0.8:
        notes.append(
            "Monthly expenses are very high relative to income—double-check the amount."
        )
    if debts > monthly_income * 0.5:
        notes.append(
            "Monthly debts are high relative to income—capacity reduced by synthetic rules."
        )

    # 3) Convert repayment capacity -> principal capacity at assessed/stress rate
    assessed_rate = float(b.rate_assessment or 0.0) + float(
        b.serviceability_buffer or 0.0
    )
    r = assessed_rate / 12.0  # monthly rate
    n = int((b.loan_term_years or 30) * 12)

    if max_monthly_repay <= 0 or n <= 0:
        max_loan = 0.0
    else:
        if r <= 0:
            # Zero/invalid rate fallback: simple linear sum
            max_loan = max_monthly_repay * n
        else:
            # PV of an annuity: PV = PMT * (1 - (1+r)^-n) / r
            annuity_factor = (1.0 - (1.0 + r) ** (-n)) / r
            max_loan = max_monthly_repay * annuity_factor

    # 4) LVR/LMI & eligibility vs target
    lvr = None
    requires_lmi = False
    eligible = max_loan > 0.0

    if b.target_price and b.savings is not None:
        target_price = float(b.target_price)
        deposit = float(b.savings)
        needed_loan = max(0.0, target_price - deposit)

        if target_price > 0:
            lvr = needed_loan / target_price
            if lvr > 0.90:
                requires_lmi = True
                notes.append("Deposit below 10% (LVR > 90%). Synthetic rule flags LMI.")

        if needed_loan > max_loan:
            eligible = False
            notes.append("Target price exceeds synthetic borrowing capacity.")

    # 5) No surplus → not eligible
    if max_monthly_repay <= 0:
        eligible = False
        if "Insufficient monthly surplus under synthetic rules." not in notes:
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

# ----------------------------
# Agent Backgrounds & Prompts
# ----------------------------

AGENT_BACKGROUNDS = {
    "eligibility": """You are Deloitte’s EligibilityAgent.
Background:
- You DO NOT ask the user questions. Treat the USER content as a trigger only.
- You only summarize what is already in SYSTEM FACTS (borrower fields + eligibility_outcome).
- You never provide financial advice; you only summarize the synthetic calculator result.
- End with a clear next step (e.g., "/next" to view listings).""",
    "listings": """You are Deloitte’s ListingsAgent.
Background:
- You only show properties passed in SYSTEM FACTS under `current_listings`. Never invent rows.
- Do NOT suggest changing filters or searching new suburbs. You can only use the details already provided via the form.
- Allowed CTA: only “/pick <#>” (for area details). Never mention /book, /save, /search, /set, or “widen the search”.
- Show 3–5 options, sorted sensibly, each with price, type, suburb and a short blurb. End with a single clear “/pick <#>” hint.""",
    "area": """You are Deloitte’s AreaAgent.
Background:
- You show nearby Schools, PTV and Restaurants for the picked listing’s suburb (from `area_info`).
- If the user asks for a category, show only that category; if none specified, show a concise mix.
- If a requested category has no data, say so briefly and suggest another.""",
}

ELIGIBILITY_PROMPT = """
ROLE: EligibilityAgent

BACKGROUND:
{{$background}}

SYSTEM FACTS (JSON):
{{$facts}}

RECENT MESSAGES:
{{$history}}

USER:
{{$user_input}}

GUIDELINES:
- DO NOT ask the user for any missing information. Ignore USER content beyond being a trigger to respond.
- Rely only on SYSTEM FACTS. If `eligibility_outcome` is present, summarize clearly:
  • Max borrowing capacity (A$)
  • Estimated max monthly repayment (A$)
  • Target price (A$) and savings/deposit (A$) if available
  • LVR (%, if available) and whether LMI is required
  • Any notable notes from `eligibility_outcome.notes`
- If `eligibility_outcome` is missing, say: "I’m waiting for the eligibility form to be submitted." and explain the next step briefly.
- Prompt-only policy overlays (demo purposes):
    • If savings >= 1,000,000 → add a line: "Outcome: Cash-sufficient — loan not required. YOU ARE VERY RICH!".
- End with one next action (e.g., "/next" to see listings).

OUTPUT STYLE (strict)
- Use exactly this compact 5–6 line card.
- No extra sentences before/after.
- Money: thousands separators, no cents (e.g., A$290,741). Repayments: add “/mo”.
- Percent: 1 decimal (e.g., 81.3%).
- Pull values from SYSTEM FACTS → eligibility_outcome and borrower fields.
- If there are no notes, omit the Notes line.

REPLY:
"""

LISTINGS_PROMPT = """
ROLE: ListingsAgent

BACKGROUND:
{{$background}}

SYSTEM FACTS (JSON):
{{$facts}}

RECENT MESSAGES:
{{$history}}

USER:
{{$user_input}}

GUIDELINES:
- Use ONLY `current_listings` from SYSTEM FACTS. Never invent or infer properties. If empty, say so briefly.
- Default output: show 3–5 options as a numbered list. For each item show exactly: A$<price> — <title> — <suburb>. Do NOT include bedrooms, bathrooms, car spaces, land size, or other specs unless the user explicitly asks or after they use /pick.
- Provide one short, helpful blurb (1 sentence) based only on the title/location; do not infer amenities or specs.
- Allowed CTA: only “/pick <#>”. Never mention /book, /save, /search, /set, or “widen the search”.
- Do not ask the user to change suburbs/budget in chat. If they request changes, say: “I can only use the details you submitted. To change them, please /reset and refill the form.”
- Keep the response concise and action-oriented. End with a single line prompt like: Choose one with /pick <#>.

REPLY:
"""

AREA_PROMPT = """
ROLE: AreaAgent

BACKGROUND:
{{$background}}

SYSTEM FACTS (JSON):
{{$facts}}

RECENT MESSAGES:
{{$history}}

USER:
{{$user_input}}

GUIDELINES:
- If the user asked for specific categories, show only those from `area_info` / `requested_categories`.
- If none specified, show a concise mix from schools, PTV, and restaurants.
- If a requested category is empty, mention it briefly and suggest another.
- End with one clear next action (e.g., ask for a category or "/next" to finish).

REPLY:
"""


def build_prompt_config(service: str, service_id: str, *, role: str):
    template_map = {
        "eligibility": ELIGIBILITY_PROMPT,
        "listings": LISTINGS_PROMPT,
        "area": AREA_PROMPT,
    }
    template = template_map[role]

    if service.lower() == "openai":
        exec_settings = OpenAIChatPromptExecutionSettings(
            service_id=service_id,
            ai_model_id=os.getenv("OPENAI_CHAT_MODEL_ID", "gpt-5"),
        )
    else:
        exec_settings = AzureChatPromptExecutionSettings(
            service_id=service_id,
            ai_model_id=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt-35-turbo"),
            temperature=0.2,
            max_completion_tokens=800,
        )

    return PromptTemplateConfig(
        template=template,
        name=f"{role}_reply",
        template_format="semantic-kernel",
        input_variables=[
            InputVariable(name="facts", description="json facts", is_required=True),
            InputVariable(name="history", description="chat history", is_required=True),
            InputVariable(
                name="user_input", description="user message", is_required=True
            ),
            InputVariable(
                name="background", description="agent background", is_required=True
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

        def _f(x):
            try:
                return float(x)
            except Exception:
                return None

        s = _f(base.get("savings"))
        tp = _f(base.get("target_price"))
        cash_sufficient = bool(
            (s is not None) and (s >= 1_000_000 or (tp is not None and s >= tp))
        )
        base["policy_overlays"] = {"cash_sufficient": cash_sufficient}

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
    elig_cfg = build_prompt_config(cfg.service, service_id, role="eligibility")
    list_cfg = build_prompt_config(cfg.service, service_id, role="listings")
    area_cfg = build_prompt_config(cfg.service, service_id, role="area")

    elig_func = kernel.add_function("reply", "eligibility_agent", elig_cfg)
    list_func = kernel.add_function("reply", "listings_agent", list_cfg)
    area_func = kernel.add_function("reply", "area_agent", area_cfg)

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

        # Choose agent by phase
        if sess.phase == Phase.ELIG:
            func = elig_func
            background = AGENT_BACKGROUNDS["eligibility"]
        elif sess.phase == Phase.LIST:
            func = list_func
            background = AGENT_BACKGROUNDS["listings"]
        elif sess.phase == Phase.AREA:
            func = area_func
            background = AGENT_BACKGROUNDS["area"]
        else:
            func = elig_func
            background = AGENT_BACKGROUNDS["eligibility"]

        # Ask the agent LLM to reply
        reply = await kernel.invoke(
            func,
            KernelArguments(
                facts=json.dumps(facts, ensure_ascii=False),
                history=hist,
                user_input=user,
                background=background,
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


def current_listings(sess, *, max_rows=8):
    rows = list(PROPERTIES)

    loc = (sess.borrower.location_hint or "").strip().lower()
    if loc:
        by_loc = [r for r in rows if (r.get("suburb") or "").strip().lower() == loc]
        rows = by_loc or rows

    # Optional type filter (leave as-is if you don't use it)
    if sess.preferred_type:
        t = sess.preferred_type.strip().lower()
        rows = [r for r in rows if (r.get("type") or "").strip().lower() == t] or rows

    cap = float(sess.borrower.target_price or 0)

    def price(r):
        return float(r.get("price") or 0)

    NEAR_BAND = 0.25  # 25% instead of 10%
    OVER_TAKE = 2  # take up to 2 closest over-budget items

    if cap:
        under = sorted([r for r in rows if price(r) <= cap], key=price)
        over = sorted([r for r in rows if price(r) > cap], key=lambda r: price(r) - cap)
        near = [r for r in over if price(r) <= cap * (1 + NEAR_BAND)]
        rows = (under + near[:OVER_TAKE])[:max_rows]
    else:
        rows.sort(key=price)

    return rows


if __name__ == "__main__":
    try:
        asyncio.run(run_cli())
    except KeyboardInterrupt:
        print("\nInterrupted.")
