import asyncio
import os
import secrets
import importlib.util
from dataclasses import asdict
from typing import Dict, Any
from collections import defaultdict

from flask import Flask, render_template, request, jsonify, session as flask_session
from semantic_kernel import Kernel
from semantic_kernel.contents import ChatHistory
from semantic_kernel.functions import KernelArguments

import logging
import requests
import sys

# ---- Dynamic import of your existing SK script (works even with hyphen in filename)
AGENT_FILE = os.environ.get("AGENT_FILE", "main.py")

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
logging.basicConfig(level=logging.INFO)

# Store per-browser session state in-memory (OK for demo)
_sessions: Dict[str, Dict[str, Any]] = {}

# ---------------------------
# DATAVERSE SETTINGS (hard-code here for now)
# ---------------------------
DYN_URL = "https://org88ea65ee.api.crm6.dynamics.com"
DYN_API_VERSION = "9.2"
WEBAPIURL = f"{DYN_URL}/api/data/v{DYN_API_VERSION}/"

# Option A: paste a short-lived bearer (from Postman) to skip token flow (for quick tests)
HARDCODE_BEARER = "1eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsIng1dCI6IkpZaEFjVFBNWl9MWDZEQmxPV1E3SG4wTmVYRSIsImtpZCI6IkpZaEFjVFBNWl9MWDZEQmxPV1E3SG4wTmVYRSJ9.eyJhdWQiOiJodHRwczovL29yZzg4ZWE2NWVlLmFwaS5jcm02LmR5bmFtaWNzLmNvbSIsImlzcyI6Imh0dHBzOi8vc3RzLndpbmRvd3MubmV0LzJlOTQ4ZWFmLWI1MmQtNDY2Ni04ZGFhLWNiYjMwZjgwZWM1ZC8iLCJpYXQiOjE3NTU3NTU5MTgsIm5iZiI6MTc1NTc1NTkxOCwiZXhwIjoxNzU1NzU5ODY1LCJhY2N0IjowLCJhY3IiOiIxIiwiYWlvIjoiQVVRQXUvOFpBQUFBSUxncDJVT3lFcXBkeUJpVVNXYjlHaGRVM2FDUkQyKzFCUWFFRnJnT3owUHI1UEU1TjQ1bzc0V2pybWh3UWY1RUFzYWFCSC93alovdUJlVFhtVXJYZ2c9PSIsImFtciI6WyJwd2QiXSwiYXBwaWQiOiI1MWY4MTQ4OS0xMmVlLTRhOWUtYWFhZS1hMjU5MWY0NTk4N2QiLCJhcHBpZGFjciI6IjAiLCJmYW1pbHlfbmFtZSI6IkRlIEZvbnNla2EiLCJnaXZlbl9uYW1lIjoiU2hlYW4iLCJpZHR5cCI6InVzZXIiLCJpcGFkZHIiOiIyMDAxOjgwMDM6ZGMwYjplZDAwOmZjZDM6NWY5ODo4M2E1OmFmNDYiLCJsb2dpbl9oaW50IjoiTy5DaVJoT1RJMU5UWXpOUzB3TWpBeUxUUmtZbUl0WVdGa09TMDVNVGxsTVRkaFptWTFPRGNTSkRKbE9UUTRaV0ZtTFdJMU1tUXRORFkyTmkwNFpHRmhMV05pWWpNd1pqZ3daV00xWkJva2MyUmxabTl1YzJWcllVQmtaV3h2YVhSMFpYQXViMjV0YVdOeWIzTnZablF1WTI5dElEaz0iLCJuYW1lIjoiU2hlYW4gRGUgRm9uc2VrYSIsIm9pZCI6ImE5MjU1NjM1LTAyMDItNGRiYi1hYWQ5LTkxOWUxN2FmZjU4NyIsInB1aWQiOiIxMDAzMjAwNDZBMDI2QjE2IiwicmgiOiIxLkFXY0FyNDZVTGkyMVprYU5xc3V6RDREc1hRY0FBQUFBQUFBQXdBQUFBQUFBQUFBMEFjaG5BQS4iLCJzY3AiOiJ1c2VyX2ltcGVyc29uYXRpb24iLCJzaWQiOiIwMDdkNGUzOS1jNjEyLWNkNmItZDYxOS05ZGFiM2IyNTEyZDgiLCJzdWIiOiIyd0RxZ0dPYm1kemhwak5RX3ZMS2F2Q3ZVQVc0ckdHMTVuTW5hSkJDV3c4IiwidGVuYW50X3JlZ2lvbl9zY29wZSI6Ik9DIiwidGlkIjoiMmU5NDhlYWYtYjUyZC00NjY2LThkYWEtY2JiMzBmODBlYzVkIiwidW5pcXVlX25hbWUiOiJzZGVmb25zZWthQGRlbG9pdHRlcC5vbm1pY3Jvc29mdC5jb20iLCJ1cG4iOiJzZGVmb25zZWthQGRlbG9pdHRlcC5vbm1pY3Jvc29mdC5jb20iLCJ1dGkiOiJma04ySEd1QzUwdW5lMlN4Nlpwa0FBIiwidmVyIjoiMS4wIiwieG1zX2Z0ZCI6IklETVJCYTAtNF9RaTJjUF9FZDJXbWJDRVlDYXk5Z1RLelRueE9lSzhWdE1CWVhWemRISmhiR2xoWXkxa2MyMXoiLCJ4bXNfaWRyZWwiOiIxIDEyIn0.Iu6_2nAW2zCtb43xdwGG5p4pMkFwyWHYXPGK3Ph3-pXklvELs7cTH3KItDPSNqypK6_8-3pOvj8nVTJyzpM8Qi5-cf96O0d6e_5qeLQEIrOIhZbVkP64Qy33fIr8Xyd801bF0Vhl5rC44wix0GPxt8lWIFPdAYhmdENT4TXsVrUo-OhWMhyOEt8WS9jsMhwKTa3jgwnlVxZcrsLekh_X2sejjo-iPexn9L87eIJV-VpGvZ0bWqYSIYDkH7gptoyHNECsWxE1unVoqo7358m0tXNt9QDYPVhzwKraRtp38Rhnusdeqc5M38vYbhbmRlSEgPRhXRxccr3fRifrP1J3dA"
BEARER = ""

# Option B: proper client-credentials flow (recommended)
TENANT_ID = "common"
CLIENT_ID = "51f81489-12ee-4a9e-aaaa-a2591f4598ab"


def _token_endpoint(tenant: str) -> str:
    return f"https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token"


# --- helpers (put near the top of app.py) ---
def _as_float(v):
    if v is None:
        return None
    try:
        # allow "150,000.00" or "150000" strings
        return float(str(v).replace(",", ""))
    except ValueError:
        return None


def get_access_token() -> str:
    if HARDCODE_BEARER:
        app.logger.info("Using hard-coded bearer token for Dataverse.")
        return HARDCODE_BEARER
    if not (CLIENT_ID and TENANT_ID):
        raise RuntimeError("CLIENT_ID/TENANT_ID not set for Dataverse token flow.")
    token_url = _token_endpoint(TENANT_ID)
    scope = f"{DYN_URL}/.default"
    resp = requests.post(
        token_url,
        data={
            "client_id": CLIENT_ID,
            "scope": scope,
            "grant_type": "client_credentials",
        },
        timeout=20,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def dataverse_create_contact(
    access_token: str,
    *,
    firstname: str,
    lastname: str,
    cr54b_income: float,
    cr54b_debts: float,
    cr54b_expense: float,
    cr54b_savings: float,
    cr54b_target_price: float,
    cr54b_suburb: str,
) -> dict:
    url = f"{WEBAPIURL}contacts"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json",
        "Content-Type": "application/json; charset=utf-8",
        "OData-MaxVersion": "4.0",
        "OData-Version": "4.0",
    }
    payload = {
        "firstname": firstname,
        "lastname": lastname,
        "cr54b_income": cr54b_income,
        "cr54b_debts": cr54b_debts,
        "cr54b_expenses": cr54b_expense,
        "cr54b_savings": cr54b_savings,
        "cr54b_target_price": cr54b_target_price,
        "cr54b_suburb": cr54b_suburb,
    }

    r = requests.post(url, headers=headers, json=payload, timeout=20)
    # Dataverse usually returns 204 with the new entity URL in OData-EntityId header
    if r.status_code in (201, 204):
        entity_url = r.headers.get("OData-EntityId") or r.headers.get("odata-entityid")
        contact_id = None
        if entity_url:
            import re as _re

            m = _re.search(r"\(([^)]+)\)", entity_url)
            if m:
                contact_id = m.group(1)
        return {"ok": True, "id": contact_id, "entityUrl": entity_url}

    try:
        r.raise_for_status()
    except Exception:
        pass
    return {"ok": False, "status": r.status_code, "text": r.text}


# ---- Raw Dataverse fetchers
def dataverse_get_contacts(access_token: str, top: int = 5) -> dict:
    url = f"{WEBAPIURL}contacts?$select=contactid,fullname,emailaddress1&$top={top}"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json",
        "OData-MaxVersion": "4.0",
        "OData-Version": "4.0",
    }
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    return r.json()


def dataverse_get_properties(access_token: str, top: int = 5) -> dict:
    url = (
        f"{WEBAPIURL}cr54b_properties"
        f"?$select=cr54b_title,cr54b_price,cr54b_type,cr54b_city,cr54b_suburb,cr54b_state,"
        f"cr54b_bed,cr54b_bath,cr54b_car,cr54b_propertyid&$top={top}"
    )
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json",
        "OData-MaxVersion": "4.0",
        "OData-Version": "4.0",
    }
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    return r.json()


def dataverse_get_ptv(access_token: str, top: int = 5) -> dict:
    url = f"{WEBAPIURL}cr54b_ptvs?$select=cr54b_name,cr54b_suburb&$top={top}"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json",
        "OData-MaxVersion": "4.0",
        "OData-Version": "4.0",
    }
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    return r.json()


def dataverse_get_restaurants(access_token: str, top: int = 5) -> dict:
    url = f"{WEBAPIURL}cr54b_restaurants?$select=cr54b_name,cr54b_suburb&$top={top}"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json",
        "OData-MaxVersion": "4.0",
        "OData-Version": "4.0",
    }
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    return r.json()


def dataverse_get_schools(access_token: str, top: int = 5) -> dict:
    url = f"{WEBAPIURL}cr54b_schools?$select=cr54b_name,cr54b_suburb&$top={top}"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json",
        "OData-MaxVersion": "4.0",
        "OData-Version": "4.0",
    }
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    return r.json()


# ---------------------------
# Mapper: Dataverse -> agentic_core.PROPERTIES / AREA_INFO
# ---------------------------
def _normalize_type(s: str) -> str:
    t = (s or "").strip().lower()
    if t == "unit":  # map unit → apartment for your regexes
        return "apartment"
    return t


def _key_suburb(s: str) -> str:
    # Normalize suburb key (trim + preserve original spelling from CE)
    return (s or "").strip()


def _map_properties(props_json: dict) -> list[dict]:
    rows = []
    for it in props_json.get("value", []):
        rows.append(
            {
                "id": it.get("cr54b_propertyid")
                or it.get("id")
                or it.get("cr54b_title"),
                "title": (it.get("cr54b_title") or "").strip(),
                "price": it.get("cr54b_price") or 0,
                "type": _normalize_type(it.get("cr54b_type") or ""),
                "city": (it.get("cr54b_city") or "").strip(),
                "suburb": _key_suburb(it.get("cr54b_suburb") or ""),
                "state": (it.get("cr54b_state") or "").strip(),
                "lat": None,  # CE schema sample doesn't include coords
                "lon": None,
                "bed": it.get("cr54b_bed"),
                "bath": it.get("cr54b_bath"),
                "car": it.get("cr54b_car"),
            }
        )
    return rows


def _map_area_info(
    ptv_json: dict,
    schools_json: dict,
    restaurants_json: dict,
    property_rows: list[dict],
) -> dict:
    area: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: {"schools": [], "ptv": [], "restaurants": []}
    )

    # seed keys from properties so AREA_INFO has entries for all suburbs with listings
    for p in property_rows:
        _ = area[_key_suburb(p.get("suburb") or "")]

    for it in ptv_json.get("value", []):
        suburb = _key_suburb(it.get("cr54b_suburb"))
        name = (it.get("cr54b_name") or "").strip()
        if suburb and name and name not in area[suburb]["ptv"]:
            area[suburb]["ptv"].append(name)

    for it in schools_json.get("value", []):
        suburb = _key_suburb(it.get("cr54b_suburb"))
        name = (it.get("cr54b_name") or "").strip()
        if suburb and name and name not in area[suburb]["schools"]:
            area[suburb]["schools"].append(name)

    for it in restaurants_json.get("value", []):
        suburb = _key_suburb(it.get("cr54b_suburb"))
        name = (it.get("cr54b_name") or "").strip()
        if suburb and name and name not in area[suburb]["restaurants"]:
            area[suburb]["restaurants"].append(name)

    # cast defaultdict to regular dict for cleanliness
    return {k: v for k, v in area.items()}


def refresh_dataverse_into_agent(
    top_props: int = 50, top_support: int = 100, access_token: str | None = None
) -> dict:
    """
    Pull CE data and write into agentic_core.PROPERTIES and agentic_core.AREA_INFO.
    Returns a small summary for logging/UI.
    """
    token = access_token or get_access_token()
    props = dataverse_get_properties(token, top=top_props)
    ptv = dataverse_get_ptv(token, top=top_support)
    schools = dataverse_get_schools(token, top=top_support)
    restaurants = dataverse_get_restaurants(token, top=top_support)

    prop_rows = _map_properties(props)
    area_info = _map_area_info(ptv, schools, restaurants, prop_rows)

    # Swap into the agent module so current_listings/Area use live data
    agentic_core.PROPERTIES = prop_rows
    print("agentic_core.PROPERTIES")
    print(agentic_core.PROPERTIES)
    agentic_core.AREA_INFO = area_info
    print("agentic_core.AREA_INFO")
    print(agentic_core.AREA_INFO)

    app.logger.info(
        "Dataverse sync -> PROPERTIES=%d, AREA_INFO suburbs=%d",
        len(prop_rows),
        len(area_info),
    )
    return {
        "properties": len(prop_rows),
        "suburbs": len(area_info),
    }


# ---------------------------
# Optional: endpoints
# ---------------------------


def _pick_bearer_from_request_or_default() -> str:
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1].strip()
    return get_access_token()


@app.post("/api/dynamics/contacts")
def api_create_contact():
    data = request.get_json(force=True) or {}

    firstname = (data.get("firstname") or data.get("first_name") or "").strip()
    lastname = (data.get("lastname") or data.get("last_name") or "").strip()
    income = _as_float(data.get("cr54b_income"))
    debts = _as_float(data.get("cr54b_debts"))
    expenses = _as_float(data.get("cr54b_expenses") or data.get("cr54b_expense"))
    savings = _as_float(data.get("cr54b_savings"))
    target = _as_float(data.get("cr54b_target_price"))
    suburb = (data.get("cr54b_suburb") or data.get("suburb") or "").strip()

    # validate
    if not firstname or not lastname:
        return jsonify({"ok": False, "error": "firstname/lastname required"}), 400
    for label, val in [
        ("income", income),
        ("debts", debts),
        ("expenses", expenses),
        ("savings", savings),
        ("target price", target),
    ]:
        if val is None:
            return jsonify({"ok": False, "error": f"{label} required"}), 400
    if not suburb:
        return jsonify({"ok": False, "error": "suburb required"}), 400

    try:
        token = _pick_bearer_from_request_or_default()
        result = dataverse_create_contact(
            token,
            firstname=firstname,
            lastname=lastname,
            cr54b_income=income,
            cr54b_debts=debts,
            cr54b_expense=expenses,
            cr54b_savings=savings,
            cr54b_target_price=target,
            cr54b_suburb=suburb,
        )
        if not result.get("ok"):
            return (
                jsonify({"ok": False, "error": result.get("text", "Dataverse error")}),
                502,
            )
        return jsonify(result)
    except Exception as e:
        app.logger.exception("create contact failed")
        return jsonify({"ok": False, "error": str(e)}), 500


@app.get("/api/dynamics/properties")
def api_dynamics_properties():
    try:
        top = int(request.args.get("top", 5))
    except ValueError:
        top = 5
    try:
        token = _pick_bearer_from_request_or_default()
        data = dataverse_get_properties(token, top=top)
        return jsonify(data)
    except Exception as e:
        app.logger.exception("properties call failed")
        return jsonify({"error": str(e)}), 500


@app.get("/api/dynamics/ptv")
def api_dynamics_ptv():
    try:
        top = int(request.args.get("top", 5))
    except ValueError:
        top = 5
    try:
        token = _pick_bearer_from_request_or_default()
        data = dataverse_get_ptv(token, top=top)
        return jsonify(data)
    except Exception as e:
        app.logger.exception("ptv call failed")
        return jsonify({"error": str(e)}), 500


@app.get("/api/dynamics/restaurants")
def api_dynamics_restaurants():
    try:
        top = int(request.args.get("top", 5))
    except ValueError:
        top = 5
    try:
        token = _pick_bearer_from_request_or_default()
        data = dataverse_get_restaurants(token, top=top)
        return jsonify(data)
    except Exception as e:
        app.logger.exception("restaurants call failed")
        return jsonify({"error": str(e)}), 500


@app.get("/api/dynamics/schools")
def api_dynamics_schools():
    try:
        top = int(request.args.get("top", 5))
    except ValueError:
        top = 5
    try:
        token = _pick_bearer_from_request_or_default()
        data = dataverse_get_schools(token, top=top)
        return jsonify(data)
    except Exception as e:
        app.logger.exception("schools call failed")
        return jsonify({"error": str(e)}), 500


@app.post("/api/dynamics/sync")
def api_dynamics_sync():
    try:
        props = int(request.args.get("props", 50))
    except ValueError:
        props = 50
    try:
        support = int(request.args.get("support", 100))
    except ValueError:
        support = 100

    try:
        # ⬇️ use the Authorization header if present
        token = _pick_bearer_from_request_or_default()
        summary = refresh_dataverse_into_agent(
            top_props=props, top_support=support, access_token=token
        )
        return jsonify({"ok": True, "summary": summary})
    except Exception as e:
        app.logger.exception("Dataverse sync failed")
        return jsonify({"ok": False, "error": str(e)}), 500


# ---------------------------
# Your SK web wrapper (unchanged)
# ---------------------------
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
                facts["area_info"] = agentic_core.AREA_INFO.get(listing["suburb"], {})
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
                return "Moving to Listings Agent. Ask me anything about the houses in Melbourne!"
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
                    return f"Selected listing #{idx}. Chat to me about PTV, Restaurants or Schools in the area!"
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
        cmd_reply = self._handle_commands(text)
        if cmd_reply is not None:
            self.chat.add_user_message(text)
            self.chat.add_assistant_message(cmd_reply)
            return {
                "reply": cmd_reply,
                "phase": self.sess.phase,
                "phaseClass": self._phase_color(),
                "facts": self._build_facts(),
            }

        self.sess.borrower, updates = parse_message_into_borrower(
            text, self.sess.borrower
        )
        if "preferred_type" in updates:
            self.sess.preferred_type = updates.get("preferred_type")

        self._maybe_compute_eligibility()

        self.chat.add_user_message(text)
        hist = self._history_str(self.chat)

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
    # One-time CE → agent data sync on first page load (simple guard flag; no Flask hooks needed)
    if not app.config.get("_DV_AGENT_SYNCED"):
        try:
            summary = refresh_dataverse_into_agent(top_props=50, top_support=100)
            app.logger.info("Seeded agent data from CE on first load: %s", summary)
        except Exception as e:
            app.logger.warning(
                "CE -> agent data seed failed (continuing with built-ins): %s", e
            )
        app.config["_DV_AGENT_SYNCED"] = True

    return render_template("index.html")


@app.post("/api/message")
def api_message():
    data = request.get_json(force=True)
    user_text = (data or {}).get("message", "").strip()
    if not user_text:
        return jsonify({"error": "empty"}), 400
    agent = get_webagent()
    payload = asyncio.run(agent.send_async(user_text))
    return jsonify(payload)


# --- helpers
def _num(x, field):
    if x is None or x == "":
        raise ValueError(f"{field} required")
    try:
        return float(x)
    except Exception:
        raise ValueError(f"{field} must be a number")


def _int(x, field):
    v = _num(x, field)
    return int(round(v))


@app.post("/api/session/seed")
def api_session_seed():
    """
    Accepts a full borrower profile from the form and stores it in the current session.
    Required JSON body:
      first_name, last_name, location_hint (suburb),
      target_price, gross_annual_income, monthly_debts, monthly_expenses, savings, dependents
    """
    data = request.get_json(force=True) or {}
    try:
        agent = get_webagent()
        b = agent.sess.borrower

        # strings
        b.first_name = (data.get("first_name") or "").strip()
        b.last_name = (data.get("last_name") or "").strip()
        b.location_hint = (data.get("location_hint") or "").strip()

        # numbers
        b.target_price = _num(data.get("target_price"), "target_price")
        b.gross_annual_income = _num(
            data.get("gross_annual_income"), "gross_annual_income"
        )
        b.monthly_debts = _num(data.get("monthly_debts"), "monthly_debts")
        b.monthly_expenses = _num(data.get("monthly_expenses"), "monthly_expenses")
        b.savings = _num(data.get("savings"), "savings")
        b.dependents = _int(data.get("dependents"), "dependents")

        # compute eligibility now that we have everything
        agent._maybe_compute_eligibility()

        return jsonify(
            {
                "ok": True,
                "phase": agent.sess.phase,
                "phaseClass": agent._phase_color(),
                "facts": agent._build_facts(),
            }
        )
    except Exception as e:
        app.logger.exception("seed session failed")
        return jsonify({"ok": False, "error": str(e)}), 400


@app.get("/api/debug/datasets")
def api_debug_datasets():
    import agentic_core as ac

    return jsonify(
        {
            "properties_count": len(ac.PROPERTIES),
            "sample_suburbs": sorted(
                {(r.get("suburb") or "").strip() for r in ac.PROPERTIES}
            )[:10],
            "area_info_keys": list(ac.AREA_INFO.keys())[:10],
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
