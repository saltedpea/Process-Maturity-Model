"""Agentic ProcessIQ layer: evidence tools, classification, and notifications.

The LLM only gathers and interprets evidence. The official scoring engine in
processiq_core.py remains the only classifier.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional
from urllib import error, request

import requests

from processiq_core import (
    ALL_CODES,
    CRITERIA,
    CRITERION_INDEX,
    PASS_THRESHOLD,
    assess_process,
    catalog_for_prompt,
    level_label,
    next_actions,
    score_level,
    unmet_codes,
)


MIN_MET_CONFIDENCE = 0.7
MAX_AGENT_STEPS = 8

PROVIDERS: Dict[str, Dict[str, str]] = {
    "OpenAI": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
    },
    "Groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "openai/gpt-oss-20b",
    },
    "Hugging Face": {
        "base_url": "https://router.huggingface.co/v1",
        "model": "openai/gpt-oss-20b",
    },
    "OpenRouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "openai/gpt-4o-mini",
    },
    "Custom": {
        "base_url": "",
        "model": "",
    },
}

PROVIDER_MODELS: Dict[str, tuple[str, ...]] = {
    "OpenAI": ("gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"),
    "Groq": ("openai/gpt-oss-20b", "openai/gpt-oss-120b"),
    "Hugging Face": ("openai/gpt-oss-20b", "openai/gpt-oss-120b"),
    "OpenRouter": ("openai/gpt-4o-mini", "openai/gpt-oss-20b"),
}


@dataclass
class EvidenceRecord:
    code: str
    met: bool
    rationale: str
    confidence: float
    source: str = "agent"


@dataclass
class Notification:
    id: str
    title: str
    body: str
    process_name: str
    level: int
    label: str
    next_actions: List[str]
    created_at: str
    read: bool = False
    webhook_ok: Optional[bool] = None


@dataclass
class AgentTurn:
    assistant_text: str
    tool_trace: List[str]
    notification: Optional[Notification] = None
    assessment: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class AgentSession:
    process_name: str = ""
    evidence: Dict[str, EvidenceRecord] = field(default_factory=dict)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    notifications: List[Notification] = field(default_factory=list)
    last_assessment: Optional[Dict[str, Any]] = None
    last_notification_id: Optional[str] = None

    def results(self) -> Dict[str, bool]:
        return {code: self.evidence[code].met if code in self.evidence else False for code in ALL_CODES}

    def unread_count(self) -> int:
        return sum(1 for item in self.notifications if not item.read)


def welcome_message() -> str:
    return (
        "I'm the ProcessIQ assessment agent. Describe the process and the evidence "
        "you already have — BPMN, owners, RACI, systems, KPIs, Celonis, WAVE, automation. "
        "I autocorrect terms like BPMN, Celonis, RACI, and WAVE, autofill matching "
        "criteria, then run the 80% sequential scoring model. You will get a notification "
        "when classification finishes.\n\n"
        "Describe a process, or pick a template. What should we assess?"
    )


def system_prompt() -> str:
    return (
        "You are ProcessIQ's evidence-collection agent.\n"
        "Your job is to interview the user, record criterion evidence with tools, "
        "then classify using the official engine.\n\n"
        "Hard rules:\n"
        "- Never invent a maturity level or a percentage. Only classify_process() may score.\n"
        "- A criterion is met only with explicit current evidence. Vague, planned, or "
        "aspirational statements are not met.\n"
        "- If evidence is missing or unclear, record met=false or ask a follow-up.\n"
        "- Do not mark met=true with confidence below 0.7.\n"
        "- Ask at most 3 focused questions per turn. Start at Level 2, then 3, 4, 5.\n"
        "- After the user has described the process or asks to finish, call "
        "classify_process. Classification automatically creates a notification.\n"
        "- If the user names the process, call set_process_name.\n"
        "- Be concise. Cite criterion codes when you record them.\n\n"
        f"{catalog_for_prompt()}"
    )


TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "set_process_name",
            "description": "Store the business process name being assessed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "process_name": {"type": "string"},
                },
                "required": ["process_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "record_evidence",
            "description": (
                "Record one official criterion as met or not met, with rationale. "
                "Use only catalog codes such as C2.1. Default to not met when unsure."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "met": {"type": "boolean"},
                    "rationale": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["code", "met", "rationale", "confidence"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_status",
            "description": "Inspect recorded evidence, live scores, and remaining gaps.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "classify_process",
            "description": (
                "Run the official ProcessIQ scoring engine and create a completion "
                "notification. Call this when evidence gathering is complete enough "
                "or the user asks to classify."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def resolve_api_key(manual_key: str = "") -> str:
    if manual_key.strip():
        return manual_key.strip()
    for name in ("OPENAI_API_KEY", "GROQ_API_KEY", "HF_TOKEN", "HUGGINGFACE_API_KEY", "OPENROUTER_API_KEY"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def _status_payload(session: AgentSession) -> Dict[str, Any]:
    results = session.results()
    preview = assess_process(results)
    evidence_rows = []
    for code in ALL_CODES:
        level, criterion = CRITERION_INDEX[code]
        record = session.evidence.get(code)
        evidence_rows.append({
            "code": code,
            "level": level,
            "name": criterion.name,
            "status": "met" if record and record.met else "not_met" if record else "pending",
            "confidence": record.confidence if record else None,
            "rationale": record.rationale if record else "",
        })
    return {
        "process_name": session.process_name or "(unnamed)",
        "preview_level": preview["level"],
        "preview_label": preview["label"],
        "preview_scores": preview["scores"],
        "blocked_by": preview["blocked_by"],
        "recorded": len(session.evidence),
        "pending": [code for code in ALL_CODES if code not in session.evidence],
        "evidence": evidence_rows,
    }


def _record_evidence(session: AgentSession, args: Mapping[str, Any]) -> Dict[str, Any]:
    code = str(args.get("code", "")).strip().upper()
    if code not in CRITERION_INDEX:
        return {"ok": False, "error": f"Unknown criterion {code}. Use an official catalog code."}

    rationale = str(args.get("rationale", "")).strip()
    if not rationale:
        return {"ok": False, "error": "Rationale is required. Evidence was not recorded."}

    try:
        confidence = float(args.get("confidence", 0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    met = bool(args.get("met", False))
    override = ""
    if met and confidence < MIN_MET_CONFIDENCE:
        met = False
        override = (
            f"Confidence {confidence:.2f} is below {MIN_MET_CONFIDENCE:.2f}, "
            "so this was recorded as not met."
        )

    session.evidence[code] = EvidenceRecord(
        code=code,
        met=met,
        rationale=rationale,
        confidence=confidence,
        source="agent",
    )
    level, criterion = CRITERION_INDEX[code]
    payload = {
        "ok": True,
        "code": code,
        "met": met,
        "confidence": confidence,
        "level": level,
        "name": criterion.name,
        "rationale": rationale,
    }
    if override:
        payload["guardrail"] = override
    return payload


def _post_webhook(webhook_url: str, notification: Notification) -> Optional[bool]:
    if not webhook_url.strip():
        return None
    body = json.dumps({
        "event": "processiq.assessment.completed",
        "notification": asdict(notification),
    }).encode("utf-8")
    req = request.Request(
        webhook_url.strip(),
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=12) as response:
            return 200 <= response.status < 300
    except error.URLError:
        return False


def seed_manual_evidence(session: AgentSession, results: Mapping[str, bool]) -> None:
    for code, met in results.items():
        if code not in CRITERION_INDEX:
            continue
        session.evidence[code] = EvidenceRecord(
            code=code,
            met=bool(met),
            rationale="Recorded from the manual checklist",
            confidence=1.0,
            source="manual",
        )


def apply_autofill(session: AgentSession, fills: Mapping[str, tuple[bool, str]], source: str = "autofill") -> List[str]:
    applied: List[str] = []
    for code, payload in fills.items():
        if code not in CRITERION_INDEX:
            continue
        met, rationale = payload
        existing = session.evidence.get(code)
        if existing and existing.source in {"agent", "manual"} and existing.confidence >= 0.7:
            continue
        session.evidence[code] = EvidenceRecord(
            code=code,
            met=bool(met),
            rationale=rationale,
            confidence=0.82 if met else 0.9,
            source=source,
        )
        applied.append(f"{code}={'met' if met else 'not met'}")
    return applied


def create_notification(
    session: AgentSession,
    assessment: Mapping[str, Any],
    webhook_url: str = "",
    results: Mapping[str, bool] | None = None,
) -> Notification:
    process_name = session.process_name or "Unnamed process"
    level = int(assessment["level"])
    label = str(assessment["label"])
    scores = assessment["scores"]
    actions = next_actions(results if results is not None else session.results(), level)
    score_line = ", ".join(
        f"L{lvl} {score:.0f}%" for lvl, score in scores.items()
    )
    action_text = "; ".join(actions) if actions else "No immediate gaps in the next gated level."
    notification = Notification(
        id=str(uuid.uuid4()),
        title=f"ProcessIQ: {process_name} classified as Level {level} — {label}",
        body=(
            f"{process_name} is Level {level} ({label}). "
            f"Scores: {score_line}. Pass mark is {PASS_THRESHOLD:.0f}% with sequential gates. "
            f"Next actions: {action_text}"
        ),
        process_name=process_name,
        level=level,
        label=label,
        next_actions=actions,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    notification.webhook_ok = _post_webhook(webhook_url, notification)
    session.notifications.insert(0, notification)
    session.last_notification_id = notification.id
    return notification


def classify_and_notify(
    session: AgentSession,
    webhook_url: str = "",
    results: Mapping[str, bool] | None = None,
) -> Dict[str, Any]:
    if not session.process_name.strip():
        session.process_name = "Unnamed process"
    used = results if results is not None else session.results()
    assessment = assess_process(used)
    session.last_assessment = assessment
    notification = create_notification(session, assessment, webhook_url, results=used)
    return {
        "ok": True,
        "process_name": session.process_name,
        "level": assessment["level"],
        "label": assessment["label"],
        "scores": assessment["scores"],
        "blocked_by": assessment["blocked_by"],
        "unmet": unmet_codes(session.results(), int(assessment["level"])),
        "next_actions": notification.next_actions,
        "notification_id": notification.id,
        "webhook_delivered": notification.webhook_ok,
    }


def dispatch_tool(
    session: AgentSession,
    name: str,
    raw_args: str,
    webhook_url: str,
) -> Dict[str, Any]:
    try:
        args = json.loads(raw_args or "{}")
    except json.JSONDecodeError:
        args = {}
    if name == "set_process_name":
        session.process_name = str(args.get("process_name", "")).strip()
        return {"ok": True, "process_name": session.process_name}
    if name == "record_evidence":
        return _record_evidence(session, args)
    if name == "get_status":
        return _status_payload(session)
    if name == "classify_process":
        return classify_and_notify(session, webhook_url)
    return {"ok": False, "error": f"Unknown tool {name}"}


def explain_llm_error(status: int, body: str, base_url: str) -> str:
    text = re.sub(r"<[^>]+>", " ", body or "")
    text = " ".join(text.split())[:300]
    lowered = f"{text} {base_url}".lower()
    if status == 403 and ("1010" in text or "cloudflare" in lowered or "access denied" in lowered):
        return (
            "The model provider blocked this request (Cloudflare 1010). "
            "That usually happens on a work network. Try: "
            "1) switch Provider to OpenAI or OpenRouter, "
            "2) use a personal hotspot, or "
            "3) confirm the API key is a live Groq/OpenAI key, not a password."
        )
    if status in {401, 403}:
        return (
            "The API key was rejected. Create a new key at the provider site, "
            "paste it into the API key box, and make sure the Provider dropdown matches that key."
        )
    if status == 429:
        return "The provider rate-limited this key. Wait a minute and try again, or switch provider."
    if status == 404:
        if "groq.com" in base_url:
            return (
                "Groq no longer serves that model on a free key. "
                "Set Model to openai/gpt-oss-20b or openai/gpt-oss-120b, then send the message again. "
                "Llama 3.3 70B is enterprise-only now."
            )
        return (
            f"The model name is not available on this provider. "
            f"Pick a model from the dropdown. Endpoint: {base_url}"
        )
    return f"LLM request failed ({status}): {text or 'no details from the provider'}"


def _chat_request(
    messages: List[Dict[str, Any]],
    api_key: str,
    base_url: str,
    model: str,
) -> Dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "tools": TOOLS,
        "tool_choice": "auto",
        "temperature": 0.2,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "ProcessIQ/1.0 (assessment-agent)",
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "ProcessIQ",
    }
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60)
    except requests.RequestException as exc:
        raise RuntimeError(f"Could not reach the LLM endpoint: {exc}") from exc
    if response.status_code >= 400:
        raise RuntimeError(explain_llm_error(response.status_code, response.text, base_url))
    return response.json()


def run_agent_turn(
    session: AgentSession,
    user_text: str,
    api_key: str,
    base_url: str,
    model: str,
    webhook_url: str = "",
) -> AgentTurn:
    if not session.messages:
        session.messages.append({"role": "system", "content": system_prompt()})
    session.messages.append({"role": "user", "content": user_text})

    trace: List[str] = []
    created_notification: Optional[Notification] = None
    latest_assessment = session.last_assessment

    for _ in range(MAX_AGENT_STEPS):
        data = _chat_request(session.messages, api_key, base_url, model)
        choice = data["choices"][0]["message"]
        session.messages.append(choice)
        tool_calls = choice.get("tool_calls") or []
        if not tool_calls:
            return AgentTurn(
                assistant_text=choice.get("content") or "I need a bit more evidence to continue.",
                tool_trace=trace,
                notification=created_notification,
                assessment=latest_assessment,
            )

        for call in tool_calls:
            function = call["function"]
            name = function["name"]
            result = dispatch_tool(session, name, function.get("arguments", "{}"), webhook_url)
            if name == "classify_process" and result.get("ok"):
                latest_assessment = session.last_assessment
                if session.notifications:
                    created_notification = session.notifications[0]
            summary = result.get("guardrail") or result.get("label") or result.get("process_name") or result.get("code")
            trace.append(f"{name}: {summary}")
            session.messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "content": json.dumps(result),
            })

    return AgentTurn(
        assistant_text="I recorded the latest evidence. Ask me to classify when you want the official score.",
        tool_trace=trace,
        notification=created_notification,
        assessment=latest_assessment,
    )


def evidence_table(session: AgentSession) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    results = session.results()
    for level, criteria in CRITERIA.items():
        current = score_level(criteria, results)
        for criterion in criteria:
            record = session.evidence.get(criterion.code)
            rows.append({
                "Level": f"{level} — {level_label(level)}",
                "Code": criterion.code,
                "Criterion": criterion.name,
                "Status": "Met" if record and record.met else "Not met" if record else "Pending",
                "Confidence": f"{record.confidence:.0%}" if record else "—",
                "Rationale": record.rationale if record else "",
                "Level score": f"{current:.0f}%",
            })
    return rows
