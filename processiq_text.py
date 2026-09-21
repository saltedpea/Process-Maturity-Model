"""Domain autocorrect and heuristic autofill for ProcessIQ chat.

These helpers run before any LLM call so the live app can still extract
evidence from messy text. They never invent scores.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import get_close_matches
from typing import Dict, List, Sequence, Tuple

DOMAIN_TERMS: Tuple[str, ...] = (
    "BPMN",
    "Celonis",
    "WAVE",
    "RACI",
    "PAM",
    "PIG",
    "KPI",
    "KPIs",
    "Signavio",
    "ARIS",
    "cockpit",
    "conformance",
    "subprocess",
    "ownership",
    "automation",
    "Action Flow",
    "Orchestration Flow",
    "Process Owner",
    "Quality Manager",
)

COMMON_TYPOS: Dict[str, str] = {
    "bmpn": "BPMN",
    "bpnm": "BPMN",
    "bpm": "BPMN",
    "celonus": "Celonis",
    "celonis": "Celonis",
    "celonis.": "Celonis",
    "celonis,": "Celonis",
    "selonis": "Celonis",
    "raic": "RACI",
    "raci": "RACI",
    "wavee": "WAVE",
    "waive": "WAVE",
    "kpi's": "KPIs",
    "kpis": "KPIs",
    "signavio": "Signavio",
    "signavo": "Signavio",
    "signaviao": "Signavio",
    "aris": "ARIS",
    "pam": "PAM",
    "pig": "PIG",
    "subprocesss": "subprocess",
    "sub-process": "subprocess",
    "processowner": "Process Owner",
    "actionflow": "Action Flow",
}

PROCESS_NAME_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"(?:assess|assessment of|classify|process(?: name)?(?: is|:)|for)\s+([A-Za-z0-9][A-Za-z0-9 &/\-]{2,60})", re.I),
    re.compile(r"^([A-Z][A-Za-z0-9 &/\-]{2,40})\s+(?:process|has|with)\b"),
)

FILL_RULES: Tuple[Tuple[str, Tuple[str, ...], str], ...] = (
    ("C2.1", ("bpmn", "signavio", "aris", "process model"), "Process model / BPMN evidence mentioned"),
    ("C2.2", ("naming convention", "naming standard", "quality manager", "pi team"), "Naming convention evidence mentioned"),
    ("C2.3", ("e2e owner", "end-to-end owner", "domain owner", "category owner", "responsible for e2e"), "Domain / E2E ownership mentioned"),
    ("C2.4", ("process owner", "named owner", "owner is", "owned by"), "Process Owner mentioned"),
    ("C2.5", ("subprocess owner", "sub-process owner"), "Subprocess Owner mentioned"),
    ("C3.1", ("raci",), "RACI mentioned"),
    ("C3.2", ("inputs and outputs", "input/output", "inputs/outputs"), "Inputs / outputs mentioned"),
    ("C3.3", ("systems mapped", "system mapping", "sap", "salesforce"), "Systems mentioned"),
    ("C3.4", ("org unit", "organizational unit", "department mapped"), "Org units mentioned"),
    ("C3.5", ("kpi defined", "kpis defined", "kpi facet", "defined kpis"), "KPI definitions mentioned"),
    ("C3.6", ("policy", "standard linked", "iso ", "sop"), "Standards / policies mentioned"),
    ("C3.7", ("risk and control", "risks and controls", "controls defined"), "Risks and controls mentioned"),
    ("C4.1", ("process cockpit", "celonis", "live kpi", "kpi connected"), "Celonis / cockpit mentioned"),
    ("C4.2", ("pam active", "pam linked", "conformance", "pig"), "PAM / conformance mentioned"),
    ("C4.3", ("deviation", "adherence", "friction"), "Adherence / deviations mentioned"),
    ("C5.1", ("wave", "ci initiative", "continuous improvement"), "WAVE / CI initiative mentioned"),
    ("C5.2", ("action flow", "orchestration flow", "automation opportunity"), "Automation opportunity mentioned"),
    ("C5.3", ("kpi improved", "business outcome", "measurable outcome"), "Measurable outcome mentioned"),
)

NEGATION = re.compile(r"\b(no|not|without|missing|none|don't|dont|haven't|hasnt|lacks?)\b.{0,40}%s", re.I)

TEMPLATES: Dict[str, str] = {
    "Order to Cash": (
        "Assess Order to Cash. There is a BPMN model in Signavio that passed quality checks. "
        "The naming convention was approved by the PI team. Jane is the domain and E2E owner, "
        "the Process Owner, and subprocess owners are populated. RACI, inputs and outputs, "
        "systems mapped, org units, and KPI definitions are filled. A policy is linked and "
        "risks and controls are defined. "
        "No Celonis Process Cockpit yet and nothing is in WAVE."
    ),
    "Hire to Retire": (
        "Assess Hire to Retire. BPMN exists in ARIS. Naming convention is followed. "
        "Process Owner is populated. Subprocess owners are missing. RACI is defined. "
        "No Celonis cockpit, PAM, WAVE, or automation opportunities."
    ),
    "Procure to Pay": (
        "Assess Procure to Pay. BPMN-compliant model exists. Process Owner is Alex. "
        "Systems are mapped to SAP. KPIs are defined. Celonis Process Cockpit is live "
        "with KPIs refreshing. Deviations are shown. No WAVE CI initiative yet."
    ),
}


@dataclass(frozen=True)
class Correction:
    original: str
    replacement: str


@dataclass
class ParsedDescription:
    corrected_text: str
    corrections: List[Correction]
    process_name: str
    fills: Dict[str, Tuple[bool, str]]


def _candidate_terms() -> List[str]:
    terms = list(DOMAIN_TERMS)
    terms.extend(COMMON_TYPOS.values())
    return sorted(set(terms), key=len, reverse=True)


def autocorrect(text: str) -> Tuple[str, List[Correction]]:
    if not text.strip():
        return text, []
    corrections: List[Correction] = []
    words = re.findall(r"[A-Za-z0-9'/\-]+|[^A-Za-z0-9'/\-]+", text)
    known = [term.lower() for term in _candidate_terms()]
    lookup = {term.lower(): term for term in _candidate_terms()}
    rebuilt: List[str] = []
    for token in words:
        raw = token.strip()
        key = raw.lower()
        if not re.fullmatch(r"[A-Za-z0-9'/\-]+", token):
            rebuilt.append(token)
            continue
        if key in COMMON_TYPOS:
            replacement = COMMON_TYPOS[key]
            if replacement != token:
                corrections.append(Correction(token, replacement))
            rebuilt.append(replacement)
            continue
        if key in lookup:
            rebuilt.append(lookup[key] if token.isupper() or token.istitle() else token)
            continue
        match = get_close_matches(key, known, n=1, cutoff=0.84)
        if match and abs(len(match[0]) - len(key)) <= 3:
            replacement = lookup[match[0]]
            if replacement.lower() != key:
                corrections.append(Correction(token, replacement))
                rebuilt.append(replacement)
                continue
        rebuilt.append(token)
    return "".join(rebuilt), corrections


def extract_process_name(text: str) -> str:
    for pattern in PROCESS_NAME_PATTERNS:
        found = pattern.search(text)
        if found:
            name = re.sub(r"\s+", " ", found.group(1)).strip(" .,:;")
            name = re.sub(r"\b(process|please|today)\b", "", name, flags=re.I).strip()
            if 2 < len(name) <= 60:
                return name
    return ""


def _negated(text: str, cue: str) -> bool:
    window = NEGATION.pattern % re.escape(cue)
    return re.search(window, text, re.I) is not None


def _has_cue(text: str, cue: str) -> bool:
    if len(cue) <= 4:
        return re.search(rf"\b{re.escape(cue)}\b", text, re.I) is not None
    return cue in text


def extract_fills(text: str) -> Dict[str, Tuple[bool, str]]:
    lowered = text.lower()
    fills: Dict[str, Tuple[bool, str]] = {}
    for code, cues, reason in FILL_RULES:
        if any(_negated(lowered, cue) for cue in cues):
            fills[code] = (False, f"Described as missing ({cues[0]})")
            continue
        for cue in cues:
            if _has_cue(lowered, cue):
                fills[code] = (True, reason)
                break
    return fills


def parse_description(text: str) -> ParsedDescription:
    corrected, corrections = autocorrect(text)
    fills = extract_fills(corrected)
    # Negative fills are stored with "Explicitly not present" — split met vs not later.
    return ParsedDescription(
        corrected_text=corrected,
        corrections=corrections,
        process_name=extract_process_name(corrected),
        fills=fills,
    )


def suggestion_chips() -> Sequence[str]:
    return tuple(TEMPLATES)
