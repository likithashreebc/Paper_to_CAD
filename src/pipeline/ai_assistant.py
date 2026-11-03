# -*- coding: utf-8 -*-
"""
Paper2CAD Grounded AI Circuit Assistant (Phase 6).

Provides circuit analysis, domain explanations, and interactive Q&A
grounded strictly in:
1. Reconstructed CircuitGraph data (components, nets, coordinates, values)
2. Rule-based Validation findings (errors, warnings, suggested actions)
3. SPICE simulation outputs (when available)

Design Principles:
- Never hallucinate invisible components or phantom connections.
- Configurable LLM backend (OpenAI, Anthropic, Ollama, LM Studio via OpenAI-compatible API).
- High-quality offline grounded fallback mode when no external LLM API key is present.
- Every suggested correction is surfaced for explicit user confirmation.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from circuit_data_model import CircuitGraph, ComponentModel, NodeModel
from circuit_validator import ValidationFinding


try:
    import requests
except ImportError:
    requests = None


_QUANTITY_RE = re.compile(
    r"(?P<number>[+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*"
    r"(?P<suffix>meg|[TGMkmunpfµμ]?)(?P<unit>ohms?|Ω|volts?|V|amps?|A|farads?|F|henries|H|seconds?|s|watts?|W)?",
    re.IGNORECASE,
)
_API_KEY_VALUE_RE = re.compile(r"(?:sk-[A-Za-z0-9_-]{20,}|gsk_[A-Za-z0-9_-]{20,})", re.IGNORECASE)
_ENGINEERING_PREFIXES = {
    "": 1.0,
    "T": 1e12,
    "G": 1e9,
    "M": 1e6,
    "k": 1e3,
    "K": 1e3,
    "m": 1e-3,
    "u": 1e-6,
    "μ": 1e-6,
    "µ": 1e-6,
    "n": 1e-9,
    "p": 1e-12,
    "f": 1e-15,
    "meg": 1e6,
}


def _quantity_value(text: str) -> Optional[float]:
    match = _QUANTITY_RE.fullmatch(text.strip())
    if not match:
        return None
    prefix = match.group("suffix")
    if prefix.lower() == "meg":
        multiplier = 1e6
    else:
        multiplier = _ENGINEERING_PREFIXES.get(prefix, _ENGINEERING_PREFIXES.get(prefix.lower()))
    if multiplier is None:
        return None
    return float(match.group("number")) * multiplier


def _question_quantity(question: str, aliases: str) -> Optional[float]:
    match = re.search(
        rf"(?:{aliases})\s*(?:across\b[^,;]+?\s*)?(?:=|:|is|of)?\s*"
        r"([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?\s*(?:meg|[TGMkmunpfµμ]?)"
        r"(?:ohms?|Ω|volts?|V|amps?|A|farads?|F|henries|H|seconds?|s|watts?|W)?)",
        question,
        flags=re.IGNORECASE,
    )
    return _quantity_value(match.group(1)) if match else None


def _format_calculated_value(value: float, unit: str) -> str:
    if value == 0:
        return f"0 {unit}"
    prefixes = ((1e9, "G"), (1e6, "M"), (1e3, "k"), (1, ""), (1e-3, "m"), (1e-6, "u"), (1e-9, "n"), (1e-12, "p"))
    for scale, prefix in prefixes:
        scaled = abs(value) / scale
        if 1 <= scaled < 1000:
            return f"{scaled:.4g} {prefix}{unit}".strip()
    return f"{value:.4g} {unit}"


SYSTEM_PROMPT = """You are Paper2CAD Assistant, a knowledgeable, friendly, and articulate Electrical Engineering AI assistant that communicates in a natural, clear ChatGPT style.
You analyze hand-drawn circuit diagrams converted into digital schematics and SPICE netlists by Paper2CAD Studio.

CORE COMMUNICATION RULES (STRICT):
1. CHATGPT-STYLE TONE & FLOW:
   - Speak conversationally, warmly, and clearly in natural English.
   - Jump straight to answering the user's question. Do NOT include robotic preamble (e.g. avoid "Based on the verified circuit data provided, here is a detailed analysis...").
   - NEVER output internal monologue, train-of-thought, or self-correction (never write "Wait, let's re-trace carefully..." or "Let's check KCL..."). Provide only the clean final answer.

2. CONCISENESS VS. DETAIL (ADAPTIVE):
   - BY DEFAULT: Explain ONLY what is necessary to answer the question clearly and directly. Keep answers concise (2 to 4 crisp paragraphs or clean bullet points). Highlight what the circuit is, key components, and main behavior. Do NOT write an overwhelming multi-page textbook report.
   - DETAILED EXPLANATION: ONLY if the user explicitly asks for "detail", "detailed", "step by step", "deep dive", "in-depth", "full calculation", or "derivation", provide a comprehensive step-by-step breakdown.
   - Always finish every sentence, thought, and section cleanly. NEVER cut off mid-thought or mid-sentence.

3. SPECIAL CHARACTERS & CLEAN FORMATTING (VERY IMPORTANT):
   - NO LaTeX Math Dollar Signs: NEVER use '$' or '$$'. Write math as natural plain text.
     * Correct: V5 = I * R = 2A * 10 ohms = 20V
     * Incorrect: $V_5 = I \\times R = 2A \\times 10\\Omega = 20V$
     * Correct: in DC steady state (as time -> infinity)
     * Incorrect: ($t \\to \\infty$)
     * Use standard engineering units: ohms, kΩ, uF, mH, A, mA, V.
   - NO Raw Hash Headers: Do NOT output '####' or '#' tags. Use clean bold subheaders (e.g., **Branch A (High Impedance):**) or clean bullet points.
   - NO Raw '#' Before Component IDs: Write natural component names like "the 10k resistor (R3)", "the 2A current source", or "Node 1 (Ground)" instead of "Comp #3" or "Node #1".
"""


def sanitize_chat_text(text: str) -> str:
    """
    Cleans raw LaTeX math delimiters, stray markdown hashes, and internal monologue artifacts
    to ensure the response looks like natural, polished ChatGPT output.
    """
    if not text:
        return ""

    # 1. Clean robotic preamble if model emitted one
    text = re.sub(
        r"^(?:Based on (?:the )?(?:supplied|verified|provided) [^\n]+\n+)+",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # 2. Clean self-correction / internal monologue leaks
    text = re.sub(
        r"(?i)\bwait,\s*let['’]?s\s*(?:re-?trace|re-?check|think|look)[^\n]*\n?",
        "",
        text,
    )

    # 3. Clean LaTeX math symbols
    text = re.sub(r"\\times", " * ", text)
    text = re.sub(r"\\cdot", " * ", text)
    text = re.sub(r"\\Omega\b", " ohms", text)
    text = text.replace("Ω", " ohms")
    text = re.sub(r"\\mu\b", "u", text)
    text = re.sub(r"\\to\b", " -> ", text)
    text = re.sub(r"\\infty\b", "infinity", text)
    text = re.sub(r"\\approx\b", "≈", text)
    text = re.sub(r"\\le\b", "<=", text)
    text = re.sub(r"\\ge\b", ">=", text)
    text = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"(\1 / \2)", text)

    # 4. Remove math dollar signs ($...$ and $$...$$)
    text = re.sub(r"\$\$(.*?)\$\$", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"\$([^\$\n]+)\$", r"\1", text)

    # 5. Clean #### and ### headers into bold text or clean headers
    text = re.sub(r"(?m)^####\s*(.*)$", r"**\1**", text)

    # 6. Clean raw # before component IDs
    text = re.sub(r"(?i)\bcomp(?:onent)?\s*#(\d+)\b", r"Component \1", text)
    text = re.sub(r"(?i)\bnode\s*#(\d+)\b", r"Node \1", text)
    text = re.sub(r"(?i)\b(resistor|inductor|capacitor|source|diode)\s*#(\d+)\b", r"\1 \2", text)

    # 7. Remove any trailing incomplete bullet/numbered header if truncated
    text = re.sub(r"\n\s*\d+\.\s*[^.\n:]*:\s*$", "", text)
    text = re.sub(r"\n\s*[-*]\s*[^.\n:]*:\s*$", "", text)

    # 8. If text cuts off mid-sentence without terminal punctuation, trim back to last complete sentence
    clean_t = text.strip()
    if clean_t and not clean_t.endswith((".", "!", "?", "```", ")", "*", '"')):
        last_punct = max(clean_t.rfind(". "), clean_t.rfind(".\n"), clean_t.rfind("!\n"), clean_t.rfind("?\n"))
        if last_punct > len(clean_t) * 0.55:
            text = clean_t[:last_punct + 1].strip()

    return text.strip()


class AICircuitAssistant:
    """
    AI Circuit Assistant providing grounded explanations and answering user questions
    powered by Groq (default model: qwen/qwen3.8-27b) or local grounded engine.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None
    ):
        # Dynamically reload .env if available
        try:
            from dotenv import load_dotenv
            load_dotenv(override=True)
        except Exception:
            pass

        configured_url = base_url or os.environ.get("LLM_BASE_URL")
        groq_url = os.environ.get("GROQ_BASE_URL")
        groq_key = os.environ.get("GROQ_API_KEY", "").strip()
        openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
        generic_key = os.environ.get("LLM_API_KEY", "").strip()

        if configured_url:
            self.base_url = configured_url
        elif groq_url and groq_key:
            self.base_url = groq_url
        elif groq_key and not (api_key or openai_key or generic_key):
            self.base_url = "https://api.groq.com/openai/v1"
        elif api_key or openai_key or generic_key:
            self.base_url = "https://api.openai.com/v1"
        else:
            self.base_url = None

        normalized_url = (self.base_url or "").lower()
        if "api.groq.com" in normalized_url or (groq_url and self.base_url == groq_url):
            self.provider = "Groq"
            selected_key = api_key or groq_key or generic_key
            self.model = model or os.environ.get("GROQ_MODEL") or os.environ.get("LLM_MODEL") or "qwen/qwen3.8-27b"
        elif "api.openai.com" in normalized_url:
            self.provider = "OpenAI"
            selected_key = api_key or openai_key or generic_key
            self.model = model or os.environ.get("OPENAI_MODEL") or os.environ.get("LLM_MODEL") or "gpt-4o-mini"
        elif self.base_url:
            self.provider = "OpenAI-compatible LLM"
            selected_key = api_key or generic_key
            self.model = model or os.environ.get("LLM_MODEL")
        else:
            self.provider = "Deterministic Grounded Engine"
            selected_key = None
            self.model = None

        self.api_key = selected_key or None

        if self.model and _API_KEY_VALUE_RE.fullmatch(self.model.strip()):
            self.model = (
                "gpt-4o-mini"
                if self.provider == "OpenAI"
                else "qwen/qwen3.8-27b" if self.provider == "Groq" else None
            )

    def explain_circuit(
        self,
        graph: CircuitGraph,
        findings: Optional[List[ValidationFinding]] = None,
        sim_result: Optional[Dict[str, Any]] = None,
        user_question: Optional[str] = None,
        spice_netlist: Optional[str] = None,
        extra_context: Optional[Dict[str, Any]] = None,
        history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Generates an insightful explanation or answers user questions.
        Uses Groq LLM if configured in .env; otherwise falls back to deterministic grounded analyzer.
        """
        calculation = self._calculate_component_value(graph, user_question or "")
        if calculation:
            return calculation

        # If external Groq / LLM key is configured, query LLM
        local_endpoint_without_key = (
            self.provider == "OpenAI-compatible LLM"
            and self.base_url is not None
            and not any(host in self.base_url.lower() for host in ("api.openai.com", "api.groq.com"))
        )
        if self.base_url and self.model and (self.api_key or local_endpoint_without_key) and self.api_key != "none":
            try:
                return self._call_llm(
                    graph,
                    findings or [],
                    sim_result,
                    user_question,
                    spice_netlist=spice_netlist,
                    extra_context=extra_context,
                    history=history
                )
            except Exception as e:
                fallback = self._generate_grounded_response(graph, findings or [], sim_result, user_question)
                return f"{fallback}\n\n> ⚠️ **LLM Notice**: {self.provider} request failed (`{e}`). Displaying the grounded offline analysis above. Check your provider API key and endpoint configuration."

        return self._generate_grounded_response(graph, findings or [], sim_result, user_question)

    def _call_llm(
        self,
        graph: CircuitGraph,
        findings: List[ValidationFinding],
        sim_result: Optional[Dict[str, Any]],
        question: Optional[str],
        spice_netlist: Optional[str] = None,
        extra_context: Optional[Dict[str, Any]] = None,
        history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Sends grounded concise representation to Groq OpenAI-compatible chat completion endpoint."""
        # Build intuitive node map so LLM sees intuitive names (e.g. Ground (0V) for ground)
        node_name_map = {}
        for n in graph.nodes:
            if n.is_ground or (n.name == "0" and n.is_ground):
                node_name_map[n.id] = "Ground (0V)"
            else:
                node_name_map[n.id] = f"Node_{n.id}"

        comp_lines = []
        for c in graph.components:
            conns = []
            for t in c.terminals:
                lbl = node_name_map.get(t.connected_node_id, f"Node_{t.connected_node_id}") if t.connected_node_id is not None else "Open"
                conns.append(f"{t.name} -> {lbl}")
            comp_lines.append(f"- {c.display_type} (id {c.id}, {c.value or c.label or 'unlabeled'}): {', '.join(conns)}")

        grounding_prefix = (
            f"Verified Circuit Topology ({graph.image_name}):\n"
            + "\n".join(comp_lines)
            + (
                "\nGround Reference: Ground (0V) is present.\n"
                if any(node.is_ground or node.name == "0" for node in graph.nodes)
                else "\nGround Reference: No ground reference was detected.\n"
            )
            + (f"\nSPICE Netlist:\n```spice\n{spice_netlist.strip()}\n```\n" if spice_netlist else "")
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + grounding_prefix}
        ]

        # Append previous conversation history if provided
        if history and isinstance(history, list):
            for h in history[-8:]:
                if isinstance(h, dict) and "role" in h and "content" in h:
                    messages.append({"role": h["role"], "content": h["content"]})

        # Determine if user is requesting an in-depth/detailed breakdown or a concise answer
        is_detail_requested = False
        if question and question.strip():
            q_clean = question.strip()
            detail_keywords = [
                "detail", "detailed", "step by step", "step-by-step",
                "in-depth", "deep dive", "thorough", "derive", "derivation",
                "calculate all", "full calculation", "calculate currents", "calculate voltages"
            ]
            if any(k in q_clean.lower() for k in detail_keywords):
                is_detail_requested = True

            if is_detail_requested:
                user_msg = (
                    f"{q_clean}\n\n"
                    f"(Instructions: Provide a detailed, step-by-step electrical engineering analysis. "
                    f"Write naturally like ChatGPT without any LaTeX dollar signs ($), without raw '#' symbols. "
                    f"Keep calculations clear and structured, and ensure every calculation and thought reaches a clean, complete conclusion before ending.)"
                )
            else:
                user_msg = (
                    f"{q_clean}\n\n"
                    f"(Instructions: Provide a concise, clear, natural response in ChatGPT style. "
                    f"Explain only what is directly necessary to answer the question. "
                    f"Do NOT write a massive essay. Keep it to 2-4 clean paragraphs or bullet points. "
                    f"Do not use LaTeX dollar signs ($) or raw '#' symbols. Finish your thoughts completely.)"
                )
        else:
            user_msg = (
                "Please give a concise, high-level overview of this circuit in ChatGPT style: "
                "its circuit type, main components, and how current flows. "
                "Keep it brief (2-3 short paragraphs), conversational, and natural. "
                "Do not use raw LaTeX dollar signs ($) or '#' symbols. Complete all sentences."
            )

        messages.append({"role": "user", "content": user_msg})

        # Token budget: 450 tokens for concise (safely fits Groq TPM and never cuts off), 780 for detailed
        token_limit = 780 if is_detail_requested else 450

        candidate_models = [self.model] if self.model else []
        if self.provider == "Groq" and self.api_key:
            for fallback_model in ("qwen/qwen3.8-27b", "openai/gpt-oss-20b", "openai/gpt-oss-120b"):
                if fallback_model not in candidate_models:
                    candidate_models.append(fallback_model)

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Paper2CAD-AI/1.0 (Windows; Electrical Studio)"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last_error = None
        for model_to_try in candidate_models:
            payload = {
                "model": model_to_try,
                "messages": messages,
                "temperature": 0.25,
                "max_tokens": token_limit
            }

            try:
                if requests is not None:
                    resp = requests.post(
                        f"{self.base_url.rstrip('/')}/chat/completions",
                        json=payload,
                        headers=headers,
                        timeout=25
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        msg_obj = data["choices"][0]["message"]
                        content = msg_obj.get("content", "")
                        if not content and "reasoning" in msg_obj:
                            content = msg_obj["reasoning"]
                        self.model = model_to_try
                        return sanitize_chat_text(content)
                    elif resp.status_code in (400, 404, 429):
                        last_error = f"{self.provider} API Error ({resp.status_code}): {resp.text}"
                        continue
                    else:
                        resp.raise_for_status()
                else:
                    req = urllib.request.Request(
                        f"{self.base_url.rstrip('/')}/chat/completions",
                        data=json.dumps(payload).encode("utf-8"),
                        headers=headers,
                        method="POST"
                    )
                    with urllib.request.urlopen(req, timeout=25) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                        msg_obj = data["choices"][0]["message"]
                        content = msg_obj.get("content", "")
                        if not content and "reasoning" in msg_obj:
                            content = msg_obj["reasoning"]
                        self.model = model_to_try
                        return sanitize_chat_text(content)
            except urllib.error.HTTPError as e:
                if e.code in (400, 404, 429):
                    last_error = f"{self.provider} HTTP {e.code}"
                    continue
                err_body = ""
                try:
                    err_body = e.read().decode("utf-8")
                except Exception:
                    pass
                raise RuntimeError(f"{self.provider} API Error ({e.code}): {err_body or str(e)}")
            except Exception as e:
                last_error = str(e)
                if any(k in last_error.lower() for k in ("404", "400", "429", "rate", "limit", "too many", "model")):
                    continue
                raise RuntimeError(f"Connection error to {self.provider} ({self.base_url}): {e}")

        raise RuntimeError(
            f"{self.provider} request failed for configured model '{self.model}'. "
            f"Verify LLM_MODEL is available for this provider. Last error: {last_error}"
        )

    def _generate_grounded_response(
        self,
        graph: CircuitGraph,
        findings: List[ValidationFinding],
        sim_result: Optional[Dict[str, Any]],
        question: Optional[str]
    ) -> str:
        """
        Generates a rigorous, deterministic grounded analysis of the circuit graph.
        Never hallucinates; derives strictly from topological nodes and components.
        """
        calculation = self._calculate_component_value(graph, question or "")
        if calculation:
            return calculation

        # If question is about a specific component (e.g. "What is connected to component 2?" or "resistor")
        if question:
            q_lower = question.lower()
            
            # Check for specific component ID query
            comp_match = re.search(r"(?:component|comp|#|r|c|l|v|d)\s*(\d+)", q_lower)
            if comp_match:
                cid = int(comp_match.group(1))
                comp = graph.get_component(cid)
                if comp:
                    connected_nodes = [t.connected_node_id for t in comp.terminals if t.connected_node_id is not None]
                    adjacent_comps = []
                    for nid in connected_nodes:
                        for other in graph.get_connected_components(nid):
                            if other.id != comp.id and other not in adjacent_comps:
                                adjacent_comps.append(other)

                    adj_str = ", ".join(f"**{c.display_type} {c.id}** ({c.value or c.label or 'unlabeled'})" for c in adjacent_comps) or "No adjacent components (floating net)"
                    out = (
                        f"### Component {comp.id}: {comp.display_type.capitalize()} ({comp.value or comp.label or 'No value'})\n\n"
                        f"- **Type:** `{comp.display_type}` (confidence: {comp.confidence:.2f})\n"
                        f"- **Bounding Box:** `[{', '.join(f'{b:.1f}' for b in comp.bbox)}]`\n"
                        f"- **Connected Nodes:** {', '.join(f'Node {nid}' for nid in connected_nodes) if connected_nodes else 'None (Open/Floating)'}\n"
                        f"- **Neighboring Components:** {adj_str}\n\n"
                        f"**Terminals:**\n" +
                        "\n".join(f"- Pin `{t.name}`: Attached to Node {t.connected_node_id} at sketch pixel `({t.x:.1f}, {t.y:.1f})`" for t in comp.terminals)
                    )
                    return sanitize_chat_text(out)

            if any(word in q_lower for word in (
                "warning", "error", "rule", "issue", "problem", "wrong",
                "short", "ground", "validat", "connection",
            )):
                if not findings:
                    return "No validation errors or warnings are currently flagged on this circuit graph."
                res = ["### Active Validation Findings:\n"]
                for idx, f in enumerate(findings, 1):
                    res.append(f"**{idx}. [{f.severity.upper()}] {f.title}** ({f.rule_id})")
                    res.append(f"- **Explanation:** {f.explanation}")
                    res.append(f"- **Evidence:** {f.evidence}")
                    res.append(f"- **Suggested Action:** {f.suggested_action}\n")
                return sanitize_chat_text("\n".join(res))

            if any(word in q_lower for word in ("component", "inventory", "list", "detected")):
                return self._format_component_inventory(graph)

            if not any(word in q_lower for word in (
                "overview", "circuit do", "circuit work", "circuit function",
                "topology", "connected", "node", "net", "current flow", "behavior",
                "behaviour",
            )):
                return (
                    "I’m in offline grounded mode, so I can answer from detected circuit data but can’t handle open-ended questions. "
                    "Try asking about a component, its connections, the detected inventory, validation warnings, or a value calculation. "
                    "For general conversational circuit analysis, configure an LLM provider in the local `.env` file."
                )

        # An overview can describe the detected graph, but cannot infer intended function
        # when component values or connectivity are missing or flagged as uncertain.
        active_comps = [c for c in graph.components if c.type.lower() not in ("text", "junction")]
        families = {
            "sources": [c for c in active_comps if any(k in c.display_type.lower() for k in ("voltage", "current", "battery"))],
            "passives": [c for c in active_comps if any(k in c.display_type.lower() for k in ("resistor", "capacitor", "inductor"))],
            "semiconductors": [c for c in active_comps if any(k in c.display_type.lower() for k in ("diode", "transistor", "amplifier", "opamp", "led"))],
        }

        res = [f"## Circuit Overview: `{graph.image_name}`\n"]
        res.append(f"The hand-drawn diagram contains **{len(active_comps)} components** interconnected across **{len(graph.nodes)} electrical nodes**.\n")

        res.append("### Component Inventory:")
        for title, key in (
            ("Power Sources", "sources"),
            ("Passives (R/L/C)", "passives"),
            ("Active / Semiconductor", "semiconductors"),
        ):
            components = families[key]
            if components:
                details = ", ".join(
                    f"{c.display_type} {c.label or c.id} ({c.value or 'value unknown'})"
                    for c in components
                )
                res.append(f"- **{title}:** {details}")
        if not any(families.values()):
            res.append("- (No recognized electronic primitives)")

        res.append("\n### Electrical Network Topology:")
        has_ground = any(n.is_ground for n in graph.nodes)
        res.append(f"- **Reference Ground:** {'Detected (Node 0)' if has_ground else '⚠️ Not detected (floating)'}")
        res.append(f"- **Total Nodes:** {len(graph.nodes)} distinct equivalence nets.")

        if findings:
            res.append("\n### Validation Issues:")
            errors = [f for f in findings if f.severity == "error"]
            warnings = [f for f in findings if f.severity == "warning"]
            infos = [f for f in findings if f.severity == "info"]
            res.append(f"- **Errors:** {len(errors)}, **Warnings:** {len(warnings)}, **Notes:** {len(infos)}")
            for f in findings:
                res.append(f"- **{f.severity.upper()}: {f.title}.** {f.suggested_action}")
            res.append(
                "\nThese findings make the inferred circuit behavior uncertain. "
                "Resolve the flagged connections and ground reference before relying on a functional explanation."
            )
        elif active_comps:
            res.append("\nNo validation findings were supplied for this circuit.")

        if question:
            res.append(
                "\nThis is a topology-based summary, not a full conversational analysis. "
                "Component values or intended function that are not present in the detected data cannot be inferred."
            )

        if sim_result and sim_result.get("success"):
            res.append("\n### SPICE Simulation (DC Operating Point):")
            voltages = sim_result.get("node_voltages", {})
            if voltages:
                res.append("- **Node Voltages:** " + ", ".join(f"`V({k}) = {v:.2f}V`" for k, v in list(voltages.items())[:6]))

        return sanitize_chat_text("\n".join(res))

    @staticmethod
    def _format_component_inventory(graph: CircuitGraph) -> str:
        components = [c for c in graph.components if c.type.lower() not in ("text", "junction")]
        if not components:
            return "No electronic components were detected in this circuit."
        lines = ["Detected components:"]
        for component in components:
            name = component.label or f"Component {component.id}"
            value = component.value or "value not detected"
            lines.append(f"- {name}: {component.display_type}, {value}.")
        return "\n".join(lines)

    def _calculate_component_value(self, graph: CircuitGraph, question: str) -> Optional[str]:
        """Calculate a passive value only when the user supplies enough numeric inputs."""
        if not re.search(r"\b(?:calculate|compute|find|determine)\b", question, re.IGNORECASE):
            return None
        if not re.search(r"\b(?:value|resistance|capacitance|inductance|[RCL]\s*\d+)\b", question, re.IGNORECASE):
            return None

        component = None
        id_match = re.search(r"\b(?:component|comp)\s*#?\s*(\d+)\b", question, re.IGNORECASE)
        designator_match = re.search(r"\b([RCL])\s*(\d+)\b", question, re.IGNORECASE)
        if id_match:
            component = graph.get_component(int(id_match.group(1)))
        elif designator_match:
            prefix, index = designator_match.group(1).upper(), int(designator_match.group(2))
            matching = [
                item for item in graph.components
                if (("resistor" in item.display_type.lower() and prefix == "R")
                    or ("capacitor" in item.display_type.lower() and prefix == "C")
                    or ("inductor" in item.display_type.lower() and prefix == "L"))
            ]
            component = next(
                (item for item in matching if (item.label or "").strip().upper() == f"{prefix}{index}"),
                matching[index - 1] if 0 < index <= len(matching) else None,
            )
        else:
            kind = next(
                (kind for kind in ("resistor", "capacitor", "inductor") if kind in question.lower()),
                None,
            )
            matching = [item for item in graph.components if kind and kind in item.display_type.lower()]
            if len(matching) == 1:
                component = matching[0]

        if component is None:
            return None

        component_type = component.display_type.lower()
        voltage = _question_quantity(question, r"\b(?:voltage|volts?|V)\b")
        current = _question_quantity(question, r"\b(?:current|amps?|A|I)\b")
        power = _question_quantity(question, r"\b(?:power|watts?|W|P)\b")
        resistance = _question_quantity(question, r"\b(?:resistance|ohms?|Ω|R)\b")
        time_constant = _question_quantity(question, r"\b(?:time\s*constant|tau|τ)\b")

        result = None
        formula = ""
        substitution = ""
        unit = ""
        if "resistor" in component_type:
            if voltage is not None and current not in (None, 0):
                result, formula = voltage / current, "R = V / I"
                substitution = f"{voltage:g} V / {current:g} A"
            elif voltage is not None and power not in (None, 0):
                result, formula = voltage * voltage / power, "R = V^2 / P"
                substitution = f"({voltage:g} V)^2 / {power:g} W"
            elif current is not None and power is not None and current != 0:
                result, formula = power / (current * current), "R = P / I^2"
                substitution = f"{power:g} W / ({current:g} A)^2"
            unit = "ohm"
        elif "capacitor" in component_type and time_constant is not None and resistance not in (None, 0):
            result, formula, unit = time_constant / resistance, "C = tau / R", "F"
            substitution = f"{time_constant:g} s / {resistance:g} ohm"
        elif "inductor" in component_type and time_constant is not None and resistance is not None:
            result, formula, unit = time_constant * resistance, "L = tau * R", "H"
            substitution = f"{time_constant:g} s * {resistance:g} ohm"

        prefix = next(
            (letter for kind, letter in (
                ("resistor", "R"),
                ("capacitor", "C"),
                ("inductor", "L"),
            ) if kind in component_type),
            None,
        )
        matching_components = [
            item for item in graph.components
            if prefix and prefix == next(
                (letter for kind, letter in (
                    ("resistor", "R"),
                    ("capacitor", "C"),
                    ("inductor", "L"),
                ) if kind in item.display_type.lower()),
                None,
            )
        ]
        designator = f"{prefix}{matching_components.index(component) + 1}" if component in matching_components else None
        target = component.label or designator or f"{component.display_type} component {component.id}"
        if result is None:
            if "resistor" in component_type:
                needed = "voltage and current (R = V / I), or voltage and power (R = V^2 / P)"
            elif "capacitor" in component_type:
                needed = "the time constant and resistance (C = tau / R)"
            elif "inductor" in component_type:
                needed = "the time constant and resistance (L = tau * R)"
            else:
                return None
            return (
                f"I can calculate {target}, but the question does not provide enough numeric inputs. "
                f"Please provide {needed}. I will not guess a value from the drawing."
            )

        return (
            f"Calculated {target} = {_format_calculated_value(result, unit)} "
            f"using {formula} = {substitution}. This is a calculation only; "
            "it has not changed the component value in the circuit. Use Edit to apply it if correct."
        )
