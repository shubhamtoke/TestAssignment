import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from langchain_ollama import OllamaLLM
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass(frozen=True)
class AnalystConfig:
    """Configuration for RequirementsAnalyst.

    Values are sourced from environment variables with sensible defaults so that
    behavior can be tuned without code changes.
    """
    model: str = os.getenv("OLLAMA_MODEL", "llama3")
    timeout_seconds: int = int(os.getenv("LLM_TIMEOUT_SECONDS", "45"))
    min_story_len: int = int(os.getenv("MIN_STORY_LEN", "20"))
    min_acceptance_len: int = int(os.getenv("MIN_ACCEPTANCE_LEN", "10"))


class RequirementsAnalyst:
    """Requirements Analyst Agent that uses LangChain + Ollama to analyze user stories.

    High-level responsibilities:
    - Produce a fast, structured TEXT analysis (four sections) suitable for human review
    - Convert the TEXT analysis into a strict JSON schema for downstream automation
    - Provide robust error handling and deterministic fallbacks
    - Maintain traceability from inputs to generated requirement IDs for future test case mapping
    """

    def __init__(self, config: Optional[AnalystConfig] = None):
        """Create a new RequirementsAnalyst.

        Args:
            config: Optional AnalystConfig. If not provided, environment defaults are used.
        """
        # Configuration
        self.config = config or AnalystConfig()

        # Initialize Ollama LLM (FREE, no API key needed for local runtime)
        self.llm = OllamaLLM(model=self.config.model)

        # Define the system prompt for fast, clear TEXT analysis with explicit sections
        self.system_prompt = (
            "You are a Senior Business Analyst with 10+ years of e-commerce experience. "
            "Analyze the user story and acceptance criteria and return a concise, structured TEXT analysis. "
            "Use EXACTLY these headings in UPPERCASE, in this order, each followed by '-' bullets:\n\n"
            "FUNCTIONAL REQUIREMENTS:\n- ...\n\n"
            "NON-FUNCTIONAL REQUIREMENTS:\n- ...\n\n"
            "EDGE CASES:\n- ...\n\n"
            "GAPS IDENTIFIED:\n- ...\n\n"
            "Keep it crisp and practical. Do not include any JSON, code fences, or additional commentary."
        )

    def analyze_user_story(self, user_story: str, acceptance_criteria: str) -> str:
        """Analyze a user story and acceptance criteria and return structured TEXT.

        Robust error handling:
        - Validates inputs are non-empty and meet a minimal length threshold.
        - Handles LLM timeouts gracefully and retries.
        - If the LLM unexpectedly returns JSON, attempts to parse it; on failure, falls back to an empty structure in TEXT form.
        - Falls back to a deterministic text analysis if all LLM attempts fail.
        """

        # Validate inputs
        min_story_len = self.config.min_story_len
        min_ac_len = self.config.min_acceptance_len

        if not isinstance(user_story, str) or not user_story.strip():
            print("DEBUG: Invalid user_story (empty or None). Returning minimal text template.")
            return (
                "FUNCTIONAL REQUIREMENTS:\n- (none)\n\n"
                "NON-FUNCTIONAL REQUIREMENTS:\n- (none)\n\n"
                "EDGE CASES:\n- (none)\n\n"
                "GAPS IDENTIFIED:\n- (none)\n"
            )

        if not isinstance(acceptance_criteria, str) or not acceptance_criteria.strip():
            print("DEBUG: Invalid acceptance_criteria (empty or None). Returning minimal text template.")
            return (
                "FUNCTIONAL REQUIREMENTS:\n- (none)\n\n"
                "NON-FUNCTIONAL REQUIREMENTS:\n- (none)\n\n"
                "EDGE CASES:\n- (none)\n\n"
                "GAPS IDENTIFIED:\n- (none)\n"
            )

        if len(user_story.strip()) < min_story_len or len(acceptance_criteria.strip()) < min_ac_len:
            print(
                f"DEBUG: Input too short (story_len={len(user_story.strip())}, ac_len={len(acceptance_criteria.strip())}). Using deterministic fallback."
            )
            return self._fallback_text_analysis(user_story, acceptance_criteria, timeout_seconds=0)

        # Create the prompt for text analysis
        prompt = (
            f"{self.system_prompt}\n"
            f"USER STORY:\n{user_story}\n\n"
            f"ACCEPTANCE CRITERIA:\n{acceptance_criteria}\n"
        )

        # Retry logic with timeout
        max_attempts = 3
        timeout_seconds = self.config.timeout_seconds
        last_error: Exception | None = None

        # Debug: show prompt being sent to LLM
        print("\n--- DEBUG: Prompt Being Sent To LLM ---")
        print(prompt)
        print("--- END DEBUG PROMPT ---\n")

        for attempt in range(1, max_attempts + 1):
            print(f"DEBUG: LLM attempt {attempt}/{max_attempts} with timeout={timeout_seconds}s")
            try:
                raw_response = self._invoke_with_timeout(prompt, timeout_seconds)
                print("--- DEBUG: Raw LLM Response BEGIN ---")
                print(raw_response)
                print("--- DEBUG: Raw LLM Response END ---\n")
                # If LLM returned JSON unexpectedly, try to parse; on failure, return empty structure in TEXT
                if isinstance(raw_response, str):
                    trimmed = raw_response.strip()
                    if trimmed.startswith("{") and trimmed.endswith("}"):
                        try:
                            parsed = json.loads(trimmed)
                            print("DEBUG: LLM returned JSON; converting to TEXT output")
                            return self._json_to_text(parsed)
                        except Exception as e:
                            print(f"DEBUG: JSON parse failed: {e}; returning empty structure as TEXT")
                            empty = self._default_schema(user_story)
                            return self._json_to_text(empty)
                if isinstance(raw_response, str) and raw_response.strip():
                    return raw_response.strip()
                print("DEBUG: Empty or non-string LLM response")
                last_error = ValueError("Empty LLM response")
            except FuturesTimeout:
                print(f"DEBUG: Timeout after {timeout_seconds}s on attempt {attempt}")
                last_error = TimeoutError(f"LLM request timed out after {timeout_seconds}s (attempt {attempt}/{max_attempts})")
            except Exception as e:
                print(f"DEBUG: Exception on attempt {attempt}: {e}")
                last_error = e

            # Backoff before next attempt
            time.sleep(0.5 * attempt)

        # Fallback to deterministic text analysis if LLM fails
        try:
            print("DEBUG: Falling back to deterministic text analysis")
            return self._fallback_text_analysis(user_story, acceptance_criteria, timeout_seconds)
        except Exception:
            # If fallback also fails, return default empty structure
            print("DEBUG: Fallback text analysis also failed; returning minimal text")
            return (
                "FUNCTIONAL REQUIREMENTS:\n- (none)\n\n"
                "NON-FUNCTIONAL REQUIREMENTS:\n- (none)\n\n"
                "EDGE CASES:\n- (none)\n\n"
                "GAPS IDENTIFIED:\n- (none)\n"
            )

    def _json_to_text(self, data: Dict[str, Any]) -> str:
        """Convert a JSON structure (assignment schema) into the standard TEXT format."""
        fr_lines = []
        for item in data.get("functional_requirements", []) or []:
            desc = item.get("description") or "(unspecified)"
            fr_lines.append(f"- {desc}")
        nfr_lines = []
        for item in data.get("non_functional_requirements", []) or []:
            desc = item.get("description") or "(unspecified)"
            nfr_lines.append(f"- {desc}")
        ec_lines = []
        for item in data.get("edge_cases", []) or []:
            desc = item.get("description") or item.get("scenario") or "(unspecified)"
            ec_lines.append(f"- {desc}")
        gaps_lines = []
        for gap in data.get("gaps_identified", []) or []:
            gaps_lines.append(f"- {gap}")

        fr_block = "\n".join(fr_lines) or "- (none)"
        nfr_block = "\n".join(nfr_lines) or "- (none)"
        ec_block = "\n".join(ec_lines) or "- (none)"
        gaps_block = "\n".join(gaps_lines) or "- (none)"

        return (
            f"FUNCTIONAL REQUIREMENTS:\n{fr_block}\n\n"
            f"NON-FUNCTIONAL REQUIREMENTS:\n{nfr_block}\n\n"
            f"EDGE CASES:\n{ec_block}\n\n"
            f"GAPS IDENTIFIED:\n{gaps_block}\n"
        )

    def _invoke_with_timeout(self, prompt: str, timeout_seconds: int) -> str:
        """Invoke the LLM with a hard timeout using a thread executor."""
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self.llm.invoke, prompt)
            return future.result(timeout=timeout_seconds)

    def _fallback_text_analysis(self, user_story: str, acceptance_criteria: str, timeout_seconds: int) -> str:
        """Deterministic text analysis fallback that formats acceptance criteria as functional requirements
        and provides placeholders for other sections. This avoids any LLM dependency.
        """
        # Extract functional bullets from acceptance_criteria lines that start with '-'
        functional_lines: List[str] = []
        for ln in (acceptance_criteria or "").splitlines():
            if ln.strip().startswith("-"):
                functional_lines.append(ln.strip().lstrip("- "))

        fr_section = "\n".join([f"- {item}" for item in functional_lines]) or "- (none)"
        nfr_section = "- (add performance, security, or reliability requirements)"
        ec_section = "- (add boundary conditions or error scenarios)"
        gaps_section = "- (list unclear or missing requirements)"

        return (
            f"FUNCTIONAL REQUIREMENTS:\n{fr_section}\n\n"
            f"NON-FUNCTIONAL REQUIREMENTS:\n{nfr_section}\n\n"
            f"EDGE CASES:\n{ec_section}\n\n"
            f"GAPS IDENTIFIED:\n{gaps_section}\n"
        )

    def _parse_plain_text_sections(self, text: str) -> Dict[str, List[str]]:
        """Parse plain text analysis into sections handling flexible headers and bullets.

        Supported headers (case-insensitive):
        - "functional requirements", "non-functional requirements", "edge cases", "gaps identified"
        - Trailing ':' or '-' (or en/em dashes) after the header are allowed

        Supported bullets:
        - Lines starting with '-', '•', or '*', optionally with spaces after the bullet
        """
        sections = {
            "functional_requirements": [],
            "non_functional_requirements": [],
            "edge_cases": [],
            "gaps_identified": [],
        }

        # Normalize newlines
        lines = [ln.rstrip() for ln in text.splitlines()]

        current = None
        header_map = {
            # Functional
            "functional requirements": "functional_requirements",
            # Non-functional common variants
            "non-functional requirements": "non_functional_requirements",
            "non functional requirements": "non_functional_requirements",
            "nonfunctional requirements": "non_functional_requirements",
            # Others
            "edge cases": "edge_cases",
            "gaps identified": "gaps_identified",
        }

        def normalize_header(line: str) -> str:
            base = line.strip().lower()
            # strip trailing colon, hyphen, en dash, em dash
            base = re.sub(r"[\s:\-\u2013\u2014]+$", "", base)
            return base

        def is_bullet(line: str) -> bool:
            s = line.lstrip()
            # Accept '-', '*', '•' and also bullets without a space after for robustness
            return bool(re.match(r"^([\-\*\u2022])\s*", s))

        def extract_bullet_text(line: str) -> str:
            s = line.strip()
            # Remove leading bullets or numbering like '1.'
            s = re.sub(r"^((\-|\*|\u2022)+|\d+[\.)])\s*", "", s)
            return s.strip()

        for ln in lines:
            norm = normalize_header(ln)
            if norm in header_map:
                current = header_map[norm]
                continue

            if current and is_bullet(ln):
                sections[current].append(extract_bullet_text(ln))

        return sections

    def _format_id(self, prefix: str, index: int) -> str:
        return f"{prefix}{index:03d}"

    def text_to_json_converter(self, analysis_text: str, original_user_story: str, acceptance_criteria: str | None = None) -> Dict[str, Any]:
        """Convert TEXT analysis into the assignment's JSON schema with traceability metadata.

        Expected text format:
        - Four sections in UPPERCASE with trailing colon
          FUNCTIONAL REQUIREMENTS:, NON-FUNCTIONAL REQUIREMENTS:, EDGE CASES:, GAPS IDENTIFIED:
        - Each followed by '-' bullet items (one per line)

        Returns JSON with keys:
        - user_story: str
        - functional_requirements: list[{id, description, priority, category, testable}]
        - non_functional_requirements: list[{id, description, type, measurable}]
        - edge_cases: list[{id, description, scenario}]
        - gaps_identified: list[str]
        """
        sections = self._parse_plain_text_sections(analysis_text or "")

        # Preprocess acceptance criteria bullets if provided for traceability
        ac_bullets: List[str] = []
        if isinstance(acceptance_criteria, str) and acceptance_criteria.strip():
            for ln in acceptance_criteria.splitlines():
                s = ln.strip()
                if s.startswith("-") or s.startswith("\u2022") or s.startswith("*"):
                    ac_bullets.append(re.sub(r"^((\-|\*|\u2022)+|\d+[\.)])\s*", "", s).strip())

        # Functional requirements with defaults and test metadata
        functional_items: List[Dict[str, Any]] = []
        for idx, line in enumerate(sections.get("functional_requirements", []), start=1):
            description = self._normalize_sentence(line)
            test_types = self._infer_test_types(description)
            source_ref = self._find_source_ref(description, ac_bullets)
            functional_items.append(
                {
                    "id": self._format_id("FR", idx),
                    "description": description,
                    "priority": "High",
                    "category": "Core Functionality",
                    "testable": True,
                    "test_types": test_types,
                    "test_category": "Functional",
                    "trace_id": f"REQ-FR-{idx:03d}",
                    "source": source_ref or "analysis_text",
                }
            )

        # Non-functional requirements with type inference and test metadata
        non_functional_items: List[Dict[str, Any]] = []
        for idx, line in enumerate(sections.get("non_functional_requirements", []), start=1):
            description = self._normalize_sentence(line)
            if not description:
                continue
            # Skip placeholder bullets that indicate absence (e.g., "None mentioned")
            desc_lower = description.lower()
            if desc_lower.startswith("none") or desc_lower in {"n/a.", "na."}:
                continue
            inferred_type = self._infer_nfr_type(description)
            non_functional_items.append(
                {
                    "id": self._format_id("NFR", len(non_functional_items) + 1),
                    "description": description,
                    "type": inferred_type,
                    "measurable": True,
                    "test_types": ["NonFunctional", inferred_type],
                    "test_category": "NonFunctional",
                    "trace_id": f"REQ-NFR-{len(non_functional_items) + 1:03d}",
                    "source": "analysis_text",
                }
            )

        # If no NFRs parsed, synthesize a baseline set to ensure coverage
        if len(non_functional_items) == 0:
            synthesized = [
                ("System should complete cart interactions within 2 seconds under normal load.", "Performance"),
                ("Service should maintain 99.9% monthly uptime for cart features.", "Reliability"),
                ("Error messages should be clear and user-friendly for cart operations.", "Usability"),
            ]
            for idx, (desc, nfr_type) in enumerate(synthesized, start=1):
                non_functional_items.append(
                    {
                        "id": self._format_id("NFR", idx),
                        "description": desc,
                        "type": nfr_type,
                        "measurable": True,
                        "test_types": ["NonFunctional", nfr_type],
                        "test_category": "NonFunctional",
                        "trace_id": f"REQ-NFR-{idx:03d}",
                        "source": "synthesized",
                    }
                )

        # Edge cases (use same text for description and scenario if no clear delimiter) with metadata
        edge_case_items: List[Dict[str, Any]] = []
        for idx, line in enumerate(sections.get("edge_cases", []), start=1):
            text_line = self._normalize_sentence(line)
            edge_case_items.append(
                {
                    "id": self._format_id("EC", idx),
                    "description": text_line,
                    "scenario": text_line,
                    "test_types": ["EdgeCase"],
                    "test_category": "EdgeCase",
                    "trace_id": f"REQ-EC-{idx:03d}",
                    "source": "analysis_text",
                }
            )

        # Gaps identified
        gaps_list: List[str] = [self._normalize_sentence(ln) for ln in sections.get("gaps_identified", [])]

        # Deduplicate entries by normalized description to improve quality for testers
        functional_items = self._deduplicate_by_description(functional_items)
        non_functional_items = self._deduplicate_by_description(non_functional_items)
        edge_case_items = self._deduplicate_by_description(edge_case_items)
        gaps_list = self._deduplicate_strings(gaps_list)
        
        result: Dict[str, Any] = {
            "user_story": original_user_story,
            "functional_requirements": functional_items,
            "non_functional_requirements": non_functional_items,
            "edge_cases": edge_case_items,
            "gaps_identified": gaps_list,
        }

        # Add top-level metadata for traceability
        try:
            import datetime as _dt
            generated_at = _dt.datetime.utcnow().isoformat() + "Z"
        except Exception:
            generated_at = ""
        result["metadata"] = {
            "generated_at": generated_at,
            "model": getattr(self.llm, "model", "ollama:llama3"),
            "source": "analysis_text",
            "acceptance_criteria": ac_bullets,
            "counts": {
                "functional": len(functional_items),
                "non_functional": len(non_functional_items),
                "edge_cases": len(edge_case_items),
                "gaps": len(gaps_list),
                "total": len(functional_items) + len(non_functional_items) + len(edge_case_items) + len(gaps_list),
            },
        }

        return result

    def _infer_nfr_type(self, description: str) -> str:
        """Infer a basic NFR type from description keywords."""
        text = (description or "").lower()
        if any(k in text for k in ["performance", "latency", "response", "throughput", "speed"]):
            return "Performance"
        if any(k in text for k in ["security", "encrypt", "authentication", "authorization", "privacy", "protection"]):
            return "Security"
        if any(k in text for k in ["scalability", "concurrent", "scale", "capacity", "load", "throughput"]):
            return "Scalability"
        if any(k in text for k in ["reliability", "uptime", "availability", "fault", "resilien", "backup"]):
            return "Reliability"
        if any(k in text for k in ["usability", "accessibility", "ux", "ui", "readable", "clear", "friendly", "error message"]):
            return "Usability"
        return "Unspecified"

    def _normalize_sentence(self, raw: str) -> str:
        """Normalize description lines for consistency (trim, capitalize, period)."""
        text = (raw or "").strip()
        # Remove bullet markers again defensively
        text = re.sub(r"^((\-|\*|\u2022)+|\d+[\.)])\s*", "", text).strip()
        if not text:
            return text
        # Capitalize first character
        text = text[0].upper() + text[1:]
        # Ensure trailing period unless it already ends with punctuation
        if text[-1] not in ".!?":
            text += "."
        return text

    def _infer_test_types(self, description: str) -> List[str]:
        """Infer test types (Positive/Negative/Boundary) from description keywords."""
        d = (description or "").lower()
        types: List[str] = ["Positive"]
        if any(k in d for k in ["fail", "error", "cannot", "not allowed", "out-of-stock", "invalid", "exceed"]):
            types.append("Negative")
        if any(k in d for k in ["max", "maximum", "limit", "boundary", "min", "minimum"]):
            types.append("Boundary")
        # De-duplicate while preserving order
        seen = set()
        unique: List[str] = []
        for t in types:
            if t not in seen:
                seen.add(t)
                unique.append(t)
        return unique

    def _find_source_ref(self, description: str, ac_bullets: List[str]) -> str | None:
        """Find an acceptance criteria line that matches the description closely to aid traceability."""
        if not ac_bullets:
            return None
        desc = (description or "").lower().rstrip(".")
        # naive match by substring tokens
        for i, ac in enumerate(ac_bullets, start=1):
            ac_norm = ac.lower().rstrip(".")
            if ac_norm in desc or desc in ac_norm:
                return f"acceptance_criteria[{i}]"
        return None

    def _deduplicate_by_description(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate requirement objects by normalized description (case-insensitive)."""
        seen: set[str] = set()
        unique: List[Dict[str, Any]] = []
        for obj in items:
            desc = (obj.get("description") or "").strip().lower().rstrip(".")
            if not desc or desc in seen:
                continue
            seen.add(desc)
            unique.append(obj)
        # Reassign IDs and trace_ids sequentially to keep order consistent
        for idx, obj in enumerate(unique, start=1):
            if obj["id"].startswith("FR"):
                obj["id"] = self._format_id("FR", idx)
                obj["trace_id"] = f"REQ-FR-{idx:03d}"
            elif obj["id"].startswith("NFR"):
                obj["id"] = self._format_id("NFR", idx)
                obj["trace_id"] = f"REQ-NFR-{idx:03d}"
            elif obj["id"].startswith("EC"):
                obj["id"] = self._format_id("EC", idx)
                obj["trace_id"] = f"REQ-EC-{idx:03d}"
        return unique

    def _deduplicate_strings(self, entries: List[str]) -> List[str]:
        """Remove duplicate strings while preserving order (case-insensitive)."""
        seen: set[str] = set()
        unique: List[str] = []
        for s in entries:
            key = (s or "").strip().lower().rstrip(".")
            if not key or key in seen:
                continue
            seen.add(key)
            # Ensure consistent period
            if s and s[-1] not in ".!?":
                s = s + "."
            unique.append(s)
        return unique

    def validate_assignment_requirements(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate assignment requirements and return a structured report.

        Checks:
        - At least 10 total requirements (FR + NFR + EC + gaps)
        - Proper ID formats: FR###, NFR###, EC###
        - All required fields present (leverages _is_valid_schema)
        - No missing keys from schema (per-item required keys present)
        """
        report: Dict[str, Any] = {
            "counts": {
                "functional": 0,
                "non_functional": 0,
                "edge_cases": 0,
                "gaps": 0,
                "total": 0,
            },
            "checks": {
                "at_least_10_total": False,
                "ids_format_valid": False,
                "required_fields_present": False,
                "no_missing_keys": False,
            },
            "errors": [],
            "valid": False,
        }

        fr_list = data.get("functional_requirements", []) or []
        nfr_list = data.get("non_functional_requirements", []) or []
        ec_list = data.get("edge_cases", []) or []
        gaps_list = data.get("gaps_identified", []) or []

        report["counts"]["functional"] = len(fr_list)
        report["counts"]["non_functional"] = len(nfr_list)
        report["counts"]["edge_cases"] = len(ec_list)
        report["counts"]["gaps"] = len(gaps_list)
        report["counts"]["total"] = (
            report["counts"]["functional"]
            + report["counts"]["non_functional"]
            + report["counts"]["edge_cases"]
            + report["counts"]["gaps"]
        )

        # Check 1: at least 10 total
        report["checks"]["at_least_10_total"] = report["counts"]["total"] >= 10
        if not report["checks"]["at_least_10_total"]:
            report["errors"].append(
                f"Total requirements ({report['counts']['total']}) is less than 10"
            )

        # Check 2: ID formats
        fr_ok = all(isinstance(x.get("id"), str) and re.fullmatch(r"FR\d{3}", x["id"]) for x in fr_list)
        nfr_ok = all(isinstance(x.get("id"), str) and re.fullmatch(r"NFR\d{3}", x["id"]) for x in nfr_list)
        ec_ok = all(isinstance(x.get("id"), str) and re.fullmatch(r"EC\d{3}", x["id"]) for x in ec_list)
        report["checks"]["ids_format_valid"] = fr_ok and nfr_ok and ec_ok
        if not report["checks"]["ids_format_valid"]:
            report["errors"].append("One or more IDs do not match required formats FR###, NFR###, EC###")

        # Check 3: required fields present
        report["checks"]["required_fields_present"] = self._is_valid_schema(data)
        if not report["checks"]["required_fields_present"]:
            report["errors"].append("JSON missing required fields or incorrect types per schema")

        # Check 4: no missing keys in items (schema-required)
        def missing_keys(obj: Dict[str, Any], required: set[str]) -> set[str]:
            return required - set(obj.keys())

        fr_required = {"id", "description", "priority", "category", "testable"}
        nfr_required = {"id", "description", "type", "measurable"}
        ec_required = {"id", "description", "scenario"}

        fr_missing = [missing_keys(x, fr_required) for x in fr_list]
        nfr_missing = [missing_keys(x, nfr_required) for x in nfr_list]
        ec_missing = [missing_keys(x, ec_required) for x in ec_list]

        no_missing = all(len(m) == 0 for m in fr_missing + nfr_missing + ec_missing)
        report["checks"]["no_missing_keys"] = no_missing
        if not no_missing:
            report["errors"].append("Some items are missing required keys from the schema")

        report["valid"] = all(report["checks"].values())
        return report

    def extract_json_from_response(self, response_text: str, original_user_story: str) -> Dict[str, Any]:
        """Extract JSON from LLM response text using regex, validate structure, and handle errors.

        - Uses regex to locate JSON objects in the response.
        - Validates against the required schema.
        - On failure, returns a default empty structure.
        """
        default_result = self._default_schema(original_user_story)

        if not isinstance(response_text, str) or not response_text.strip():
            print("DEBUG: Response text is empty or non-string; returning default structure")
            return default_result

        # Try strict parse first in case model returned only JSON
        try:
            parsed = json.loads(response_text.strip())
            if self._is_valid_schema(parsed):
                print("DEBUG: Strict JSON parse succeeded and schema validated")
                return parsed
        except Exception as e:
            print(f"DEBUG: Strict JSON parse failed: {e}")
            pass

        # Regex-based extraction of JSON objects (non-greedy to try smallest valid blocks first)
        # We will iterate through all candidates and return the first valid one
        json_pattern = re.compile(r"\{[\s\S]*?\}")
        for match in json_pattern.finditer(response_text):
            candidate_text = match.group(0)
            try:
                parsed_candidate = json.loads(candidate_text)
            except json.JSONDecodeError:
                # continue trying other candidates
                continue
            if self._is_valid_schema(parsed_candidate):
                print("DEBUG: Regex-based JSON extraction succeeded and schema validated")
                return parsed_candidate

        # If nothing valid found, return default
        print("DEBUG: No valid JSON found after strict and regex-based attempts; returning default structure")
        return default_result

    def _default_schema(self, user_story_text: str) -> Dict[str, Any]:
        """Return the required default empty structure with the provided user story text."""
        return {
            "user_story": user_story_text,
            "functional_requirements": [],
            "non_functional_requirements": [],
            "edge_cases": [],
            "gaps_identified": [],
        }

    def _is_valid_schema(self, data: Dict[str, Any]) -> bool:
        """Validate the JSON structure matches the required schema types and keys.

        Required top-level keys and types:
        - user_story: str
        - functional_requirements: list of {id:str, description:str, priority:str, category:str, testable:bool}
        - non_functional_requirements: list of {id:str, description:str, type:str, measurable:bool}
        - edge_cases: list of {id:str, description:str, scenario:str}
        - gaps_identified: list[str]
        """
        if not isinstance(data, dict):
            return False

        required_keys = [
            "user_story",
            "functional_requirements",
            "non_functional_requirements",
            "edge_cases",
            "gaps_identified",
        ]
        for key in required_keys:
            if key not in data:
                return False

        if not isinstance(data["user_story"], str):
            return False

        # functional_requirements
        if not isinstance(data["functional_requirements"], list):
            return False
        for item in data["functional_requirements"]:
            if not self._has_keys_with_types(
                item,
                {
                    "id": str,
                    "description": str,
                    "priority": str,
                    "category": str,
                    "testable": bool,
                },
            ):
                return False

        # non_functional_requirements
        if not isinstance(data["non_functional_requirements"], list):
            return False
        for item in data["non_functional_requirements"]:
            if not self._has_keys_with_types(
                item,
                {
                    "id": str,
                    "description": str,
                    "type": str,
                    "measurable": bool,
                },
            ):
                return False

        # edge_cases
        if not isinstance(data["edge_cases"], list):
            return False
        for item in data["edge_cases"]:
            if not self._has_keys_with_types(
                item,
                {
                    "id": str,
                    "description": str,
                    "scenario": str,
                },
            ):
                return False

        # gaps_identified
        if not isinstance(data["gaps_identified"], list):
            return False
        for gap in data["gaps_identified"]:
            if not isinstance(gap, str):
                return False

        return True

    def _has_keys_with_types(self, obj: Dict[str, Any], spec: Dict[str, Any]) -> bool:
        if not isinstance(obj, dict):
            return False
        for key, expected_type in spec.items():
            if key not in obj:
                return False
            # Allow bool to pass isinstance check properly (bool is subclass of int in Python)
            value = obj[key]
            if expected_type is bool:
                if not isinstance(value, bool):
                    return False
            else:
                if not isinstance(value, expected_type):
                    return False
        return True

if __name__ == "__main__":
    # Initialize the analyst
    analyst = RequirementsAnalyst()
    
    # Edge-case test runner (enabled with --test CLI flag or RUN_EDGE_TESTS=1)
    def run_edge_case_tests(analyst_instance: RequirementsAnalyst) -> None:
        print("\n=== RUNNING EDGE CASE TESTS ===")
        test_cases = [
            {
                "name": "Empty user story",
                "user_story": "",
                "ac": "- Add items\n- Remove items"
            },
            {
                "name": "Very short user story",
                "user_story": "Buy",
                "ac": "- Add to cart"
            },
            {
                "name": "Malformed acceptance criteria",
                "user_story": "As a user, I want to save items for later.",
                "ac": "This is not a bullet list. 1) first; 2) second; random text"
            },
            {
                "name": "Extremely long input",
                "user_story": ("As a power user, I want to bulk add and update items so that I can manage large orders efficiently. " * 500).strip(),
                "ac": ("- Bulk add up to 100 items at once\n" * 500).strip()
            },
            {
                "name": "Special characters and emojis",
                "user_story": "As a user 😃, I want a cart that supports ©products® with ™symbols — and dashes.",
                "ac": "• Add ✓ item\n• Remove ✗ item\n- Show total — with tax"
            },
        ]

        passed = 0
        for idx, tc in enumerate(test_cases, start=1):
            print(f"\n-- Test {idx}: {tc['name']} --")
            try:
                text_out = analyst_instance.analyze_user_story(tc["user_story"], tc["ac"])
                print("Text output (first 200 chars):")
                print(text_out[:200].replace("\n", " ") + ("..." if len(text_out) > 200 else ""))
                try:
                    json_out = analyst_instance.text_to_json_converter(text_out, tc["user_story"]) 
                    # Basic shape checks
                    assert isinstance(json_out, dict)
                    assert "functional_requirements" in json_out
                    assert "non_functional_requirements" in json_out
                    assert "edge_cases" in json_out
                    assert "gaps_identified" in json_out
                    print("Result: PASS")
                    passed += 1
                except Exception as conv_err:
                    print(f"Converter error: {conv_err}")
                    print("Result: FAIL")
            except Exception as e:
                print(f"Agent error: {e}")
                print("Result: FAIL")

        print(f"\n=== EDGE CASE TEST SUMMARY: {passed}/{len(test_cases)} passed ===")

    if "--test" in sys.argv or os.getenv("RUN_EDGE_TESTS") == "1":
        run_edge_case_tests(analyst)
        sys.exit(0)

    # Input: load from sample_input.json if present, else use built-in sample
    user_story = None
    acceptance_criteria = None
    input_path = "sample_input.json"
    if os.path.exists(input_path):
        try:
            with open(input_path, "r", encoding="utf-8") as f:
                sample = json.load(f)
            user_story = sample.get("user_story")
            ac = sample.get("acceptance_criteria")
            if isinstance(ac, list):
                # Join list into dash-prefixed lines
                acceptance_criteria = "\n".join([f"- {line}" for line in ac])
            elif isinstance(ac, str):
                acceptance_criteria = ac
        except Exception as e:
            print(f"DEBUG: Failed to read {input_path}: {e}. Falling back to built-in sample.")

    if not user_story or not acceptance_criteria:
        # Built-in default
        user_story = (
            "As a customer, I want to add products to my shopping cart and modify quantities, "
            "so that I can purchase multiple items in a single order."
        )
        acceptance_criteria = """
        - User can add products to cart from product detail page
        - User can update item quantities in the cart
        - User can remove items from the cart
        - Cart displays total price including tax
        - Cart persists during the user session
        - Maximum 10 items of any single product allowed
        - Out-of-stock items cannot be added to cart
        - Cart shows item availability status
        - User receives confirmation when items are added/removed
        """
    
    # Get analysis (TEXT)
    print("Analyzing user story...")
    analysis_text = analyst.analyze_user_story(user_story, acceptance_criteria)

    # Print TEXT analysis
    print("\n=== ANALYSIS RESULTS (TEXT) ===")
    print(analysis_text)

    # Convert TEXT to JSON using converter and print
    json_result = analyst.text_to_json_converter(analysis_text, user_story)
    print("\n=== ANALYSIS RESULTS (JSON) ===")
    print(json.dumps(json_result, indent=2))

    # Save TEXT output to file
    output_txt = "analysis_output.txt"
    try:
        with open(output_txt, "w", encoding="utf-8") as f:
            f.write(analysis_text)
        print(f"\nSaved text analysis to '{output_txt}'")
    except Exception as e:
        print(f"\nFailed to save text analysis to '{output_txt}': {e}")

    # Save JSON output to file
    output_json = "analysis_output.json"
    try:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(json_result, f, ensure_ascii=False, indent=2)
        print(f"Saved JSON analysis to '{output_json}'")
    except Exception as e:
        print(f"Failed to save JSON analysis to '{output_json}': {e}")