import copy
import base64
import concurrent.futures
import html
import json
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
from openai import OpenAI
from streamlit_extras.eval_javascript import eval_javascript


st.set_page_config(page_title="AI Sprint Planner", layout="wide")


def load_icon_data_uri(candidate_paths: List[str]) -> str:
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".svg": "image/svg+xml",
        ".webp": "image/webp",
    }

    for path in candidate_paths:
        if not os.path.exists(path):
            continue
        ext = os.path.splitext(path)[1].lower()
        mime = mime_map.get(ext, "image/png")
        with open(path, "rb") as f:
            raw = f.read()
        b64 = base64.b64encode(raw).decode("utf-8")
        return f"data:{mime};base64,{b64}"

    return ""


def get_openai_client() -> Optional[OpenAI]:
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])

    raise ValueError("Could not parse JSON from model response.")


def extract_response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    chunks: List[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            if getattr(content, "type", "") == "output_text":
                text = getattr(content, "text", "")
                if text:
                    chunks.append(text)

    return "\n".join(chunks).strip()


def sprint_point_total(sprint: Dict[str, Any]) -> int:
    total = 0
    for story in sprint.get("stories", []):
        points = story.get("estimate_points", 0)
        try:
            total += int(points)
        except (TypeError, ValueError):
            continue
    return total


def get_velocity_violations(plan: Dict[str, Any], sprint_velocity: int) -> List[str]:
    violations: List[str] = []
    for idx, sprint in enumerate(plan.get("sprints", [])):
        sprint_no = sprint.get("sprint_number", idx + 1)
        total = sprint_point_total(sprint)
        if total > sprint_velocity:
            violations.append(
                f"Sprint {sprint_no} has {total} points which exceeds velocity {sprint_velocity}."
            )
    return violations


def generate_plan(
    application_idea: str,
    sprint_velocity: int,
    sprint_count: int,
) -> Dict[str, Any]:
    client = get_openai_client()
    if client is None:
        raise RuntimeError("OPENAI_API_KEY not found. Set it in environment variables or Streamlit secrets.")

    system_prompt = (
        "You are an expert agile delivery coach. "
        "Return only valid JSON with no markdown fences and no extra text."
    )

    user_prompt = f"""
Create a sprint plan for this application idea:
"{application_idea}"

Constraints:
- Sprint velocity (max story points per sprint): {sprint_velocity}
- Number of sprints: {sprint_count}
- Keep scope realistic for velocity and sprint count.

Output JSON schema exactly:
{{
  "epics": [
    {{"id": "E1", "name": "Epic name", "summary": "1-2 sentence summary"}}
  ],
  "sprints": [
    {{
      "sprint_number": 1,
      "goal": "Sprint goal",
      "stories": [
        {{
          "id": "S1",
          "title": "Story title",
          "epic_id": "E1",
                    "description": "As a <type of user>, I want to <perform some action>, so that <goal/value>",
                    "acceptance_criteria": [
                        "Background:\nGiven <common precondition 1>\nAnd <common precondition 2>",
                        "Scenario: <Primary Scenario Name>\nGiven <initial context>\nAnd <additional condition>\nWhen <user performs an action>\nAnd <another action if needed>\nThen <expected outcome>\nAnd <additional outcome>",
                        "Scenario: <Alternate / Edge Case Scenario>\nGiven <initial context>\nWhen <different action or edge condition>\nThen <expected result>"
                    ],
                    "technical_specifications": ["spec 1", "spec 2"],
                    "non_functional_requirements": ["nfr 1", "nfr 2"],
                    "test_cases": [
                        {{
                            "preconditions": "User is authenticated and on target screen",
                            "test_steps": "1) Enter valid data 2) Submit",
                            "expected_behaviour": "System accepts input and shows success confirmation"
                        }}
                    ],
          "estimate_points": 3,
          "dependencies": ["S0"],
          "priority": "High"
        }}
      ]
    }}
  ]
}}

Rules:
- Create 3 to 6 epics.
- Ensure exactly {sprint_count} sprints.
- Distribute stories across sprints with a sensible sequence.
- Include at least 3 stories per sprint.
- Story IDs must be unique.
- dependency IDs should refer to earlier stories when possible.
- The sum of estimate_points for each sprint must be <= {sprint_velocity}.
- Every story description must be in this format:
    As a <type of user>
        I want to <perform some action>
        So that <achieve some goal or value>
- Acceptance criteria must be Gherkin-oriented and include one Background and at least two Scenarios.
- Include technical_specifications and non_functional_requirements arrays for each story (can be empty arrays only if truly not applicable).
- Include test_cases array for each story with 2 to 4 rows, each containing: preconditions, test_steps, expected_behaviour.
"""

    model_name = "gpt-4o-mini"
    velocity_feedback = ""

    for attempt in range(3):
        text = ""
        prompt = user_prompt
        if velocity_feedback:
            prompt = f"{user_prompt}\n\nPrevious draft was invalid:\n{velocity_feedback}\nRegenerate a corrected plan."

        # Support both newer and older OpenAI Python SDK styles.
        if hasattr(client, "responses"):
            response = client.responses.create(
                model=model_name,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.4,
            )
            text = extract_response_text(response)
        else:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.4,
            )
            message_content = completion.choices[0].message.content
            if isinstance(message_content, str):
                text = message_content
            elif isinstance(message_content, list):
                parts: List[str] = []
                for part in message_content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        parts.append(part.get("text", ""))
                text = "\n".join(parts).strip()

        if not text:
            raise ValueError("The model response was empty.")

        data = extract_json(text)

        if "epics" not in data or "sprints" not in data:
            raise ValueError("The model response did not contain required fields: epics, sprints.")

        data = allocate_stories_to_sprints(data, sprint_count=sprint_count, sprint_velocity=sprint_velocity)
        violations = get_velocity_violations(data, sprint_velocity)
        if not violations:
            return data
        velocity_feedback = "\n".join(violations)

    raise ValueError(f"Could not generate a plan within sprint velocity. {velocity_feedback}")


def flatten_stories(plan: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    stories_by_id: Dict[str, Dict[str, Any]] = {}
    for sprint in plan.get("sprints", []):
        sprint_number = sprint.get("sprint_number")
        sprint_goal = sprint.get("goal", "")
        for story in sprint.get("stories", []):
            story_id = str(story.get("id", ""))
            if not story_id:
                continue
            stories_by_id[story_id] = {
                **story,
                "sprint_number": sprint_number,
                "sprint_goal": sprint_goal,
            }
    return stories_by_id


def epic_color(index: int) -> str:
    palette = [
        "#FFE8D6",
        "#D8F3DC",
        "#E0FBFC",
        "#EAE4E9",
        "#FFF1C1",
        "#DDE7FF",
    ]
    return palette[index % len(palette)]


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    color = hex_color.lstrip("#")
    if len(color) != 6:
        return f"rgba(0, 0, 0, {alpha})"
    r = int(color[0:2], 16)
    g = int(color[2:4], 16)
    b = int(color[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


def normalize_priority(value: Any) -> str:
    if not isinstance(value, str):
        return "Medium"
    normalized = value.strip().lower()
    if normalized in {"high", "h", "p1"}:
        return "High"
    if normalized in {"medium", "med", "m", "p2"}:
        return "Medium"
    if normalized in {"low", "l", "p3"}:
        return "Low"
    return "Medium"


def priority_rank(value: Any) -> int:
    priority = normalize_priority(value)
    if priority == "High":
        return 0
    if priority == "Medium":
        return 1
    return 2


def safe_points(value: Any) -> int:
    try:
        points = int(value)
    except (TypeError, ValueError):
        return 1
    return max(1, points)


def build_epic_meta(epics: List[Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
    meta: Dict[str, Dict[str, str]] = {}
    for idx, epic in enumerate(epics):
        epic_id = str(epic.get("id", f"E{idx+1}"))
        meta[epic_id] = {
            "epic_name": str(epic.get("name", epic_id)),
            "epic_color": epic_color(idx),
        }
    return meta


def flatten_all_stories(plan: Dict[str, Any], epic_meta: Dict[str, Dict[str, str]]) -> List[Dict[str, Any]]:
    all_stories: List[Dict[str, Any]] = []
    seen_ids = set()
    for sprint in plan.get("sprints", []):
        for story in sprint.get("stories", []):
            story_id = str(story.get("id", "")).strip()
            if not story_id or story_id in seen_ids:
                continue
            seen_ids.add(story_id)

            epic_id = str(story.get("epic_id", "")).strip()
            if not epic_id and epic_meta:
                epic_id = next(iter(epic_meta.keys()))
            if epic_id not in epic_meta:
                epic_meta[epic_id] = {
                    "epic_name": epic_id or "Unknown Epic",
                    "epic_color": epic_color(len(epic_meta)),
                }

            dependencies = story.get("dependencies", [])
            if not isinstance(dependencies, list):
                dependencies = []

            meta = epic_meta.get(epic_id, {"epic_name": "Unknown Epic", "epic_color": epic_color(0)})
            all_stories.append(
                {
                    **story,
                    "id": story_id,
                    "epic_id": epic_id,
                    "epic_name": meta["epic_name"],
                    "epic_color": meta["epic_color"],
                    "priority": normalize_priority(story.get("priority", "Medium")),
                    "estimate_points": safe_points(story.get("estimate_points", 1)),
                    "dependencies": [str(dep) for dep in dependencies if str(dep).strip()],
                }
            )
    return all_stories


def dependencies_satisfied(
    story: Dict[str, Any],
    allocated_story_ids: set,
    current_sprint_story_ids: set,
) -> bool:
    for dep in story.get("dependencies", []):
        dep_id = str(dep)
        if dep_id in allocated_story_ids or dep_id in current_sprint_story_ids:
            continue
        return False
    return True


def allocate_stories_to_sprints(plan: Dict[str, Any], sprint_count: int, sprint_velocity: int) -> Dict[str, Any]:
    def sprint_goal_from_stories(stories: List[Dict[str, Any]], sprint_no: int) -> str:
        if not stories:
            return "Stabilize carry-over work and prepare the next delivery slice."

        titles = [str(s.get("title", "")).lower() for s in stories]
        epic_names: List[str] = []
        for s in stories:
            name = str(s.get("epic_name", "")).strip()
            if name and name not in epic_names:
                epic_names.append(name)

        combined = " ".join(titles)

        theme = "core shopping flow"
        if any(k in combined for k in ["login", "register", "signup", "account", "profile", "auth"]):
            theme = "account setup and access"
        elif any(k in combined for k in ["catalog", "product", "search", "filter", "category", "listing"]):
            theme = "product discovery and browsing"
        elif any(k in combined for k in ["cart", "add to cart", "quantity", "checkout", "payment"]):
            theme = "cart and checkout journey"
        elif any(k in combined for k in ["order", "history", "tracking", "status", "delivery"]):
            theme = "post-purchase order experience"
        elif any(k in combined for k in ["admin", "manage", "settings", "configuration"]):
            theme = "operations and management controls"

        outcome = "a usable end-to-end user flow"
        if "account" in theme:
            outcome = "users can create accounts and sign in confidently"
        elif "discovery" in theme:
            outcome = "users can find relevant products quickly"
        elif "checkout" in theme:
            outcome = "users can add items and complete purchase steps"
        elif "post-purchase" in theme:
            outcome = "users can monitor and manage their orders"
        elif "operations" in theme:
            outcome = "admins can manage storefront data reliably"

        epic_hint = ""
        if epic_names:
            epic_hint = f" across {', '.join(epic_names[:2])}"

        starters = [
            "This sprint establishes",
            "This sprint delivers",
            "This sprint focuses on",
            "This sprint advances",
        ]
        starter = starters[(max(1, sprint_no) - 1) % len(starters)]

        goal = f"{starter} {theme}{epic_hint}, so {outcome}."

        words = goal.split()
        if len(words) > 30:
            goal = " ".join(words[:30])
        return goal

    updated_plan = copy.deepcopy(plan)
    epic_meta = build_epic_meta(updated_plan.get("epics", []))
    backlog = flatten_all_stories(updated_plan, epic_meta)
    backlog.sort(key=lambda s: (priority_rank(s.get("priority")), safe_points(s.get("estimate_points")), s.get("id", "")))

    new_sprints: List[Dict[str, Any]] = []
    for i in range(sprint_count):
        new_sprints.append(
            {
                "sprint_number": i + 1,
                "goal": "",
                "stories": [],
            }
        )

    remaining = backlog.copy()
    allocated_story_ids = set()

    for sprint in new_sprints:
        capacity_left = sprint_velocity
        current_story_ids = set()

        while True:
            candidate_index = -1
            for idx, story in enumerate(remaining):
                points = safe_points(story.get("estimate_points", 1))
                if points > capacity_left:
                    continue
                if not dependencies_satisfied(story, allocated_story_ids, current_story_ids):
                    continue
                candidate_index = idx
                break

            if candidate_index == -1:
                break

            chosen = remaining.pop(candidate_index)
            sprint["stories"].append(chosen)
            chosen_id = str(chosen.get("id", ""))
            if chosen_id:
                allocated_story_ids.add(chosen_id)
                current_story_ids.add(chosen_id)
            capacity_left -= safe_points(chosen.get("estimate_points", 1))

        sprint["goal"] = sprint_goal_from_stories(sprint.get("stories", []), int(sprint.get("sprint_number", 0) or 0))

    updated_plan["sprints"] = new_sprints
    updated_plan["unallocated_stories"] = remaining
    return updated_plan


def format_gherkin_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    clean = text.replace("\r", " ").replace("\n", " ")
    clean = re.sub(r"\s+", " ", clean).strip()
    if not clean:
        return ""

    # Force line breaks before common Gherkin tokens.
    clean = re.sub(r"\s+(?=(Background:|Scenario:|Given\b|And\b|When\b|Then\b))", "\n", clean)
    raw_lines = [line.strip() for line in clean.split("\n") if line.strip()]

    formatted_lines: List[str] = []
    for line in raw_lines:
        lower = line.lower()

        if lower.startswith("scenario:"):
            title = line.split(":", 1)[1].strip() if ":" in line else ""
            formatted_lines.append(f"<span class='scenario-label'>Scenario:</span> {html.escape(title)} -")
            continue

        if lower.startswith("background:"):
            formatted_lines.append("<strong>Background:</strong>")
            continue

        for token in ["Given", "And", "When", "Then"]:
            if line.lower().startswith(token.lower() + " "):
                remainder = line[len(token) :].strip()
                formatted_lines.append(f"<strong>{token}</strong> {html.escape(remainder)}")
                break
        else:
            formatted_lines.append(html.escape(line))

    return "<br>".join(formatted_lines)


def build_story_clipboard_text(story: Dict[str, Any]) -> str:
    story_id = str(story.get("id", ""))
    title = str(story.get("title", ""))
    sprint = str(story.get("sprint_number", "-"))
    priority = str(story.get("priority", "-"))
    points = str(story.get("estimate_points", "-"))

    lines: List[str] = []
    lines.append(f"{story_id}: {title} | Sprint: {sprint} | Priority: {priority} | Points: {points}")
    lines.append("")

    lines.append("Description")
    description = str(story.get("description", "")).strip()
    if description:
        lines.append(description)
    else:
        lines.append("As a <type of user>")
        lines.append("I want to <perform some action>")
        lines.append("So that <achieve some goal or value>")
    lines.append("")

    lines.append("Acceptance Criteria")
    criteria = story.get("acceptance_criteria", [])
    if isinstance(criteria, list) and criteria:
        for item in criteria:
            lines.append(str(item).strip())
            lines.append("")
    else:
        lines.append("Background:")
        lines.append("Given <common precondition 1>")
        lines.append("And <common precondition 2>")
        lines.append("")
        lines.append("Scenario: <Primary Scenario Name> -")
        lines.append("Given <initial context>")
        lines.append("And <additional condition>")
        lines.append("When <user performs an action>")
        lines.append("And <another action if needed>")
        lines.append("Then <expected outcome>")
        lines.append("And <additional outcome>")
    lines.append("")

    lines.append("Technical Specifications")
    technical_specs = story.get("technical_specifications", [])
    if isinstance(technical_specs, list) and technical_specs:
        for idx, spec in enumerate(technical_specs, start=1):
            lines.append(f"{idx}) {spec}")
    else:
        lines.append("1)")
        lines.append("2)")
    lines.append("")

    lines.append("Non Functional Requirements")
    nfrs = story.get("non_functional_requirements", [])
    if isinstance(nfrs, list) and nfrs:
        for idx, nfr in enumerate(nfrs, start=1):
            lines.append(f"{idx}) {nfr}")
    else:
        lines.append("1)")
        lines.append("2)")
    lines.append("")

    lines.append("Dependencies")
    deps = story.get("dependencies", [])
    if isinstance(deps, list) and deps:
        lines.append(", ".join(str(dep) for dep in deps))
    else:
        lines.append("None")

    lines.append("")
    lines.append("Test Cases")
    for row in build_test_cases_rows(story):
        lines.append(f"Preconditions: {row['Preconditions']}")
        lines.append(f"Test Steps: {row['Test Steps']}")
        lines.append(f"Expected Behaviour: {row['Expected Behaviour']}")
        lines.append("")

    return "\n".join(lines).strip()


def build_story_clipboard_html(story: Dict[str, Any]) -> str:
    story_id = html.escape(str(story.get("id", "")))
    title = html.escape(str(story.get("title", "")))
    sprint = html.escape(str(story.get("sprint_number", "-")))
    priority = html.escape(str(story.get("priority", "-")))
    points = html.escape(str(story.get("estimate_points", "-")))

    description = str(story.get("description", "")).strip()
    if description:
        description_html = html.escape(description)
    else:
        description_html = (
            "As a &lt;type of user&gt;<br>"
            "I want to &lt;perform some action&gt;<br>"
            "So that &lt;achieve some goal or value&gt;"
        )

    criteria_blocks: List[str] = []
    criteria = story.get("acceptance_criteria", [])
    if isinstance(criteria, list) and criteria:
        for item in criteria:
            block = format_gherkin_text(str(item))
            block = block.replace(
                "<span class='scenario-label'>Scenario:</span>",
                "<span style='color:#1d4ed8;font-weight:700;'>Scenario:</span>",
            )
            if block:
                criteria_blocks.append(f"<div style='margin:0 0 8px 0;line-height:1.25;'>{block}</div>")
    else:
        criteria_blocks.append(
            "<div style='margin:0 0 8px 0;line-height:1.25;'><strong>Background:</strong><br><strong>Given</strong> &lt;common precondition 1&gt;<br><strong>And</strong> &lt;common precondition 2&gt;</div>"
        )

    technical_specs = story.get("technical_specifications", [])
    if isinstance(technical_specs, list) and technical_specs:
        technical_html = "<br>".join(f"{idx}) {html.escape(str(spec))}" for idx, spec in enumerate(technical_specs, start=1))
    else:
        technical_html = "1)<br>2)"

    nfrs = story.get("non_functional_requirements", [])
    if isinstance(nfrs, list) and nfrs:
        nfr_html = "<br>".join(f"{idx}) {html.escape(str(nfr))}" for idx, nfr in enumerate(nfrs, start=1))
    else:
        nfr_html = "1)<br>2)"

    deps = story.get("dependencies", [])
    if isinstance(deps, list) and deps:
        deps_html = html.escape(", ".join(str(dep) for dep in deps))
    else:
        deps_html = "None"

    test_rows = build_test_cases_rows(story)
    test_rows_html = "".join(
        [
            f"<tr><td style='border:1px solid #d1d5db;padding:6px;vertical-align:top;'>{html.escape(row['Preconditions'])}</td>"
            f"<td style='border:1px solid #d1d5db;padding:6px;vertical-align:top;'>{html.escape(row['Test Steps'])}</td>"
            f"<td style='border:1px solid #d1d5db;padding:6px;vertical-align:top;'>{html.escape(row['Expected Behaviour'])}</td></tr>"
            for row in test_rows
        ]
    )

    return f"""
    <div style="font-family:Calibri, Arial, sans-serif; color:#111827; line-height:1.3;">
      <div style="font-size:16px; font-weight:700; margin-bottom:8px;">{story_id}: {title} | Sprint: {sprint} | Priority: {priority} | Points: {points}</div>

      <div style="font-weight:700; text-decoration:underline; margin:8px 0 4px 0;">Description</div>
      <div style="margin-bottom:8px;">{description_html}</div>

      <div style="font-weight:700; text-decoration:underline; margin:8px 0 4px 0;">Acceptance Criteria</div>
      {''.join(criteria_blocks)}

      <div style="font-weight:700; text-decoration:underline; margin:8px 0 4px 0;">Technical Specifications</div>
      <div style="margin-bottom:8px;">{technical_html}</div>

      <div style="font-weight:700; text-decoration:underline; margin:8px 0 4px 0;">Non Functional Requirements</div>
      <div style="margin-bottom:8px;">{nfr_html}</div>

      <div style="font-weight:700; text-decoration:underline; margin:8px 0 4px 0;">Dependencies</div>
      <div>{deps_html}</div>

            <div style="font-weight:700; text-decoration:underline; margin:8px 0 4px 0;">Test Cases</div>
            <table style="border-collapse:collapse;width:100%;font-size:13px;">
                <thead>
                    <tr>
                        <th style="border:1px solid #d1d5db;padding:6px;text-align:left;">Preconditions</th>
                        <th style="border:1px solid #d1d5db;padding:6px;text-align:left;">Test Steps</th>
                        <th style="border:1px solid #d1d5db;padding:6px;text-align:left;">Expected Behaviour</th>
                    </tr>
                </thead>
                <tbody>
                    {test_rows_html}
                </tbody>
            </table>
    </div>
    """.strip()


def build_test_cases_rows(story: Dict[str, Any]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    raw_cases = story.get("test_cases", [])
    if isinstance(raw_cases, list):
        for case in raw_cases:
            if not isinstance(case, dict):
                continue
            pre = str(case.get("preconditions", "")).strip()
            steps = str(case.get("test_steps", "")).strip()
            expected = str(case.get("expected_behaviour", "")).strip()
            if pre or steps or expected:
                rows.append(
                    {
                        "Preconditions": pre or "-",
                        "Test Steps": steps or "-",
                        "Expected Behaviour": expected or "-",
                    }
                )

    if rows:
        return rows

    # Backward-compatible fallback if older plans don't include test_cases.
    return [
        {
            "Preconditions": "Relevant setup is complete",
            "Test Steps": "Execute the primary user flow for this story",
            "Expected Behaviour": "System behaves as defined in acceptance criteria",
        },
        {
            "Preconditions": "Edge condition is prepared",
            "Test Steps": "Execute edge/negative flow",
            "Expected Behaviour": "System handles condition gracefully with expected response",
        },
    ]


def export_jira_csv(plan_data: Dict[str, Any]) -> str:
    rows: List[Dict[str, Any]] = []

    for sprint in plan_data.get("sprints", []):
        sprint_name = sprint.get("sprint", sprint.get("sprint_number", ""))
        stories = sprint.get("stories", [])
        for story in stories:
            rows.append(
                {
                    "Summary": story.get("title", ""),
                    "Issue Type": "Story",
                    "Description": story.get("description", ""),
                    "Story Points": story.get("points", story.get("estimate_points", "")),
                    "Epic Name": story.get("epic", story.get("epic_name", story.get("epic_id", ""))),
                    "Sprint": story.get("sprint", sprint_name),
                }
            )

    df = pd.DataFrame(
        rows,
        columns=["Summary", "Issue Type", "Description", "Story Points", "Epic Name", "Sprint"],
    )
    return df.to_csv(index=False)


def saved_plans_file_path() -> str:
    root_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(root_dir, "saved_plans.json")


def load_saved_plans() -> List[Dict[str, Any]]:
    path = saved_plans_file_path()
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except Exception:
        return []
    return []


def write_saved_plans(items: List[Dict[str, Any]]) -> None:
    path = saved_plans_file_path()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2)


def generate_ai_plan_name(application_idea: str) -> str:
    base_fallback = " ".join(application_idea.strip().split())
    if not base_fallback:
        base_fallback = "Sprint Planner"

    fallback_name = " ".join(base_fallback.split()[:5]).title() + " Plan"

    client = get_openai_client()
    if client is None:
        return fallback_name

    prompt = (
        "Create a concise plan title (3-6 words) for this app idea. "
        "Return title only, no punctuation decorations.\n\n"
        f"Idea: {application_idea}"
    )

    try:
        if hasattr(client, "responses"):
            resp = client.responses.create(
                model="gpt-4o-mini",
                input=[
                    {"role": "system", "content": "You generate short project plan names."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            title = extract_response_text(resp)
        else:
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You generate short project plan names."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            title = str(completion.choices[0].message.content or "").strip()

        title = " ".join(title.split())
        if title:
            return title[:80]
    except Exception:
        pass

    return fallback_name


def unique_plan_name(base_name: str, existing_names: List[str]) -> str:
    name = base_name.strip() or "Sprint Plan"
    if name not in existing_names:
        return name
    counter = 2
    while f"{name} ({counter})" in existing_names:
        counter += 1
    return f"{name} ({counter})"


def save_plan_entry(plan_name: str, application_idea: str, plan: Dict[str, Any]) -> None:
    saved = load_saved_plans()
    saved.append(
        {
            "name": plan_name,
            "application": application_idea,
            "saved_at": datetime.utcnow().isoformat() + "Z",
            "plan": plan,
        }
    )
    write_saved_plans(saved)


def get_saved_plan_by_name(plan_name: str) -> Optional[Dict[str, Any]]:
    for item in load_saved_plans():
        if str(item.get("name", "")) == plan_name:
            return item
    return None


def load_plan_into_session(plan_name: str) -> bool:
    selected_saved = get_saved_plan_by_name(plan_name)
    if not selected_saved or not isinstance(selected_saved.get("plan"), dict):
        return False

    # Copy plan payload so switching between saved plans always refreshes cleanly.
    st.session_state.plan = copy.deepcopy(selected_saved["plan"])
    st.session_state.selected_story_id = None
    st.session_state.loaded_plan_name = plan_name
    return True


def safe_key(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "-", value)


def render_story_tile(
    *,
    story: Dict[str, Any],
    story_id: str,
    story_epic_name: str,
    story_epic_color: str,
    dep_titles: List[str],
    style_key: str,
    button_key: str,
    is_dark_mode: bool,
    user_story_icon_uri: str,
) -> bool:
    tile_label = (
        f"{story.get('title', 'Untitled Story')}{' 🔗' if dep_titles else ''}\n"
        f"{story_epic_name}\n"
        f"{story_id} • {story.get('estimate_points', '?')} pts • {story.get('priority', 'Medium')}"
    )

    story_bg = story_epic_color
    text_color = "#1f2937"
    border_color = "#e2e8f0"
    hover_border = story_epic_color

    container_key = safe_key(style_key)
    with st.container(key=container_key):
        icon_css = ""
        if user_story_icon_uri:
            icon_css = f"""
            .st-key-{container_key} .stButton > button {{
                position: relative;
                padding-left: 72px;
            }}
            .st-key-{container_key} .stButton > button::before {{
                content: "";
                position: absolute;
                left: 14px;
                top: 12px;
                width: 54px;
                height: 54px;
                background-image: url('{user_story_icon_uri}');
                background-size: contain;
                background-repeat: no-repeat;
                background-position: center;
                opacity: 0.95;
            }}
            """

        st.markdown(
            f"""
            <style>
            .st-key-{container_key} .stButton > button {{
                background: {story_bg};
                border-left: 5px solid {story_epic_color};
                border-color: {border_color};
                color: {text_color};
                border-radius: 12px;
                padding: 12px 14px;
                box-shadow: 0 3px 10px rgba(15, 23, 42, 0.04);
                margin-bottom: 10px;
            }}
            .st-key-{container_key} .stButton > button:hover {{
                border-color: {hover_border};
                box-shadow: 0 8px 16px rgba(15, 23, 42, 0.10);
                filter: brightness(0.98);
                transform: translateY(-1px);
            }}
            {icon_css}
            </style>
            """,
            unsafe_allow_html=True,
        )
        return st.button(tile_label, key=button_key, use_container_width=True)


is_dark_mode = False
bg_color = "linear-gradient(to bottom right, #f8fafc, #eef2ff)"
panel_color = "#ffffff"
text_color = "#1e293b"
muted_text_color = "#64748b"
border_color = "#dbe2ea"
input_bg = "#f1f5f9"
chip_bg = "#f3f6fb"
chip_border = "#d9e2ec"
primary_btn = "linear-gradient(135deg, #3b82f6, #6366f1)"
primary_btn_hover = "linear-gradient(135deg, #2563eb, #4f46e5)"
scenario_blue = "#1d4ed8"

theme_css = """
<style>
header[data-testid="stHeader"] {
    background: transparent;
}
[data-testid="stToolbar"],
[data-testid="stStatusWidget"],
[data-testid="stHeaderActionElements"],
#MainMenu,
footer {
    visibility: hidden !important;
    display: none !important;
}
[data-testid="stAppViewContainer"],
[data-testid="stApp"],
.stApp {
    background: __BG_COLOR__;
    color: __TEXT_COLOR__;
}
.block-container {
    padding-top: 1.4rem;
    padding-bottom: 2rem;
    padding-left: 1.5rem;
    padding-right: 1.5rem;
    max-width: 100%;
}
[data-testid="stVerticalBlockBorderWrapper"] {
    background: __PANEL_COLOR__;
    border-color: __BORDER_COLOR__ !important;
}
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] div,
label,
.stCaption,
.stSubheader,
.stMetricLabel,
.stMetricValue,
h1, h2, h3, h4 {
    color: __TEXT_COLOR__;
}
.app-header {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 1rem;
}
.app-header-icon {
    width: 44px;
    height: 44px;
    border-radius: 12px;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: #ffffff;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.15rem;
    box-shadow: 0 8px 18px rgba(99, 102, 241, 0.25);
}
.app-header-title {
    font-size: 2rem;
    font-weight: 700;
    color: #1e293b;
    line-height: 1.1;
}
.section-title {
    font-size: 1.02rem;
    font-weight: 600;
    color: #334155;
    margin-top: 0.2rem;
    margin-bottom: 0.8rem;
}
.section-accent-title {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-top: 0.65rem;
    margin-bottom: 0.9rem;
}
.section-accent-line {
    width: 4px;
    height: 24px;
    border-radius: 999px;
    background: linear-gradient(180deg, #3b82f6, #8b5cf6);
}
.section-accent-text {
    font-size: 1.32rem;
    font-weight: 700;
    letter-spacing: 0.2px;
    color: #1f2a44;
}
.epic-chip {
    border-radius: 10px;
    padding: 0.75rem;
    min-height: 92px;
    border: 1px solid __BORDER_COLOR__;
}
.epic-premium-card {
    border-radius: 14px;
    padding: 16px;
    min-height: 150px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.06);
    position: relative;
    border: 1px solid rgba(15, 23, 42, 0.06);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.epic-premium-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 24px rgba(0,0,0,0.10);
}
.epic-left-strip {
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    width: 4px;
    border-radius: 14px 0 0 14px;
}
.epic-icon-dot {
    width: 30px;
    height: 30px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 10px;
    font-size: 0.9rem;
    font-weight: 700;
    color: #233045;
}
.epic-name {
    font-weight: 700;
    margin-bottom: 0.25rem;
    color: #101828;
}
.epic-summary {
    font-size: 0.85rem;
    color: #334155;
    line-height: 1.25;
}
.story-meta {
    color: __MUTED_TEXT_COLOR__;
    font-size: 0.82rem;
    margin-bottom: 0.25rem;
}
.input-label-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 6px;
    color: #334155;
    font-weight: 500;
    font-size: 1.02rem;
}
.input-label-row img {
    width: 17px;
    height: 17px;
    object-fit: contain;
}
.stTextInput input,
.stTextArea textarea,
.stNumberInput input,
[data-baseweb="select"] > div,
[data-baseweb="input"] > div,
[data-baseweb="base-input"] > div {
    background: __INPUT_BG__ !important;
    color: __TEXT_COLOR__ !important;
    border-color: __BORDER_COLOR__ !important;
}
.stTextArea textarea {
    border-radius: 12px !important;
    border: none !important;
    padding: 12px 16px !important;
}
.stTextArea textarea:focus {
    outline: 2px solid #6366f1 !important;
    box-shadow: none !important;
}
.stTextArea textarea::placeholder {
    color: #94a3b8 !important;
}
.stButton > button[kind="primary"] {
    background: __PRIMARY_BTN__ !important;
    border: none !important;
    color: #ffffff !important;
    border-radius: 12px !important;
    padding: 12px 20px !important;
    font-weight: 600 !important;
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.3) !important;
    transition: transform 0.16s ease, filter 0.16s ease !important;
}
.stButton > button[kind="primary"] * {
    color: #ffffff !important;
}
.stButton > button[kind="primary"]:hover {
    background: __PRIMARY_BTN_HOVER__ !important;
    transform: scale(1.02);
    filter: brightness(1.02);
}
.stButton > button[kind="secondary"] {
    border-radius: 10px;
    border: 1px solid __BORDER_COLOR__;
    padding: 0.7rem 0.75rem;
    text-align: left;
    white-space: pre-wrap;
    line-height: 1.3;
    font-weight: 500;
    background: __PANEL_COLOR__;
    color: __TEXT_COLOR__;
}
.detail-card {
    border: 1px solid __BORDER_COLOR__;
    border-radius: 12px;
    padding: 0.9rem;
    background: __PANEL_COLOR__;
}
.story-detail-line {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.45rem;
    margin-bottom: 0.55rem;
}
.story-detail-title {
    font-size: 1.65rem;
    font-weight: 700;
    margin-right: 0.4rem;
}
.story-detail-chip {
    font-size: 0.85rem;
    color: __TEXT_COLOR__;
    background: __CHIP_BG__;
    border: 1px solid __CHIP_BORDER__;
    border-radius: 999px;
    padding: 0.18rem 0.55rem;
}
.gherkin-compact {
    line-height: 1.25;
    margin-bottom: 0.35rem;
}
.detail-section-heading {
    font-weight: 700;
    text-decoration: underline;
    margin-top: 0.45rem;
    margin-bottom: 0.25rem;
}
.scenario-label {
    color: __SCENARIO_BLUE__;
    font-weight: 700;
}
.sprint-title-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}
.sprint-title-left {
    display: flex;
    align-items: center;
    gap: 10px;
}
.sprint-icon-circle {
    width: 28px;
    height: 28px;
    border-radius: 9px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.82rem;
    color: #1f2a44;
    background: rgba(59,130,246,0.15);
}
.sprint-title-text {
    font-size: 1.8rem;
    font-weight: 700;
    line-height: 1.05;
    color: #1f2a44;
}
.sprint-points-badge {
    font-size: 0.82rem;
    font-weight: 700;
    border-radius: 999px;
    padding: 0.28rem 0.62rem;
    background: #e2e8f0;
    color: #334155;
}
.sprint-goal-text {
    font-size: 0.86rem;
    color: #64748b;
    line-height: 1.55;
    margin-top: 0.35rem;
    margin-bottom: 0.8rem;
}
.sprint-story-divider {
    border-top: 1px solid #e2e8f0;
    margin-top: 6px;
    margin-bottom: 10px;
}
.sprint-empty-text {
    color: #94a3b8;
    font-size: 0.9rem;
}
.st-key-sprints_grid [data-testid="stVerticalBlock"] {
    gap: 20px;
}
.st-key-plan_inputs_card {
    background: #ffffff !important;
    border: 1px solid #d7e0ec !important;
    border-radius: 16px !important;
    box-shadow: 0 26px 54px rgba(15, 23, 42, 0.24), 0 10px 24px rgba(15, 23, 42, 0.16) !important;
    padding: 14px 16px 12px 16px !important;
    margin-bottom: 18px;
}
.st-key-plan_inputs_card [data-testid="stVerticalBlock"] {
    gap: 0.55rem;
}
.st-key-velocity_card,
.st-key-sprint_count_card {
    background: #f8fafc;
    border-radius: 12px;
    padding: 10px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 2px 8px rgba(15,23,42,0.06), inset 0 1px 2px rgba(0,0,0,0.04);
}
.st-key-export_row .stSelectbox [data-baseweb="select"] > div {
    border-radius: 10px !important;
    background: #f8fafc !important;
    border: 1px solid #dbe4f1 !important;
    box-shadow: 0 2px 8px rgba(15,23,42,0.05) !important;
}
.st-key-export_row .stButton > button[kind="secondary"] {
    border: 1px solid #93c5fd !important;
    color: #1e3a8a !important;
    background: #dbeafe !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 8px rgba(15,23,42,0.05) !important;
}
.st-key-export_row .stButton > button[kind="secondary"]:hover {
    background: #bfdbfe !important;
    color: #1e3a8a !important;
}
.st-key-plan_inputs_card .stButton > button[kind="secondary"] {
    border: 1px solid #93c5fd !important;
    color: #1e3a8a !important;
    background: #dbeafe !important;
}
.st-key-plan_inputs_card .stButton > button[kind="secondary"]:hover {
    background: #bfdbfe !important;
    color: #1e3a8a !important;
}
.st-key-export_row [data-testid="stHorizontalBlock"] {
    justify-content: flex-end;
    align-items: center;
    gap: 12px;
}
.st-key-header_help_btn [data-testid="stVerticalBlock"] {
    align-items: flex-end;
}
.st-key-header_help_btn .stButton > button {
    min-width: 78px;
    padding: 0.25rem 0.7rem !important;
    margin-top: 0.15rem;
}
</style>
"""
theme_css = (
    theme_css.replace("__BG_COLOR__", bg_color)
    .replace("__PANEL_COLOR__", panel_color)
    .replace("__TEXT_COLOR__", text_color)
    .replace("__MUTED_TEXT_COLOR__", muted_text_color)
    .replace("__BORDER_COLOR__", border_color)
    .replace("__INPUT_BG__", input_bg)
    .replace("__CHIP_BG__", chip_bg)
    .replace("__CHIP_BORDER__", chip_border)
    .replace("__PRIMARY_BTN__", primary_btn)
    .replace("__PRIMARY_BTN_HOVER__", primary_btn_hover)
    .replace("__SCENARIO_BLUE__", scenario_blue)
)
st.markdown(theme_css, unsafe_allow_html=True)

@st.dialog("How to Use AI Sprint Planner")
def show_help_dialog() -> None:
    st.markdown(
        """
AI Sprint Planner helps you quickly convert an application idea into structured Agile deliverables like epics, user stories, and sprint plans.

**Step 1: Enter Application Details**
Provide a clear description of the application or feature you want to build. The more specific you are, the better the generated plan.

**Step 2: Configure Planning Inputs**
Set your Sprint Velocity (story points per sprint) and Number of Sprints based on your team's capacity.

**Step 3: Generate Plan**
Click Generate Plan to automatically create:

- Epics grouped by functionality
- User stories with descriptions and acceptance criteria
- Sprint-wise distribution based on capacity

Step 4: Explore Details
Click on any story to view detailed information including acceptance criteria and technical notes.

Step 5: Export
Use the Export To option to download your plan in a format compatible with tools like Jira.

**Tip:** Provide domain-specific inputs (for example, payment gateway with fraud detection) for more accurate results.
        """
    )
    if st.button("Close", key="help_close_btn"):
        st.rerun()


with st.container(key="header_bar"):
    header_left, header_right = st.columns([11, 1])
    with header_left:
        st.markdown(
            """
            <div class="app-header">
                <div class="app-header-icon">✦</div>
                <div class="app-header-title">AI Sprint Planner</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with header_right:
        with st.container(key="header_help_btn"):
            if st.button("Help", key="open_help_dialog"):
                st.session_state.show_help_dialog = True

if "plan" not in st.session_state:
    st.session_state.plan = None
if "selected_story_id" not in st.session_state:
    st.session_state.selected_story_id = None
if "clipboard_copy_pending" not in st.session_state:
    st.session_state.clipboard_copy_pending = False
if "clipboard_copy_text" not in st.session_state:
    st.session_state.clipboard_copy_text = ""
if "clipboard_copy_html" not in st.session_state:
    st.session_state.clipboard_copy_html = ""
if "clipboard_copy_nonce" not in st.session_state:
    st.session_state.clipboard_copy_nonce = 0
if "jira_csv_data" not in st.session_state:
    st.session_state.jira_csv_data = ""
if "loaded_plan_name" not in st.session_state:
    st.session_state.loaded_plan_name = ""
if "load_plan_choice" not in st.session_state:
    st.session_state.load_plan_choice = "-- Select saved plan --"
if "show_help_dialog" not in st.session_state:
    st.session_state.show_help_dialog = False

if st.session_state.show_help_dialog:
    st.session_state.show_help_dialog = False
    show_help_dialog()

root_dir = os.path.dirname(os.path.abspath(__file__))
application_icon_uri = load_icon_data_uri(
    [
        os.path.join(root_dir, "Application.png"),
        os.path.join(root_dir, "application.png"),
    ]
)
velocity_icon_uri = load_icon_data_uri(
    [
        os.path.join(root_dir, "Velocity.png"),
        os.path.join(root_dir, "Velocity.jpg"),
        os.path.join(root_dir, "velocity.png"),
        os.path.join(root_dir, "velocity.jpg"),
    ]
)
sprints_icon_uri = load_icon_data_uri(
    [
        os.path.join(root_dir, "Numberofsprints.png"),
        os.path.join(root_dir, "NumberOfSprints.png"),
        os.path.join(root_dir, "numberofsprints.png"),
    ]
)


# Top input section
with st.container(key="plan_inputs_card"):
    st.markdown('<div class="section-title">Plan Inputs</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns([3.0, 1.2, 1.2, 1.8])
    with c1:
        st.markdown(
            f"<div class='input-label-row'>{('<img src=' + chr(34) + application_icon_uri + chr(34) + ' />') if application_icon_uri else ''}<span>Application/Functionality</span></div>",
            unsafe_allow_html=True,
        )
        application_idea = st.text_area(
            "Application/Functionality",
            placeholder="Describe your app idea, target users, and core value proposition.",
            height=120,
            max_chars=10000,
            label_visibility="collapsed",
        )
    with c2:
        with st.container(key="velocity_card"):
            st.markdown(
                f"<div class='input-label-row'>{('<img src=' + chr(34) + velocity_icon_uri + chr(34) + ' />') if velocity_icon_uri else ''}<span>Sprint Velocity</span></div>",
                unsafe_allow_html=True,
            )
            sprint_velocity = st.number_input(
                "Sprint Velocity",
                min_value=1,
                max_value=200,
                value=20,
                step=1,
                label_visibility="collapsed",
            )
    with c3:
        with st.container(key="sprint_count_card"):
            st.markdown(
                f"<div class='input-label-row'>{('<img src=' + chr(34) + sprints_icon_uri + chr(34) + ' />') if sprints_icon_uri else ''}<span>Number of Sprints</span></div>",
                unsafe_allow_html=True,
            )
            sprint_count = st.number_input(
                "Number of Sprints",
                min_value=1,
                max_value=6,
                value=4,
                step=1,
                label_visibility="collapsed",
            )
    with c4:
        st.caption(" ")
        generate_clicked = st.button("Generate Plan", type="primary", use_container_width=True)
        saved_names = [str(item.get("name", "")) for item in load_saved_plans() if str(item.get("name", "")).strip()]

        save_clicked = st.button(
            "Save Plan",
            use_container_width=True,
            disabled=st.session_state.plan is None,
        )
        if save_clicked and st.session_state.plan is not None:
            base_name = generate_ai_plan_name(application_idea)
            final_name = unique_plan_name(base_name, saved_names)
            save_plan_entry(final_name, application_idea, st.session_state.plan)
            st.success(f"Saved plan as: {final_name}")
            saved_names = [
                str(item.get("name", ""))
                for item in load_saved_plans()
                if str(item.get("name", "")).strip()
            ]

        load_options = ["-- Select saved plan --"] + saved_names
        if st.session_state.load_plan_choice not in load_options:
            st.session_state.load_plan_choice = "-- Select saved plan --"

        load_choice = st.selectbox(
            "Load Plan",
            options=load_options,
            key="load_plan_choice",
        )
        if load_choice != "-- Select saved plan --" and load_choice != st.session_state.loaded_plan_name:
            if load_plan_into_session(load_choice):
                st.success(f"Loaded plan: {load_choice}")

if generate_clicked:
    if not application_idea.strip():
        st.warning("Please enter an application idea first.")
    else:
        progress_holder = st.empty()
        status_holder = st.empty()

        try:
            progress_holder.progress(0, text="Generating sprint plan... 0%")

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    generate_plan,
                    application_idea.strip(),
                    int(sprint_velocity),
                    int(sprint_count),
                )

                start_time = time.perf_counter()
                synthetic_progress = 0
                while not future.done():
                    elapsed = time.perf_counter() - start_time

                    # Milestone feel: move to 33%, pause, then 69%, pause, then glide upward.
                    if elapsed < 1.4:
                        target = int((elapsed / 1.4) * 33)
                    elif elapsed < 2.2:
                        target = 33
                    elif elapsed < 3.6:
                        target = 33 + int(((elapsed - 2.2) / 1.4) * (69 - 33))
                    elif elapsed < 4.4:
                        target = 69
                    else:
                        tail = int((elapsed - 4.4) * 4)
                        target = min(92, 69 + tail)

                    synthetic_progress = max(synthetic_progress, target)
                    progress_holder.progress(
                        synthetic_progress,
                        text=f"Generating sprint plan... {synthetic_progress}%",
                    )
                    time.sleep(0.10)

                plan = future.result()

            progress_holder.progress(100, text="Generating sprint plan... 100%")
            status_holder.success("Sprint plan generated")
            st.session_state.plan = plan
            st.session_state.selected_story_id = None
            st.session_state.loaded_plan_name = ""
        except Exception as exc:
            progress_holder.empty()
            st.error(f"Failed to generate plan: {exc}")

plan_data = st.session_state.plan

if plan_data:
    epic_icon_uri = load_icon_data_uri(
        [
            os.path.join(root_dir, "Epic.png"),
            os.path.join(root_dir, "epic.png"),
        ]
    )
    user_story_icon_uri = load_icon_data_uri(
        [
            os.path.join(root_dir, "UserStory.png"),
            os.path.join(root_dir, "Userstory.png"),
            os.path.join(root_dir, "userstory.png"),
        ]
    )

    epics: List[Dict[str, Any]] = plan_data.get("epics", [])
    sprints: List[Dict[str, Any]] = plan_data.get("sprints", [])
    unallocated_stories = plan_data.get("unallocated_stories", [])
    all_stories = flatten_stories(plan_data)
    for extra_story in unallocated_stories:
        extra_id = str(extra_story.get("id", "")).strip()
        if not extra_id:
            continue
        all_stories[extra_id] = {
            **extra_story,
            "sprint_number": "Extra",
            "sprint_goal": "Deferred due to current capacity/dependency limits.",
        }

    with st.container(key="export_row"):
        export_spacer, export_c1, export_c2, export_c3 = st.columns([4.8, 0.95, 1.9, 0.95])
        with export_spacer:
            st.write("")
        with export_c1:
            st.markdown(
                "<div style='font-weight:600;color:#334155;text-align:right;padding-right:10px;'>Export To</div>",
                unsafe_allow_html=True,
            )
        with export_c2:
            export_target = st.selectbox("Export To", options=["JIRA"], key="export_target", label_visibility="collapsed")
        with export_c3:
            go_clicked = st.button("Go", use_container_width=True)

            if st.session_state.jira_csv_data:
                st.download_button(
                    label="Download JIRA CSV",
                    data=st.session_state.jira_csv_data,
                    file_name="jira_export.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="download_jira_csv_button",
                )

    if go_clicked:
        if not plan_data:
            st.info("Generate a plan before exporting")
        elif export_target == "JIRA":
            st.session_state.jira_csv_data = export_jira_csv(plan_data)

    # Epics horizontal display
    st.markdown(
        '<div class="section-accent-title"><span class="section-accent-line"></span><span class="section-accent-text">EPICS</span></div>',
        unsafe_allow_html=True,
    )
    if epics:
        epic_cols = st.columns(len(epics))
        for idx, epic in enumerate(epics):
            with epic_cols[idx]:
                bg = epic_color(idx)
                strip = hex_to_rgba(bg, 0.95).replace("0.95", "1")
                icon_bg = hex_to_rgba(bg, 0.58)
                st.markdown(
                    f"""
                    <div class="epic-premium-card" style="background:{bg};">
                      <span class="epic-left-strip" style="background:{strip};"></span>
                                            <span class="epic-icon-dot" style="background:{icon_bg};">{'<img src="' + epic_icon_uri + '" style="width:16px;height:16px;object-fit:contain;" />' if epic_icon_uri else '□'}</span>
                      <div class="epic-name">{epic.get("id", "")}: {epic.get("name", "")}</div>
                      <div class="epic-summary">{epic.get("summary", "")}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        st.info("No epics found in generated plan.")

    # Sprint columns with story cards
    st.markdown(
        '<div class="section-accent-title"><span class="section-accent-line"></span><span class="section-accent-text">SPRINTS</span></div>',
        unsafe_allow_html=True,
    )
    if sprints:
        sprint_palette = ["#3b82f6", "#22c55e", "#ec4899", "#a855f7", "#06b6d4", "#f59e0b"]
        with st.container(key="sprints_grid"):
            columns_per_row = 4
            for row_start in range(0, len(sprints), columns_per_row):
                sprint_cols = st.columns(columns_per_row)
                for col_idx in range(columns_per_row):
                    sprint_idx = row_start + col_idx
                    if sprint_idx >= len(sprints):
                        continue
                    sprint = sprints[sprint_idx]
                    accent = sprint_palette[sprint_idx % len(sprint_palette)]

                    with sprint_cols[col_idx]:
                        sprint_key = safe_key(f"sprint-card-{sprint_idx}")
                        st.markdown(
                            f"""
                            <style>
                            .st-key-{sprint_key} [data-testid="stVerticalBlockBorderWrapper"] {{
                                border-radius: 16px !important;
                                border: 1px solid rgba(148,163,184,0.35) !important;
                                border-top: 4px solid {accent} !important;
                                box-shadow: 0 10px 25px rgba(0,0,0,0.08) !important;
                                padding: 12px !important;
                                min-height: 420px;
                                background: #ffffff !important;
                            }}
                            </style>
                            """,
                            unsafe_allow_html=True,
                        )

                        with st.container(border=True, key=sprint_key):
                            sprint_no = sprint.get("sprint_number", sprint_idx + 1)
                            stories = sprint.get("stories", [])
                            total_points = sprint_point_total(sprint)

                            st.markdown(
                                f'''
                                <div class="sprint-title-row">
                                    <div class="sprint-title-left">
                                        <span class="sprint-icon-circle">◷</span>
                                        <span class="sprint-title-text">Sprint {sprint_no}</span>
                                    </div>
                                    <span class="sprint-points-badge">{total_points} / {int(sprint_velocity)} pts</span>
                                </div>
                                ''',
                                unsafe_allow_html=True,
                            )
                            st.markdown(
                                f'<div class="sprint-goal-text">{sprint.get("goal", "")}</div>',
                                unsafe_allow_html=True,
                            )
                            st.markdown('<div class="sprint-story-divider"></div>', unsafe_allow_html=True)

                            if not stories:
                                st.markdown('<div class="sprint-empty-text">No stories</div>', unsafe_allow_html=True)
                                continue

                            for story_idx, story in enumerate(stories):
                                story_id = story.get("id", f"story-{sprint_idx}-{story_idx}")
                                story_epic_id = str(story.get("epic_id", ""))
                                story_epic_name = str(story.get("epic_name", story_epic_id or "Unknown Epic"))
                                story_epic_color = str(story.get("epic_color", epic_color(story_idx)))
                                dep_ids = [str(dep) for dep in story.get("dependencies", [])]
                                dep_titles: List[str] = []
                                for dep_id in dep_ids:
                                    dep_title = all_stories.get(dep_id, {}).get("title", dep_id)
                                    dep_titles.append(f"{dep_id}: {dep_title}")

                                if render_story_tile(
                                    story=story,
                                    story_id=str(story_id),
                                    story_epic_name=story_epic_name,
                                    story_epic_color=story_epic_color,
                                    dep_titles=dep_titles,
                                    style_key=f"story-tile-style-{story_id}-{sprint_idx}-{story_idx}",
                                    button_key=f"story-tile-{story_id}-{sprint_idx}-{story_idx}",
                                    is_dark_mode=is_dark_mode,
                                    user_story_icon_uri=user_story_icon_uri,
                                ):
                                    st.session_state.selected_story_id = str(story_id)
    else:
        st.info("No sprints found in generated plan.")

    if unallocated_stories:
        st.markdown('<div class="section-title">EXTRA STORIES</div>', unsafe_allow_html=True)
        st.caption("Stories that could not fit within sprint capacity/dependency limits in this run.")
        with st.container(border=True):
            extra_columns_per_row = 4
            for row_start in range(0, len(unallocated_stories), extra_columns_per_row):
                extra_cols = st.columns(extra_columns_per_row)
                for col_idx in range(extra_columns_per_row):
                    story_idx = row_start + col_idx
                    if story_idx >= len(unallocated_stories):
                        continue

                    story = unallocated_stories[story_idx]
                    story_id = story.get("id", f"extra-{story_idx}")
                    story_epic_id = str(story.get("epic_id", ""))
                    story_epic_name = str(story.get("epic_name", story_epic_id or "Unknown Epic"))
                    story_epic_color = str(story.get("epic_color", epic_color(story_idx)))
                    dep_ids = [str(dep) for dep in story.get("dependencies", [])]
                    dep_titles: List[str] = []
                    for dep_id in dep_ids:
                        dep_title = all_stories.get(dep_id, {}).get("title", dep_id)
                        dep_titles.append(f"{dep_id}: {dep_title}")

                    with extra_cols[col_idx]:
                        if render_story_tile(
                            story=story,
                            story_id=str(story_id),
                            story_epic_name=story_epic_name,
                            story_epic_color=story_epic_color,
                            dep_titles=dep_titles,
                            style_key=f"extra-story-tile-style-{story_id}-{story_idx}",
                            button_key=f"extra-story-tile-{story_id}-{story_idx}",
                            is_dark_mode=is_dark_mode,
                            user_story_icon_uri=user_story_icon_uri,
                        ):
                            st.session_state.selected_story_id = str(story_id)

    # Story details section
    st.markdown('<div class="section-title">STORY DETAILS</div>', unsafe_allow_html=True)
    selected_id = st.session_state.selected_story_id

    if selected_id and selected_id in all_stories:
        story = all_stories[selected_id]
        with st.container(border=True):
            h1, h2 = st.columns([0.94, 0.06])
            with h2:
                if st.button("📋", key=f"copy-story-{selected_id}", help="Copy story details", use_container_width=True):
                    st.session_state.clipboard_copy_text = build_story_clipboard_text(story)
                    st.session_state.clipboard_copy_html = build_story_clipboard_html(story)
                    st.session_state.clipboard_copy_pending = True
                    st.session_state.clipboard_copy_nonce += 1

            detail_title = html.escape(f"{story.get('id', '')}: {story.get('title', '')}")
            detail_sprint = html.escape(str(story.get("sprint_number", "-")))
            detail_priority = html.escape(str(story.get("priority", "-")))
            detail_points = html.escape(str(story.get("estimate_points", "-")))
            with h1:
                st.markdown(
                    f'''
                    <div class="story-detail-line">
                        <span class="story-detail-title">{detail_title}</span>
                        <span class="story-detail-chip">Sprint: {detail_sprint}</span>
                        <span class="story-detail-chip">Priority: {detail_priority}</span>
                        <span class="story-detail-chip">Points: {detail_points}</span>
                    </div>
                    ''',
                    unsafe_allow_html=True,
                )

            st.markdown('<div class="detail-section-heading">Description</div>', unsafe_allow_html=True)
            description = str(story.get("description", "")).strip()
            if description:
                st.markdown(description)
            else:
                st.write("As a <type of user>")
                st.write("I want to <perform some action>")
                st.write("So that <achieve some goal or value>")

            st.markdown('<div class="detail-section-heading">Acceptance Criteria</div>', unsafe_allow_html=True)
            criteria = story.get("acceptance_criteria", [])
            if criteria:
                for item in criteria:
                    formatted = format_gherkin_text(str(item))
                    if formatted:
                        st.markdown(f'<div class="gherkin-compact">{formatted}</div>', unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div class="gherkin-compact"><strong>Background:</strong><br><strong>Given</strong> &lt;common precondition 1&gt;<br><strong>And</strong> &lt;common precondition 2&gt;</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    '<div class="gherkin-compact"><span class="scenario-label">Scenario:</span> &lt;Primary Scenario Name&gt; -<br><strong>Given</strong> &lt;initial context&gt;<br><strong>And</strong> &lt;additional condition&gt;<br><strong>When</strong> &lt;user performs an action&gt;<br><strong>And</strong> &lt;another action if needed&gt;<br><strong>Then</strong> &lt;expected outcome&gt;<br><strong>And</strong> &lt;additional outcome&gt;</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    '<div class="gherkin-compact"><span class="scenario-label">Scenario:</span> &lt;Alternate / Edge Case Scenario&gt; -<br><strong>Given</strong> &lt;initial context&gt;<br><strong>When</strong> &lt;different action or edge condition&gt;<br><strong>Then</strong> &lt;expected result&gt;</div>',
                    unsafe_allow_html=True,
                )

            st.markdown('<div class="detail-section-heading">Technical Specifications</div>', unsafe_allow_html=True)
            technical_specs = story.get("technical_specifications", [])
            if technical_specs:
                for idx, spec in enumerate(technical_specs, start=1):
                    st.write(f"{idx}) {spec}")
            else:
                st.write("1)")
                st.write("2)")

            st.markdown('<div class="detail-section-heading">Non Functional Requirements</div>', unsafe_allow_html=True)
            nfrs = story.get("non_functional_requirements", [])
            if nfrs:
                for idx, nfr in enumerate(nfrs, start=1):
                    st.write(f"{idx}) {nfr}")
            else:
                st.write("1)")
                st.write("2)")

            st.markdown('<div class="detail-section-heading">Dependencies</div>', unsafe_allow_html=True)
            deps = story.get("dependencies", [])
            if deps:
                st.write(", ".join(str(dep) for dep in deps))
            else:
                st.write("None")

            st.markdown('<div class="detail-section-heading">Test Cases</div>', unsafe_allow_html=True)
            test_rows = build_test_cases_rows(story)
            st.table(test_rows)
    else:
        st.write("Click any story card in a sprint column to see details here.")

    if st.session_state.clipboard_copy_pending and st.session_state.clipboard_copy_text:
        plain_json = json.dumps(st.session_state.clipboard_copy_text)
        html_json = json.dumps(st.session_state.clipboard_copy_html or st.session_state.clipboard_copy_text)
        copy_expression = f"""
            (async () => {{
                const plainText = {plain_json};
                const richHtml = {html_json};
                if (navigator.clipboard && window.ClipboardItem) {{
                    const item = new ClipboardItem({{
                        'text/plain': new Blob([plainText], {{ type: 'text/plain' }}),
                        'text/html': new Blob([richHtml], {{ type: 'text/html' }})
                    }});
                    await navigator.clipboard.write([item]);
                    return 'ok';
                }}
                await navigator.clipboard.writeText(plainText);
                return 'ok';
            }})().catch((e) => 'error:' + (e && e.message ? e.message : e))
        """
        copy_result = eval_javascript(
            copy_expression,
            key=f"story-clipboard-copy-{st.session_state.clipboard_copy_nonce}",
        )
        if isinstance(copy_result, str):
            if copy_result == "ok":
                st.toast("Story details copied to clipboard")
            else:
                st.warning("Clipboard copy failed in browser. You can still select and copy the text manually.")
            st.session_state.clipboard_copy_pending = False
else:
    pass