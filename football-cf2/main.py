import os
import json
import re
import logging
import functions_framework
from flask import render_template, request, jsonify
from google import genai
from google.genai.types import (
    Tool, GoogleSearch, GenerateContentConfig,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Vertex AI ─────────────────────────────────────────────────────────────────
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "your-gcp-project-id")
LOCATION   = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
MODEL_ID   = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

os.environ["GOOGLE_CLOUD_PROJECT"]      = PROJECT_ID
os.environ["GOOGLE_CLOUD_LOCATION"]     = LOCATION
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"

client = genai.Client()

# Create the drop down
LEAGUE_ONE_TEAMS = [
    "Birmingham City", "Blackpool", "Bolton Wanderers", "Bristol Rovers",
    "Burton Albion", "Cambridge United", "Charlton Athletic", "Crawley Town",
    "Exeter City", "Huddersfield Town", "Leyton Orient", "Lincoln City",
    "Mansfield Town", "Northampton Town", "Peterborough United",
    "Reading", "Rotherham United", "Shrewsbury Town", "Stevenage",
    "Stockport County", "Wigan Athletic", "Wrexham", "Wycombe Wanderers",
]

TEAM_REPORT_SECTIONS = [
    {
        "id": "recent_results", "title": "Recent Results", "icon": "⚽",
        "prompt": (
            "Find the last 10 completed match results for {club} in the current season. "
            "Return a JSON array where each item has: date (YYYY-MM-DD), competition, "
            "opponent, score (e.g. '2-1'), venue ('Home'/'Away'), result ('Win'/'Draw'/'Loss'). "
            "Order most-recent first."
        ),
    },
    {
        "id": "injuries", "title": "Injuries & Availability", "icon": "🏥",
        "prompt": (
            "Search for the latest injury news for {club} football club right now. "
            "Return a JSON array where each item has: player_name, injury_type, "
            "expected_return (e.g. '2-3 weeks' or 'Unknown'), status ('Doubtful'/'Out'/'Returning'). "
            "If no injuries reported, return []."
        ),
    },
    {
        "id": "suspensions", "title": "Suspensions", "icon": "🟥",
        "prompt": (
            "Search for any current player suspensions for {club} football club. "
            "Return a JSON array where each item has: player_name, reason, matches_remaining (number or null). "
            "If none, return []."
        ),
    },
    {
        "id": "style_of_play", "title": "Style of Play", "icon": "🎯",
        "prompt": (
            "Describe {club}'s current style of play and tactical setup this season based on recent matches. "
            "Return a JSON object with: formation (e.g. '4-3-3'), press_intensity ('High'/'Medium'/'Low'), "
            "attacking_style (short text), defensive_shape (short text), "
            "key_strengths (array of up to 4 strings), key_weaknesses (array of up to 4 strings), "
            "summary (2-3 sentence paragraph)."
        ),
    },
    {
        "id": "top_scorers", "title": "Top Scorers", "icon": "🏆",
        "prompt": (
            "Find the top goal scorers for {club} in the current season. "
            "Return a JSON array (up to 6 players) where each item has: "
            "player_name, goals (number), assists (number), appearances (number). "
            "Order by goals descending."
        ),
    },
    {
        "id": "next_fixture", "title": "Next Fixture", "icon": "📅",
        "prompt": (
            "Find {club}'s next upcoming fixture. "
            "Return a JSON object with: date (YYYY-MM-DD), time (HH:MM or 'TBC'), "
            "opponent, venue ('Home'/'Away'), competition, stadium."
        ),
    },
]

ALLOWED_SITES = [
    "bbc.co.uk", "skysports.com", "efl.com", "espn.com",
    "flashscore.com", "sofascore.com", "soccerway.com",
    "transfermarkt.com", "whoscored.com", "theguardian.com",
]

# ── Agent ─────────────────────────────────────────────────────────────────────
def call_agent(prompt: str):
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
        config=GenerateContentConfig(
            tools=[Tool(google_search=GoogleSearch())],
            temperature=1.0,
        ),
    )
    raw = re.sub(r"^```(?:json)?\s*", "", response.text.strip())
    raw = re.sub(r"\s*```$", "", raw.strip())
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"(\[.*\]|\{.*\})", raw, re.DOTALL)
        try:    data = json.loads(m.group()) if m else None
        except: data = None

    sources = []
    try:
        meta = response.candidates[0].grounding_metadata
        if meta and meta.grounding_chunks:
            for chunk in meta.grounding_chunks:
                if chunk.web:
                    domain = chunk.web.uri.split("/")[2].replace("www.", "")
                    if any(s in domain for s in ALLOWED_SITES):
                        sources.append({"title": chunk.web.title, "url": chunk.web.uri})
    except Exception:
        pass
    return data, sources


# ── Entrypoint ────────────────────────────────────────────────────────────────
@functions_framework.http
def scoutiq(request):
    """
    All routing is done via the `page` query parameter because Cloud Functions
    receive every request at the same URL — path-based routing doesn't work.

      ?            → homepage
      ?page=report&club=X  → team report page
      ?page=api&club=X&section=Y  → JSON API (called by JS)
      ?page=referee / stadium / h2h  → coming soon
    """
    page = request.args.get("page", "")

    # ── Homepage
    if not page or page == "home":
        return render_template("index.html", teams=LEAGUE_ONE_TEAMS)

    # ── Team report page
    if page == "report":
        club = request.args.get("club", "").strip()
        if not club or club not in LEAGUE_ONE_TEAMS:
            return render_template("index.html", teams=LEAGUE_ONE_TEAMS, error="Please select a valid team.")
        return render_template(
            "report_team.html",
            club=club,
            sections=TEAM_REPORT_SECTIONS,
            teams=LEAGUE_ONE_TEAMS,
        )

    # ── JSON API — called by the page JS for each section
    if page == "api":
        club       = request.args.get("club", "").strip()
        section_id = request.args.get("section", "").strip()

        if not club or not section_id:
            return jsonify({"error": "Missing parameters"}), 400

        section = next((s for s in TEAM_REPORT_SECTIONS if s["id"] == section_id), None)
        if not section:
            return jsonify({"error": "Unknown section"}), 404

        prompt = (
            section["prompt"].format(club=club)
            + "\n\nReturn ONLY raw JSON — no markdown, no backticks, no explanation."
            + f"\n\nPrefer sources from: {', '.join(ALLOWED_SITES)}"
        )
        try:
            data, sources = call_agent(prompt)
            return jsonify({"section": section_id, "data": data, "sources": sources})
        except Exception as e:
            logger.error("Section %s error: %s", section_id, e)
            return jsonify({"error": str(e)}), 500

    # ── Coming soon pages
    labels = {"referee": "Referee", "stadium": "Stadium", "h2h": "Head to Head", "team": "Team"}
    if page in labels:
        if page == "team":
            return render_template("index.html", teams=LEAGUE_ONE_TEAMS)
        return render_template("coming_soon.html", report_type=labels[page], teams=LEAGUE_ONE_TEAMS)

    return "<h1>404</h1>", 404
