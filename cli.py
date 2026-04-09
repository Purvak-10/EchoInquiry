#!/usr/bin/env python3
"""
Research Agent — Rich Terminal CLI
====================================
Run:
    python cli.py                          # interactive prompt
    python cli.py "who created the atom bomb"   # pass query directly
    python cli.py scheduler start          # start 30-day scheduler
    python cli.py scheduler stop           # stop scheduler
    python cli.py scheduler status         # check scheduler status
    python cli.py scheduler check-now      # manually trigger recheck

Features
--------
• Live pipeline stream — every node printed as it completes
• Full research report printed in sections:
    - Executive Summary
    - Detailed Research Sections
    - Hypotheses (tabular)
    - Key Conclusions
    - Contradictions (tabular, colour-coded severity)
    - Research Gaps
    - All Sources / Citations (tabular)
    - Confidence Score
    - Follow-up Recommendations
• Email prompt offered after EVERY pipeline step and at the end
• Follow-up Q&A chat loop
• Save report as JSON + plain-text
• 30-day scheduler for source rechecking
"""

# Load .env FIRST — before any other project imports pick up config

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='transformers')


from dotenv import load_dotenv
load_dotenv()

import asyncio
import json
import logging
import os
import sys
import textwrap
import threading
import time
import uuid
from datetime import datetime
from typing import Optional

# ── Silence noisy library loggers so only WARNING+ reaches the terminal ──────
logging.basicConfig(level=logging.WARNING)
for _noisy in [
    "httpx", "httpcore", "urllib3", "sentence_transformers",
    "huggingface_hub", "transformers", "pinecone", "boto3",
    "botocore", "s3transfer", "memory.vector_store",
    "langfuse", "langchain", "langgraph",
]:
    logging.getLogger(_noisy).setLevel(logging.ERROR)

# ══════════════════════════════════════════════════════════════════════════════
# ANSI colour helpers
# ══════════════════════════════════════════════════════════════════════════════
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
ITALIC = "\033[3m"
UL     = "\033[4m"
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
MAGENTA= "\033[95m"
CYAN   = "\033[96m"
WHITE  = "\033[97m"
BG_DARK= "\033[40m"

def clr(text, *codes):
    return "".join(codes) + str(text) + RESET

def hr(char="═", width=76, colour=CYAN):
    print(clr(char * width, colour))

def hr2(char="─", width=76, colour=BLUE):
    print(clr(char * width, colour))

def banner_box(text, width=76):
    inner = width - 2
    print(clr("╔" + "═" * inner + "╗", CYAN + BOLD))
    pad  = (inner - len(text)) // 2
    line = " " * pad + text + " " * (inner - pad - len(text))
    print(clr("║", CYAN + BOLD) + clr(line, WHITE + BOLD) + clr("║", CYAN + BOLD))
    print(clr("╚" + "═" * inner + "╝", CYAN + BOLD))

def section_header(text, char="▶", colour=BLUE + BOLD):
    print()
    print(clr(f" {char} {text} ", colour + UL))
    hr2(width=76)

def wrap_print(text, width=72, indent=4, colour=WHITE):
    prefix = " " * indent
    for para in str(text).split("\n"):
        para = para.strip()
        if not para:
            print()
            continue
        for line in textwrap.wrap(para, width):
            print(prefix + clr(line, colour))


# ══════════════════════════════════════════════════════════════════════════════
# Spinner
# ══════════════════════════════════════════════════════════════════════════════
class Spinner:
    FRAMES = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]

    def __init__(self, msg=""):
        self.msg    = msg
        self._stop  = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)

    def _spin(self):
        i = 0
        while not self._stop.is_set():
            frame = self.FRAMES[i % len(self.FRAMES)]
            sys.stdout.write(f"\r  {clr(frame, CYAN + BOLD)}  {self.msg}   ")
            sys.stdout.flush()
            time.sleep(0.09)
            i += 1

    def start(self):
        self._thread.start()
        return self

    def stop(self, ok=True, done_msg=""):
        self._stop.set()
        self._thread.join()
        icon = clr("✓", GREEN + BOLD) if ok else clr("✗", RED + BOLD)
        label = done_msg or self.msg
        sys.stdout.write(f"\r  {icon}  {label:<68}\n")
        sys.stdout.flush()


# ══════════════════════════════════════════════════════════════════════════════
# Table renderer
# ══════════════════════════════════════════════════════════════════════════════
def print_table(headers: list, rows: list, col_colours: dict = None, max_col: int = 30):
    """
    Print a simple aligned table to the terminal.
    col_colours: {header_name: ansi_code}
    """
    if not rows:
        print(clr("  (none)", DIM))
        return

    col_colours = col_colours or {}
    col_widths  = {h: len(h) for h in headers}
    for row in rows:
        for h in headers:
            val = str(row.get(h, ""))
            col_widths[h] = min(max_col, max(col_widths[h], len(val)))

    def fmt_cell(val, h):
        s = str(val)
        if len(s) > max_col:
            s = s[:max_col - 1] + "…"
        return s.ljust(col_widths[h])

    # Header row
    header_line = "  │ " + " │ ".join(
        clr(fmt_cell(h, h), BOLD + YELLOW) for h in headers
    ) + " │"
    sep_line = "  ├─" + "─┼─".join("─" * col_widths[h] for h in headers) + "─┤"
    top_line = "  ┌─" + "─┬─".join("─" * col_widths[h] for h in headers) + "─┐"
    bot_line = "  └─" + "─┴─".join("─" * col_widths[h] for h in headers) + "─┘"

    print(clr(top_line, BLUE))
    print(header_line)
    print(clr(sep_line, BLUE))

    for i, row in enumerate(rows):
        cells = []
        for h in headers:
            val  = row.get(h, "")
            colour = col_colours.get(h, WHITE)
            # special: map verdict/severity to colours inline
            if h.upper() in ("VERDICT", "SEVERITY"):
                v = str(val).upper()
                c = {
                    "SUPPORTED":       GREEN,
                    "WEAKLY_SUPPORTED":YELLOW,
                    "CONTESTED":       MAGENTA,
                    "UNSUPPORTED":     RED,
                    "HIGH":            RED,
                    "MEDIUM":          YELLOW,
                    "LOW":             GREEN,
                }.get(v, WHITE)
                cells.append(clr(fmt_cell(val, h), c + BOLD))
            else:
                cells.append(clr(fmt_cell(val, h), colour))
        print("  │ " + " │ ".join(cells) + " │")

    print(clr(bot_line, BLUE))


# ══════════════════════════════════════════════════════════════════════════════
# Email helper (thin wrapper around utils.email_sender)
# ══════════════════════════════════════════════════════════════════════════════
def _offer_email(
    step_label: str,
    final_report: dict,
    state: dict,
    query: str,
    already_sent_to: set,
    force: bool = False,
):
    """
    Ask user if they want to email the current report.
    Skipped automatically when stdin is not a real terminal (piped/non-interactive).
    already_sent_to: set of addresses already emailed (to avoid duplicates).
    force: if True, always ask regardless of whether it was already offered.
    """
    if not sys.stdin.isatty():
        return  # non-interactive mode — skip silently
    print()
    choice = input(
        clr(
            f"  📧 [{step_label}] Send report so far by email? [y/N]: ",
            BOLD + MAGENTA,
        )
    ).strip().lower()

    if choice not in ("y", "yes"):
        return

    to_email = input(clr("     Recipient email address: ", CYAN)).strip()
    if not to_email:
        print(clr("     No address entered — skipping.", DIM))
        return

    try:
        from utils.email_sender import send_report_email
    except ImportError as e:
        print(clr(f"     Email sender not available: {e}", RED))
        return

    subject = f"Research Report: {query[:60]} [{datetime.now():%Y-%m-%d}]"
    sp = Spinner(f"Sending email to {to_email}").start()
    ok, msg = send_report_email(to_email, subject, final_report, state)
    sp.stop(ok, msg)
    if ok:
        already_sent_to.add(to_email)


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline node labels + progress tracker
# ══════════════════════════════════════════════════════════════════════════════
_NODE_META = {
    "query_parser": {
        "num": "1/9", "label": "Query Understanding",
        "icon": "🔍", "desc": "Parsing intent, domain, keywords",
    },
    "research_planner": {
        "num": "2/9", "label": "Research Planning",
        "icon": "📋", "desc": "Building task graph & strategy",
    },
    "hypothesis_generation": {
        "num": "3/9", "label": "Hypothesis Formation",
        "icon": "💡", "desc": "Generating testable hypotheses",
    },
    "retriever": {
        "num": "4/9", "label": "Evidence Retrieval",
        "icon": "🌐", "desc": "Searching sources (Web / Academic APIs)",
    },
    "credibility_scorer": {
        "num": "5/9", "label": "Credibility Scoring",
        "icon": "⚖️ ", "desc": "Scoring & ranking every source",
    },
    "hypothesis_evaluation": {
        "num": "6/9", "label": "Hypothesis Evaluation",
        "icon": "🧪", "desc": "Testing hypotheses against evidence",
    },
    "contradiction_detector": {
        "num": "7/9", "label": "Contradiction Detection",
        "icon": "⚡", "desc": "Finding conflicting claims",
    },
    "synthesis_engine": {
        "num": "8/9", "label": "Evidence Synthesis",
        "icon": "🔗", "desc": "Synthesising consensus findings",
    },
    "output_generator": {
        "num": "9/9", "label": "Report Generation",
        "icon": "📝", "desc": "Writing the final research report",
    },
}


def _print_node_start(node_name: str):
    meta = _NODE_META.get(node_name, {
        "num": "?/9", "label": node_name,
        "icon": "⚙️ ", "desc": "",
    })
    print()
    print(
        clr(f"  [{meta['num']}]", CYAN + BOLD) +
        f"  {meta['icon']}  " +
        clr(meta["label"], WHITE + BOLD) +
        clr(f"  —  {meta['desc']}", DIM)
    )


def _print_node_done(node_name: str, payload: dict, cumulative: dict):
    """Print per-node detail line after completion."""
    if node_name == "query_parser":
        pq = payload.get("parsed_query", {})
        print(
            clr("        └─ ", DIM) +
            clr(f"intent={pq.get('intent','?')}  "
                f"domain={pq.get('domain','?')}  "
                f"academic={pq.get('is_academic','?')}  "
                f"keywords={pq.get('keywords',[])}",
                DIM + ITALIC)
        )

    elif node_name == "research_planner":
        plan = payload.get("research_plan", {})
        depth = plan.get("estimated_depth", "?")
        hyp_n = plan.get("recommended_hypothesis_count", "?")
        print(
            clr("        └─ ", DIM) +
            clr(f"depth={depth}  recommended_hypotheses={hyp_n}", DIM + ITALIC)
        )

    elif node_name == "hypothesis_generation":
        hs = cumulative.get("hypotheses", [])
        print(
            clr("        └─ ", DIM) +
            clr(f"{len(hs)} hypothesis(es) generated", DIM + ITALIC)
        )

    elif node_name == "retriever":
        srcs = cumulative.get("retrieved_sources", [])
        print(
            clr("        └─ ", DIM) +
            clr(f"{len(srcs)} source(s) retrieved", DIM + ITALIC)
        )

    elif node_name == "credibility_scorer":
        srcs = cumulative.get("retrieved_sources", [])
        if srcs:
            best = max(srcs, key=lambda s: s.get("credibility_score", 0))
            print(
                clr("        └─ ", DIM) +
                clr(
                    f"best credibility={best.get('credibility_score',0):.2f}  "
                    f"title='{str(best.get('title',''))[:55]}'",
                    DIM + ITALIC)
            )

    elif node_name == "hypothesis_evaluation":
        hs = cumulative.get("hypotheses", [])
        supported = sum(
            1 for h in hs
            if str(h.get("verdict", "")).lower() in ("supported", "weakly_supported")
        )
        print(
            clr("        └─ ", DIM) +
            clr(
                f"{supported}/{len(hs)} hypothesis(es) supported",
                DIM + ITALIC)
        )

    elif node_name == "contradiction_detector":
        cons = cumulative.get("contradictions", [])
        high = sum(1 for c in cons if str(c.get("severity","")).lower() == "high")
        print(
            clr("        └─ ", DIM) +
            clr(
                f"{len(cons)} contradiction(s) found  ({high} high severity)",
                DIM + ITALIC)
        )

    elif node_name == "synthesis_engine":
        syn = cumulative.get("synthesis", {})
        conf = syn.get("confidence_score", 0)
        print(
            clr("        └─ ", DIM) +
            clr(f"confidence={int(conf * 100)}%", DIM + ITALIC)
        )

    elif node_name == "output_generator":
        report = cumulative.get("final_report", {})
        secs   = len(report.get("sections", []))
        cites  = len(report.get("citations", []))
        conf   = report.get("confidence_overall", 0)
        print(
            clr("        └─ ", DIM) +
            clr(
                f"sections={secs}  citations={cites}  "
                f"confidence={int(conf * 100)}%",
                DIM + ITALIC)
        )


# ══════════════════════════════════════════════════════════════════════════════
# Report renderer (full detail)
# ══════════════════════════════════════════════════════════════════════════════
def _confidence_bar(score: float, width: int = 30) -> str:
    filled = int(score * width)
    bar    = "█" * filled + "░" * (width - filled)
    colour = GREEN if score >= 0.7 else (YELLOW if score >= 0.4 else RED)
    return clr(bar, colour) + clr(f" {int(score * 100)}%", BOLD + colour)


def print_full_report(final_report: dict, state: dict, query: str):
    """Print the complete, richly-formatted research report to the terminal."""

    print()
    print()
    hr("═", 76, CYAN + BOLD)
    banner_box("RESEARCH REPORT")
    hr("═", 76, CYAN + BOLD)

    # ── Meta block ────────────────────────────────────────────────────────────
    pq   = state.get("parsed_query", {})
    srcs = state.get("retrieved_sources", [])
    cons = state.get("contradictions", [])
    hs   = state.get("hypotheses", [])
    conf = final_report.get("confidence_overall", state.get("synthesis", {}).get("confidence_score", 0))

    print()
    print(clr("  Query   :", BOLD + YELLOW), clr(query, WHITE))
    print(clr("  Domain  :", BOLD + YELLOW), clr(pq.get("domain","?") + f"  (academic={pq.get('is_academic','?')})", WHITE))
    print(clr("  Session :", BOLD + YELLOW), clr(state.get("session_id","?"), DIM))
    print(clr("  Date    :", BOLD + YELLOW), clr(datetime.now().strftime("%Y-%m-%d %H:%M UTC"), DIM))
    print()
    print(clr("  Sources Retrieved :", BOLD + CYAN), len(srcs))
    print(clr("  Hypotheses Tested :", BOLD + CYAN), len(hs))
    print(clr("  Contradictions    :", BOLD + CYAN), len(cons))
    print(clr("  Confidence        :", BOLD + CYAN), _confidence_bar(float(conf)))

    # ── 1. Executive Summary ─────────────────────────────────────────────────
    section_header("1.  EXECUTIVE SUMMARY", "📄")
    wrap_print(final_report.get("executive_summary", "Not available."), colour=WHITE)

    # ── 2. Detailed Research Sections ────────────────────────────────────────
    sections = final_report.get("sections", [])
    if sections:
        section_header("2.  DETAILED RESEARCH FINDINGS", "🔬")
        for i, sec in enumerate(sections):
            heading = sec.get("heading", f"Section {i+1}")
            content = sec.get("content", "")
            print()
            print(clr(f"  ▸ {heading}", BOLD + YELLOW))
            wrap_print(content, colour=WHITE)
            supporting = sec.get("supporting_source_ids", [])
            if supporting:
                print(clr(f"    [Sources: {', '.join(str(s) for s in supporting)}]", DIM))

    # ── 3. Hypotheses (tabular) ──────────────────────────────────────────────
    hvs = final_report.get("hypotheses_verdict", [])
    section_header("3.  HYPOTHESES & VERDICTS", "🧪")
    if hvs:
        rows = [
            {
                "#":         str(i + 1),
                "Verdict":   h.get("verdict", "").upper(),
                "Hypothesis": h.get("statement", ""),
                "Evidence Summary": h.get("summary", ""),
            }
            for i, h in enumerate(hvs)
        ]
        print_table(
            headers=["#", "Verdict", "Hypothesis", "Evidence Summary"],
            rows=rows,
            max_col=32,
        )
    else:
        print(clr("  No hypotheses recorded.", DIM))

    # ── 4. Key Conclusions ───────────────────────────────────────────────────
    conclusions = final_report.get("key_conclusions", [])
    section_header("4.  KEY CONCLUSIONS", "✅")
    if conclusions:
        for i, c in enumerate(conclusions):
            print()
            print(clr(f"  {i+1:>2}.", BOLD + YELLOW), end=" ")
            wrap_print(c, indent=6, colour=WHITE)
    else:
        print(clr("  No conclusions recorded.", DIM))

    # ── 5. Contradictions ────────────────────────────────────────────────────
    cflagged = final_report.get("contradictions_flagged", [])
    section_header("5.  CONTRADICTIONS DETECTED", "⚡")
    if cflagged:
        rows = [
            {
                "#":        str(i + 1),
                "Severity": c.get("severity", "").upper(),
                "Summary":  c.get("summary", ""),
                "Recommended Action": c.get("action", ""),
            }
            for i, c in enumerate(cflagged)
        ]
        print_table(
            headers=["#", "Severity", "Summary", "Recommended Action"],
            rows=rows,
            max_col=28,
        )
    else:
        print(clr("  No contradictions flagged under current detector limits.", GREEN))

    # ── 6. Research Gaps ─────────────────────────────────────────────────────
    gaps = final_report.get("research_gaps", [])
    section_header("6.  RESEARCH GAPS", "🕳 ")
    if gaps:
        for i, g in enumerate(gaps):
            print(clr(f"  {i+1:>2}. ", BOLD + YELLOW) + clr(g, WHITE))
    else:
        print(clr("  No research gaps identified.", DIM))

    # ── 7. All Sources / Citations ───────────────────────────────────────────
    cites = final_report.get("citations", [])
    section_header(f"7.  SOURCES & CITATIONS  ({len(cites)} total)", "📚")
    if cites:
        rows = [
            {
                "#":      str(i + 1),
                "Title":  c.get("title", ""),
                "Authors":c.get("authors", ""),
                "Year":   str(c.get("year", "")),
                "DOI / URL": c.get("doi") or c.get("url") or "N/A",
            }
            for i, c in enumerate(cites)
        ]
        print_table(
            headers=["#", "Title", "Authors", "Year", "DOI / URL"],
            rows=rows,
            max_col=26,
        )
    else:
        print(clr("  No citations available.", DIM))

    # ── 8. Confidence Score ──────────────────────────────────────────────────
    section_header("8.  CONFIDENCE SCORE", "📊")
    print()
    print(f"  Overall Confidence: {_confidence_bar(float(conf), 40)}")
    print()
    synth_conf = state.get("synthesis", {}).get("confidence_score", 0)
    if synth_conf and abs(synth_conf - float(conf)) > 0.01:
        print(f"  Synthesis stage: {_confidence_bar(float(synth_conf), 40)}")
    print()
    interp = (
        clr("HIGH — strong, consistent evidence found.", GREEN + BOLD)
        if float(conf) >= 0.7
        else clr("MODERATE — evidence found but with gaps or conflicts.", YELLOW + BOLD)
        if float(conf) >= 0.4
        else clr("LOW — limited or conflicting evidence; treat with caution.", RED + BOLD)
    )
    print(f"  Interpretation: {interp}")

    # ── 9. Follow-up Recommendations ─────────────────────────────────────────
    fuqs = final_report.get("follow_up_questions", [])
    section_header("9.  FOLLOW-UP RECOMMENDATIONS", "🔭")
    if fuqs:
        for i, q in enumerate(fuqs):
            print(clr(f"  {i+1:>2}. ", BOLD + CYAN) + clr(q, WHITE))
    else:
        print(clr("  No follow-up questions generated.", DIM))

    print()
    hr("═", 76, CYAN + BOLD)
    print()


# ══════════════════════════════════════════════════════════════════════════════
# Save report to disk
# ══════════════════════════════════════════════════════════════════════════════
def _save_report(state: dict, final_report: dict, query: str) -> Optional[str]:
    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_q = "".join(
        c for c in query[:40] if c.isalnum() or c in " _-"
    ).strip().replace(" ", "_")
    basename = f"research_report_{safe_q}_{ts}"
    json_path = f"{basename}.json"
    txt_path  = f"{basename}.txt"

    payload = {
        "query":       query,
        "session_id":  state.get("session_id", ""),
        "generated_at":datetime.now().isoformat(),
        "final_report":final_report,
        "synthesis":   state.get("synthesis", {}),
        "hypotheses":  state.get("hypotheses", []),
        "contradictions": state.get("contradictions", []),
        "retrieved_sources": state.get("retrieved_sources", []),
        "parsed_query":state.get("parsed_query", {}),
    }

    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    # plain text
    try:
        from utils.email_sender import build_plaintext
        txt = build_plaintext(final_report)
        with open(txt_path, "w") as f:
            f.write(f"RESEARCH REPORT\nQuery: {query}\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            f.write(txt)
        print(clr(f"  ✓  Plain-text report → {txt_path}", GREEN))
    except Exception:
        pass

    print(clr(f"  ✓  JSON report        → {json_path}", GREEN))
    return json_path


# ══════════════════════════════════════════════════════════════════════════════
# Follow-up Q&A chat loop
# ══════════════════════════════════════════════════════════════════════════════
def _followup_chat(final_report: dict, state: dict, query: str, already_sent_to: set):
    try:
        from agents.followup_agent import FollowupAgent
    except ImportError as e:
        print(clr(f"  Follow-up agent unavailable: {e}", YELLOW))
        return

    session_id = state.get("session_id", "cli-unknown")
    agent = FollowupAgent(session_id=session_id, report_data=final_report)
    chat_history = []

    section_header("FOLLOW-UP CHAT", "💬")
    print(clr("  Ask anything about this research. Type 'done' or 'exit' to stop.", DIM))
    print()

    while True:
        try:
            user_q = input(clr("  You  ▸ ", BOLD + CYAN)).strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_q:
            continue
        if user_q.lower() in ("done", "exit", "quit", "q", "bye"):
            break

        sp = Spinner("Thinking…").start()
        try:
            answer = agent.answer(user_q, chat_history)
            sp.stop(True, "Answer ready")
        except Exception as e:
            sp.stop(False, str(e))
            answer = f"Error: {e}"

        chat_history.append({"role": "user",      "content": user_q})
        chat_history.append({"role": "assistant",  "content": answer})

        print()
        print(clr("  Agent ▸", BOLD + GREEN))
        wrap_print(answer, indent=4, colour=WHITE)
        print()

        # email is offered after the full follow-up session ends, not per-answer


# ══════════════════════════════════════════════════════════════════════════════
# Scheduler Commands
# ══════════════════════════════════════════════════════════════════════════════

def _handle_scheduler_commands():
    """Handle scheduler management commands"""
    from scheduler.living_doc_scheduler import living_doc_scheduler
    
    if len(sys.argv) < 3:
        print(clr("\n  Scheduler Commands:", BOLD + CYAN))
        print(clr("    scheduler start      ", WHITE), "Start 30-day recheck scheduler")
        print(clr("    scheduler stop       ", WHITE), "Stop the scheduler")
        print(clr("    scheduler status     ", WHITE), "Check scheduler status")
        print(clr("    scheduler check-now  ", WHITE), "Manually trigger source recheck")
        print()
        return

    command = sys.argv[2].lower()

    if command == "start":
        print(clr("\n  Starting scheduler…", BOLD + YELLOW))
        try:
            living_doc_scheduler.start()
            print(clr("  ✅ Scheduler started successfully", GREEN))
            status = living_doc_scheduler.get_status()
            print(clr(f"  Next check: {status.get('next_check_at')}", DIM))
        except Exception as e:
            print(clr(f"  ❌ Failed to start scheduler: {e}", RED))
            sys.exit(1)

    elif command == "stop":
        print(clr("\n  Stopping scheduler…", BOLD + YELLOW))
        try:
            living_doc_scheduler.stop()
            print(clr("  ✅ Scheduler stopped successfully", GREEN))
        except Exception as e:
            print(clr(f"  ❌ Failed to stop scheduler: {e}", RED))
            sys.exit(1)

    elif command == "status":
        print(clr("\n  Scheduler Status:", BOLD + CYAN))
        try:
            status = living_doc_scheduler.get_status()
            print(clr(f"  Status: {status['status']}", 
                  GREEN if status['running'] else YELLOW))
            print(clr(f"  Running: {status['running']}", 
                  GREEN if status['running'] else YELLOW))
            if status.get('last_check_at'):
                print(clr(f"  Last check: {status['last_check_at']}", DIM))
            if status.get('next_check_at'):
                print(clr(f"  Next check: {status['next_check_at']}", CYAN))
            if status.get('jobs'):
                print(clr(f"  Active jobs: {len(status['jobs'])}", WHITE))
                for job in status['jobs']:
                    print(clr(f"    • {job['name']}", DIM))
                    print(clr(f"      Next run: {job['next_run']}", DIM))
        except Exception as e:
            print(clr(f"  ❌ Failed to get status: {e}", RED))
            sys.exit(1)

    elif command == "check-now":
        print(clr("\n  Triggering manual source recheck…", BOLD + YELLOW))
        try:
            result = living_doc_scheduler.trigger_manual_check()
            if result['status'] == 'success':
                print(clr("  ✅ Manual recheck completed", GREEN))
            else:
                print(clr(f"  ⚠️  {result['message']}", YELLOW))
        except Exception as e:
            print(clr(f"  ❌ Manual recheck failed: {e}", RED))
            sys.exit(1)

    else:
        print(clr(f"\n  ❌ Unknown scheduler command: {command}", RED))
        sys.exit(1)

    print()


# ══════════════════════════════════════════════════════════════════════════════
# Main CLI
# ══════════════════════════════════════════════════════════════════════════════
def _load_pipeline():
    from graph.research_graph import stream_research  # noqa: F401
    return stream_research


def main():
    # ── Handle scheduler commands first ────────────────────────────────────────
    if len(sys.argv) > 1 and sys.argv[1] == "scheduler":
        _handle_scheduler_commands()
        return

    # ── Banner ────────────────────────────────────────────────────────────────
    print()
    banner_box("RESEARCH AGENT  ·  Terminal Mode  v2.0")
    print(
        clr(
            "  Powered by Ollama · LangGraph · DuckDuckGo · "
            "PubMed · Pinecone · DynamoDB",
            DIM,
        )
    )
    print()

    # ── Load pipeline ─────────────────────────────────────────────────────────
    sp = Spinner("Loading pipeline modules…").start()
    try:
        stream_research = _load_pipeline()
        sp.stop(True, "Pipeline loaded — all modules OK")
    except Exception as e:
        sp.stop(False, f"Pipeline load failed: {e}")
        print(clr(f"\n  ❌  {e}", RED))
        sys.exit(1)

    # ── Get query ─────────────────────────────────────────────────────────────
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(clr("  Query : ", BOLD + YELLOW) + clr(query, WHITE))
    else:
        print(clr("  Enter your research question:", BOLD + YELLOW))
        print(clr("  (Ctrl+C to quit)\n", DIM))
        try:
            query = input(clr("  ▶  ", CYAN + BOLD)).strip()
        except (EOFError, KeyboardInterrupt):
            print(clr("\n  Goodbye!", DIM))
            sys.exit(0)

    if not query:
        print(clr("  No query entered. Exiting.", YELLOW))
        sys.exit(0)

    # ── Pipeline live stream ──────────────────────────────────────────────────
    print()
    hr("═", 76, CYAN + BOLD)
    banner_box(f"RESEARCHING: {query[:60]}")
    hr("═", 76, CYAN + BOLD)
    print()

    cumulative: dict   = {}
    final_report: dict = {}
    already_sent_to    = set()
    active_spinner: Optional[Spinner] = None

    try:
        for event in stream_research(query, fast_mode=False):
            node_name = event.get("node", "")

            # ── Skip meta events ───────────────────────────────────────────────
            if node_name in ("__done__", "post_pipeline_warning"):
                if active_spinner:
                    active_spinner.stop(True, "Done")
                    active_spinner = None
                if node_name == "__done__":
                    fr = event.get("final_report") or {}
                    if fr:
                        final_report = fr
                continue

            # ── Stop previous spinner ──────────────────────────────────────────
            if active_spinner:
                meta = _NODE_META.get(node_name, {})
                active_spinner.stop(True, meta.get("label", "Done"))
                active_spinner = None

            payload = event.get("payload", {})
            cumulative.update(payload)
            if "session_id" in event:
                cumulative["session_id"] = event["session_id"]

            # Print node completion detail
            _print_node_done(node_name, payload, cumulative)

            # Track final_report as it builds
            if payload.get("final_report"):
                final_report = payload["final_report"]

            # ── Start spinner for next node ────────────────────────────────────
            next_nodes = {
                "query_parser":           "research_planner",
                "research_planner":       "hypothesis_generation",
                "hypothesis_generation":  "retriever",
                "retriever":              "credibility_scorer",
                "credibility_scorer":     "hypothesis_evaluation",
                "hypothesis_evaluation":  "contradiction_detector",
                "contradiction_detector": "synthesis_engine",
                "synthesis_engine":       "output_generator",
            }
            next_node = next_nodes.get(node_name)
            if next_node:
                _print_node_start(next_node)
                next_meta   = _NODE_META.get(next_node, {})
                active_spinner = Spinner(next_meta.get("desc", next_node)).start()

        if active_spinner:
            active_spinner.stop(True, "Complete")

    except KeyboardInterrupt:
        if active_spinner:
            active_spinner.stop(False, "Interrupted")
        print(clr("\n  Research interrupted.", YELLOW))
    except Exception as e:
        if active_spinner:
            active_spinner.stop(False, str(e))
        print(clr(f"\n  ❌  Pipeline error: {e}", RED))
        import traceback
        traceback.print_exc()

    # ── If final_report is empty, pull from cumulative ────────────────────────
    if not final_report:
        final_report = cumulative.get("final_report", {})

    # ── Print full report ────────────────────────────────────────────────────
    if final_report:
        print_full_report(final_report, cumulative, query)
    else:
        print(clr("\n  ⚠  No final report was generated.", YELLOW))

    # ── Display storage status ────────────────────────────────────────────────
    section_header("CLOUD STORAGE", "☁️ ")
    session_id = cumulative.get("session_id", "unknown")
    n_sources = len(cumulative.get("retrieved_sources", []))
    n_hypotheses = len(cumulative.get("hypotheses", []))
    n_contradictions = len(cumulative.get("contradictions", []))
    print()
    print(clr("  Persisted to Cloud:", BOLD + CYAN))
    print(clr(f"    • DynamoDB Session", GREEN), f"(ID: {session_id[:16]}...)")
    print(clr(f"    • {n_hypotheses} hypothesis(es)", GREEN))
    print(clr(f"    • {n_contradictions} contradiction(s)", GREEN))
    if n_sources > 0:
        print(clr(f"    • {n_sources} source(s) → Pinecone vector store", GREEN))
    print(clr(f"    • Knowledge graph indexed", GREEN))
    print()

    # ── Email after full report ───────────────────────────────────────────────
    section_header("EMAIL REPORT", "📧")
    _offer_email("Full Report", final_report, cumulative, query, already_sent_to)

    # ── Save report ───────────────────────────────────────────────────────────
    print()
    section_header("SAVE REPORT", "💾")
    try:
        if not sys.stdin.isatty():
            _save_report(cumulative, final_report, query)  # auto-save when non-interactive
        else:
            save_choice = input(
                clr("  Save report to disk (JSON + TXT)? [Y/n]: ", BOLD + YELLOW)
            ).strip().lower()
            if save_choice not in ("n", "no"):
                _save_report(cumulative, final_report, query)
    except (EOFError, KeyboardInterrupt):
        pass

    # ── Follow-up Q&A ────────────────────────────────────────────────────────
    print()
    try:
        if sys.stdin.isatty():
            chat_choice = input(
                clr("  Start follow-up Q&A about this research? [Y/n]: ", BOLD + YELLOW)
            ).strip().lower()
            if chat_choice not in ("n", "no"):
                _followup_chat(final_report, cumulative, query, already_sent_to)
    except (EOFError, KeyboardInterrupt):
        pass

    # ── Final email offer ─────────────────────────────────────────────────────
    _offer_email("Final — End of Session", final_report, cumulative, query, already_sent_to)

    print()
    hr("═", 76, CYAN + BOLD)
    print(clr("  Research session complete. Goodbye! 👋", BOLD + GREEN))
    hr("═", 76, CYAN + BOLD)
    print()


if __name__ == "__main__":
    main()
