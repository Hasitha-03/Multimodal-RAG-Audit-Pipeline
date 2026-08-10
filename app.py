from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

# MUST BE THE ABSOLUTE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="SEC 10-K Multimodal Audit Engine", layout="wide")

st.title("SEC 10-K Multimodal Audit & Verification Pipeline")

# ----------------------------------------------------------------------------
# IMPORTANT: this file deliberately imports NOTHING from src/ (no torch, no
# sentence-transformers, no VLM provider SDKs). All pipeline work happens in
# separate subprocesses (prepare_run.py / generate_matrix.py / query_item.py)
# so that PyTorch's C-extension internals never load inside Streamlit's own
# multi-threaded server process. If a subprocess segfaults, it dies on its
# own — Streamlit and the browser session stay up and can show the error.
# ----------------------------------------------------------------------------

RUNS_DIR = Path("data/runs")
REGISTRY_PATH = RUNS_DIR / "registry.json"
PROJECT_ROOT = Path(__file__).resolve().parent


# ----------------------------------------------------------------------------
# Run registry
# ----------------------------------------------------------------------------

def load_registry() -> list[dict]:
    if not REGISTRY_PATH.exists():
        return []
    with open(REGISTRY_PATH) as f:
        return json.load(f)


def save_registry(entries: list[dict]) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, "w") as f:
        json.dump(entries, f, indent=2)


def register_run(run_id: str, scope_sheet_name: str, contractor_names: list[str]) -> None:
    entries = load_registry()
    entries.append(
        {
            "run_id": run_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "scope_sheet_name": scope_sheet_name,
            "contractor_names": contractor_names,
        }
    )
    save_registry(entries)


# ----------------------------------------------------------------------------
# Subprocess plumbing — every pipeline stage runs in its own OS process.
# A background thread here only reads the child's stdout line-by-line and
# feeds it into a queue.Queue for the Streamlit main thread to display; the
# thread itself never touches torch/sentence-transformers.
# ----------------------------------------------------------------------------

def _stream_subprocess(
    cmd: list[str],
    progress_queue: "queue.Queue[str]",
    result_holder: dict,
    ok_sentinel: str,
    err_sentinel: str,
    env: Optional[dict] = None,
) -> None:
    """Run `cmd` as a subprocess, forward its stdout lines into
    progress_queue, and set result_holder['ok'] based on the sentinel line
    and exit code. Runs on a background thread; safe because it does no
    torch/ML work itself — it only shells out and reads pipes."""
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        saw_ok = False
        error_lines: list[str] = []
        all_lines: list[str] = []
        for line in proc.stdout:
            line = line.rstrip("\n")
            progress_queue.put(line)
            all_lines.append(line)
            if line.startswith(ok_sentinel):
                saw_ok = True
            elif line.startswith(err_sentinel) or error_lines:
                error_lines.append(line)  # also captures traceback lines that follow

        returncode = proc.wait()

        if returncode == 0 and saw_ok:
            result_holder["ok"] = True
        else:
            result_holder["ok"] = False
            if returncode < 0:
                # Negative returncode means the child died from a signal —
                # e.g. -11 is SIGSEGV. Surface this distinctly since it's
                # exactly the crash class this subprocess design exists to
                # contain: the child dies, but Streamlit stays up.
                result_holder["error"] = (
                    f"Subprocess terminated by signal {-returncode} "
                    f"(likely a native-library crash, e.g. segfault). "
                    f"Streamlit itself did not crash. Last output before the crash:\n"
                    + "\n".join((error_lines or all_lines)[-15:])
                )
            else:
                result_holder["error"] = "\n".join(error_lines) or f"Subprocess exited with code {returncode}."

    except Exception as e:
        result_holder["ok"] = False
        result_holder["error"] = f"Failed to launch subprocess: {type(e).__name__}: {e}"


def _subprocess_env() -> dict:
    """Child env: inherit everything, but make sure the API keys currently
    in this process's environment are passed through explicitly (they
    already are via inheritance, but this keeps the intent visible and
    gives one place to add overrides later)."""
    env = os.environ.copy()
    return env


# ----------------------------------------------------------------------------
# Stage 0/1/2 ONLY — "prepare" a run (fast, no VLM calls)
# ----------------------------------------------------------------------------

def prepare_run_in_background(
    run_id: str,
    scope_sheet_path: Path,
    contractor_pdf_paths: dict[str, Path],
    progress_queue: "queue.Queue[str]",
    result_holder: dict,
) -> None:
    run_dir = RUNS_DIR / run_id
    cmd = [
        sys.executable, str(PROJECT_ROOT / "prepare_run.py"),
        "--run-dir", str(run_dir),
        "--scope-sheet", str(scope_sheet_path),
    ]
    for name, path in contractor_pdf_paths.items():
        cmd += ["--contractor", f"{name}={path}"]

    _stream_subprocess(cmd, progress_queue, result_holder, "PREPARE_OK", "PREPARE_ERROR", env=_subprocess_env())


# ----------------------------------------------------------------------------
# On-demand single-item query (Query tab) — Stage 3 -> 3.5
# ----------------------------------------------------------------------------

def query_cache_path(run_id: str) -> Path:
    return RUNS_DIR / run_id / "query_cache.json"


def load_query_cache(run_id: str) -> dict:
    path = query_cache_path(run_id)
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def save_query_cache(run_id: str, cache: dict) -> None:
    path = query_cache_path(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(cache, f, indent=2)


def cache_key(line_item_id: str, contractor_name: str) -> str:
    return f"{line_item_id}::{contractor_name}"


def run_single_query(
    run_id: str,
    line_item_id: str,
    line_item_text: str,
    contractor_name: str,
    api_key: str,
    provider_type: str = "openrouter",
    model_override: Optional[str] = "openai/gpt-4o-mini",
) -> dict:
    """Runs query_item.py as a subprocess and parses its single-line JSON
    stdout. Blocking (used from within a st.spinner in the Query tab) —
    fine since a single line-item query is a few seconds, not a batch job."""
    cmd = [
        sys.executable, str(PROJECT_ROOT / "query_item.py"),
        "--run-dir", str(RUNS_DIR / run_id),
        "--line-item-id", line_item_id,
        "--line-item-text", line_item_text,
        "--contractor", contractor_name,
        "--provider", provider_type,
    ]
    if model_override:
        cmd += ["--model", model_override]

    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        env=_subprocess_env(),
    )

    if proc.returncode < 0:
        raise RuntimeError(
            f"Query subprocess terminated by signal {-proc.returncode} "
            f"(likely a native-library crash). Streamlit itself did not crash.\n"
            f"stderr:\n{proc.stderr}"
        )
    if proc.returncode != 0:
        raise RuntimeError(f"Query subprocess failed:\n{proc.stderr}")

    stdout = proc.stdout.strip()
    if not stdout:
        raise RuntimeError(f"Query subprocess produced no output.\nstderr:\n{proc.stderr}")

    try:
        return json.loads(stdout.splitlines()[-1])
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Could not parse query result JSON: {e}\nRaw stdout:\n{stdout}")


# ----------------------------------------------------------------------------
# Optional batch path (Browse tab "Generate Full Matrix" button)
# ----------------------------------------------------------------------------

def generate_full_matrix(
    run_id: str,
    api_key: str,
    progress_queue: "queue.Queue[str]",
    result_holder: dict,
    provider_type: str = "openrouter",
    model_override: Optional[str] = "openai/gpt-4o-mini",
) -> None:
    run_dir = RUNS_DIR / run_id
    cmd = [
        sys.executable, str(PROJECT_ROOT / "generate_matrix.py"),
        "--run-dir", str(run_dir),
        "--provider", provider_type,
    ]
    if model_override:
        cmd += ["--model", model_override]

    _stream_subprocess(cmd, progress_queue, result_holder, "MATRIX_OK", "MATRIX_ERROR", env=_subprocess_env())


# ----------------------------------------------------------------------------
# App State Session Keys
# ----------------------------------------------------------------------------

for key, default in [
    ("prep_progress_queue", None),
    ("prep_progress_log", []),
    ("prep_result_holder", {}),
    ("prep_thread", None),
    ("active_run_id", None),
    ("batch_progress_queue", None),
    ("batch_progress_log", []),
    ("batch_result_holder", {}),
    ("batch_thread", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

tab_run, tab_browse, tab_query = st.tabs(["Upload & Prepare", "Browse Results", "Query a Line Item"])


# ----------------------------------------------------------------------------
# Tab 1: Upload & Prepare
# ----------------------------------------------------------------------------

with tab_run:
    st.header("Upload files and prepare an audit run")
    st.caption(
        "Dense Vector Retrieval (sentence-transformers) + Vision-Language Model (OpenRouter) + RapidFuzz Grounding"
    )

    scope_sheet_file = st.file_uploader("Master Audit Checklist (JSON or PDF)", type=["json", "pdf"], key="scope_upload")
    contractor_files = st.file_uploader(
        "SEC 10-K Corporate Filings (PDFs)", type=["pdf"], accept_multiple_files=True, key="contractor_upload"
    )

    prep_running = st.session_state.prep_thread is not None and st.session_state.prep_thread.is_alive()

    if st.button("Prepare Audit Run", disabled=prep_running):
        if not scope_sheet_file:
            st.error("Please upload a Master Audit Checklist file.")
        elif not contractor_files:
            st.error("Please upload at least one SEC 10-K corporate filing PDF.")
        else:
            run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
            run_dir = RUNS_DIR / run_id
            run_dir.mkdir(parents=True, exist_ok=True)

            scope_sheet_path = run_dir / scope_sheet_file.name
            with open(scope_sheet_path, "wb") as f:
                f.write(scope_sheet_file.getbuffer())

            contractor_pdf_paths = {}
            for cf in contractor_files:
                contractor_name = Path(cf.name).stem
                cpath = run_dir / "uploads" / cf.name
                cpath.parent.mkdir(parents=True, exist_ok=True)
                with open(cpath, "wb") as f:
                    f.write(cf.getbuffer())
                contractor_pdf_paths[contractor_name] = cpath

            st.session_state.prep_progress_queue = queue.Queue()
            st.session_state.prep_progress_log = []
            st.session_state.prep_result_holder = {}
            st.session_state.active_run_id = run_id

            new_result_holder = st.session_state.prep_result_holder
            new_progress_queue = st.session_state.prep_progress_queue

            thread = threading.Thread(
                target=prepare_run_in_background,
                args=(run_id, scope_sheet_path, contractor_pdf_paths, new_progress_queue, new_result_holder),
                daemon=True,
            )
            st.session_state.prep_thread = thread
            thread.start()
            register_run(run_id, scope_sheet_file.name, list(contractor_pdf_paths.keys()))
            st.rerun()

    active_thread = st.session_state.prep_thread
    if active_thread is not None:
        while not st.session_state.prep_progress_queue.empty():
            st.session_state.prep_progress_log.append(st.session_state.prep_progress_queue.get())

        st.code("\n".join(st.session_state.prep_progress_log[-30:]) or "Starting...")

        if active_thread.is_alive():
            time.sleep(0.5)
            st.rerun()
        else:
            if st.session_state.prep_thread is active_thread:
                result = st.session_state.prep_result_holder
                if result.get("ok"):
                    st.success(
                        f"Run '{st.session_state.active_run_id}' prepared! "
                        f"Go to 'Query a Line Item' to get instant audit extraction results."
                    )
                elif result.get("ok") is False:
                    st.error("Preparation failed:")
                    st.code(result.get("error", "(no error details captured)"))
                st.session_state.prep_thread = None


# ----------------------------------------------------------------------------
# Shared: run selector + ground truth / contractor loader
# ----------------------------------------------------------------------------

def _select_run(tab_name: str) -> Optional[str]:
    registry = load_registry()
    if not registry:
        st.info("No prepared runs yet — use the 'Upload & Prepare' tab first.")
        return None
    options = [e["run_id"] for e in reversed(registry)]
    default_index = options.index(st.session_state.active_run_id) if st.session_state.active_run_id in options else 0
    return st.selectbox("Select an Audit Run:", options, index=default_index, key=f"run_select_{tab_name}")


def _load_line_items(run_id: str) -> list[dict]:
    gt_path = RUNS_DIR / run_id / "scope_ground_truth.json"
    if not gt_path.exists():
        return []
    with open(gt_path) as f:
        return json.load(f)["line_items"]


def _load_contractor_names(run_id: str) -> list[str]:
    contractors_dir = RUNS_DIR / run_id / "contractors"
    if not contractors_dir.exists():
        return []
    return sorted(p.name for p in contractors_dir.iterdir() if p.is_dir())


# ----------------------------------------------------------------------------
# Tab 2: Browse Results
# ----------------------------------------------------------------------------

with tab_browse:
    st.header("Browse Comparative Audit Results")
    run_id = _select_run("browse")

    if run_id:
        matrix_path = RUNS_DIR / run_id / "scope_matrix.json"
        csv_path = RUNS_DIR / run_id / "scope_matrix.csv"
        batch_running = st.session_state.batch_thread is not None and st.session_state.batch_thread.is_alive()

        if not matrix_path.exists():
            st.info(
                "No full matrix has been generated for this run yet. Querying individual "
                "items (Query tab) doesn't require this — this button is only for a complete "
                "CSV export or full-table view."
            )
            api_key = os.environ.get("OPENROUTER_API_KEY") or st.text_input(
                "OPENROUTER_API_KEY (not found in environment):", type="password", key="batch_api_key"
            )
            if st.button("Generate Full Audit Matrix", disabled=batch_running or not api_key):
                st.session_state.batch_progress_queue = queue.Queue()
                st.session_state.batch_progress_log = []
                st.session_state.batch_result_holder = {}
                thread = threading.Thread(
                    target=generate_full_matrix,
                    args=(run_id, api_key, st.session_state.batch_progress_queue, st.session_state.batch_result_holder),
                    daemon=True,
                )
                st.session_state.batch_thread = thread
                thread.start()
                st.rerun()

        if st.session_state.batch_thread is not None:
            while not st.session_state.batch_progress_queue.empty():
                st.session_state.batch_progress_log.append(st.session_state.batch_progress_queue.get())
            st.code("\n".join(st.session_state.batch_progress_log[-30:]) or "Starting...")
            if st.session_state.batch_thread.is_alive():
                time.sleep(0.5)
                st.rerun()
            else:
                result = st.session_state.batch_result_holder
                if result.get("ok"):
                    st.success("Full matrix generated!")
                elif "error" in result:
                    st.error("Matrix generation failed:")
                    st.code(result["error"])
                st.session_state.batch_thread = None

        if matrix_path.exists():
            with open(matrix_path) as f:
                data = json.load(f)
            matrix = data["line_items"]

            if matrix and "contractors" in matrix[0]:
                contractor_names = sorted(matrix[0]["contractors"].keys())
            else:
                contractor_names = []

            rows = []
            for item in matrix:
                gt = item.get("ground_truth", {})
                row = {
                    "Requirement ID": item["line_item_id"],
                    "Description": item["description"],
                    "Section": item.get("section"),
                    "GT Status": gt.get("status"),
                    "GT Expected Figure": gt.get("cost"),
                }
                contractors_dict = item.get("contractors", {})
                for name in contractor_names:
                    c = contractors_dict.get(name)
                    row[f"{name} Status"] = c["status"] if c else None
                    row[f"{name} Figure"] = c["cost_value"] if c else None
                    row[f"{name} Needs Review?"] = c["needs_human_review"] if c else None
                rows.append(row)

            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, height=500)

            if csv_path.exists():
                with open(csv_path, "rb") as f:
                    st.download_button("Download Full Audit CSV", f, file_name="audit_matrix.csv")


# ----------------------------------------------------------------------------
# Tab 3: Query a Line Item
# ----------------------------------------------------------------------------

with tab_query:
    st.header("Look up one audit requirement x corporate filing — on demand")
    run_id = _select_run("query")

    if run_id:
        line_items = _load_line_items(run_id)
        contractor_names = _load_contractor_names(run_id)

        if not line_items or not contractor_names:
            st.warning("This run has no checklist items or no prepared corporate filings yet.")
        else:
            item_options = {f"{i['line_item_id']}: {i['description'][:70]}": i for i in line_items}
            
            selected_item_label = st.selectbox("Select Audit Requirement", list(item_options.keys()))
            selected_contractor = st.selectbox("Select Target Company / Filing", contractor_names)

            item = item_options[selected_item_label]

            gt = item.get("ground_truth", {})
            gt_status = gt.get("status", "N/A")
            gt_cost = gt.get("cost", "N/A")
            gt_notes = gt.get("notes", "N/A")

            st.subheader("Ground Truth Baseline")
            st.write(
                f"**Status:** {gt_status}  |  "
                f"**Expected Metric/Cost:** {gt_cost}  |  "
                f"**Notes:** {gt_notes}"
            )

            cache = load_query_cache(run_id)
            key = cache_key(item["line_item_id"], selected_contractor)
            cached = cache.get(key)

            st.subheader(f"{selected_contractor}'s Extraction Result")

            col_query, col_status = st.columns([1, 3])
            with col_query:
                query_clicked = st.button(
                    "Get Result" if cached is None else "Re-run Query",
                    key=f"query_btn_{key}",
                )

            if query_clicked:
                api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("GEMINI_API_KEY")
                provider_type = "gemini" if os.environ.get("GEMINI_API_KEY") else "openrouter"
                
                if not api_key:
                    st.error("API Key not found. Set OPENROUTER_API_KEY or GEMINI_API_KEY in environment.")
                else:
                    with st.spinner(f"Querying '{selected_contractor}' for this requirement..."):
                        t0 = time.time()
                        try:
                            result = run_single_query(
                                run_id, item["line_item_id"], item["description"], selected_contractor, api_key, provider_type=provider_type
                            )
                            elapsed = time.time() - t0
                            cache[key] = result
                            save_query_cache(run_id, cache)
                            st.caption(f"Query completed in {elapsed:.1f}s.")
                            cached = result
                        except Exception as e:
                            st.error(f"Query failed: {type(e).__name__}: {e}")
                            cached = None

            if cached is None:
                st.info("No result yet for this pair — click 'Get Result' above.")
            else:
                result = cached
                status_color = {"INCLUDED": "green", "EXCLUDED": "red", "NOT_MENTIONED": "gray"}.get(result.get("status"), "gray")
                st.markdown(f"**Status:** :{status_color}[{result.get('status')}]  |  **Confidence:** {result.get('confidence')}")
                st.write(f"**Extracted Figure:** {result.get('cost_value')} ({result.get('cost_type')})")
                st.write(f"**Comments:** {result.get('comments')}")

                st.markdown("**Citation:**")
                st.write(f"Page {result.get('citation_page_number')}")
                st.code(result.get("citation_verbatim") or "(none)")
                if result.get("citation_section_label"):
                    st.caption(f"Section: {result.get('citation_section_label')}")

                if result.get("needs_human_review"):
                    st.error(f"⚠️ Flagged for human review — {result.get('flag_reason')}")
                else:
                    st.success(f"✓ Grounding score: {result.get('grounding_score')}")

                st.caption(
                    f"Retrieval tier: {result.get('retrieval_tier')} | "
                    f"Pages sent: {result.get('candidate_pages_sent')} | "
                    f"(cached — instant on re-select)"
                )
