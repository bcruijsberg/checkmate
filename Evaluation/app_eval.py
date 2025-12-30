# app.py
# Streamlit UI to browse rows one-by-one and edit:
# checkable, explanation, question, details_text, alerts
#
# Run:
#   streamlit run app.py
#
# Notes:
# - Loads a CSV (default separator ";")
# - Edits are stored in session_state and written back to the DataFrame
# - Save options: download edited CSV or overwrite the input file (optional toggle)

import json
import csv
from io import StringIO
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Fact-check dataset editor", layout="wide")

EDIT_COLS = ["checkable", "explanation", "question", "details_text", "alerts"]
CHECKABLE_OPTIONS = ["POTENTIALLY CHECKABLE", "UNCHECKABLE"]


import ast
import json
import pandas as pd

def _parse_alerts_to_list(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return value

    s = str(value).strip()
    if not s:
        return []

    # If it looks like a list, try JSON first, then Python literal
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = json.loads(s)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            parsed = ast.literal_eval(s)
            return parsed if isinstance(parsed, list) else []

    # Otherwise treat as newline-separated text
    return [line.strip("- ").strip() for line in s.splitlines() if line.strip()]



def _alerts_list_to_text(alerts_list):
    if not alerts_list:
        return ""
    return "\n".join(str(a) for a in alerts_list)

def _detect_sep(sample_text: str) -> str:
    candidates = [";", ",", "\t"]
    counts = {c: sample_text.count(c) for c in candidates}
    return max(counts, key=counts.get)

def _load_csv(uploaded_file, sep=";", encoding="utf-8", auto_sep=True, on_bad_lines="warn"):
    """
    Robust CSV loader for messy text fields (newlines, delimiters inside text, etc.)
    Uses python engine (more tolerant) and supports on_bad_lines.
    """
    uploaded_file.seek(0)
    raw = uploaded_file.read()
    text = raw.decode(encoding, errors="replace") if isinstance(raw, (bytes, bytearray)) else str(raw)

    if auto_sep:
        sep = _detect_sep(text[:5000])

    buf = StringIO(text)

    return pd.read_csv(
        buf,
        sep=sep,
        engine="python",
        quotechar='"',
        escapechar="\\",
        quoting=csv.QUOTE_MINIMAL,
        on_bad_lines=on_bad_lines,  # "error" | "warn" | "skip"
    )


def _ensure_columns(df):
    for c in EDIT_COLS:
        if c not in df.columns:
            df[c] = "" if c != "alerts" else "[]"
    return df


def _init_session(df, source_name):
    st.session_state.df = df
    st.session_state.source_name = source_name
    st.session_state.row_idx = 0
    st.session_state.dirty = False


def _get_row(df, idx):
    idx = max(0, min(idx, len(df) - 1))
    return df.iloc[idx], idx


def _write_back(df, idx, updates):
    for k, v in updates.items():
        df.at[df.index[idx], k] = v
    return df


st.title("Dataset browser & editor")

with st.sidebar:
    st.header("Load data")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    sep = st.selectbox("Separator", options=[";", ",", "\t"], index=0)
    encoding = st.text_input("Encoding", value="utf-8")

    st.divider()
    st.header("Save options")
    allow_overwrite = st.checkbox("Allow overwriting the uploaded file on disk", value=False)
    overwrite_path = st.text_input(
        "Overwrite path (only if running locally and you want to save to a file path)",
        value="edited_dataset.csv",
        disabled=not allow_overwrite,
    )
    auto_sep = st.checkbox("Auto-detect delimiter", value=True)
    bad_lines = st.selectbox("On bad lines", ["error", "warn", "skip"], index=2)

if uploaded and "df" not in st.session_state:
    df0 = _load_csv(uploaded, sep=sep, encoding=encoding, auto_sep=auto_sep, on_bad_lines=bad_lines)
    df0 = _ensure_columns(df0)
    _init_session(df0, getattr(uploaded, "name", "uploaded.csv"))

if "df" not in st.session_state:
    st.info("Upload a CSV to start.")
    st.stop()

df = st.session_state.df

# Top controls
colA, colB, colC, colD, colE = st.columns([1.2, 1.2, 2.0, 1.2, 1.2])

with colA:
    if st.button("◀ Prev", use_container_width=True) and st.session_state.row_idx > 0:
        st.session_state.row_idx -= 1

with colB:
    if st.button("Next ▶", use_container_width=True) and st.session_state.row_idx < len(df) - 1:
        st.session_state.row_idx += 1

with colC:
    jump = st.number_input(
        "",
        min_value=1,
        max_value=max(1, len(df)),
        value=st.session_state.row_idx + 1,
        step=1,
        label_visibility="collapsed",
    )
    # Sync jump to state
    st.session_state.row_idx = int(jump) - 1

with colE:
    show_raw = st.checkbox("Show raw row JSON", value=False)

# ✅ get the row first
row, idx = _get_row(df, st.session_state.row_idx)

# ✅ then compute translated flags
has_translated_col = "translated" in df.columns
translated_has_text = False
if has_translated_col:
    v = row.get("translated")
    translated_has_text = isinstance(v, str) and v.strip() != ""

claim_title = row.get("claim", "")

if has_translated_col and translated_has_text:
    claim_title = f"✅ {claim_title}"

st.subheader(claim_title)


rating = row.get("rating", "")
st.markdown(f"**Rating:** {rating}")

url = row.get("url", "")
if isinstance(url, str) and url.strip():
    st.markdown(f"**Source URL:** [{url}]({url})")


st.subheader("Editable fields")

# Prepare editable values
current_checkable = row.get("checkable", "")
current_expl = row.get("explanation", "")
current_q = row.get("question", "")
current_details = row.get("details_text", "")
current_alerts_raw = row.get("alerts", "[]")

alerts_list = _parse_alerts_to_list(current_alerts_raw)
alerts_text = _alerts_list_to_text(alerts_list)

left, right = st.columns([1, 1])

with left:
    checkable = st.selectbox(
        "checkable",
        options=CHECKABLE_OPTIONS,
        index=CHECKABLE_OPTIONS.index(current_checkable) if current_checkable in CHECKABLE_OPTIONS else 0,
    )
    explanation = st.text_area("explanation", value=str(current_expl or ""), height=140)
    question = st.text_area("question", value=str(current_q or ""), height=120)

with right:
    details_text = st.text_area("details_text", value=str(current_details or ""), height=260)
    alerts_edit = st.text_area(
        "alerts (one per line)",
        value=alerts_text,
        height=170,
        help="Edit as newline-separated list. Will be stored as JSON list in the dataset.",
    )

# Apply edits button
apply_col1, apply_col2, apply_col3 = st.columns([1.2, 1.2, 4])
with apply_col1:
    apply_now = st.button("Apply edits to this row", type="primary", use_container_width=True)
with apply_col2:
    revert_row = st.button("Revert row to saved", use_container_width=True)

if revert_row:
    st.experimental_rerun()

if apply_now:
    # Convert alerts text to JSON list string
    alerts_lines = [ln.strip() for ln in alerts_edit.splitlines() if ln.strip()]
    alerts_json = json.dumps(alerts_lines, ensure_ascii=False)

    updates = {
        "checkable": checkable,
        "explanation": explanation,
        "question": question,
        "details_text": details_text,
        "alerts": alerts_json,
    }
    st.session_state.df = _write_back(st.session_state.df, idx, updates)
    st.session_state.dirty = True
    st.success("Edits applied to this row.")

st.divider()

# Download edited CSV
download_name = f"edited_{st.session_state.source_name}"
csv_bytes = st.session_state.df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download edited CSV (utf-8, comma-separated)",
    data=csv_bytes,
    file_name=download_name,
    mime="text/csv",
    use_container_width=True,
)

# Optional overwrite to disk (local)
if allow_overwrite:
    if st.button("Overwrite file on disk", use_container_width=True):
        Path(overwrite_path).parent.mkdir(parents=True, exist_ok=True)
        st.session_state.df.to_csv(overwrite_path, index=False)
        st.success(f"Saved to: {overwrite_path}")

if show_raw:
    st.subheader("Raw row (current DataFrame values)")
    st.json(st.session_state.df.iloc[idx].to_dict())
