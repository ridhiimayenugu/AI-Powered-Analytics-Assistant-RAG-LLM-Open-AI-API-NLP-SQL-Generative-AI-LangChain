import os
import re
import html as _html
import streamlit as st

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from db_utils import (
    get_engine,
    list_user_databases,
    fetch_schema_summary,
    run_sql,
)

load_dotenv()

st.set_page_config(page_title="SQL Server RAG Assistant", layout="wide")
st.title("SQL Server RAG Assistant")

# -----------------------------
# Helpers / guardrails
# -----------------------------
DESTRUCTIVE_PATTERNS = [
    r"\bdrop\b",
    r"\btruncate\b",
    r"\bdelete\b",
    r"\bupdate\b",
    r"\binsert\b",
    r"\balter\b",
    r"\bcreate\b",
    r"\bgrant\b",
    r"\brevoke\b",
    r"\bexecute\b",
    r"\bexec\b",
]


def looks_destructive(sql: str) -> bool:
    s = sql.lower()
    return any(re.search(p, s) for p in DESTRUCTIVE_PATTERNS)


def extract_sql(text: str) -> str:
    m = re.search(r"```sql\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*(.*?)```", text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def df_to_markdown_sample(df, max_rows=20, max_cols=12) -> str:
    if df is None:
        return ""
    d = df.copy()
    if d.shape[1] > max_cols:
        d = d.iloc[:, :max_cols]
    if d.shape[0] > max_rows:
        d = d.head(max_rows)
    return d.to_markdown(index=False)


def build_sql_system_prompt(schema_summary: str, db_name: str) -> str:
    return f"""
You are a senior data analyst writing SQL Server T-SQL.

Rules:
- Output ONLY a single SQL SELECT statement. No explanations.
- SQL Server syntax only.
- Use TOP (N), NOT LIMIT.
- Qualify tables with schema when possible (e.g., dbo.Table).
- Prefer simple, readable SQL.
- Do not use destructive statements (DROP/DELETE/UPDATE/INSERT/TRUNCATE/ALTER/CREATE).
- If the question is ambiguous, make a reasonable assumption and still produce a SELECT query.

Target database: {db_name}

Database schema (tables and columns):
{schema_summary}
""".strip()


def build_answer_system_prompt(
    schema_summary: str, db_name: str, extra: str = ""
) -> str:
    return f"""
You are a helpful assistant answering questions about a SQL Server database.

Rules:
- If you do NOT have enough data to answer confidently, say what is missing and suggest what query/info would be needed.
- Do NOT invent table/column names not present in the schema.
- Keep answers short and directly useful.

Target database: {db_name}

Schema reference:
{schema_summary}

{extra}
""".strip()


def clipboard_button(label: str, text: str, key: str):
    safe_text = _html.escape(text).replace("\n", "\\n")
    btn_id = f"copybtn_{key}"
    st.components.v1.html(
        f"""
        <div style="display:flex; align-items:center; gap:8px; margin: 4px 0 10px 0;">
          <button id="{btn_id}" style="
              padding:6px 10px;
              border-radius:8px;
              border:1px solid #ddd;
              background:#fff;
              cursor:pointer;">
            {_html.escape(label)}
          </button>
          <span id="{btn_id}_status" style="font-size:12px; color:#666;"></span>
        </div>

        <script>
        const btn = document.getElementById("{btn_id}");
        const status = document.getElementById("{btn_id}_status");
        const txt = "{safe_text}";
        btn.addEventListener("click", async () => {{
          try {{
            await navigator.clipboard.writeText(txt.replace(/\\\\n/g, "\\n"));
            status.textContent = "Copied!";
            setTimeout(() => status.textContent = "", 1200);
          }} catch (e) {{
            status.textContent = "Copy failed (browser blocked).";
            setTimeout(() => status.textContent = "", 2000);
          }}
        }});
        </script>
        """,
        height=48,
    )


# -----------------------------
# Session state defaults
# -----------------------------
DEFAULTS = {
    "question": "",
    "generated_sql": "",
    "sql_editor": "",
    "last_df": None,
    "last_answer": "",
    "chat_history": [],
    "filter_tables": "",
    "chosen_table": None,
    "__reset_requested__": False,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


def request_reset():
    st.session_state["__reset_requested__"] = True


# IMPORTANT: perform reset BEFORE any widgets are created
if st.session_state.get("__reset_requested__", False):
    keys_to_clear = [
        "question",
        "generated_sql",
        "sql_editor",
        "last_df",
        "last_answer",
        "chat_history",
        "filter_tables",
        "chosen_table",
    ]
    for k in keys_to_clear:
        if k in st.session_state:
            del st.session_state[k]
    st.session_state["__reset_requested__"] = False
    st.rerun()

# -----------------------------
# Sidebar: Connection + Model + Full Schema
# -----------------------------
with st.sidebar:
    st.header("Connection")
    try:
        engine_master = get_engine("master")
        dbs = list_user_databases(engine_master)
    except Exception as e:
        st.error(f"Could not connect to SQL Server (master): {e}")
        st.stop()

    default_db = os.getenv("SQL_SERVER_DB", dbs[0] if dbs else "master")
    selected_db = st.selectbox(
        "Select database",
        options=dbs if dbs else [default_db],
        index=(dbs.index(default_db) if dbs and default_db in dbs else 0),
        key="selected_db",
    )

    st.divider()
    st.header("Model")
    model_name = st.text_input(
        "OPENAI_MODEL", value=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), key="model_name"
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1, key="temperature")
    st.caption("Ensure OPENAI_API_KEY is set in .env")


# -----------------------------
# Schema caching
# -----------------------------
@st.cache_data(show_spinner=False)
def _cached_schema(db_name: str):
    engine = get_engine(db_name)
    return fetch_schema_summary(engine, max_tables=250, include_views=False)


# -----------------------------
# Top controls
# -----------------------------
top_left, top_right = st.columns([1, 1], gap="large")

with top_left:
    st.subheader("System Information")
    st.write(f"**Selected DB:** `{selected_db}`")
    st.write("**Available User Databases:** " + ", ".join(dbs) if dbs else "None found")
    refresh = st.button(
        "Refresh schema", help="Re-load schema metadata for the selected database"
    )

with top_right:
    st.subheader("Ask a question")
    mode = st.radio(
        "Mode",
        options=[
            "Run SQL & Answer (recommended)",
            "Answer only (no SQL)",
            "Generate SQL only",
        ],
        horizontal=True,
        key="mode",
    )

    st.text_area(
        "Question",
        placeholder="Example: What were total sales by month in 2014?",
        height=120,
        key="question",
    )

    show_sql = st.checkbox("Show SQL editor", value=False, key="show_sql")
    auto_run = st.checkbox(
        "Auto-run after SQL generate",
        value=False,
        help="When enabled (and mode is Generate SQL only), SQL will run immediately after generation.",
        key="auto_run",
    )

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        go = st.button("Go", width="stretch")
    with c2:
        run_only = st.button(
            "Run SQL", width="stretch", help="Runs whatever SQL is in the editor"
        )
    with c3:
        clear = st.button("Clear", width="stretch")

if clear:
    request_reset()
    st.rerun()

if refresh:
    _cached_schema.clear()
    # also clear stale chosen table when schema is refreshed
    if "chosen_table" in st.session_state:
        del st.session_state["chosen_table"]

try:
    schema_summary, tables_struct = _cached_schema(selected_db)
except Exception as e:
    st.error(f"Failed to read schema from {selected_db}: {e}")
    st.stop()

with st.sidebar:
    st.divider()
    with st.expander("Full schema (tables + columns)", expanded=False):
        st.caption("Tip: use Table browser on the main page for a cleaner view.")
        st.container(height=450).code(
            schema_summary if schema_summary else "No schema data (after filters).",
            language="text",
        )

# -----------------------------
# Main layout
# -----------------------------
left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("Schema Tools")
    st.text_input(
        "Filter tables",
        value=st.session_state.get("filter_tables", ""),
        placeholder="Type to filter (e.g., FactInternetSales, Sales., dbo.Customer)",
        key="filter_tables",
    )
    filter_text = (st.session_state.get("filter_tables") or "").strip().lower()

    st.markdown("#### Table browser")
    if tables_struct:
        table_labels_all = [f"{t['schema']}.{t['name']}" for t in tables_struct]
        table_labels = (
            [x for x in table_labels_all if filter_text in x.lower()]
            if filter_text
            else table_labels_all
        )

        if not table_labels:
            st.warning("No tables match your filter.")
            if "chosen_table" in st.session_state:
                del st.session_state["chosen_table"]
        else:
            current_choice = st.session_state.get("chosen_table")
            if current_choice not in table_labels:
                st.session_state["chosen_table"] = table_labels[0]

            chosen = st.selectbox(
                "Pick a table to view columns", options=table_labels, key="chosen_table"
            )

            chosen_table = next(
                (t for t in tables_struct if f"{t['schema']}.{t['name']}" == chosen),
                None,
            )

            if chosen_table is None:
                st.warning(
                    "Selected table could not be found. Please pick another table."
                )
            else:
                cols = chosen_table.get("columns", [])
                st.write(f"**{chosen}** — {len(cols)} columns")
                st.dataframe(
                    [{"column": c["name"], "type": c["type"]} for c in cols],
                    width="stretch",
                    hide_index=True,
                )
    else:
        st.info("No tables found after filters.")
        if "chosen_table" in st.session_state:
            del st.session_state["chosen_table"]

with right:
    st.subheader("SQL / Results / Answer")

    if show_sql:
        st.text_area(
            "SQL (editable)",
            value=st.session_state.get("generated_sql", ""),
            height=200,
            placeholder="SQL will appear here when generated. You can also paste your own SELECT.",
            key="generated_sql",
        )
        if st.session_state.get("generated_sql", "").strip():
            clipboard_button("Copy SQL", st.session_state["generated_sql"], key="sql")
            with st.expander("Show SQL as code block", expanded=False):
                st.code(st.session_state["generated_sql"], language="sql")
    else:
        if st.session_state.get("generated_sql", "").strip():
            st.caption("SQL is hidden. Enable 'Show SQL editor' to view/copy it.")

    st.markdown("#### Results")
    if st.session_state.get("last_df") is not None:
        st.dataframe(st.session_state["last_df"], width="stretch")
        st.download_button(
            "Download CSV",
            data=st.session_state["last_df"].to_csv(index=False).encode("utf-8"),
            file_name=f"{selected_db}_results.csv",
            mime="text/csv",
        )
    else:
        st.info("No results yet.")

    st.markdown("#### Answer")
    if st.session_state.get("last_answer"):
        st.write(st.session_state["last_answer"])
    else:
        st.caption("No answer yet.")

    st.divider()
    st.subheader("Chat with your data (follow-ups)")

    for role, msg in st.session_state.get("chat_history", []):
        st.chat_message(role).write(msg)

    chat_input = st.chat_input(
        "Ask a follow-up (e.g., 'break it down by year', 'only 2014')"
    )


# -----------------------------
# LLM / execution functions
# -----------------------------
def llm_client():
    return ChatOpenAI(
        model=st.session_state["model_name"],
        temperature=st.session_state["temperature"],
    )


def generate_sql(question: str) -> str:
    llm = llm_client()
    system_prompt = build_sql_system_prompt(schema_summary, selected_db)
    resp = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=question)]
    )
    sql = extract_sql(resp.content)

    if looks_destructive(sql):
        raise ValueError("Model produced potentially destructive SQL.")
    if not sql.strip().lower().startswith("select"):
        raise ValueError("Model did not produce a SELECT statement.")
    return sql


def run_sql_query(sql: str):
    if looks_destructive(sql):
        raise ValueError("Blocked: destructive SQL detected.")
    engine = get_engine(selected_db)
    return run_sql(engine, sql, max_rows=2000)


def answer_from_schema_only(question: str) -> str:
    llm = llm_client()
    system_prompt = build_answer_system_prompt(schema_summary, selected_db)
    resp = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=question)]
    )
    return resp.content.strip()


def answer_from_results(question: str, sql: str, df) -> str:
    llm = llm_client()
    sample_md = df_to_markdown_sample(df, max_rows=20, max_cols=12)
    extra = f"""
You also have ACTUAL query results.

SQL executed:
{sql}

Result sample:
{sample_md}

Instructions:
- Answer the user's question using the results.
- If more data would be needed, say what and suggest the next query.
""".strip()

    system_prompt = build_answer_system_prompt(schema_summary, selected_db, extra=extra)
    resp = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=question)]
    )
    return resp.content.strip()


def chat_answer(message: str) -> str:
    llm = llm_client()
    df = st.session_state.get("last_df")
    last_sql = st.session_state.get("generated_sql", "")
    result_note = ""
    if df is not None and len(df) > 0:
        result_note = f"""
Last executed SQL:
{last_sql}

Last result sample:
{df_to_markdown_sample(df)}
""".strip()

    system_prompt = build_answer_system_prompt(
        schema_summary, selected_db, extra=result_note
    )

    messages = [SystemMessage(content=system_prompt)]
    for role, txt in st.session_state.get("chat_history", [])[-10:]:
        if role == "user":
            messages.append(HumanMessage(content=txt))
        else:
            messages.append(AIMessage(content=txt))
    messages.append(HumanMessage(content=message))

    resp = llm.invoke(messages)
    return resp.content.strip()


# -----------------------------
# Button actions
# -----------------------------
if go and st.session_state["question"].strip():
    q = st.session_state["question"].strip()
    try:
        if st.session_state["mode"] == "Answer only (no SQL)":
            st.session_state["last_df"] = None
            st.session_state["last_answer"] = answer_from_schema_only(q)

        elif st.session_state["mode"] == "Generate SQL only":
            st.session_state["last_df"] = None
            st.session_state["last_answer"] = ""
            sql = generate_sql(q)
            st.session_state["generated_sql"] = sql

            if st.session_state.get("auto_run"):
                df = run_sql_query(sql)
                st.session_state["last_df"] = df
                st.session_state["last_answer"] = answer_from_results(q, sql, df)

        else:  # Run SQL & Answer
            sql = generate_sql(q)
            st.session_state["generated_sql"] = sql
            df = run_sql_query(sql)
            st.session_state["last_df"] = df
            st.session_state["last_answer"] = answer_from_results(q, sql, df)

        st.rerun()
    except Exception as e:
        st.error(str(e))

if run_only and st.session_state.get("generated_sql", "").strip():
    try:
        sql = st.session_state["generated_sql"].strip()
        df = run_sql_query(sql)
        st.session_state["last_df"] = df

        if st.session_state["mode"] == "Run SQL & Answer (recommended)":
            q = st.session_state["question"].strip() or "Summarize the query results."
            st.session_state["last_answer"] = answer_from_results(q, sql, df)

        st.rerun()
    except Exception as e:
        st.error(str(e))

if chat_input:
    st.session_state["chat_history"].append(("user", chat_input))
    try:
        ans = chat_answer(chat_input)
        st.session_state["chat_history"].append(("assistant", ans))
        st.rerun()
    except Exception as e:
        st.error(str(e))
