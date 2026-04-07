import streamlit as st
import pandas as pd
import time
import sqlite3
import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipeline.predict import predict

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Spam Detector",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── DB setup ─────────────────────────────────────────────────
DB_PATH = "data/logs.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS email_logs (
            id              TEXT PRIMARY KEY,
            timestamp       TEXT,
            sender          TEXT,
            subject         TEXT,
            label           TEXT,
            spam_confidence REAL,
            ham_confidence  REAL,
            feedback        TEXT DEFAULT NULL
        )
    """)
    conn.commit()
    conn.close()

def save_to_db(results):
    conn = sqlite3.connect(DB_PATH)
    for r in results:
        if not r.get("id") or not r.get("label"):
            continue
        conn.execute("""
            INSERT OR IGNORE INTO email_logs
            (id, timestamp, sender, subject, label, spam_confidence, ham_confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            r.get("id", "unknown"),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            r.get("sender", "Unknown"),
            r.get("subject", "No subject"),
            r.get("label", "unknown"),
            r.get("spam_confidence", 0.0),
            r.get("ham_confidence", 0.0)
        ))
    conn.commit()
    conn.close()

def load_logs():
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql("SELECT * FROM email_logs ORDER BY timestamp DESC", conn)
    conn.close()
    return df
def update_feedback(email_id, correct_label):
    """Flip the label in DB and append corrected email to training CSV."""
    conn = sqlite3.connect(DB_PATH)

    # Update feedback column in DB
    conn.execute("""
        UPDATE email_logs
        SET feedback = ?, label = ?
        WHERE id = ?
    """, (f"corrected_to_{correct_label}", correct_label, email_id))
    conn.commit()

    # Fetch the subject + body preview to append to training data
    row = conn.execute(
        "SELECT subject FROM email_logs WHERE id = ?", (email_id,)
    ).fetchone()
    conn.close()

    # Append corrected sample to training CSV
    if row:
        import csv
        csv_path = "data/spam.csv"
        with open(csv_path, "a", newline="", encoding="latin-1") as f:
            writer = csv.writer(f)
            writer.writerow([correct_label, row[0]])
        print(f"Appended correction to {csv_path}: [{correct_label}] {row[0]}")

def get_feedback_stats():
    """Returns count of corrections made."""
    conn = sqlite3.connect(DB_PATH)
    count = conn.execute(
        "SELECT COUNT(*) FROM email_logs WHERE feedback IS NOT NULL"
    ).fetchone()[0]
    conn.close()
    return count
def retrain_model():
    """Re-runs train.py on updated dataset. Returns (success, message)."""
    import subprocess
    try:
        result = subprocess.run(
            [sys.executable, "pipeline/train.py"],
            capture_output=True,
            text=True,
            timeout=120
        )
        if result.returncode == 0:
            # Clear predict cache so new model loads immediately
            import importlib
            import pipeline.predict as pred_module
            importlib.reload(pred_module)
            return True, result.stdout
        else:
            return False, result.stderr
    except subprocess.TimeoutExpired:
        return False, "Retraining timed out after 120 seconds."
    except Exception as e:
        return False, str(e)

init_db()

# ── Try importing Gmail fetcher ───────────────────────────────
gmail_available = False
try:
    from pipeline.email_fetcher import fetch_and_predict
    gmail_available = True
except Exception as e:
    gmail_available = False
    gmail_import_error = str(e)

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.title("📧 Spam Detector")
    st.markdown("---")

    st.subheader("Settings")
    fetch_limit = st.slider("Emails to fetch", 5, 50, 20)
    threshold   = st.slider(
        "Spam threshold", 0.1, 1.0, 0.6, 0.05,
        help="Higher = stricter. Lower = catches more spam."
    )

    st.markdown("---")
    st.subheader("Auto Refresh")
    auto_refresh = st.toggle("Enable auto-refresh", value=False)
    if auto_refresh:
        try:
            from streamlit_autorefresh import st_autorefresh
            refresh_secs = st.selectbox("Interval", [30, 60, 120, 300], index=1)
            st_autorefresh(interval=refresh_secs * 1000, key="auto_refresher")
            st.success(f"Refreshing every {refresh_secs}s")
        except ImportError:
            st.warning("Run: pip install streamlit-autorefresh")

    st.markdown("---")
    manual_refresh = st.button("🔄 Refresh Inbox Now", use_container_width=True)

    st.markdown("---")
    st.subheader("Model Retraining")

    corrections = get_feedback_stats()
    st.metric("Corrections collected", corrections)

    if corrections > 0:
        st.caption(f"{corrections} emails marked as wrong — ready to retrain.")
    else:
        st.caption("Mark emails as wrong to improve the model.")

    retrain_btn = st.button(
        "🧠 Retrain Model Now",
        use_container_width=True,
        disabled=(corrections == 0),
        help="Retrains on original + corrected emails"
    )

    if retrain_btn:
        with st.spinner("Retraining... this takes ~30 seconds"):
            success, message = retrain_model()
        if success:
            st.success("Model retrained successfully!")
            st.cache_data.clear()
            lines = [l for l in message.split("\n") if l.strip()]
            for line in lines[-6:]:
                st.caption(line)
        else:
            st.error("Retraining failed.")
            st.code(message)

    st.markdown("---")
    st.caption("Built with TF-IDF + Naive Bayes")

# ── Fetch helper ─────────────────────────────────────────────
@st.cache_data(ttl=60, show_spinner=False)
def get_emails(limit):
    return fetch_and_predict(max_results=limit)

if manual_refresh:
    st.cache_data.clear()

# ── Tabs ─────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📥 Live Inbox", "🚫 Spam Log", "🔍 Manual Test"])


# ════════════════════════════════════════════════════════════
# TAB 1 — Live Inbox
# ════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Live Inbox")

    if not gmail_available:
        st.error("Gmail not connected. Check credentials.json and token.json.")
        st.info("You can still use the Manual Test tab while fixing Gmail.")
    else:
        with st.spinner("Fetching emails from Gmail..."):
            try:
                results = get_emails(fetch_limit)
                if results:
                    save_to_db(results)
            except Exception as e:
                st.error(f"Could not fetch emails: {e}")
                results = []

        if results:
            total      = len(results)
            spam_count = sum(1 for r in results if r.get("is_spam", False))
            ham_count  = total - spam_count
            spam_pct   = (spam_count / total * 100) if total > 0 else 0

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total fetched",  total)
            col2.metric("Spam detected",  spam_count,
                        delta=f"{spam_pct:.1f}%", delta_color="inverse")
            col3.metric("Clean (ham)",    ham_count)
            col4.metric("Threshold",      f"{threshold:.0%}")

            st.markdown("---")

            df = pd.DataFrame(results)
            display_df = df[["sender", "subject", "date", "label",
                              "spam_confidence", "ham_confidence"]].copy()
            display_df["spam_confidence"] = (
                display_df["spam_confidence"] * 100
            ).round(1).astype(str) + "%"
            display_df["ham_confidence"] = (
                display_df["ham_confidence"] * 100
            ).round(1).astype(str) + "%"
            display_df.columns = [
                "Sender", "Subject", "Date",
                "Label", "Spam %", "Ham %"
            ]

            def color_rows(row):
                color = (
                    "background-color: #ffecec;"
                    if row["Label"] == "spam"
                    else "background-color: #ecffec;"
                )
                return [color] * len(row)

            st.dataframe(
                display_df.style.apply(color_rows, axis=1),
                use_container_width=True,
                height=400
            )

            st.markdown("---")
            st.subheader("Email Details")

            select_options = range(len(results))
            selected = st.selectbox(
                "Select an email to inspect",
                options=select_options,
                format_func=lambda i: (
                    f"[{results[i].get('label','?').upper()}] "
                    f"{results[i].get('subject', 'No subject')[:60]}"
                )
            )

            if selected is not None and selected < len(results):
                r = results[selected]

                if "label" not in r:
                    st.warning("This email could not be analysed.")
                else:
                    is_spam = r.get("is_spam", False)
                    badge   = "🔴 SPAM" if is_spam else "🟢 HAM"
                    spam_conf = float(r.get("spam_confidence", 0.0))

                    st.markdown(f"**Status:** {badge}")
                    st.markdown(f"**From:** {r.get('sender', 'Unknown')}")
                    st.markdown(f"**Subject:** {r.get('subject', 'No subject')}")
                    st.markdown(f"**Date:** {r.get('date', 'Unknown')}")
                    st.progress(
                        spam_conf,
                        text=f"Spam confidence: {spam_conf * 100:.1f}%"
                    )

                    d1, d2, d3 = st.columns(3)
                    d1.metric("Label",      r.get("label", "unknown").upper())
                    d2.metric("Spam score", f"{spam_conf * 100:.1f}%")
                    d3.metric("Threshold",  f"{r.get('threshold_used', 0.6)*100:.0f}%")

                    with st.expander("Show email body preview"):
                        st.text(r.get("body_preview", "No preview available"))
        else:
            st.info("No unread emails found, or inbox is empty.")


# ════════════════════════════════════════════════════════════
# TAB 2 — Spam Log
# ════════════════════════════════════════════════════════════
with tab2:
    st.subheader("All Scanned Emails")

    logs = load_logs()

    if not logs.empty:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total scanned", len(logs))
        c2.metric(
            "Total spam",
            len(logs[logs["label"] == "spam"]) if "label" in logs.columns else 0
        )
        c3.metric(
            "Total ham",
            len(logs[logs["label"] == "ham"]) if "label" in logs.columns else 0
        )
        c4.metric("Corrections", get_feedback_stats())

        st.markdown("---")

        f1, f2 = st.columns(2)
        with f1:
            label_filter = st.selectbox("Filter by label", ["All", "spam", "ham"])
        with f2:
            search_term = st.text_input("Search subject or sender")

        filtered = logs.copy()
        if label_filter != "All":
            filtered = filtered[filtered["label"] == label_filter]
        if search_term:
            mask = (
                filtered["subject"].str.contains(search_term, case=False, na=False) |
                filtered["sender"].str.contains(search_term, case=False, na=False)
            )
            filtered = filtered[mask]

        st.markdown("---")

        # Show each email row with a feedback button
        for _, row in filtered.iterrows():
            col_info, col_btn = st.columns([5, 1])

            with col_info:
                label_badge = "🔴 SPAM" if row["label"] == "spam" else "🟢 HAM"
                feedback_tag = ""
                if pd.notna(row.get("feedback")) and row["feedback"]:
                    feedback_tag = " ✏️ corrected"
                st.markdown(
                    f"**{label_badge}{feedback_tag}** — {row['subject'][:60]}  \n"
                    f"<span style='font-size:12px;color:gray;'>"
                    f"{row['sender'][:50]} &nbsp;|&nbsp; "
                    f"Spam: {row['spam_confidence']*100:.1f}% &nbsp;|&nbsp; "
                    f"{row['timestamp']}</span>",
                    unsafe_allow_html=True
                )

            with col_btn:
                # Flip label direction
                correct_label = "ham" if row["label"] == "spam" else "spam"
                btn_label     = "Mark HAM" if row["label"] == "spam" else "Mark SPAM"
                btn_key       = f"feedback_{row['id']}"

                if st.button(btn_label, key=btn_key):
                    update_feedback(row["id"], correct_label)
                    st.success(f"Marked as {correct_label.upper()}. Thanks!")
                    st.rerun()

            st.divider()

        st.markdown("---")
        csv = filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇ Download log as CSV",
            csv, "spam_log.csv", "text/csv"
        )
    else:
        st.info("No emails scanned yet. Go to Live Inbox tab to fetch emails.")

# ════════════════════════════════════════════════════════════
# TAB 3 — Manual Test
# ════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Test Any Email Manually")
    st.caption("Paste any email subject + body below to check if it's spam.")

    sample_spam = (
        "WINNER!! You have been selected for a $1000 gift card. "
        "Call NOW to claim your FREE prize! Limited time offer!"
    )
    sample_ham = (
        "Hi, just checking in about tomorrow's meeting. "
        "Can you send over the agenda when you get a chance? Thanks!"
    )

    col_a, col_b = st.columns(2)
    if col_a.button("Load spam example"):
        st.session_state["manual_input"] = sample_spam
    if col_b.button("Load ham example"):
        st.session_state["manual_input"] = sample_ham

    user_input = st.text_area(
        "Email text",
        value=st.session_state.get("manual_input", ""),
        height=180,
        max_chars=3000,
        placeholder="Paste your email subject + body here..."
    )

    predict_btn = st.button(
        "Analyse Email", type="primary", use_container_width=True
    )

    if predict_btn and user_input.strip():
        with st.spinner("Analysing..."):
            time.sleep(0.3)
            try:
                result = predict(user_input)

                st.markdown("---")

                if result.get("is_spam"):
                    st.error("🚨 SPAM DETECTED")
                else:
                    st.success("✅ CLEAN EMAIL (Ham)")

                r1, r2 = st.columns(2)
                with r1:
                    st.markdown("**Spam confidence**")
                    st.progress(
                        float(result.get("spam_confidence", 0)),
                        text=f"{result.get('spam_confidence', 0) * 100:.1f}%"
                    )
                with r2:
                    st.markdown("**Ham confidence**")
                    st.progress(
                        float(result.get("ham_confidence", 0)),
                        text=f"{result.get('ham_confidence', 0) * 100:.1f}%"
                    )

                st.markdown("---")
                m1, m2, m3 = st.columns(3)
                m1.metric("Label",      result.get("label", "unknown").upper())
                m2.metric("Spam score", f"{result.get('spam_confidence', 0)*100:.1f}%")
                m3.metric("Threshold",  f"{result.get('threshold_used', 0.6)*100:.0f}%")

            except Exception as e:
                st.error(f"Prediction failed: {e}")

    elif predict_btn and not user_input.strip():
        st.warning("Please enter some text before analysing.")