# app/ui/app.py
from __future__ import annotations

import os
import pathlib
import requests
import streamlit as st
from PIL import Image

# ---------------------------
# Config & assets
# ---------------------------
def _find_logo_path() -> pathlib.Path | None:
    env_logo = os.environ.get("APP_LOGO")
    if env_logo:
        p = pathlib.Path(env_logo.replace("\\", "/"))
        if p.exists():
            return p
    for candidate in (
        pathlib.Path("Arabic model/logo.png"),
        pathlib.Path("app/ui/assets/moc_logo.png"),
        pathlib.Path("logo.png"),
    ):
        if candidate.exists():
            return candidate
    return None

def _resolve_api_default() -> str:
    env = os.environ.get("API_BASE_URL")
    if env:
        return env
    try:
        return st.secrets["API_BASE_URL"]  # type: ignore
    except Exception:
        return "http://127.0.0.1:8000"

LOGO_PATH = _find_logo_path()
API_DEFAULT = _resolve_api_default()

_page_icon = None
try:
    if LOGO_PATH is not None:
        _page_icon = Image.open(LOGO_PATH)
except Exception:
    _page_icon = None

st.set_page_config(
    page_title="Mini Al-Jazari",
    layout="centered",
    page_icon=_page_icon,
)

# ---------------------------
# Style (subtle facelift)
# ---------------------------
st.markdown(
    """
<style>
/* RTL base */
html, body, [class*="css"] { direction: rtl; text-align: right; }
code, pre, .stMarkdown pre { direction: ltr; text-align: left; }

/* Page background + typography */
body {
  background: linear-gradient(180deg, #fafafa 0%, #f5f7fb 60%, #eef1f6 100%);
}
h1, h2, h3 { letter-spacing: 0.2px; }

/* Card containers */
.alz-card {
  background: #ffffffAA;
  border: 1px solid #e9eef5;
  border-radius: 16px;
  padding: 18px 18px 14px 18px;
  box-shadow: 0 4px 18px rgba(17, 24, 39, 0.06);
}

/* Section header chip */
.alz-chip {
  display: inline-block;
  font-size: 12px;
  padding: 4px 10px;
  border-radius: 999px;
  background: #eef2ff;
  border: 1px solid #e5e7ff;
  color: #3949ab;
  margin-bottom: 6px;
}

/* Primary button (slightly rounded) */
.stButton button[kind="primary"] {
  border-radius: 10px;
  padding: 0.6rem 1.1rem;
}

/* Tabs polish */
.stTabs [data-baseweb="tab-list"] {
  gap: 6px;
}
.stTabs [data-baseweb="tab"] {
  background: #ffffff78;
  border: 1px solid #e6ecf5;
  border-radius: 12px 12px 0 0;
  padding: 10px 14px;
}

/* Small status pill */
.alz-status {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  padding: 4px 10px;
  border-radius: 999px;
  border: 1px solid #e8eef6;
  background: #ffffffd9;
}
.alz-dot {
  width: 8px; height: 8px; border-radius: 50%;
  display: inline-block;
}
.alz-dot.ok { background: #10b981; }       /* green */
.alz-dot.bad { background: #ef4444; }      /* red */

/* Top header group */
.alz-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 10px;
}
.alz-title {
  display: flex;
  align-items: center;
  gap: 12px;
}
.alz-logo {
  max-width: 140px;
  border-radius: 12px;
}

/* Subtle divider space */
.alz-spacer { height: 10px; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------
# Header (logo + title + status)
# ---------------------------
colA, colB = st.columns([6, 2], vertical_alignment="center")
with colA:
    st.markdown('<div class="alz-header">', unsafe_allow_html=True)
    st.markdown('<div class="alz-title">', unsafe_allow_html=True)
    if LOGO_PATH is not None:
        st.image(str(LOGO_PATH), caption=None, width=140)
    st.markdown("</div>", unsafe_allow_html=True)  # .alz-title
    st.title("نموذج الجزري المصغّر")
    st.caption("مبني على «مجموعة بيانات للتعرّف على اللهجات السعودية المتعددة عبر كلمات الأغاني»")
    st.markdown("</div>", unsafe_allow_html=True)  # .alz-header
with colB:
    # We'll resolve API status silently (no URL shown)
    api_url = API_DEFAULT

    status_text = "غير متصل"
    docs_count = "—"
    ok = False
    try:
        r = requests.get(f"{api_url}/", timeout=4)
        if r.ok:
            js = r.json()
            docs_count = js.get("docs_count", "—")
            status_text = f"متصل — {docs_count} مقطع"
            ok = True
    except Exception:
        ok = False

    dot_class = "ok" if ok else "bad"
    pill = f"""
    <div class="alz-status">
        <span class="alz-dot {dot_class}"></span>
        <span>{status_text}</span>
    </div>
    """
    st.markdown(pill, unsafe_allow_html=True)

st.markdown('<div class="alz-spacer"></div>', unsafe_allow_html=True)

# ---------------------------
# Tabs
# ---------------------------
tabs = st.tabs(["🧠 تصنيف السطر", "🔎 استرجاع (RAG)", "ℹ️ عن"])

# -------- Tab 1: Classify --------
with tabs[0]:
    st.markdown('<div class="alz-chip">تحليل نص</div>', unsafe_allow_html=True)
    st.markdown('<div class="alz-card">', unsafe_allow_html=True)

    st.subheader("أدخل بيتًا أو سطرًا شعريًا للتحليل")
    text = st.text_area("النص:", "", height=100)

    go = st.button("تحليل", type="primary")
    if go:
        if text.strip():
            try:
                with st.spinner("جارٍ التحليل..."):
                    r = requests.post(
                        f"{API_DEFAULT}/classify",
                        json={"text": text},
                        headers={"Content-Type": "application/json; charset=utf-8"},
                        timeout=15,
                    )
                    r.raise_for_status()
                    data = r.json()
                c1, c2 = st.columns(2)
                with c1:
                    st.write(f"**اللهجة:** {data.get('dialect','—')}")
                    dc = data.get("dialect_confidence")
                    if dc is not None:
                        st.write(f"**موثوقية اللهجة:** {dc:.3f}")
                with c2:
                    st.write(f"**الثيم:** {data.get('theme','—')}")
                    conf = data.get("confidence")
                    if conf is not None:
                        st.write(f"**موثوقية الثيم:** {conf:.3f}")
            except Exception as e:
                st.error(f"خطأ في الاستدعاء: {e}")
        else:
            st.warning("اكتب نصًا أولًا.")

    st.markdown('</div>', unsafe_allow_html=True)  # .alz-card

# -------- Tab 2: Ask / RAG --------
with tabs[1]:
    st.markdown('<div class="alz-chip">بحث ذكي</div>', unsafe_allow_html=True)
    st.markdown('<div class="alz-card">', unsafe_allow_html=True)

    st.subheader("اسأأدخل ما تُريد البحث عنه (مثل: بيت عن البحر)")
    question = st.text_input("سؤالك:", "")
    k = st.number_input("عدد النتائج", min_value=1, max_value=10, value=3, step=1)
    c1, c2 = st.columns(2)
    with c1:
        theme_hint = st.selectbox("الثيم (اختياري)", ["", "غزل", "وطنية", "رياضية", "دينية"])
    with c2:
        dialect_filter = st.selectbox("اللهجة (اختياري)", ["", "حجازي", "نجدي", "شمالي", "جنوبي", "شرقاوي"])

    if st.button("اسأل", type="primary"):
        if not question.strip():
            st.warning("اكتب سؤالك أولاً.")
        else:
            payload = {
                "question": question,
                "k": int(k),
                "show_sources": True,
                "theme_hint": theme_hint or None,
                "dialect_filter": dialect_filter or None,
            }
            try:
                with st.spinner("جارٍ الاسترجاع..."):
                    r = requests.post(
                        f"{API_DEFAULT}/ask",
                        json=payload,
                        headers={"Content-Type": "application/json; charset=utf-8"},
                        timeout=30,
                    )
                    r.raise_for_status()
                    data = r.json()

                # Display answer only (no sources expander)
                st.text_area("الإجابة", data.get("answer", ""), height=180)

            except Exception as e:
                st.error(f"خطأ في الاستدعاء: {e}")

    st.markdown('</div>', unsafe_allow_html=True)


# -------- Tab 3: About --------
with tabs[2]:
    st.markdown('<div class="alz-chip">حول المنتج</div>', unsafe_allow_html=True)
    st.markdown('<div class="alz-card">', unsafe_allow_html=True)

    st.info("تنويه: هذا المنتج هو مجرد نموذج أولي (Prototype) ويُمثّل تطبيقًا عمليًا لفكرة بحثية علمية. النموذج قيد البناء والتقييم.")
    st.markdown(
        """
نموذج مصغّر يعرض:
- تصنيف **الثيم** (وطنية/غزل/دينية/رياضية) + **اللهجة**.
- استرجاع مقاطع ذات صلة بالسؤال مع عوامل ذكاء (نية السؤال، الثيم المقصود، تصفية اللهجة).
"""
    )
    st.caption("© 2025 — Abrar Sebiany — FastAPI + Streamlit")
    st.markdown('</div>', unsafe_allow_html=True)  # .alz-card

# ---------------------------
# Advanced settings (collapsed, optional)
# ---------------------------
    st.caption("إعدادات مخصصة للمطورين. لا تُعرض في الواجهة العامة.")
    # Allow changing API URL without displaying it elsewhere
    api_override = st.text_input("API URL", API_DEFAULT, help="لن يُعرض في الواجهة. للاختبار فقط.")
    if api_override and api_override != API_DEFAULT:
        st.session_state["api_base_url"] = api_override
        # Note: For simplicity, we re-use API_DEFAULT on this run.
        # On next interactions, you can swap to st.session_state["api_base_url"] where needed.
        st.success("تم تحديث عنوان الـ API مؤقتًا لهذه الجلسة.")
