# app/ui/app.py
import streamlit as st
import requests
import json
import os
import pathlib
from PIL import Image

# Allow configuring API base URL via env var or (optionally) Streamlit secrets
API_DEFAULT = os.environ.get("API_BASE_URL")
if not API_DEFAULT:
    try:
        # Access secrets only if available; otherwise ignore
        API_DEFAULT = st.secrets["API_BASE_URL"]
    except Exception:
        API_DEFAULT = "http://127.0.0.1:8000"
# Try loading logo for page icon (avoid backslash escapes; support env + fallbacks)
_page_icon = None

def _find_logo_path() -> pathlib.Path | None:
    # 1) Environment override
    env_logo = os.environ.get("APP_LOGO")
    if env_logo:
        p = pathlib.Path(env_logo.replace("\\", "/"))
        if p.exists():
            return p
    # 2) Common relative paths
    for candidate in (
        pathlib.Path("Arabic model/logo.png"),
        pathlib.Path("app/ui/assets/moc_logo.png"),
        pathlib.Path("logo.png"),
    ):
        if candidate.exists():
            return candidate
    return None

LOGO_PATH = _find_logo_path()
try:
    if LOGO_PATH is not None:
        _page_icon = Image.open(LOGO_PATH)
except Exception:
    _page_icon = None

st.set_page_config(page_title="Mini Al-Jazari", layout="centered", page_icon=_page_icon)

# RTL styling for Arabic
st.markdown("""
<style>
html, body, [class*="css"]  { direction: rtl; text-align: right; }
code, pre, .stMarkdown pre { direction: ltr; text-align: left; }
</style>
""", unsafe_allow_html=True)

has_logo = LOGO_PATH is not None
if has_logo:
    # Show logo first, then title under it
    st.image(str(LOGO_PATH), caption=None, width=140)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.title("نموذج الجزري المصغّر ")
else:
    st.title("نموذج الجزري المصغّر ")

# Sidebar: API status
st.sidebar.header("حالة الخادم (API)")
api_url = st.sidebar.text_input("API URL", API_DEFAULT)
try:
    r = requests.get(f"{api_url}/", timeout=5)
    if r.ok:
        js = r.json()
        st.sidebar.success(f"✅ يعمل — {js.get('docs_count', '—')} مقطع")
    else:
        st.sidebar.error("⚠️ الخادم لا يرد")
except Exception as e:
    st.sidebar.error("⚠️ تعذر الاتصال بالخادم")
    st.stop()

tabs = st.tabs(["🧠 تصنيف السطر", "🔎 استرجاع (RAG)", "ℹ️ عن"])

# -------- Tab 1: Classify --------
with tabs[0]:
    st.subheader("أدخل سطرًا لتحليل اللهجة والثيم")
    text = st.text_area("النص:", "عن خطا تعتذرلي ولك الرضا حتى ترضى", height=100)
    if st.button("تحليل"):
        if text.strip():
            try:
                with st.spinner("جارٍ التحليل..."):
                    r = requests.post(f"{api_url}/classify",
                                      json={"text": text},
                                      headers={"Content-Type": "application/json; charset=utf-8"},
                                      timeout=15)
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
                # تم إخفاء عرض الكلمات المفتاحية وتفاصيل التحليل حسب الطلب
            except Exception as e:
                st.error(f"خطأ في الاستدعاء: {e}")
        else:
            st.warning("اكتب نصًا أولًا.")

# -------- Tab 2: Ask / RAG --------
with tabs[1]:
    st.subheader("اسأل عن مقاطع ثقافية (RAG)")
    question = st.text_input("سؤالك:", "ابي بيت غزلي عن الاعتذار")
    k = st.number_input("عدد النتائج", min_value=1, max_value=10, value=3, step=1)

    c1, c2 = st.columns(2)
    with c1:
        theme_hint = st.selectbox("الثيم (اختياري)", ["", "غزل", "وطنية", "رياضية", "دينية"])
    with c2:
        dialect_filter = st.selectbox("اللهجة (اختياري)", ["", "Hijazi", "Najdi", "Shamali", "Janoubi"])


    if st.button("اسأل"):
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
                    r = requests.post(f"{api_url}/ask",
                                      json=payload,
                                      headers={"Content-Type": "application/json; charset=utf-8"},
                                      timeout=30)
                    r.raise_for_status()
                    data = r.json()
                st.text_area("الإجابة", data.get("answer",""), height=180)

                with st.expander("المصادر (Top-K)"):
                    for s in data.get("sources", []):
                        st.markdown(
                            f"**{s.get('title','')}** — {s.get('writer','')}  \n"
                            f"الثيم: `{s.get('theme','')}` · اللهجة: `{s.get('dialect','')}` · "
                            f"الدرجة: {s.get('score',0):.3f} · intent_hits: {s.get('intent_hits',0)}  \n\n"
                            f"> {s.get('text','')}"
                        )

            except Exception as e:
                st.error(f"خطأ في الاستدعاء: {e}")

# -------- Tab 3: About --------
with tabs[2]:
    st.info("تنويه: هذا المنتج هو مجرد نموذج أولي (Prototype) ويُمثّل تطبيقًا عمليًا لفكرة بحثية علمية. النموذج قيد البناء والتقييم.")
    st.markdown("""
نموذج مصغّر يعرض:
- تصنيف **الثيم** (وطنية/غزل/دينية/رياضية) + **اللهجة**.
- استرجاع مقاطع ذات صلة بالسؤال مع عوامل ذكاء (نية السؤال، الثيم المقصود، تصفية اللهجة).
""")
    st.caption("© 2025 — Abrar Sebiany — FastAPI + Streamlit")
