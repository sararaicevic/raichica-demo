import os

import streamlit as st
from dotenv import load_dotenv
from PIL import Image, UnidentifiedImageError

from model_utils import CONFIDENCE_THRESHOLD, MODEL_PATH, load_class_names, load_model, run_hybrid_diagnosis

load_dotenv()

st.set_page_config(page_title="rAIchica", page_icon="🍅", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --text-main: #000000;
        --text-muted: #000000;
    }
    .main {
        background: radial-gradient(circle at top right, #fff3e6 0%, #f7fbf7 45%, #eef7ff 100%);
    }
    html, body, [class*="css"], .stApp, .stMarkdown, .stText, p, span, label, div, h1, h2, h3, h4, h5, h6,
    .stSelectbox label, .stFileUploader label, .stCameraInput label {
        color: var(--text-main) !important;
    }
    .title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #1f3a27;
        margin-bottom: 0;
    }
    .subtitle {
        font-size: 1rem;
        color: #3d5a47;
        margin-top: 0.2rem;
    }
    .plan-card {
        background: #ffffff;
        border: 1px solid #dfe7f1;
        border-radius: 10px;
        padding: 12px 14px;
        margin-bottom: 10px;
        color: var(--text-main);
    }
    .plan-title {
        font-weight: 700;
        margin-bottom: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def format_plan_value(value):
    if isinstance(value, dict):
        rows = []
        for key, item in value.items():
            rows.append(f"- **{key.replace('_', ' ').title()}:** {format_plan_value(item)}")
        return "\n".join(rows)
    if isinstance(value, list):
        return "\n".join(f"- {format_plan_value(item)}" for item in value)
    return str(value)

if "api_calls" not in st.session_state:
    st.session_state.api_calls = 0

MAX_API_CALLS_PER_SESSION = 12

st.markdown("<p class='title'>🍅 rAIchica</p>", unsafe_allow_html=True)
st.markdown(
    "<p class='subtitle'>Hybrid AI dijagnostika biljnih bolesti za low-resource uslove.</p>",
    unsafe_allow_html=True,
)

logo_path = "assets/logo.png"
if os.path.exists(logo_path):
    st.image(logo_path, width=100)

crop_type = st.selectbox("Odaberi kulturu", ["Tomato", "Potato", "Pepper", "Cucumber", "Other"])
uploaded_file = st.file_uploader("Ili uploaduj sliku lista", type=["jpg", "jpeg", "png", "webp"])
camera_file = st.camera_input("Fotografiši list biljke")

model = load_model()
class_names = load_class_names()

input_file = uploaded_file if uploaded_file is not None else camera_file

if input_file:
    if not input_file.type.startswith("image/"):
        st.error("Neispravan format. Dozvoljene su samo slike.")
        st.stop()

    if getattr(input_file, "size", 0) <= 0:
        st.error("Uploadovani fajl je prazan. Pošalji validnu sliku.")
        st.stop()

    try:
        image = Image.open(input_file)
        image.verify()
        input_file.seek(0)
        image = Image.open(input_file).convert("RGB")
    except (UnidentifiedImageError, OSError):
        st.error("Fajl nije validna slika ili je oštećen. Pokušaj drugi JPG/PNG fajl.")
        st.stop()

    if st.button("Pokreni dijagnostiku", type="primary"):
        if st.session_state.api_calls >= MAX_API_CALLS_PER_SESSION:
            st.error("Dostignut je limit API poziva za ovu sesiju. Osvježi stranicu za reset.")
            st.stop()

        with st.spinner("Skeniram list..."):
            diagnosis, plan = run_hybrid_diagnosis(
                image=image,
                crop_type=crop_type,
                class_names=class_names,
                threshold=CONFIDENCE_THRESHOLD,
            )
            st.session_state.api_calls += 1

        left_col, right_col = st.columns([1, 1])

        with left_col:
            st.image(image, caption="Ulazna fotografija", use_column_width=True)

        with right_col:
            st.subheader("Rezultat dijagnoze")
            st.write(f"**Dijagnoza:** {diagnosis.diagnosis}")
            st.write(f"**Pouzdanost:** {diagnosis.confidence:.2f}")
            st.write(f"**Izvor:** `{diagnosis.source}`")

            if diagnosis.confidence < CONFIDENCE_THRESHOLD:
                st.warning("Niska pouzdanost lokalnog modela, aktiviran fallback sloj.")

            if diagnosis.is_invalid_input:
                st.error("Ovo nije biljka, molim vas uslikajte list.")

        st.info(
            "Eco-Impact: Ovim preciznim prskanjem uštedjeli ste oko 0.5L pesticida "
            "i smanjili CO2 otisak za ~2kg."
        )

        if plan and not diagnosis.is_invalid_input:
            st.subheader("Agronom preporuka")
            sections = [
                ("Hemijski tretman", plan.get("chemical_treatment", "N/A")),
                ("Organske/Bio alternative", plan.get("organic_alternatives", "N/A")),
                ("Prevencija", plan.get("prevention", "N/A")),
                ("Eco-impact procjena", plan.get("eco_impact", "N/A")),
            ]
            for title, value in sections:
                st.markdown(f"<div class='plan-card'><div class='plan-title'>{title}</div></div>", unsafe_allow_html=True)
                st.markdown(format_plan_value(value))

if model is None and not os.path.exists(MODEL_PATH):
    st.warning("Model `raichica_v1.h5` nije pronađen. Tier 1 će biti preskočen dok ne dodaš model fajl.")

st.caption(f"API calls (session): {st.session_state.api_calls}/{MAX_API_CALLS_PER_SESSION}")
