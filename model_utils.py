import base64
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import streamlit as st
from PIL import Image
from openai import OpenAI

try:
    import h5py
except Exception:
    h5py = None

try:
    import tensorflow as tf
except Exception:
    tf = None

MODEL_PATH = "raichica_v1.h5"
CONFIDENCE_THRESHOLD = 0.80
DEFAULT_CLASS_NAMES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
]


@dataclass
class DiagnosisResult:
    diagnosis: str
    confidence: float
    source: str
    is_invalid_input: bool = False


@dataclass
class InputCheckResult:
    is_plant_leaf: bool
    reason: str = ""


@st.cache_resource

def load_model(model_path: str = MODEL_PATH):
    if not Path(model_path).exists():
        return None
    if tf is None:
        st.warning(
            "TensorFlow is not available in this environment. "
            "Tier 1 will be skipped, and the app will use Vision API fallback."
        )
        return None
    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except Exception:
        try:
            return _load_model_with_h5_compat(model_path)
        except Exception:
            st.warning(
                "Local model could not be loaded due to a Keras/TensorFlow format mismatch. "
                "Tier 1 will be skipped, and the app will use Vision API fallback."
            )
            return None


def _load_model_with_h5_compat(model_path: str):
    if tf is None or h5py is None:
        raise RuntimeError("TensorFlow/h5py not available")
    # Some .h5 files saved with newer Keras cannot be deserialized by TF 2.15.
    # Rebuild the known architecture and load weights by layer name.
    with h5py.File(model_path, "r") as h5_file:
        model_config = h5_file.attrs.get("model_config")
        if model_config is None:
            raise

        if isinstance(model_config, bytes):
            model_config = model_config.decode("utf-8")

    config = json.loads(model_config)
    layers = config.get("config", {}).get("layers", [])

    rescaling = next((layer for layer in layers if layer.get("class_name") == "Rescaling"), {})
    dense_hidden = next((layer for layer in layers if layer.get("class_name") == "Dense" and layer.get("config", {}).get("name") == "dense"), {})
    dense_output = next((layer for layer in layers if layer.get("class_name") == "Dense" and layer.get("config", {}).get("name") == "dense_1"), {})
    dropout = next((layer for layer in layers if layer.get("class_name") == "Dropout"), {})

    rescaling_cfg = rescaling.get("config", {})
    hidden_cfg = dense_hidden.get("config", {})
    output_cfg = dense_output.get("config", {})
    dropout_cfg = dropout.get("config", {})

    base_model = tf.keras.applications.MobileNetV2(
        weights=None,
        include_top=False,
        input_shape=(224, 224, 3),
    )
    base_model._name = "mobilenetv2_1.00_224"

    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(224, 224, 3), name="input_layer_1"),
            tf.keras.layers.Rescaling(
                scale=rescaling_cfg.get("scale", 1.0 / 255.0),
                offset=rescaling_cfg.get("offset", 0.0),
                name="rescaling",
            ),
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling2d"),
            tf.keras.layers.Dense(
                hidden_cfg.get("units", 128),
                activation=hidden_cfg.get("activation", "relu"),
                name="dense",
            ),
            tf.keras.layers.Dropout(
                rate=dropout_cfg.get("rate", 0.2),
                name="dropout",
            ),
            tf.keras.layers.Dense(
                output_cfg.get("units", len(DEFAULT_CLASS_NAMES)),
                activation=output_cfg.get("activation", "softmax"),
                name="dense_1",
            ),
        ],
        name=config.get("config", {}).get("name", "sequential"),
    )

    model.load_weights(model_path, by_name=True, skip_mismatch=False)
    return model


def load_class_names(path: str = "class_names.txt") -> List[str]:
    class_file = Path(path)
    if not class_file.exists():
        return DEFAULT_CLASS_NAMES

    names = [line.strip() for line in class_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    return names or DEFAULT_CLASS_NAMES


def preprocess_image(image: Image.Image, target_size=(224, 224)) -> np.ndarray:
    if tf is None:
        raise RuntimeError("TensorFlow is not available for local preprocessing")
    image = image.convert("RGB").resize(target_size)
    arr = np.asarray(image).astype("float32")
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def image_to_base64(image: Image.Image, image_format: str = "JPEG") -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=image_format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def get_openai_client() -> Optional[OpenAI]:
    api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    if not api_key:
        return None
    try:
        return OpenAI(api_key=api_key)
    except Exception as exc:
        st.error(f"OpenAI client init failed: {exc}")
        return None


def local_prediction(image: Image.Image, class_names: List[str], model=None) -> Optional[DiagnosisResult]:
    if model is None:
        model = load_model()
    if model is None:
        return None

    batch = preprocess_image(image)
    prediction = model.predict(batch, verbose=0)[0]
    confidence = float(np.max(prediction))
    idx = int(np.argmax(prediction))

    diagnosis = class_names[idx] if idx < len(class_names) else f"class_{idx}"
    return DiagnosisResult(diagnosis=diagnosis, confidence=confidence, source="tier1_local_cnn")


def vision_fallback_prediction(client: OpenAI, image_b64: str) -> DiagnosisResult:
    system_prompt = (
        "Ti si botanički dijagnostičar. Identifikuj biljnu vrstu i patologiju. "
        "Ako na slici nije biljka/list, vrati tačno: Invalid Input. "
        "Odgovori kratko, format: <Biljka> - <Bolest ili Healthy>."
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analiziraj ovu sliku biljke."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                ],
            },
        ],
    )

    content = (response.choices[0].message.content or "").strip()
    if content.lower() == "invalid input":
        return DiagnosisResult(
            diagnosis="Invalid Input",
            confidence=0.0,
            source="tier2_vision_llm",
            is_invalid_input=True,
        )

    return DiagnosisResult(diagnosis=content, confidence=0.79, source="tier2_vision_llm")


def validate_plant_input(client: OpenAI, image_b64: str) -> InputCheckResult:
    system_prompt = (
        "You are an image gatekeeper for a plant disease app. "
        "Return JSON with keys: is_plant_leaf (boolean), reason (string). "
        "Set is_plant_leaf=true only if the image clearly contains a plant leaf/crop suitable for disease diagnosis."
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Is this a plant leaf image for diagnosis?"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                ],
            },
        ],
    )

    raw = response.choices[0].message.content or "{}"
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return InputCheckResult(is_plant_leaf=True, reason="input_check_parse_failed")

    return InputCheckResult(
        is_plant_leaf=bool(parsed.get("is_plant_leaf", True)),
        reason=str(parsed.get("reason", "")),
    )


def agronomist_plan(client: OpenAI, diagnosis: str, crop_type: str, region: str = "Western Balkans") -> dict:
    system_prompt = (
        "Ti si agronomski savjetnik za Crnu Goru i Srbiju. "
        "Vrati JSON sa ključevima: chemical_treatment, organic_alternatives, "
        "prevention, eco_impact. "
        "Koristi preparate dostupne regionalno (npr. Ridomil, Acrobat, Proplant) samo kad su smisleni."
    )

    user_prompt = (
        f"Dijagnoza: {diagnosis}\n"
        f"Kultura: {crop_type}\n"
        f"Region: {region}\n"
        "Daj konkretan plan tretmana, kratko i praktično."
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    raw = response.choices[0].message.content or "{}"
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "chemical_treatment": "Nije moguće parsirati odgovor modela.",
            "organic_alternatives": "N/A",
            "prevention": "N/A",
            "eco_impact": "N/A",
        }


def run_hybrid_diagnosis(image: Image.Image, crop_type: str, class_names: List[str], threshold: float = CONFIDENCE_THRESHOLD) -> tuple[DiagnosisResult, Optional[dict]]:
    client = get_openai_client()
    image_b64 = image_to_base64(image)

    if client is not None:
        input_check = validate_plant_input(client=client, image_b64=image_b64)
        if not input_check.is_plant_leaf:
            return (
                DiagnosisResult(
                    diagnosis="Invalid Input",
                    confidence=0.0,
                    source="tier0_input_guard",
                    is_invalid_input=True,
                ),
                None,
            )

    local_result = local_prediction(image=image, class_names=class_names)

    if local_result and local_result.confidence >= threshold:
        diagnosis = local_result
    else:
        if client is None:
            fallback_diagnosis = local_result.diagnosis if local_result else "Model i API nijesu dostupni"
            fallback_confidence = local_result.confidence if local_result else 0.0
            return (
                DiagnosisResult(
                    diagnosis=fallback_diagnosis,
                    confidence=fallback_confidence,
                    source="fallback_unavailable",
                ),
                None,
            )

        diagnosis = vision_fallback_prediction(client, image_b64)

    if diagnosis.is_invalid_input:
        return diagnosis, None

    if client is None:
        return diagnosis, None

    plan = agronomist_plan(client=client, diagnosis=diagnosis.diagnosis, crop_type=crop_type)
    return diagnosis, plan
