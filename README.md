# rAIchica

Hybrid AI app for diagnosing plant diseases using a tiered architecture:
1. **Tier 1:** Local `MobileNetV2` model (`raichica_v1.h5`)
2. **Tier 2:** `gpt-4o` Vision fallback when confidence is low
3. **Tier 3:** `gpt-4o-mini` for treatment advice and prevention planning

## Python Version

This project is intended to run with **Python 3.11.x**.

Recommended version:
- `Python 3.11.9`

Why:
- TensorFlow is compatible with Python 3.11
- Newer versions (for example, Python 3.14) can cause dependency issues

If you use `pyenv`, set the project version like this:

```bash
pyenv install 3.11.9
pyenv local 3.11.9
python --version
```

Expected output:

```bash
Python 3.11.9
```

## Local Setup

### 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run the app

```bash
python -m streamlit run app.py
```

## Required Files

- `raichica_v1.h5` in the project root (local model)
- `class_names.txt` in the project root (label mappings)
- `.streamlit/secrets.toml` with your `OPENAI_API_KEY`

## `class_names.txt`

This file must contain one class name per line, in the same order used during model training.

Contents:

```text
Tomato___Bacterial_spot
Tomato___Early_blight
Tomato___Late_blight
Tomato___Leaf_Mold
Tomato___healthy
Potato___Early_blight
Potato___Late_blight
Potato___healthy
```

## Training (Google Colab)

This app uses a model trained on the **PlantVillage** dataset.

### Steps

1. Download the PlantVillage dataset.
2. Load and preprocess images:

```python
dataset = tf.keras.utils.image_dataset_from_directory(
    "PlantVillage/",
    image_size=(224, 224),
    batch_size=32
)
```

3. Use `MobileNetV2` as the transfer-learning backbone:

```python
base_model = tf.keras.applications.MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)
```

4. Add custom layers:

```python
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(num_classes, activation="softmax")
])
```

5. Train for 5-10 epochs:

```python
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
model.fit(dataset, epochs=5)
```

6. Save the model:

```python
model.save("raichica_v1.h5")
```

## Deployment

This app can be deployed on **Streamlit Community Cloud**.

### Steps

1. Push your project to GitHub.
2. Connect it to Streamlit Cloud and set the entry point to `app.py`.
3. Add your secrets (for example, `OPENAI_API_KEY`) in the Streamlit secrets settings.

## Troubleshooting

### TensorFlow installation fails

Ensure you are using **Python 3.11.x** and reinstall from `requirements.txt`.
This project uses TensorFlow CPU `2.16.2` for better compatibility with newer Keras-saved models.

### `streamlit: command not found`

Activate your virtual environment:

```bash
source .venv/bin/activate
```

## Notes

- If the local model (`raichica_v1.h5`) is missing, the app falls back to the Vision API flow.
- The app does not store user images on the server.
- API fallback usage is limited to control API costs during live demos.

## License

This project is intended for educational, research, and hackathon use.
