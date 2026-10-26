import streamlit as st
from ultralytics import YOLO
from PIL import Image
from collections import Counter
from pathlib import Path

# Advisory knowledge base
ADVISORY = {
    "chilli anthracnose": {
        "type": "Fungal disease",
        "symptoms": "Dark, sunken lesions may appear on fruits.",
        "prevention": [
            "Remove affected fruits and plant debris.",
            "Avoid overhead watering and prolonged leaf wetness.",
            "Maintain proper plant spacing and airflow.",
            "Use only locally approved disease-management products after consulting an agricultural expert."
        ]
    },
    "chilli yellowing": {
        "type": "Plant symptom",
        "symptoms": "Leaves may appear pale or yellow.",
        "prevention": [
            "Check soil moisture and drainage.",
            "Inspect leaves for insects and other visible symptoms.",
            "Check nutrient availability and root health.",
            "Identify the underlying cause before applying treatment."
        ]
    },
    "chilli veinal mottle virus": {
        "type": "Suspected viral disease",
        "symptoms": "Leaves may show mottling or vein-related discoloration.",
        "prevention": [
            "Seek expert confirmation of suspected infection.",
            "Monitor for aphids and other relevant virus vectors.",
            "Follow local guidance for removing confirmed infected plants.",
            "Use integrated vector management; pesticides do not cure a viral infection."
        ]
    },
    "chilli whitefly": {
        "type": "Insect pest",
        "symptoms": "Small white insects may be visible beneath leaves.",
        "prevention": [
            "Inspect the undersides of leaves regularly.",
            "Monitor whitefly populations using suitable sticky traps.",
            "Remove heavily infested plant material where appropriate.",
            "Use locally recommended integrated pest management."
        ]
    }
}

# Map model class names to advisory keys
CLASS_MAPPING = {
    "chilli anthracnose": "chilli anthracnose",
    "chilli yellowish": "chilli yellowing",
    "chilli yellowing": "chilli yellowing",
    "chilli veinal mottle virus": "chilli veinal mottle virus",
    "chilli whitefly": "chilli whitefly",
}

st.set_page_config(
    page_title="Chilli AI Detector",
    page_icon="🌶️",
    layout="wide"
)

st.title("🌶️ AI-Based Chilli Pest & Disease Detection")
st.write(
    "Upload a chilli plant image to detect conditions learned by "
    "the trained YOLO11 model and view general advisory guidance."
)

# Confirm this path matches your actual best.pt file.
MODEL_PATH = Path(
    r"D:\AI ml project data set\runs\detect\training_results\chilli_yolo11n\weights\best.pt"
)

@st.cache_resource
def load_model(model_path):
    return YOLO(str(model_path))

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.info(
        "Locate best.pt in File Explorer and update MODEL_PATH "
        "to its actual location."
    )
    st.stop()

model = load_model(MODEL_PATH)

uploaded_file = st.file_uploader(
    "Upload a chilli plant image",
    type=["jpg", "jpeg", "png"]
)

confidence = st.slider(
    "Detection confidence threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.25,
    step=0.05
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.subheader("Uploaded image")
    st.image(image, use_container_width=True)

    if st.button("🔍 Detect pests and diseases", type="primary"):
        with st.spinner("Analysing image..."):
            result = model.predict(
                source=image,
                conf=confidence,
                imgsz=640,
                verbose=False
            )[0]

        annotated_image = result.plot()
        annotated_image = Image.fromarray(annotated_image[:, :, ::-1])

        st.subheader("Detection results")
        st.image(
            annotated_image,
            caption="Model predictions with bounding boxes",
            use_container_width=True
        )

        # Detection summary
        st.subheader("Detected conditions")
        detected_classes = set()

        if result.boxes is None or len(result.boxes) == 0:
            st.warning(
                "No objects were detected above the selected confidence "
                "threshold. This does not prove the plant is healthy."
            )
        else:
            names = result.names
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            scores = result.boxes.conf.cpu().numpy()
            counts = Counter(class_ids.tolist())

            st.write(f"**Total detections:** {len(class_ids)}")
            for class_id, count in counts.items():
                st.write(f"**{names[class_id]}:** {count} detection(s)")

            st.subheader("Individual detection details")
            for class_id, score in zip(class_ids, scores):
                st.write(
                    f"**{names[class_id]}** — "
                    f"Confidence: {score * 100:.2f}%"
                )

            detected_classes = {
                str(names[class_id]).strip().lower()
                for class_id in counts.keys()
            }

        # Advisory system - runs only after prediction
        st.divider()
        st.header("🌱 Pest & Disease Advisory")

        if not detected_classes:
            st.info(
                "No advisory can be selected because no condition was "
                "detected above the threshold. Try another clear image "
                "or consult an agricultural expert."
            )
        else:
            for detected_name in sorted(detected_classes):
                normalized_name = detected_name.strip().lower().replace("_", " ")
                advisory_key = CLASS_MAPPING.get(
                    normalized_name,
                    normalized_name
                )
                advice = ADVISORY.get(advisory_key)

                if advice is None:
                    for key, value in ADVISORY.items():
                        if key in normalized_name or normalized_name in key:
                            advice = value
                            break

                if advice:
                    with st.expander(
                        f"Advice for {detected_name.title()}",
                        expanded=True
                    ):
                        st.write(f"**Category:** {advice['type']}")
                        st.write(f"**Possible symptoms:** {advice['symptoms']}")
                        st.write("**Prevention and management:**")
                        for item in advice["prevention"]:
                            st.markdown(f"- {item}")
                else:
                    st.warning(
                        f"No advisory is configured for '{detected_name}'. "
                        "Add this model class to CLASS_MAPPING and ADVISORY."
                    )

        st.info(
            "AI predictions are not confirmed diagnoses. This advisory "
            "provides general guidance; consult a local agricultural "
            "expert before applying treatments."
        )
