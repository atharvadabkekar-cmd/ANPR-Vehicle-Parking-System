from __future__ import annotations
from pathlib import Path
import io
import pandas as pd
import streamlit as st
from PIL import Image

# The dashboard and standalone YOLO detection can run without MySQL.
try:
    from main import ParkingProcessor, read_image_file
    BACKEND_IMPORT_ERROR = None
except Exception as exc:
    ParkingProcessor = None
    read_image_file = None
    BACKEND_IMPORT_ERROR = str(exc)

st.set_page_config(page_title="Smart Parking | ANPR", page_icon="P", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
.hero {padding: 1.4rem 1.6rem; border-radius: 16px; margin-bottom: 1.2rem;
background: linear-gradient(120deg, #14213d, #264653); color: white;}
.hero h1 {margin: 0; color: white; font-size: 2rem;}
.hero p {margin: .45rem 0 0 0; color: #e8edf5;}
div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.25); padding: 1rem; border-radius: 12px;}
</style>
""", unsafe_allow_html=True)

RATES = [
    {"Duration": "1–5 hours", "Rate per hour": "₹100"},
    {"Duration": "6–9 hours", "Rate per hour": "₹200"},
    {"Duration": "10 hours or more", "Rate per hour": "₹500"},
]
DEMO_VEHICLES = [
    {"Plate number": "MH12AB1234", "Customer": "Demo Customer 1", "Vehicle": "Car", "Slot": "P-03", "Status": "Parked"},
    {"Plate number": "MH14CD5678", "Customer": "Demo Customer 2", "Vehicle": "Two-wheeler", "Slot": "P-07", "Status": "Parked"},
    {"Plate number": "MH12EF9012", "Customer": "Demo Customer 3", "Vehicle": "SUV", "Slot": "P-11", "Status": "Parked"},
]

def money(value):
    return f"₹{float(value):,.2f}"

def find_weights():
    folder = Path(__file__).resolve().parent
    for path in [folder / "best.pt", folder / "runs" / "anpr_yolo26n" / "weights" / "best.pt"]:
        if path.is_file():
            return path
    return None

@st.cache_resource
def load_yolo_model(weights_path):
    from ultralytics import YOLO
    return YOLO(weights_path)

@st.cache_resource
def get_ocr_reader():
    import easyocr
    return easyocr.Reader(["en"], gpu=False)

@st.cache_resource
def get_service():
    if ParkingProcessor is None:
        raise RuntimeError("Could not import the main.py backend: " + str(BACKEND_IMPORT_ERROR))
    return ParkingProcessor()

def try_get_database_state():
    try:
        return get_service().get_parking_state(), None
    except Exception as exc:
        return None, str(exc)

def detect_plate(uploaded_file):
    import cv2
    import numpy as np
    weights = find_weights()
    if weights is None:
        raise FileNotFoundError("Cannot find best.pt. Put best.pt beside app.py.")
    image_rgb = np.array(Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB"))
    result = load_yolo_model(str(weights)).predict(source=image_rgb, conf=0.25, verbose=False)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return image_rgb, None, 0.0, None, "YOLO did not detect a number plate."
    confidence_values = result.boxes.conf.cpu().numpy()
    index = int(confidence_values.argmax())
    confidence = float(confidence_values[index])
    x1, y1, x2, y2 = result.boxes.xyxy[index].cpu().numpy().astype(int)
    h, w = image_rgb.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    crop = image_rgb[y1:y2, x1:x2]
    preview = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 200, 80), 3)
    cv2.putText(preview, f"License plate {confidence:.2f}", (x1, max(25, y1-10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 80), 2, cv2.LINE_AA)
    preview = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
    text = None
    try:
        reader = get_ocr_reader()
        reads = reader.readtext(crop)
        if reads:
            text = max(reads, key=lambda item: item[2])[1].strip().upper()
    except Exception:
        pass
    return preview, text, confidence, crop, None

def dashboard():
    st.markdown("""
    <div class="hero"><h1>Smart Parking Management</h1>
    <p>Automatic Number Plate Recognition · Parking overview</p></div>
    """, unsafe_allow_html=True)
    state, error = try_get_database_state()
    if state is not None:
        counts = state.get("capacity") or {}
        total = int(counts.get("total_slots") or 0)
        available = int(counts.get("available_slots") or 0)
        occupied = int(counts.get("occupied_slots") or 0)
        active = state.get("active_sessions", [])
        st.caption("LIVE DATA — read from MySQL")
        rows = [{
            "Plate number": x.get("number_plate", ""), "Customer": x.get("customer_name", ""),
            "Phone": x.get("phone") or "—", "Vehicle type": x.get("vehicle_type", ""),
            "Slot": x.get("slot_id", ""), "Duration (hours)": x.get("duration_hours", ""),
            "Charge": money(x.get("charge", 0)), "Entry time": x.get("entry_time", ""),
            "Expected exit": x.get("expected_exit_time", "")
        } for x in active]
    else:
        total, occupied, available = 20, len(DEMO_VEHICLES), 20-len(DEMO_VEHICLES)
        active = DEMO_VEHICLES
        rows = DEMO_VEHICLES
        st.warning("DEMO MODE — MySQL is not connected. Counts and records below are examples, not saved data.")
        with st.expander("Why is live data unavailable?"):
            st.code(error or "Database unavailable")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total slots", total)
    c2.metric("Available", available)
    c3.metric("Occupied", occupied)
    c4.metric("Occupancy", f"{occupied / total * 100:.1f}%" if total else "0%")
    st.subheader("Currently parked vehicles")
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No vehicles are currently parked.")

def vehicle_entry():
    st.title("Vehicle Entry")
    st.write("Upload a vehicle image. YOLO uses your trained best.pt model to locate the number plate.")
    with st.form("vehicle_entry_form"):
        left, right = st.columns(2)
        with left:
            customer = st.text_input("Customer name *")
            phone = st.text_input("Phone number (optional)")
            vehicle = st.selectbox("Vehicle type", ["Two-wheeler", "Car", "SUV", "Van", "Truck", "Other"])
        with right:
            duration = st.number_input("Parking duration (hours)", min_value=1, max_value=72, value=2, step=1)
            rate = 100 if duration <= 5 else (200 if duration <= 9 else 500)
            st.metric("Estimated parking charge", money(duration * rate))
            st.caption("1–5 hours: ₹100/hour · 6–9 hours: ₹200/hour · 10+ hours: ₹500/hour")
        image_file = st.file_uploader("Upload vehicle image", type=["jpg", "jpeg", "png"])
        submitted = st.form_submit_button("Detect number plate", type="primary", use_container_width=True)
    if not submitted:
        return
    if image_file is None:
        st.error("Please upload a JPG or PNG vehicle image.")
        return
    try:
        with st.spinner("Running trained YOLO detector..."):
            preview, plate_text, confidence, crop, error = detect_plate(image_file)
        st.image(preview, caption="YOLO number-plate detection", use_container_width=True)
        if error:
            st.warning(error)
            return
        a, b = st.columns(2)
        a.metric("Detection confidence", f"{confidence*100:.1f}%")
        b.metric("OCR text", plate_text if plate_text else "Not read")
        if plate_text:
            st.success(f"Plate text read by OCR: {plate_text}")
        else:
            st.info("The plate was located, but OCR did not read clear text. Try a closer, sharper image.")
        st.divider()
        st.subheader("Save parking entry")
        state, db_error = try_get_database_state()
        if state is None:
            st.warning("YOLO detection worked, but this vehicle was NOT saved. MySQL is unavailable; slot allocation and permanent records require the database.")
            with st.expander("Database connection details"):
                st.code(db_error or "No database connection")
        elif not customer.strip():
            st.info("Enter the customer name and submit again to save the entry.")
        else:
            try:
                import cv2
                import numpy as np
                image_file.seek(0)
                image = read_image_file(image_file) if read_image_file else cv2.imdecode(np.frombuffer(image_file.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)
                with st.spinner("Saving entry to MySQL..."):
                    saved = get_service().create_parking_entry(
                        customer_name=customer.strip(), phone=phone.strip() or None,
                        vehicle_type=vehicle, duration_hours=int(duration), image=image)
                st.success(saved.get("message", "Parking entry saved."))
                x, y, z = st.columns(3)
                x.metric("Saved plate", saved.get("number_plate", plate_text or "—"))
                y.metric("Allocated slot", f"Slot {saved.get('slot_id', '—')}")
                z.metric("Parking charge", money(saved.get("charge", duration*rate)))
                st.caption("The existing backend runs its own plate recognition before saving.")
            except Exception as exc:
                st.error("YOLO detection worked, but the backend could not save the entry.")
                st.code(str(exc))
    except Exception as exc:
        st.error("Could not run number-plate detection.")
        st.code(str(exc))
        st.info("Confirm best.pt exists and ultralytics, opencv-python and Pillow are installed.")

def vehicle_checkout():
    st.title("Vehicle Checkout")
    state, error = try_get_database_state()
    if state is None:
        st.warning("Checkout needs MySQL. It is disabled in demo mode so we do not pretend a checkout was saved.")
        with st.expander("Connection details"):
            st.code(error or "No database connection")
        st.caption("You can still test the dashboard and YOLO plate detection.")
        return
    st.write("Enter the plate number of a currently parked vehicle.")
    with st.form("checkout_form"):
        plate = st.text_input("Number plate", placeholder="e.g. MH12AB1234")
        submitted = st.form_submit_button("Complete checkout", type="primary", use_container_width=True)
    if submitted:
        if not plate.strip():
            st.error("Please enter a number plate.")
            return
        try:
            result = get_service().checkout(plate.strip())
            st.success(result.get("message", "Checkout completed."))
            a, b, c = st.columns(3)
            a.metric("Number plate", result.get("number_plate", plate.upper()))
            b.metric("Released slot", f"Slot {result.get('slot_id', '—')}")
            c.metric("Recorded charge", money(result.get("charge", 0)))
            st.write(f"**Session ID:** {result.get('session_id', '—')}")
            st.write(f"**Entry time:** {result.get('entry_time', '—')}")
            st.write(f"**Actual exit:** {result.get('actual_exit_time', '—')}")
        except Exception as exc:
            st.error(str(exc))

def parking_rates():
    st.title("Parking Rates")
    st.dataframe(pd.DataFrame(RATES), use_container_width=True, hide_index=True)
    st.caption("Examples: 3 hours = ₹300; 7 hours = ₹1,400; 10 hours = ₹5,000.")

with st.sidebar:
    st.title("Smart Parking")
    st.caption("ANPR-enabled parking management")
    page = st.radio("Navigation", ["Dashboard", "Vehicle Entry / YOLO", "Vehicle Checkout", "Parking Rates"],
                    label_visibility="collapsed")
    st.divider()
    if st.button("Refresh dashboard", use_container_width=True):
        st.rerun()

if page == "Dashboard":
    dashboard()
elif page == "Vehicle Entry / YOLO":
    vehicle_entry()
elif page == "Vehicle Checkout":
    vehicle_checkout()
else:
    parking_rates()
