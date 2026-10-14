from __future__ import annotations

import pandas as pd
import streamlit as st

from main import ParkingProcessor, read_image_file

st.set_page_config(page_title="Smart Parking | ANPR", page_icon="🅿️", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
.hero {padding: 1.4rem 1.6rem; border-radius: 16px; margin-bottom: 1.2rem;
       background: linear-gradient(120deg, #14213d, #264653); color: white;}
.hero h1 {margin: 0; color: white; font-size: 2rem;}
.hero p {margin: .45rem 0 0 0; color: #e8edf5;}
div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.25);
       padding: 1rem; border-radius: 12px;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_service():
    # Reuse the ANPR model instead of loading it on every Streamlit rerun.
    return ParkingProcessor()


service = get_service()


def money(value):
    return f"₹{float(value):,.2f}"


def dashboard():
    st.markdown("""
    <div class="hero">
      <h1>Smart Parking Management</h1>
      <p>Automatic Number Plate Recognition · Live parking overview</p>
    </div>
    """, unsafe_allow_html=True)
    try:
        state = service.get_parking_state()
        counts = state["capacity"] or {}
        total = int(counts.get("total_slots") or 0)
        available = int(counts.get("available_slots") or 0)
        occupied = int(counts.get("occupied_slots") or 0)
        active = state.get("active_sessions", [])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total slots", total)
        c2.metric("Available", available)
        c3.metric("Occupied", occupied)
        c4.metric("Occupancy", f"{occupied / total * 100:.1f}%" if total else "0%")

        st.subheader("Currently parked vehicles")
        if active:
            rows = [{
                "Plate number": x.get("number_plate", ""),
                "Customer": x.get("customer_name", ""),
                "Phone": x.get("phone") or "—",
                "Vehicle type": x.get("vehicle_type", ""),
                "Slot": x.get("slot_id", ""),
                "Duration (hours)": x.get("duration_hours", ""),
                "Charge": money(x.get("charge", 0)),
                "Entry time": x.get("entry_time", ""),
                "Expected exit": x.get("expected_exit_time", ""),
            } for x in active]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("No vehicles are currently parked.")
    except Exception as exc:
        st.error("Could not load parking information from MySQL.")
        st.code(str(exc))
        st.info("Check that MySQL is running and your database credentials are configured.")


def vehicle_entry():
    st.title("Vehicle Entry")
    st.write("Enter customer details and upload a clear vehicle image containing the number plate.")
    with st.form("vehicle_entry_form"):
        left, right = st.columns(2)
        with left:
            customer_name = st.text_input("Customer name *", placeholder="Enter full name")
            phone = st.text_input("Phone number (optional)", placeholder="Enter phone number")
            vehicle_type = st.selectbox("Vehicle type *", [
                "Two-wheeler", "Car", "SUV", "Van", "Truck", "Other"
            ])
        with right:
            duration = st.number_input("Parking duration (hours) *", min_value=1,
                                       max_value=72, value=2, step=1)
            rate = 100 if duration <= 5 else (200 if duration <= 9 else 500)
            st.metric("Estimated parking charge", money(duration * rate))
            st.caption("1–5 hours: ₹100/hour · 6–9 hours: ₹200/hour · 10+ hours: ₹500/hour")
        image_file = st.file_uploader("Upload vehicle image *", type=["jpg", "jpeg", "png"])
        submitted = st.form_submit_button("Detect plate & allocate slot", type="primary",
                                          use_container_width=True)

    if submitted:
        if not customer_name.strip():
            st.error("Please enter the customer name.")
            return
        if image_file is None:
            st.error("Please upload a vehicle image.")
            return
        try:
            image = read_image_file(image_file)
            with st.spinner("Detecting the plate and allocating a parking slot..."):
                result = service.create_parking_entry(
                    customer_name=customer_name,
                    phone=phone,
                    vehicle_type=vehicle_type,
                    duration_hours=int(duration),
                    image=image,
                )
            st.success(result.get("message", "Parking entry created successfully."))
            a, b, c = st.columns(3)
            a.metric("Detected plate", result["number_plate"])
            b.metric("Allocated slot", f"Slot {result['slot_id']}")
            c.metric("Parking charge", money(result["charge"]))
            st.write(f"**Session ID:** {result['session_id']}")
            st.write(f"**Entry time:** {result['entry_time']}")
            st.write(f"**Expected exit:** {result['expected_exit_time']}")
        except Exception as exc:
            st.error(str(exc))


def vehicle_checkout():
    st.title("Vehicle Checkout")
    st.write("Enter the number plate of a vehicle that is currently parked.")
    with st.form("checkout_form"):
        plate = st.text_input("Number plate", placeholder="e.g. MH12AB1234")
        submitted = st.form_submit_button("Complete checkout", type="primary",
                                          use_container_width=True)
    if submitted:
        if not plate.strip():
            st.error("Please enter a number plate.")
            return
        try:
            with st.spinner("Completing checkout..."):
                result = service.checkout(plate)
            st.success(result.get("message", "Checkout completed."))
            a, b, c = st.columns(3)
            a.metric("Number plate", result["number_plate"])
            b.metric("Released slot", f"Slot {result['slot_id']}")
            c.metric("Recorded charge", money(result["charge"]))
            st.write(f"**Session ID:** {result['session_id']}")
            st.write(f"**Entry time:** {result['entry_time']}")
            st.write(f"**Expected exit:** {result['expected_exit_time']}")
            st.write(f"**Actual exit:** {result['actual_exit_time']}")
        except Exception as exc:
            st.error(str(exc))


def parking_rates():
    st.title("Parking Rates")
    st.dataframe(pd.DataFrame([
        {"Duration": "1–5 hours", "Rate per hour": "₹100"},
        {"Duration": "6–9 hours", "Rate per hour": "₹200"},
        {"Duration": "10 hours or more", "Rate per hour": "₹500"},
    ]), use_container_width=True, hide_index=True)
    st.caption("Example: 3 hours = ₹300; 7 hours = ₹1,400; 10 hours = ₹5,000.")


with st.sidebar:
    st.title("🅿️ Smart Parking")
    st.caption("ANPR-enabled parking management")
    page = st.radio("Navigation", [
        "Dashboard", "Vehicle Entry", "Vehicle Checkout", "Parking Rates"
    ], label_visibility="collapsed")
    st.divider()
    if st.button("Refresh dashboard", use_container_width=True):
        st.rerun()

if page == "Dashboard":
    dashboard()
elif page == "Vehicle Entry":
    vehicle_entry()
elif page == "Vehicle Checkout":
    vehicle_checkout()
else:
    parking_rates()
