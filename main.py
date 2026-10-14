"""Smart parking central processor: Streamlit-facing service + ANPR + MySQL.

Expected project files:
    main.py
    processor.py

Install dependencies with:
    pip install mysql-connector-python streamlit opencv-python easyocr ultralytics

Set MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE in the
process environment. Create the database first; this module creates its tables.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timedelta
from typing import Any, Optional

import cv2
import mysql.connector
from mysql.connector import Error

from processor import ANPRsystem, PLATE_CLASS_ID

# Hourly rates by selected duration bracket (INR per hour).
# Boundaries are non-overlapping: 1-5 hours => Rs. 100/hour,
# 6-9 hours => Rs. 200/hour, and 10 hours or more => Rs. 500/hour.
DURATION_PRICES: dict[str, float] = {
    "1-5 hours": 100.00,
    "6-9 hours": 200.00,
    "10+ hours": 500.00,
}

# Configure the total capacity here. Table creation does not automatically
# invent slot numbers; initialize_slots() adds slots up to this capacity.
DEFAULT_CAPACITY = int(os.getenv("PARKING_CAPACITY", "20"))


def mysql_connection():
    """Open a MySQL connection using environment-based credentials."""
    database = os.getenv("MYSQL_DATABASE", "smart_parking")
    return mysql.connector.connect(
        host=os.getenv("MYSQL_HOST", "localhost"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        user=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASSWORD", ""),
        database=database,
        autocommit=False,
    )


class ParkingProcessor:
    """Coordinates plate recognition, slot allocation, and database state."""

    def __init__(self, anpr: Optional[ANPRsystem] = None):
        # Instantiate once and reuse this object across Streamlit reruns by
        # caching ParkingProcessor in the Streamlit app with st.cache_resource.
        self.anpr = anpr if anpr is not None else ANPRsystem()

    def setup_database(self, capacity: int = DEFAULT_CAPACITY) -> None:
        """Create required tables and ensure slots 1..capacity exist."""
        if capacity < 1:
            raise ValueError("Parking capacity must be at least 1")

        conn = mysql_connection()
        try:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS parking_slots (
                    slot_id INT PRIMARY KEY,
                    status ENUM('AVAILABLE', 'OCCUPIED') NOT NULL DEFAULT 'AVAILABLE',
                    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
                        ON UPDATE CURRENT_TIMESTAMP
                ) ENGINE=InnoDB
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS parking_sessions (
                    session_id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    customer_name VARCHAR(120) NOT NULL,
                    phone VARCHAR(30) NULL,
                    vehicle_type VARCHAR(40) NOT NULL,
                    number_plate VARCHAR(20) NOT NULL,
                    slot_id INT NOT NULL,
                    duration_hours INT NOT NULL,
                    charge DECIMAL(10,2) NOT NULL,
                    entry_time DATETIME NOT NULL,
                    expected_exit_time DATETIME NOT NULL,
                    actual_exit_time DATETIME NULL,
                    status ENUM('ACTIVE', 'COMPLETED') NOT NULL DEFAULT 'ACTIVE',
                    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_active_plate (number_plate, status),
                    INDEX idx_slot_status (slot_id, status),
                    CONSTRAINT fk_session_slot FOREIGN KEY (slot_id)
                        REFERENCES parking_slots(slot_id)
                ) ENGINE=InnoDB
            """)
            for slot_id in range(1, capacity + 1):
                cur.execute(
                    "INSERT IGNORE INTO parking_slots (slot_id, status) VALUES (%s, 'AVAILABLE')",
                    (slot_id,),
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @staticmethod
    def get_charge(duration_hours: int) -> float:
        """Calculate total charge from the hourly rate for the selected duration bracket."""
        if isinstance(duration_hours, bool) or not isinstance(duration_hours, int):
            raise ValueError("Duration must be a whole number of hours.")
        if duration_hours < 1:
            raise ValueError("Duration must be at least 1 hour.")

        if duration_hours <= 5:
            hourly_rate = DURATION_PRICES["1-5 hours"]
        elif duration_hours < 10:
            hourly_rate = DURATION_PRICES["6-9 hours"]
        else:
            hourly_rate = DURATION_PRICES["10+ hours"]

        return duration_hours * hourly_rate

    def extract_number_plate(self, image) -> str:
        """Run the uploaded processor.py detection/OCR pipeline on an OpenCV BGR image."""
        if image is None or not hasattr(image, "shape") or image.size == 0:
            raise ValueError("A valid image/frame is required for number-plate detection")

        detections = self.anpr.detect_frame(image)
        for result in detections:
            if result.boxes is None or len(result.boxes) == 0:
                continue
            for box in result.boxes:
                # Only crop boxes whose class matches the configured plate class.
                class_id = int(box.cls[0].item()) if box.cls is not None else -1
                if class_id != PLATE_CLASS_ID:
                    continue
                xyxy = box.xyxy
                roi = self.anpr.crop_frame(image, xyxy)
                if roi is None or roi.size == 0:
                    continue
                processed_roi = self.anpr.preocess_roi(roi)
                raw_text = self.anpr.ocr_processing(processed_roi)
                plate = self._normalize_plate(raw_text)
                if plate:
                    return plate

        raise ValueError("No readable number plate detected. Please try a clearer image/frame.")

    @staticmethod
    def _normalize_plate(raw_text: Any) -> str:
        # OCR may include spaces or punctuation; retain only letters and digits.
        return re.sub(r"[^A-Z0-9]", "", str(raw_text).upper())

    def create_parking_entry(
        self,
        *,
        customer_name: str,
        vehicle_type: str,
        duration_hours: int,
        image,
        phone: Optional[str] = None,
    ) -> dict[str, Any]:
        """Detect plate, allocate a free slot, and atomically save the entry.

        No session is inserted if detection fails or no slot is available.
        The slot update and session insert are committed in the same transaction.
        """
        customer_name = (customer_name or "").strip()
        vehicle_type = (vehicle_type or "").strip()
        if not customer_name:
            raise ValueError("Customer name is required")
        if not vehicle_type:
            raise ValueError("Vehicle type is required")
        charge = self.get_charge(int(duration_hours))

        # Detection is performed before opening a transaction so model inference
        # does not hold database row locks.
        plate = self.extract_number_plate(image)
        now = datetime.now()
        expected_exit = now + timedelta(hours=int(duration_hours))

        conn = mysql_connection()
        try:
            cur = conn.cursor(dictionary=True)
            conn.start_transaction()

            cur.execute(
                "SELECT session_id FROM parking_sessions "
                "WHERE number_plate = %s AND status = 'ACTIVE' LIMIT 1 FOR UPDATE",
                (plate,),
            )
            if cur.fetchone():
                raise ValueError(f"Vehicle {plate} already has an active parking session")

            # FOR UPDATE serializes competing allocations so two users cannot
            # claim the same available slot at the same time.
            cur.execute(
                "SELECT slot_id FROM parking_slots WHERE status = 'AVAILABLE' "
                "ORDER BY slot_id ASC LIMIT 1 FOR UPDATE"
            )
            slot = cur.fetchone()
            if not slot:
                raise ValueError("Parking is full: no available slots")
            slot_id = int(slot["slot_id"])

            cur.execute(
                "UPDATE parking_slots SET status = 'OCCUPIED' WHERE slot_id = %s AND status = 'AVAILABLE'",
                (slot_id,),
            )
            if cur.rowcount != 1:
                raise RuntimeError("Slot allocation conflict; please retry")

            cur.execute("""
                INSERT INTO parking_sessions
                    (customer_name, phone, vehicle_type, number_plate, slot_id,
                     duration_hours, charge, entry_time, expected_exit_time, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 'ACTIVE')
            """, (
                customer_name, (phone or "").strip() or None, vehicle_type, plate,
                slot_id, int(duration_hours), charge, now, expected_exit,
            ))
            session_id = cur.lastrowid
            conn.commit()
            return {
                "success": True,
                "message": "Parking slot allocated successfully",
                "session_id": session_id,
                "customer_name": customer_name,
                "number_plate": plate,
                "vehicle_type": vehicle_type,
                "slot_id": slot_id,
                "duration_hours": int(duration_hours),
                "charge": charge,
                "entry_time": now,
                "expected_exit_time": expected_exit,
                "status": "ACTIVE",
            }
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def checkout(self, number_plate: str) -> dict[str, Any]:
        """Complete an active session and make its slot available again."""
        plate = self._normalize_plate(number_plate)
        if not plate:
            raise ValueError("A number plate is required")

        conn = mysql_connection()
        try:
            cur = conn.cursor(dictionary=True)
            conn.start_transaction()
            cur.execute(
                "SELECT session_id, slot_id, entry_time, expected_exit_time, charge "
                "FROM parking_sessions WHERE number_plate = %s AND status = 'ACTIVE' "
                "ORDER BY entry_time DESC LIMIT 1 FOR UPDATE",
                (plate,),
            )
            session = cur.fetchone()
            if not session:
                raise ValueError(f"No active parking session found for {plate}")

            exit_time = datetime.now()
            cur.execute(
                "UPDATE parking_sessions SET status = 'COMPLETED', actual_exit_time = %s "
                "WHERE session_id = %s AND status = 'ACTIVE'",
                (exit_time, session["session_id"]),
            )
            cur.execute(
                "UPDATE parking_slots SET status = 'AVAILABLE' WHERE slot_id = %s",
                (session["slot_id"],),
            )
            conn.commit()
            return {
                "success": True,
                "message": "Checkout completed; slot is now available",
                "session_id": session["session_id"],
                "number_plate": plate,
                "slot_id": session["slot_id"],
                "entry_time": session["entry_time"],
                "expected_exit_time": session["expected_exit_time"],
                "actual_exit_time": exit_time,
                "charge": float(session["charge"]),
                "status": "COMPLETED",
            }
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @staticmethod
    def get_parking_state() -> dict[str, Any]:
        """Return capacity, free/occupied counts, and active parking sessions."""
        conn = mysql_connection()
        try:
            cur = conn.cursor(dictionary=True)
            cur.execute("""
                SELECT
                    COUNT(*) AS total_slots,
                    COALESCE(SUM(status = 'AVAILABLE'), 0) AS available_slots,
                    COALESCE(SUM(status = 'OCCUPIED'), 0) AS occupied_slots
                FROM parking_slots
            """)
            counts = cur.fetchone()
            cur.execute("""
                SELECT session_id, customer_name, phone, vehicle_type, number_plate,
                       slot_id, duration_hours, charge, entry_time, expected_exit_time, status
                FROM parking_sessions WHERE status = 'ACTIVE' ORDER BY entry_time DESC
            """)
            active_sessions = cur.fetchall()
            return {"capacity": counts, "active_sessions": active_sessions}
        finally:
            conn.close()


def read_image_file(uploaded_file):
    """Convert a Streamlit UploadedFile or image path to an OpenCV BGR image."""
    if isinstance(uploaded_file, (str, os.PathLike)):
        image = cv2.imread(os.fspath(uploaded_file))
    else:
        data = uploaded_file.getvalue()
        import numpy as np
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not read the image. Upload a valid JPG or PNG file.")
    return image


# Optional standalone smoke-test: `python main.py` initializes schema and prints state.
if __name__ == "__main__":
    service = ParkingProcessor()
    try:
        service.setup_database()
        print("Database tables are ready.")
        print(service.get_parking_state())
    except (Error, ValueError, RuntimeError) as exc:
        print(f"Smart parking startup failed: {exc}")
        raise
