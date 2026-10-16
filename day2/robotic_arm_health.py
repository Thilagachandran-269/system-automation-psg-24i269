import argparse
import csv
import math
import re
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
HISTORY_FILE = BASE_DIR / "arm_joint_history.csv"
MAINTENANCE_LOG = BASE_DIR / "maintenance_log.txt"
JOINTS = ("shoulder", "elbow", "wrist")
FEATURES = ("torque_nm", "temperature_c", "vibration_mm_s")

# These limits are deterministic trip points, not model features.
IMMEDIATE_STOP_LIMITS = {
    "torque_nm": 180.0,
    "temperature_c": 100.0,
    "vibration_mm_s": 12.0,
}
WEAR_SCHEDULE_THRESHOLD = 0.70
WEAR_MONITOR_THRESHOLD = 0.40

HIGH_RISK_NOTE_TERMS = (
    "grinding",
    "crack",
    "cracked",
    "binding",
    "metallic debris",
    "burning smell",
)
MONITOR_NOTE_TERMS = (
    "rattle",
    "unusual noise",
    "warm",
    "play in",
    "loose",
    "grease leak",
    "vibration",
)
ARM_ID_PATTERN = re.compile(r"\bARM[- ]([A-Z0-9]+)\b", re.IGNORECASE)


@dataclass(frozen=True)
class Reading:
    torque_nm: float
    temperature_c: float
    vibration_mm_s: float

    def as_features(self):
        return [self.torque_nm, self.temperature_c, self.vibration_mm_s]


@dataclass(frozen=True)
class Decision:
    arm_id: str
    status: str
    reason: str
    wear_probability: float
    note_signal: str


class KNearestWearModel:
    """A small standardized k-NN classifier trained from labeled CSV history."""

    def __init__(self, neighbors=9):
        self.neighbors = neighbors
        self.classes_ = (0, 1)
        self.rows = []
        self.means = ()
        self.scales = ()

    def fit(self, history_file):
        with history_file.open(newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            required = set(FEATURES) | {"wear_label"}
            missing = required - set(reader.fieldnames or ())
            if missing:
                raise ValueError(
                    f"Historical dataset is missing columns: {sorted(missing)}"
                )
            rows = []
            for line_number, row in enumerate(reader, start=2):
                try:
                    features = tuple(float(row[name]) for name in FEATURES)
                    label = int(row["wear_label"])
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"Invalid historical data on CSV line {line_number}"
                    ) from error
                if (
                    label not in self.classes_
                    or not all(math.isfinite(value) and value >= 0 for value in features)
                ):
                    raise ValueError(
                        f"Invalid historical data on CSV line {line_number}"
                    )
                rows.append((features, label))
        if len(rows) < 50:
            raise ValueError("Historical dataset must contain at least 50 readings")
        if {label for _, label in rows} != set(self.classes_):
            raise ValueError("Historical data must include healthy and worn labels")

        self.means = tuple(
            statistics.fmean(features[index] for features, _ in rows)
            for index in range(len(FEATURES))
        )
        self.scales = tuple(
            statistics.pstdev(features[index] for features, _ in rows) or 1.0
            for index in range(len(FEATURES))
        )
        self.rows = [
            (self._scale(features), label) for features, label in rows
        ]
        self.neighbors = min(self.neighbors, len(self.rows))
        return self

    def _scale(self, features):
        return tuple(
            (value - mean) / scale
            for value, mean, scale in zip(features, self.means, self.scales)
        )

    def predict_proba(self, feature_rows):
        probabilities = []
        for features in feature_rows:
            scaled = self._scale(features)
            nearest = sorted(
                (
                    (
                        sum((value - known) ** 2 for value, known in zip(scaled, row)),
                        label,
                    )
                    for row, label in self.rows
                ),
                key=lambda item: item[0],
            )[: self.neighbors]
            wear_count = Counter(label for _, label in nearest)[1]
            wear_probability = wear_count / len(nearest)
            probabilities.append([1 - wear_probability, wear_probability])
        return probabilities


def load_notes(log_file):
    """Index plain-text log sentences by the arm identifier they mention."""
    notes_by_arm = {}
    with log_file.open(encoding="utf-8") as log:
        for line_number, line in enumerate(log, start=1):
            text = line.strip()
            if not text:
                continue
            match = ARM_ID_PATTERN.search(text)
            if match is None:
                raise ValueError(
                    f"Maintenance log line {line_number} has no ARM-ID: {text}"
                )
            arm_id = f"ARM-{match.group(1).upper()}"
            notes_by_arm.setdefault(arm_id, []).append(text)
    return notes_by_arm


def classify_notes(notes):
    text = " ".join(notes).lower()
    if any(term in text for term in HIGH_RISK_NOTE_TERMS):
        return "schedule"
    if any(term in text for term in MONITOR_NOTE_TERMS):
        return "monitor"
    return "none"


def read_reading(values, arm_id, joint):
    try:
        reading = Reading(
            torque_nm=float(values["torque_nm"]),
            temperature_c=float(values["temperature_c"]),
            vibration_mm_s=float(values["vibration_mm_s"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid sensor data for {arm_id} {joint}") from error

    if not all(math.isfinite(value) for value in reading.as_features()):
        raise ValueError(f"Non-finite sensor data for {arm_id} {joint}")
    if any(value < 0 for value in reading.as_features()):
        raise ValueError(f"Negative sensor data for {arm_id} {joint}")
    return reading


class RoboticArmHealthMonitor:
    def __init__(self, history_file=HISTORY_FILE, log_file=MAINTENANCE_LOG):
        self.model = KNearestWearModel().fit(history_file) if history_file else None
        self.notes_by_arm = load_notes(log_file) if log_file else {}

    @staticmethod
    def immediate_stop_reason(readings):
        for joint, reading in readings.items():
            for feature, limit in IMMEDIATE_STOP_LIMITS.items():
                value = getattr(reading, feature)
                if value >= limit:
                    return f"{joint} {feature}={value:g} reached trip limit {limit:g}"
        return None

    @staticmethod
    def spoof_reason(readings, previous_readings):
        if previous_readings is None:
            return None
        for joint, reading in readings.items():
            previous = previous_readings.get(joint)
            if previous is None:
                continue
            sudden_torque_drop = (
                previous.torque_nm >= 35
                and reading.torque_nm <= previous.torque_nm * 0.45
            )
            corroborating_stress = (
                reading.temperature_c >= 70 or reading.vibration_mm_s >= 6
            )
            if sudden_torque_drop and corroborating_stress:
                return (
                    f"{joint} torque dropped over 55% from the prior reading "
                    "while temperature or vibration remains elevated"
                )
        return None

    def evaluate_arm(self, arm_id, current, previous=None):
        arm_id = arm_id.upper()
        if set(current) != set(JOINTS):
            raise ValueError(f"{arm_id} must include readings for {', '.join(JOINTS)}")
        readings = {
            joint: read_reading(current[joint], arm_id, joint) for joint in JOINTS
        }
        prior_readings = None
        if previous is not None:
            prior_readings = {
                joint: read_reading(previous[joint], arm_id, joint)
                for joint in previous
                if joint in JOINTS
            }

        note_signal = classify_notes(self.notes_by_arm.get(arm_id, []))
        stop_reason = self.immediate_stop_reason(readings)
        if stop_reason:
            return Decision(arm_id, "EMERGENCY_STOP", stop_reason, 0.0, note_signal)

        spoof_reason = self.spoof_reason(readings, prior_readings)
        if spoof_reason:
            return Decision(
                arm_id,
                "EMERGENCY_STOP",
                f"Possible spoofed torque sensor: {spoof_reason}",
                0.0,
                note_signal,
            )

        if self.model is None:
            if note_signal == "schedule":
                return Decision(
                    arm_id,
                    "SCHEDULE_MAINTENANCE",
                    "Technician note reports a high-risk mechanical symptom",
                    0.0,
                    note_signal,
                )
            if note_signal == "monitor":
                return Decision(
                    arm_id,
                    "MONITOR",
                    "Technician note reports a symptom requiring observation",
                    0.0,
                    note_signal,
                )
            raise RuntimeError(
                "Wear model is unavailable; non-emergency sensor-only decisions "
                "cannot be made"
            )

        probabilities = self.model.predict_proba(
            [reading.as_features() for reading in readings.values()]
        )
        wear_class_index = self.model.classes_.index(1)
        wear_probability = max(row[wear_class_index] for row in probabilities)

        if wear_probability >= WEAR_SCHEDULE_THRESHOLD:
            return Decision(
                arm_id,
                "SCHEDULE_MAINTENANCE",
                "Predictive wear probability exceeds scheduling threshold",
                wear_probability,
                note_signal,
            )
        if note_signal == "schedule":
            return Decision(
                arm_id,
                "SCHEDULE_MAINTENANCE",
                "Technician note reports a high-risk mechanical symptom",
                wear_probability,
                note_signal,
            )
        if wear_probability >= WEAR_MONITOR_THRESHOLD:
            return Decision(
                arm_id,
                "MONITOR",
                "Predictive wear probability exceeds monitoring threshold",
                wear_probability,
                note_signal,
            )
        if note_signal == "monitor":
            return Decision(
                arm_id,
                "MONITOR",
                "Technician note reports a symptom requiring observation",
                wear_probability,
                note_signal,
            )
        return Decision(arm_id, "SAFE", "No current rule or model alert", wear_probability, note_signal)


def normal_readings(torque=48, temperature=48, vibration=1.8):
    return {
        joint: {
            "torque_nm": torque,
            "temperature_c": temperature,
            "vibration_mm_s": vibration,
        }
        for joint in JOINTS
    }


def run_demo(monitor):
    scenarios = [
        (
            "SAFE",
            normal_readings(),
            None,
            "ARM-04",
        ),
        (
            "MONITOR",
            normal_readings(),
            None,
            "ARM-01",
        ),
        (
            "SCHEDULE_MAINTENANCE",
            normal_readings(130, 88, 9.0),
            None,
            "ARM-05",
        ),
        (
            "EMERGENCY_STOP",
            {
                **normal_readings(),
                "wrist": {
                    "torque_nm": 70,
                    "temperature_c": 102,
                    "vibration_mm_s": 5,
                },
            },
            None,
            "ARM-06",
        ),
        (
            "EMERGENCY_STOP (spoof defense)",
            normal_readings(30, 78, 7.5),
            normal_readings(90, 78, 7.5),
            "ARM-07",
        ),
    ]
    print("Example arm decisions:")
    for label, current, previous, arm_id in scenarios:
        decision = monitor.evaluate_arm(arm_id, current, previous)
        print(
            f"{label}: {decision.arm_id} -> {decision.status} "
            f"(wear={decision.wear_probability:.2f}; {decision.reason})"
        )


def main():
    parser = argparse.ArgumentParser(description="Robotic arm joint health monitor")
    parser.add_argument("--history", type=Path, default=HISTORY_FILE)
    parser.add_argument("--log", type=Path, default=MAINTENANCE_LOG)
    parser.add_argument(
        "--demo", action="store_true", help="show examples of all decisions and spoof detection"
    )
    args = parser.parse_args()
    monitor = RoboticArmHealthMonitor(args.history, args.log)
    run_demo(monitor)


if __name__ == "__main__":
    main()
