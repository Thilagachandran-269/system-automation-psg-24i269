import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from robotic_arm_health import (
    JOINTS,
    RoboticArmHealthMonitor,
    classify_notes,
    normal_readings,
)


class RoboticArmHealthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.monitor = RoboticArmHealthMonitor()

    def test_all_four_decisions_are_reachable(self):
        safe = self.monitor.evaluate_arm("ARM-04", normal_readings())
        monitor = self.monitor.evaluate_arm("ARM-01", normal_readings())
        schedule = self.monitor.evaluate_arm(
            "ARM-03", normal_readings(130, 88, 9)
        )
        dangerous = normal_readings()
        dangerous["wrist"]["temperature_c"] = 100
        emergency = self.monitor.evaluate_arm("ARM-06", dangerous)

        self.assertEqual(safe.status, "SAFE")
        self.assertEqual(monitor.status, "MONITOR")
        self.assertEqual(schedule.status, "SCHEDULE_MAINTENANCE")
        self.assertEqual(emergency.status, "EMERGENCY_STOP")

    def test_immediate_stop_does_not_require_model_prediction(self):
        readings = normal_readings()
        readings["elbow"]["vibration_mm_s"] = 12
        original_predict = self.monitor.model.predict_proba
        self.monitor.model.predict_proba = lambda _values: (_ for _ in ()).throw(
            AssertionError("model must not be called for an immediate stop")
        )
        try:
            result = self.monitor.evaluate_arm("ARM-02", readings)
        finally:
            self.monitor.model.predict_proba = original_predict
        self.assertEqual(result.status, "EMERGENCY_STOP")

    def test_immediate_stop_works_without_a_model(self):
        monitor = RoboticArmHealthMonitor(history_file=None)
        dangerous = normal_readings()
        dangerous["wrist"]["temperature_c"] = 100
        result = monitor.evaluate_arm("ARM-04", dangerous)
        self.assertEqual(result.status, "EMERGENCY_STOP")
        with self.assertRaisesRegex(RuntimeError, "Wear model is unavailable"):
            monitor.evaluate_arm("ARM-04", normal_readings())

    def test_spoofed_torque_drop_with_stress_stops_arm(self):
        result = self.monitor.evaluate_arm(
            "ARM-02",
            normal_readings(torque=30, temperature=78, vibration=7.5),
            normal_readings(torque=90, temperature=78, vibration=7.5),
        )
        self.assertEqual(result.status, "EMERGENCY_STOP")
        self.assertIn("spoofed torque", result.reason)

    def test_notes_are_an_independent_signal(self):
        self.assertEqual(classify_notes(["ARM-01 shoulder light rattle"]), "monitor")
        self.assertEqual(classify_notes(["ARM-03 wrist grinding sound"]), "schedule")
        self.assertEqual(classify_notes(["ARM-04 shift checklist completed"]), "none")

    def test_every_joint_is_required(self):
        with self.assertRaises(ValueError):
            self.monitor.evaluate_arm("ARM-01", {"shoulder": normal_readings()["shoulder"]})

    def test_log_rejects_notes_without_arm_identifier(self):
        with TemporaryDirectory() as directory:
            log = Path(directory) / "notes.txt"
            log.write_text("The wrist sounded unusual.", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no ARM-ID"):
                RoboticArmHealthMonitor(log_file=log)


if __name__ == "__main__":
    unittest.main()
