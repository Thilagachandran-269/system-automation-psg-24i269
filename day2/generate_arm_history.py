import csv
import random
from datetime import datetime, timedelta
from pathlib import Path


OUTPUT_FILE = Path(__file__).resolve().parent / "arm_joint_history.csv"
JOINTS = ("shoulder", "elbow", "wrist")


def generate_history(output_file=OUTPUT_FILE, samples_per_joint=120, seed=24):
    rng = random.Random(seed)
    start = datetime(2026, 1, 1)

    with output_file.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            (
                "timestamp",
                "arm_id",
                "joint",
                "torque_nm",
                "temperature_c",
                "vibration_mm_s",
                "wear_label",
            )
        )

        for index in range(samples_per_joint):
            for joint_index, joint in enumerate(JOINTS):
                worn = index % 2 == 1
                if worn:
                    torque = rng.uniform(82, 151)
                    temperature = rng.uniform(68, 94)
                    vibration = rng.uniform(4.2, 10.8)
                else:
                    torque = rng.uniform(28, 88)
                    temperature = rng.uniform(34, 73)
                    vibration = rng.uniform(0.3, 4.8)

                writer.writerow(
                    (
                        (start + timedelta(hours=index, minutes=joint_index)).isoformat(
                            sep=" "
                        ),
                        f"TRAIN-{index % 12 + 1:02}",
                        joint,
                        f"{torque:.2f}",
                        f"{temperature:.2f}",
                        f"{vibration:.2f}",
                        int(worn),
                    )
                )


if __name__ == "__main__":
    generate_history()
    print(f"Generated synthetic historical readings at {OUTPUT_FILE}")
