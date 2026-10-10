"""
DAY 1 · HANDS-ON — Build Your Rule Engine
23IF02 System Automation

Complete implementation of Steps 1–8.
"""

import random

import time

LOG_FILE = "day1_log.txt"


# ---------------------------------------------------------------------------
# STEP 1 & STEP 6 — Generate simulated sensor data
# ---------------------------------------------------------------------------
def generate_sensor_data():
    """Return simulated sensor readings."""
    return {
        "temperature": random.randint(50, 95),
        "pressure": round(random.uniform(1.0, 5.0), 2),
        "vibration": round(random.uniform(0.1, 5.0), 2),
        "current": round(random.uniform(5.0, 15.0), 2)
    }


# ---------------------------------------------------------------------------
# STEP 2, STEP 7 & STEP 8 — Automation rules
# ---------------------------------------------------------------------------
RULES = [
    {
        "name": "EMERGENCY_HIGH_TEMP",
        "priority": 1,
        "condition": lambda r: r["temperature"] > 90,
        "action": "EMERGENCY_STOP"
    },

    {
        "name": "HIGH_VIBRATION",
        "priority": 2,
        "condition": lambda r: r["vibration"] > 4.0,
        "action": "STOP_MACHINE"
    },

    {
        "name": "HIGH_CURRENT",
        "priority": 3,
        "condition": lambda r: r["current"] > 13,
        "action": "REDUCE_LOAD"
    },

    {
        "name": "HIGH_TEMPERATURE",
        "priority": 4,
        "condition": lambda r: r["temperature"] > 80,
        "action": "COOL_MACHINE"
    },

    {
        "name": "HIGH_PRESSURE",
        "priority": 5,
        "condition": lambda r: r["pressure"] > 4.0,
        "action": "CHECK_PRESSURE"
    }
]


# ---------------------------------------------------------------------------
# STEP 3 — Detect machine state from the rules
# ---------------------------------------------------------------------------
def evaluate_rules(reading, rules):
    """
    Return (rule_name, action) for the first matching rule.
    Rules are checked in priority order.
    """
    sorted_rules = sorted(rules, key=lambda r: r["priority"])

    for rule in sorted_rules:
        if rule["condition"](reading):
            return rule["name"], rule["action"]

    return "NO_RULE_FIRED", "CONTINUE_NORMAL"


# ---------------------------------------------------------------------------
# STEP 4 — Trigger an action when a rule fires
# ---------------------------------------------------------------------------
def take_action(action):
    """Return a human-readable message for the action."""

    actions = {
        "EMERGENCY_STOP": "Emergency stop activated!",
        "STOP_MACHINE": "Machine stopped due to unsafe condition.",
        "REDUCE_LOAD": "Machine load reduced.",
        "COOL_MACHINE": "Cooling system activated.",
        "CHECK_PRESSURE": "Pressure is high. Inspection required.",
        "CONTINUE_NORMAL": "Machine operating normally."
    }

    return actions.get(action, f"No handler for action: {action}")


# ---------------------------------------------------------------------------
# STEP 5 — Log every decision made
# ---------------------------------------------------------------------------
def log_decision(reading, rule_name, action):
    """Append one line to the log file."""

    with open(LOG_FILE, "a") as file:
        file.write(
            f"Reading: {reading} | "
            f"Rule: {rule_name} | "
            f"Action: {action}\n"
        )


# ---------------------------------------------------------------------------
# MAIN PROGRAM
# ---------------------------------------------------------------------------
def main():

    for tick in range(1, 11):

        reading = generate_sensor_data()

        rule_name, action = evaluate_rules(reading, RULES)

        outcome = take_action(action)

        print(f"[Tick {tick:02d}] SENSE -> {reading}")
        print(f"           THINK -> {rule_name}")
        print(f"           ACT   -> {outcome}\n")

        log_decision(reading, rule_name, action)

        time.sleep(0.2)


if __name__ == "__main__":
    main()