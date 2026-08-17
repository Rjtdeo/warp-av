"""
Warp AV - Mission Log Review Tool

Reads a JSONL telemetry file and prints a short,
human-readable mission report.

Example:

    python tools/review_mission.py logs/mission_0001.jsonl
"""

import json
import os
import sys


def load_log(filename):
    records = []

    with open(filename, "r") as file:
        for line_number, line in enumerate(file, start=1):

            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))

            except json.JSONDecodeError as error:
                print(
                    f"Warning: could not read "
                    f"line {line_number}: {error}"
                )

    return records


def find_event(records, event_name):
    for record in records:
        if record.get("event") == event_name:
            return record

    return None


def find_events(records):
    return [
        record
        for record in records
        if "event" in record
    ]


def find_ticks(records):
    return [
        record
        for record in records
        if "pose" in record
    ]


def format_yes_no(value):
    return "YES" if value else "NO"


def main():

    # ========================================================
    # Check command-line argument
    # ========================================================

    if len(sys.argv) != 2:

        print("")
        print("Usage:")
        print(
            "python tools/review_mission.py "
            "logs/mission_0001.jsonl"
        )
        print("")

        sys.exit(1)

    filename = sys.argv[1]

    # ========================================================
    # Check file exists
    # ========================================================

    if not os.path.exists(filename):

        print(
            f"\nERROR: File not found: {filename}\n"
        )

        sys.exit(1)

    # ========================================================
    # Load log
    # ========================================================

    records = load_log(filename)

    if not records:

        print("\nERROR: Log file is empty.\n")
        sys.exit(1)

    events = find_events(records)
    ticks = find_ticks(records)

    mission_name = os.path.splitext(
        os.path.basename(filename)
    )[0]

    # ========================================================
    # Find important events
    # ========================================================

    mission_started = find_event(
        records,
        "mission_started"
    )

    route_planned = find_event(
        records,
        "route_planned"
    )

    autonomy_engaged = find_event(
        records,
        "autonomy_engaged"
    )

    perception_failure = find_event(
        records,
        "perception_failure"
    )

    safety_intervention = find_event(
        records,
        "safety_intervention"
    )

    safe_stop = find_event(
        records,
        "vehicle_safe_stop"
    )

    autonomy_disengaged = find_event(
        records,
        "autonomy_disengaged"
    )

    mission_failed = find_event(
        records,
        "mission_failed"
    )

    mission_completed = find_event(
        records,
        "mission_completed"
    )

    emergency_stop = find_event(
        records,
        "emergency_stop"
    )

    # ========================================================
    # Determine mission result
    # ========================================================

    if mission_failed:

        result = "FAILED"

    elif mission_completed:

        result = "COMPLETED"

    else:

        # Try final telemetry state
        result = "UNKNOWN"

        if ticks:

            final_mission_state = ticks[-1].get(
                "mission",
                "unknown"
            )

            result = final_mission_state.upper()

    # ========================================================
    # Find speed information
    # ========================================================

    max_speed_mps = 0.0

    for tick in ticks:

        speed = tick.get(
            "pose",
            {}
        ).get(
            "speed",
            0.0
        )

        if speed > max_speed_mps:
            max_speed_mps = speed

    max_speed_kmh = max_speed_mps * 3.6

    # --------------------------------------------------------
    # Speed at safety intervention
    # --------------------------------------------------------

    intervention_speed_kmh = None

    if safety_intervention:

        intervention_data = (
            safety_intervention.get(
                "data",
                {}
            )
        )

        intervention_speed_kmh = (
            intervention_data.get(
                "speed_kmh"
            )
        )

    # ========================================================
    # Find braking information
    # ========================================================

    max_brake = 0.0

    safety_stop_ticks = []

    for tick in ticks:

        command = tick.get(
            "command",
            {}
        )

        brake = command.get(
            "brake",
            0.0
        )

        if brake > max_brake:
            max_brake = brake

        if tick.get("behavior") == "safety_stop":
            safety_stop_ticks.append(tick)

    # ========================================================
    # Final vehicle speed
    # ========================================================

    final_speed_mps = None

    if ticks:

        final_speed_mps = ticks[-1].get(
            "pose",
            {}
        ).get(
            "speed"
        )

    final_speed_kmh = None

    if final_speed_mps is not None:
        final_speed_kmh = (
            final_speed_mps * 3.6
        )

    # ========================================================
    # Failure time
    # ========================================================

    failure_elapsed = None

    if perception_failure:

        failure_elapsed = (
            perception_failure
            .get("data", {})
            .get("elapsed_sec")
        )

    # ========================================================
    # Safe-stop time
    # ========================================================

    stop_response_time = None

    if perception_failure and safe_stop:

        failure_timestamp = (
            perception_failure.get("t")
        )

        stop_timestamp = (
            safe_stop.get("t")
        )

        if (
            failure_timestamp is not None
            and stop_timestamp is not None
        ):

            stop_response_time = (
                stop_timestamp
                - failure_timestamp
            )

    # ========================================================
    # Failure reason
    # ========================================================

    failure_reason = "None"

    if mission_failed:

        failure_reason = (
            mission_failed.get(
                "description",
                "Unknown"
            )
        )

    # ========================================================
    # Safety information
    # ========================================================

    safety_state = "N/A"
    driving_allowed = None

    if safety_intervention:

        intervention_data = (
            safety_intervention.get(
                "data",
                {}
            )
        )

        safety_state = (
            intervention_data.get(
                "safety_state",
                "unknown"
            )
        )

        driving_allowed = (
            intervention_data.get(
                "driving_allowed"
            )
        )

    # ========================================================
    # Print report
    # ========================================================

    print("")
    print(
        "===================================================="
    )
    print(
        "              WARP AV MISSION REVIEW"
    )
    print(
        "===================================================="
    )

    print(
        f"\nMission log:       {mission_name}"
    )

    print(
        f"Result:            {result}"
    )

    print(
        f"Telemetry records: {len(ticks)}"
    )

    print(
        f"Event records:     {len(events)}"
    )

    print("")
    print(
        "---------------- MISSION SETUP --------------------"
    )

    print(
        f"Mission started:      "
        f"{format_yes_no(mission_started is not None)}"
    )

    print(
        f"Route planned:        "
        f"{format_yes_no(route_planned is not None)}"
    )

    print(
        f"Autonomy engaged:     "
        f"{format_yes_no(autonomy_engaged is not None)}"
    )

    print("")
    print(
        "---------------- VEHICLE DATA ---------------------"
    )

    print(
        f"Maximum speed:        "
        f"{max_speed_kmh:.2f} km/h"
    )

    if intervention_speed_kmh is not None:

        print(
            f"Speed at failure:     "
            f"{intervention_speed_kmh:.2f} km/h"
        )

    if final_speed_kmh is not None:

        print(
            f"Final speed:          "
            f"{final_speed_kmh:.3f} km/h"
        )

    print(
        f"Maximum brake:        "
        f"{max_brake:.2f}"
    )

    print("")
    print(
        "---------------- FAILURE TEST ---------------------"
    )

    print(
        f"Failure injected:     "
        f"{format_yes_no(perception_failure is not None)}"
    )

    if failure_elapsed is not None:

        print(
            f"Failure time:         "
            f"{failure_elapsed:.2f} sec"
        )

    print(
        f"Failure reason:       "
        f"{failure_reason}"
    )

    print("")
    print(
        "---------------- SAFETY RESPONSE ------------------"
    )

    print(
        f"Safety intervention: "
        f"{format_yes_no(safety_intervention is not None)}"
    )

    print(
        f"Safety state:         "
        f"{safety_state}"
    )

    if driving_allowed is not None:

        print(
            f"Driving allowed:      "
            f"{driving_allowed}"
        )

    print(
        f"Safety braking ticks: "
        f"{len(safety_stop_ticks)}"
    )

    print(
        f"Safe stop achieved:   "
        f"{format_yes_no(safe_stop is not None)}"
    )

    if stop_response_time is not None:

        print(
            f"Failure → stop time:  "
            f"{stop_response_time:.3f} sec"
        )

    print(
        f"Autonomy disengaged:  "
        f"{format_yes_no(autonomy_disengaged is not None)}"
    )

    print(
        f"Emergency stop used:  "
        f"{format_yes_no(emergency_stop is not None)}"
    )

    print("")
    print(
        "---------------- EVENT TIMELINE -------------------"
    )

    if events:

        first_timestamp = events[0].get(
            "t",
            0.0
        )

        for event in events:

            timestamp = event.get(
                "t",
                first_timestamp
            )

            relative_time = (
                timestamp - first_timestamp
            )

            event_name = event.get(
                "event",
                "unknown"
            )

            description = event.get(
                "description",
                ""
            )

            print(
                f"+{relative_time:6.2f}s | "
                f"{event_name:22} | "
                f"{description}"
            )

    print("")
    print(
        "===================================================="
    )

    if (
        result == "FAILED"
        and safe_stop is not None
        and final_speed_kmh is not None
        and final_speed_kmh < 0.5
    ):

        print(
            " SAFETY RESULT: PASS"
        )

        print(
            " Failure detected and vehicle reached safe stop."
        )

    elif result == "COMPLETED":

        print(
            " MISSION RESULT: COMPLETED"
        )

    else:

        print(
            " SAFETY RESULT: REVIEW REQUIRED"
        )

    print(
        "===================================================="
    )
    print("")


if __name__ == "__main__":
    main()
