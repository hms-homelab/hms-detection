#!/usr/bin/env python3
"""
E2E test for confidence gate on MQTT result/context publishing.

Uses live hms-detection with front_door camera (reliably detects parked cars
at ~86% confidence).

Scenarios:
  1. gate=0.70, car in classes → 86% > 70%  → result + context + recording
  2. gate=1.00, car in classes → 86% < 100% → no result, no context, no recording

Usage:
  sudo python3 e2e_confidence_gate_test.py
  sudo python3 e2e_confidence_gate_test.py --high-only
  sudo python3 e2e_confidence_gate_test.py --low-only
"""

import argparse
import json
import os
import subprocess
import sys
import time
import threading
from pathlib import Path

import paho.mqtt.client as mqtt

MQTT_BROKER = "192.168.2.15"
MQTT_PORT = 1883
MQTT_USER = "aamat"
MQTT_PASS = "exploracion"
TOPIC_PREFIX = "yolo_detection"
CAMERA_ID = "front_door"
SERVICE_USER = "aamat"

CONFIG_SRC = "/opt/yolo_detection/config.yaml"
CONFIG_DST = "/opt/hms-detection/config.yaml"
EVENTS_DIR = "/mnt/ssd/events"
SNAPSHOTS_DIR = "/mnt/ssd/snapshots"

MOTION_HOLD_SECONDS = 10
SETTLE_SECONDS = 25


class MqttCollector:
    def __init__(self, camera_id):
        self.camera_id = camera_id
        self.messages = {}
        self.client = mqtt.Client(client_id=f"e2e_test_{os.getpid()}")
        self.client.username_pw_set(MQTT_USER, MQTT_PASS)
        self.client.on_message = self._on_message
        self._lock = threading.Lock()

    def _on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
        except json.JSONDecodeError:
            payload = msg.payload.decode()
        with self._lock:
            self.messages.setdefault(msg.topic, []).append(payload)
        print(f"    MQTT: {msg.topic} -> {json.dumps(payload)[:120]}...")

    def start(self):
        self.client.connect(MQTT_BROKER, MQTT_PORT, 60)
        self.client.subscribe(f"{TOPIC_PREFIX}/{self.camera_id}/#", qos=1)
        self.client.loop_start()

    def stop(self):
        self.client.loop_stop()
        self.client.disconnect()

    def get(self, subtopic):
        topic = f"{TOPIC_PREFIX}/{self.camera_id}/{subtopic}"
        with self._lock:
            return list(self.messages.get(topic, []))


def send_motion(camera_id, action, post_roll=5):
    subprocess.run([
        "mosquitto_pub", "-h", MQTT_BROKER, "-u", MQTT_USER, "-P", MQTT_PASS,
        "-t", f"camera/event/motion/{action}",
        "-m", json.dumps({"camera_id": camera_id, "post_roll_seconds": post_roll})
    ], check=True)


def wait_for_health(timeout=30):
    import urllib.request
    for i in range(timeout):
        time.sleep(1)
        try:
            req = urllib.request.Request("http://localhost:8000/health")
            with urllib.request.urlopen(req, timeout=2) as resp:
                if b'"healthy"' in resp.read():
                    time.sleep(3)
                    return True
        except Exception:
            pass
    return False


def set_config(classes, gate):
    """Set front_door classes and gate, preserving YAML formatting."""
    # Restore clean base first
    subprocess.run(["git", "checkout", "config.yaml"],
                   cwd="/opt/yolo_detection", check=True,
                   capture_output=True)

    with open(CONFIG_SRC) as f:
        lines = f.readlines()

    classes_str = "[" + ", ".join(classes) + "]"
    new_lines = []
    in_front_door = False

    for line in lines:
        stripped = line.rstrip("\n")
        # Detect front_door section
        if stripped == "  front_door:":
            in_front_door = True
            new_lines.append(line)
            continue
        # Detect end of front_door section (next top-level or camera key)
        if in_front_door and (line[0:1] not in (" ", "\n", "") or
                              (line.startswith("  ") and not line.startswith("    ") and line.strip() and line.strip()[0] != '#')):
            in_front_door = False

        if in_front_door and stripped.lstrip().startswith("classes:"):
            new_lines.append(f"    classes: {classes_str}\n")
            new_lines.append(f"    immediate_notification_confidence: {gate}\n")
        elif in_front_door and "immediate_notification_confidence" in stripped:
            pass  # skip existing gate line
        else:
            new_lines.append(line)

    with open(CONFIG_SRC, "w") as f:
        f.writelines(new_lines)

    # Fix ownership (script runs as root, service runs as aamat)
    import pwd
    uid = pwd.getpwnam(SERVICE_USER).pw_uid
    gid = pwd.getpwnam(SERVICE_USER).pw_gid

    subprocess.run(["cp", CONFIG_SRC, CONFIG_DST], check=True)
    os.chown(CONFIG_DST, uid, gid)
    os.chown(CONFIG_SRC, uid, gid)


def restart_service():
    subprocess.run(["systemctl", "restart", "hms-detection"], check=True)
    if not wait_for_health(30):
        raise RuntimeError("hms-detection failed to become healthy")
    print("    Service healthy")


def restore_config():
    subprocess.run(["git", "checkout", "config.yaml"],
                   cwd="/opt/yolo_detection", check=True,
                   capture_output=True)
    subprocess.run(["cp", CONFIG_SRC, CONFIG_DST], check=True)


def recent_files(directory, prefix, since=None, seconds=90):
    cutoff = since if since else (time.time() - seconds)
    return sorted(
        [p for p in Path(directory).glob(f"{prefix}*") if p.stat().st_mtime > cutoff],
        key=lambda p: p.stat().st_mtime, reverse=True
    )


def run_scenario(name, classes, gate, expect_result, expect_context,
                 expect_recording, expect_snapshot):
    print(f"\n{'='*60}")
    print(f"SCENARIO: {name}")
    print(f"  classes={classes}, gate={gate}")
    print(f"  expect: result={expect_result}, context={expect_context}, "
          f"recording={expect_recording}, snapshot={expect_snapshot}")
    print(f"{'='*60}")

    print("\n[1] Updating config and restarting...")
    set_config(classes, gate)
    restart_service()

    collector = MqttCollector(CAMERA_ID)
    collector.start()
    time.sleep(1)

    # Mark start time for file age checks
    scenario_start = time.time()

    print(f"\n[2] Triggering {CAMERA_ID} motion event...")
    send_motion(CAMERA_ID, "start", 5)
    print(f"    Holding {MOTION_HOLD_SECONDS}s...")
    time.sleep(MOTION_HOLD_SECONDS)
    print("    Sending motion_stop...")
    send_motion(CAMERA_ID, "stop")
    print(f"    Waiting {SETTLE_SECONDS}s for completion...")
    time.sleep(SETTLE_SECONDS)

    collector.stop()
    results = collector.get("result")
    contexts = collector.get("context")
    recordings = recent_files(EVENTS_DIR, f"{CAMERA_ID}_", since=scenario_start)
    snapshots = recent_files(SNAPSHOTS_DIR, f"{CAMERA_ID}_", since=scenario_start)

    passed = True

    def check(label, actual, expected):
        nonlocal passed
        ok = actual == expected
        if not ok:
            passed = False
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}: got={actual}, expected={expected}")

    print(f"\n[3] Results:")
    print(f"    /result messages: {len(results)}")
    print(f"    /context messages: {len(contexts)}")
    print(f"    Recordings: {[p.name for p in recordings]}")
    print(f"    Snapshots: {[p.name for p in snapshots]}")

    if results:
        confs = [d.get("confidence", 0) for d in results[0].get("detections", [])]
        print(f"    Detected: {results[0].get('detected_objects')}, confs={confs}")
    if contexts:
        print(f"    Context: {contexts[0].get('context', '')}")

    print(f"\n[4] Assertions:")
    check("/result published", len(results) > 0, expect_result)
    check("/context published", len(contexts) > 0, expect_context)
    check("Recording exists", len(recordings) > 0, expect_recording)
    check("Snapshot exists", len(snapshots) > 0, expect_snapshot)

    return passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--high-only", action="store_true")
    parser.add_argument("--low-only", action="store_true")
    args = parser.parse_args()

    if os.geteuid() != 0:
        print("ERROR: Run with sudo")
        sys.exit(1)

    all_passed = True

    try:
        if not args.low_only:
            ok = run_scenario(
                name="HIGH: car ~86%, gate=0.70 -> result + context + recording",
                classes=["person", "dog", "package", "car"],
                gate=0.70,
                expect_result=True,
                expect_context=True,
                expect_recording=True,
                expect_snapshot=True,
            )
            all_passed = all_passed and ok

        if not args.high_only:
            ok = run_scenario(
                name="LOW: car ~86%, gate=1.00 -> no result, no context, no recording, snapshot kept",
                classes=["person", "dog", "package", "car"],
                gate=1.00,
                expect_result=False,
                expect_context=False,
                expect_recording=False,
                expect_snapshot=True,
            )
            all_passed = all_passed and ok

    finally:
        print("\n[*] Restoring default config...")
        restore_config()
        restart_service()

    print(f"\n{'='*60}")
    print("ALL SCENARIOS PASSED" if all_passed else "SOME SCENARIOS FAILED")
    print(f"{'='*60}")
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
