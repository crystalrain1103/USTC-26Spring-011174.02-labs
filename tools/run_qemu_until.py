#!/usr/bin/env python3
"""Run QEMU until a success marker, failure marker, process exit, or timeout.

This is intentionally small and generic because Makefile tests should not wait
for the full timeout after the guest kernel has already printed a panic.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import re
import select
import subprocess
import sys
import time
from pathlib import Path


def set_nonblocking(fd: int) -> None:
    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)


def decode_command_text(text: str) -> bytes:
    return text.replace("\\n", "\n").encode()


def extract_matching_line(text: str, patterns: list[re.Pattern[str]]) -> str:
    for line in text.splitlines():
        for pattern in patterns:
            if pattern.search(line):
                return line.strip()
    return ""


def terminate(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=3)
    except Exception:
        try:
            proc.kill()
            proc.wait(timeout=3)
        except Exception:
            pass


def main() -> int:
    ap = argparse.ArgumentParser(description="run QEMU until success/failure output appears")
    ap.add_argument("--timeout", type=float, required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--commands", default="")
    ap.add_argument("--success", action="append", default=[])
    ap.add_argument("--fail", action="append", default=[])
    ap.add_argument("qemu_cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    qemu_cmd = args.qemu_cmd
    if qemu_cmd and qemu_cmd[0] == "--":
        qemu_cmd = qemu_cmd[1:]
    if not qemu_cmd:
        print("run_qemu_until.py: missing QEMU command", file=sys.stderr)
        return 2
    if not args.success:
        print("run_qemu_until.py: at least one --success pattern is required", file=sys.stderr)
        return 2

    success_patterns = [re.compile(p) for p in args.success]
    fail_patterns = [re.compile(p) for p in args.fail]

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    proc = subprocess.Popen(
        qemu_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        close_fds=True,
    )
    assert proc.stdout is not None
    set_nonblocking(proc.stdout.fileno())

    if args.commands and proc.stdin is not None:
        try:
            proc.stdin.write(decode_command_text(args.commands))
            proc.stdin.flush()
        except BrokenPipeError:
            pass

    output = bytearray()
    deadline = time.monotonic() + args.timeout
    outcome = "timeout"
    reason = f"timeout after {args.timeout:g}s"

    with log_path.open("wb") as logf:
        try:
            while proc.poll() is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break

                r, _, _ = select.select([proc.stdout], [], [], min(0.2, remaining))
                if proc.stdout in r:
                    while True:
                        try:
                            chunk = os.read(proc.stdout.fileno(), 65536)
                        except BlockingIOError:
                            break
                        if not chunk:
                            break
                        output.extend(chunk)
                        logf.write(chunk)
                    logf.flush()

                    text = output.decode("utf-8", errors="replace")
                    success_line = extract_matching_line(text, success_patterns)
                    if success_line:
                        outcome = "success"
                        reason = success_line
                        break

                    fail_line = extract_matching_line(text, fail_patterns)
                    if fail_line:
                        outcome = "failure"
                        reason = fail_line
                        break

            if proc.poll() is not None and outcome == "timeout":
                outcome = "exit"
                reason = f"QEMU exited with status {proc.returncode}"

            # Capture any final bytes produced just before termination.
            try:
                time.sleep(0.05)
                while proc.stdout is not None:
                    try:
                        chunk = os.read(proc.stdout.fileno(), 65536)
                    except BlockingIOError:
                        break
                    if not chunk:
                        break
                    output.extend(chunk)
                    logf.write(chunk)
                logf.flush()
            except OSError:
                pass
        finally:
            terminate(proc)
            try:
                if proc.stdin is not None:
                    proc.stdin.close()
            except OSError:
                pass

    final_text = output.decode("utf-8", errors="replace")
    if outcome == "failure":
        reason = extract_matching_line(final_text, fail_patterns) or reason
    elif outcome == "success":
        reason = extract_matching_line(final_text, success_patterns) or reason

    if outcome == "success":
        return 0

    print(f"QEMU test stopped: {outcome}: {reason}", file=sys.stderr)
    print(f"see log: {log_path}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
