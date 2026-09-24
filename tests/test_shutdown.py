"""Ctrl-C must end the server promptly, even while a Tuner tool is waiting on something."""

import os
import subprocess
import sys
import time

SCRIPT = """
import asyncio, time, sys
from sqlmodel import Session
from slm.db import Job, Project, engine
from slm.tuner import tools
from slm.tuner.session import tuner
from slm.api.app import shutdown_services

with Session(engine()) as s:
    p = Project(name="p"); s.add(p); s.commit(); s.refresh(p)
    job = Job(project_id=p.id, kind="sft", status="queued"); s.add(job); s.commit(); s.refresh(job)

# A tool blocked in _wait on a job that will never finish, on the Tuner's tool thread.
loop = tuner._event_loop()
fut = asyncio.run_coroutine_threadsafe(asyncio.to_thread(tools._wait, job.id, 3600), loop)
time.sleep(0.5)
t0 = time.time()
shutdown_services()
took = time.time() - t0
import threading
alive = [t.name for t in threading.enumerate() if t.name.startswith("tuner-tools") and t.is_alive()]
assert not alive, alive  # the blocked wait returned instead of running for an hour
assert took < 6, took
print(f"shutdown {took:.1f}s")
"""


def test_server_shutdown_does_not_wait_for_a_blocked_tool(tmp_path):
    # Regression: Ctrl-C hung at "Exception ignored on threading shutdown ... t.join()" until a
    # second Ctrl-C, whenever a tool thread was still polling a job.
    env = os.environ | {"SLM_WORKSPACE": str(tmp_path)}
    t0 = time.time()
    proc = subprocess.run([sys.executable, "-c", SCRIPT], env=env, capture_output=True, text=True, timeout=60)
    took = time.time() - t0
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "shutdown" in proc.stdout and "Exception ignored" not in proc.stderr
    assert took < 20, f"process took {took:.0f}s to exit"
