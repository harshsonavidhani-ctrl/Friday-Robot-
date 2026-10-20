"""Run ONE version of Friday's exploration in ONE simulated room.

    [BODY=1] [CAP_V=0.125] python run_one.py <code_root> <world> <out.json> <seconds> [run flags...]

<code_root> is a git worktree of the version under test; its rl/ is imported, so
the control loop is that version's own.

Plain mode: the version's own --simulate (the world moves exactly as commanded,
null motor driver; the real driver is replaced by one that raises).

BODY=1: the version's REAL motor driver, talking to body.FakeBoard (a stand-in
board; `serial.Serial` is replaced before anything can import it, and FakeBoard
refuses every port name but its own), with the measured drivetrain model moving
the world, and the loop on its real-hardware branch (gyro, executed(), unstick).
CAP_V (BODY only): a top forward speed at the wheels, m/s -- the experiment for
"drive at the 4 Sept 03:52 speed" (0.125).
"""
import os
import sys

root, world, out, seconds = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
extra = sys.argv[5:]
BODY = os.environ.get("BODY") == "1"
here = os.path.dirname(os.path.abspath(__file__))
os.chdir(root)
sys.path.insert(0, os.path.join(root, "rl"))
sys.path.insert(0, root)
sys.path.insert(0, here)

import body  # noqa: E402
import simworlds  # noqa: E402

if BODY:
    import serial  # noqa: E402
    serial.Serial = body.FakeBoard            # nothing in this process opens a real port

import pi_deploy  # noqa: E402  (the version under test)

pi_deploy.SimulatedWorld = simworlds.make_world_class(pi_deploy.SimulatedWorld, world, out)

if BODY:
    pi_deploy.resolve_serial_port = lambda *a, **k: body.FAKE_PORT
else:
    class _NoRealMotors:
        def __init__(self, *a, **k):
            raise RuntimeError("A/B sim tried to build the REAL motor driver -- refused")
    pi_deploy.FridayMotorDriver = _NoRealMotors

import friday_localnav_deploy as fld  # noqa: E402

if BODY:
    cls = fld.LeRobotExperimentalControlLoop
    _run = cls.run

    def run(self, *a, **k):
        w = self.sim
        assert w is not None and isinstance(self.motors, pi_deploy.FridayMotorDriver)
        assert isinstance(self.motors._ser, body.FakeBoard)
        self.sim = None                       # the real-hardware branch from here on
        cap = float(os.environ.get("CAP_V", "0") or 0)
        if cap > 0:
            _drive = self.motors.drive

            def drive(v, omega, *a, **k):
                return _drive(min(v, cap), omega, *a, **k)
            self.motors.drive = drive
        body.start_physics(w)
        return _run(self, *a, **k)

    cls.run = run
    argv = ["--simulate", "--motor", "friday", "--i-am-watching",
            "--motor-max-duty", "0.85", "--motor-min-duty", "80"]
else:
    argv = ["--simulate", "--motor", "null"]
argv += ["--no-voice", "--duration", seconds] + extra
rc = fld.main(argv) or 0
if simworlds.LAST[0] is not None:
    simworlds.LAST[0].dump()
print("A/B run finished rc", rc, flush=True)
sys.stdout.flush()
sys.stderr.flush()
os._exit(int(rc))        # a lingering non-daemon thread must not hold the run open
