# MILE on Franka — Block-Stacking Tutorial

**MILE** (Model-based Intervention Learning) lets a robot learn from a human who intervenes *only when the robot is about to fail*. This repo adapts MILE to a Franka Panda block-stacking task and packages it as a 2-hour hands-on tutorial.

Paper: https://liralab.usc.edu/mile/

> **Summer-school participants:** start at [docs/tutorial/00-setup.md](docs/tutorial/00-setup.md).

---

## Quick start (Docker — recommended)

Everything runs inside a single Docker image built on top of hucebot's `franka-humble` base.

```bash
# One-time: build the hucebot base image (not on any registry)
git clone https://github.com/hucebot/multipanda_ros2
cd multipanda_ros2 && docker compose build && cd -

# Build the MILE image, start the service, fetch artifacts
make build
make up
make fetch-artifacts   # downloads trained models from GitHub Release

# Verify the environment is ready
make tutorial-check    # expected: "tutorial-check OK — you're ready."
```

---

## The tutorial (2 hours)

See [docs/tutorial/03-walkthrough.md](docs/tutorial/03-walkthrough.md) for the full flow. Quick reference:

| Make verb | What it does |
|---|---|
| `make tutorial-check` | Assert imports + artifacts are present |
| `make tutorial-check-loss` | Green-light test for the MILE-loss exercise |
| `make tutorial-metaworld` | Part 1: MetaWorld peg-insert with synthetic expert |
| `make tutorial-fake` | Part 2: Franka fake backend smoke test |
| `make sim-up` | Start the multipanda MuJoCo stacking sim |
| `make tutorial-collect-train` | Part 3: keyboard teleop → collect → train |
| `make eval-base` | Evaluate the base policy (run before and after training) |

**Docs:**
- [00-setup.md](docs/tutorial/00-setup.md) — pre-session setup (homework)
- [01-concepts.md](docs/tutorial/01-concepts.md) — 5-min MILE primer
- [02-loss-exercise.md](docs/tutorial/02-loss-exercise.md) — implement the MILE loss
- [03-walkthrough.md](docs/tutorial/03-walkthrough.md) — full tutorial flow
- [04-teleop.md](docs/tutorial/04-teleop.md) — keyboard / gamepad reference
- [05-troubleshooting.md](docs/tutorial/05-troubleshooting.md) — FAQ
- [06-instructor-runbook.md](docs/tutorial/06-instructor-runbook.md) — instructor/staff guide

---

## Dev setup (host conda, no Docker)

```bash
conda create -n mile python=3.10
pip install -r requirements.txt   # installs the pinned MetaWorld v2 commit
pip install -e .
```

Do **not** `pip install metaworld` — the PyPI release is v3 and requires `gymnasium>=1.1`, which breaks the stack. Install via `requirements.txt` only.

---

## Key make verbs (full list)

```bash
make build              # build the Docker image
make up / make down     # start / stop the persistent sim service
make shell              # interactive in-container shell (env sourced)
make sim-up             # launch the multipanda MuJoCo stacking sim (headless)
make sim-gui            # live MuJoCo window (needs xhost +local:root)
make collect-mediocre   # scripted demos → sim_demos_mediocre.npz
make base-policy        # BC-train the mediocre base policy
make eval-base          # measure base policy sim success rate
make mile               # iterative MILE run (sim, joystick intervener)
make pose-test          # 20 pose-layer unit tests (no ROS/hardware needed)
make joystick-check     # verify gamepad reads inside the container
make view-twin          # live MuJoCo digital twin from AprilTag + joint states
make apriltag-up        # launch camera + apriltag_ros + calibration static tf (MILE_CAMERA=realsense|webcam, default realsense)
make calibrate-camera   # eye-to-hand calibration → config/camera_calib.yaml
make franka-up          # start the real FR3 controller (franka_ros2 container)
make mile-real          # iterative MILE on the real FR3
make eval-real          # policy eval on the real FR3
make eval-expert-real   # scripted expert (not a learned policy) eval on the real FR3
```

---

## Architecture

The system jointly trains two networks and deploys only the policy:

- **`mile/computational_model.py`** — probit intervention model; `COST_LOOKUP` maps each env to `[cost, cdf_scale]`.
- **`mile/algorithm.py`** — `InterventionTrainer` jointly trains policy π_θ and mental model π̃_ξ. Loss = BCE on ν + Gaussian NLL of human action on ν=1 steps.
- **`mile_franka/`** — Franka adaptation: `FrankaEnv`, `RobotBackend` ABC, `AprilTagPoseSource`, `KeyboardDevice` / `JoystickDevice`, `Collector`, tutorial scaffolding.
- **`scripts/train_mile.py`** — offline and iterative training modes. `scripts/tutorial_train.py` injects the participant's loss before training.

Config files: `config/franka_sim.yaml` (sim, joystick), `config/franka_real.yaml` (real FR3), `config/tutorial_metaworld.yaml` (Part 1), `config/tutorial_franka.yaml` (Part 3).
