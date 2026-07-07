MILE_REAL_STACK ?= multipanda             ## real controller stack: multipanda (default) or fr3; override per-invocation or export in your shell
DC      := docker compose -f docker/docker-compose.yml
ENVSH   := source scripts/in_container_env.sh
# $(call RUN,cmd) — run cmd inside the sim container (non-interactive; usable from host or make shell)
RUN      = $(DC) exec sim bash -lc '$(ENVSH) && $(1)'
RUND     = $(DC) exec -d sim bash -lc '$(ENVSH) && $(1)'
# Kill any leftover process still publishing to the arm (prior run or stale rclpy context).
KILLCLIENTS = self=$$$$; pgrep -f "train_mile.py|eval_base_policy_(sim|real)|eval_expert_real|eval_mile|franka_sim_rollout_record|build_base_policy|collect_synthetic_interventions" | grep -vx $$self | xargs -r kill 2>/dev/null; sleep 2; true
TS      := $(shell date -u +%Y%m%dT%H%M%S)
DEMOS   ?= output_dir/franka/sim_demos_mediocre.npz   ## base-policy input; override: make base-policy DEMOS=path.npz
GRIP_FORCE  ?= 40
CLOSE_WIDTH ?= 0.0
OPEN_WIDTH  ?= 0.08
# Resolve which gripper action is live and bail if none found.
GRIP_NS = ns=$$(ros2 action list 2>/dev/null | grep -E "/grasp$$" | head -1 | sed "s|/grasp||"); if [ -z "$$ns" ]; then echo "No gripper action server found -- is the controller/sim running?"; exit 1; fi; echo "gripper: $$ns"
MILE_CAMERA     ?= webcam                 ## camera driver: realsense or webcam (USB/UVC, e.g. Logitech); webcam here (no RealSense on this box)
FRANKA_CTR      ?= multipanda        ## fr3 controller container name
MULTIPANDA_CTR  ?= realtime_franka_humble  ## hucebot multipanda container name; override if yours differs
ROBOT_IP        ?= 176.16.0.1               ## robot FCI IP; the enp2s0-facing address on this lab's dedicated robot subnet
LOAD_GRIPPER    ?= true
# ROS2 workspace path inside each controller container -- override if yours differs.
FRANKA_WS       ?= /ros2_ws                  ## workspace path inside $(FRANKA_CTR)
MULTIPANDA_WS   ?= /home/user/humble_ws      ## workspace path inside $(MULTIPANDA_CTR)
FRANKA_SRC       = source /opt/ros/humble/setup.bash && source $(strip $(FRANKA_WS))/install/setup.bash && export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
MULTIPANDA_SRC   = source /opt/ros/humble/setup.bash && source $(strip $(MULTIPANDA_WS))/install/setup.bash && export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# tutorial-teleop and tutorial-collect-train need keyboard/stdin (pygame window,
# keyboard intervener) and must be run from inside `make shell`. The guard below enforces this.
CONTAINER_GUARD = @test -d /home/user/mile-code || { echo "ERROR: Run 'make shell' first, then run this command inside the container."; exit 1; }

.PHONY: \
  build up down shell \
  tutorial-check tutorial-check-loss tutorial-check-intervention-model tutorial-check-scripted-intervener \
  tutorial-metaworld eval-metaworld \
  tutorial-fake \
  sim-up sim-gui tutorial-teleop eval-base tutorial-collect-train eval-mile \
  franka-up apriltag-up eval-real eval-expert-real \
  spacemouse-check joystick-check vive-check pose-test tune-cost \
  real-home-smoke close-gripper open-gripper view-tags view-twin calibrate-camera calibrate-intrinsics franka-shell \
  collect-mediocre collect-expert base-policy mile mile-real fetch-artifacts

# ── Docker / session ──────────────────────────────────────────────────────────
# Run these from the host to manage the container.

build:                       ## build the Docker image
	DOCKER_BUILDKIT=0 $(DC) build

up:                          ## start the persistent sim container
	$(DC) up -d

down:                        ## stop the container
	$(DC) down

shell:                       ## open an interactive shell inside the container (env sourced, cd'd into repo; host display passed for pygame teleop)
	$(DC) exec -e DISPLAY=$$DISPLAY sim bash -lc '$(ENVSH) && exec bash'

# ── Tutorial: setup checks ────────────────────────────────────────────────────
# Callable from the host or from inside make shell.

tutorial-check:              ## assert imports + artifacts are present
	$(call RUN,python3 scripts/tutorial_check.py)

tutorial-check-loss:         ## run tests for the MILE-loss exercise
	$(DC) exec sim bash -c 'cd /home/user/mile-code && python3 -m pytest tests/test_loss_exercise.py -v'

tutorial-check-intervention-model: ## run tests for the intervention model exercise
	$(DC) exec sim bash -c 'cd /home/user/mile-code && python3 -m pytest tests/test_intervention_model_exercise.py -v'

tutorial-check-scripted-intervener: ## run tests for the scripted intervener exercise
	$(DC) exec sim bash -c 'cd /home/user/mile-code && python3 -m pytest tests/test_scripted_intervener_exercise.py -v'

# ── Tutorial: Part 1 — MetaWorld ──────────────────────────────────────────────
# Interactive (needs stdin for training output). Run from inside make shell.

tutorial-metaworld:          ## Part 1: MetaWorld peg-insert synthetic loop (uses your loss)
	$(call RUN,cd scripts && python3 tutorial_train.py --config ../config/tutorial_metaworld.yaml)

eval-metaworld:              ## Part 1: evaluate a trained MILE policy on MetaWorld; override MODEL=path/to/dir EPISODES=100
	$(call RUN,python3 scripts/eval_mile.py \
	  --trained_model $${MODEL:-output_dir} \
	  --num_episodes $${EPISODES:-100} \
	  --video_dir output_dir/metaworld_eval_videos_$(TS))

# ── Tutorial: Part 2 — Franka fake backend ────────────────────────────────────
# Interactive. Run from inside make shell.

tutorial-fake:               ## Part 2: smoke-test that the Franka environment loads correctly
	$(call RUN,python3 scripts/smoke_franka_env.py)

# ── Tutorial: Part 3 — Franka sim ─────────────────────────────────────────────
# sim-up / sim-gui / eval-base / eval-mile use $(call RUN,...) — callable from host.
# tutorial-teleop and tutorial-collect-train are interactive — run from inside make shell.

sim-up:                      ## launch the stacking sim headless (detached); wait ~10s before next step
	$(call RUND,bash scripts/sim_up.sh)
	@echo "sim launching headless; give it ~10s"

sim-gui:                     ## open a live MuJoCo viewer on the host display (host prereq: xhost +local:root)
	@if [ "$$DISPLAY" != "$${DISPLAY#localhost:}" ] || [ "$$DISPLAY" != "$${DISPLAY#/}" ]; then \
		docker exec -u root mile_sim bash -c \
			'which x11vnc >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -q x11vnc; }'; \
		python3 scripts/view_twin_macos.py \
			--viewer-cmd 'bash scripts/sim_gui.sh' \
			--pkill-pattern franka_sim_stacking; \
	else \
		echo "Host prereq (once per login): xhost +local:root"; \
		$(DC) exec -e DISPLAY=$$DISPLAY sim bash -lc '$(ENVSH) && bash scripts/sim_gui.sh'; \
	fi

tutorial-teleop:             ## Part 3 practice: free-play keyboard teleop in sim (Ctrl-C to exit; data not saved)
	$(CONTAINER_GUARD)
	python3 scripts/tutorial_teleop.py

eval-base:                   ## Part 3: evaluate the base policy in sim (run before training)
	$(call RUN,python3 scripts/eval_base_policy_sim.py --episodes 5 \
	  --video_dir output_dir/franka/eval_videos_$(TS))

tutorial-collect-train:      ## Part 3: keyboard teleop → collect interventions → train (uses your loss)
	$(CONTAINER_GUARD)
	cd scripts && python3 tutorial_train.py --config ../config/tutorial_franka.yaml

eval-mile:                   ## Part 3: evaluate the MILE-trained policy in sim (run after training)
	$(call RUN,python3 scripts/eval_base_policy_sim.py --episodes 5 \
	  --policy output_dir/franka/policy \
	  --video_dir output_dir/franka/eval_mile_videos_$(TS))

# ── Tutorial: Part 4 — Real FR3 ───────────────────────────────────────────────
# All callable from the host via docker exec.

franka-up:                   ## launch / verify the real controller stack; MILE_REAL_STACK=multipanda (default) just checks $(MULTIPANDA_CTR) is up (nothing launched from here); =fr3 starts mile_bringup in $(FRANKA_CTR); override ROBOT_IP=.. LOAD_GRIPPER=false MULTIPANDA_CTR=..
	@if [ "$(strip $(MILE_REAL_STACK))" = "multipanda" ]; then \
	  docker ps --format '{{.Names}}' | grep -qx "$(strip $(MULTIPANDA_CTR))" \
	    || { echo "Container $(MULTIPANDA_CTR) not running -- start it (from your multipanda_ros2 checkout): docker compose up -d"; exit 1; }; \
	  echo "multipanda controller running ($(MULTIPANDA_CTR)) — nothing to launch from here"; \
	else \
	  docker ps --format '{{.Names}}' | grep -qx $(FRANKA_CTR) \
	    || { echo "Container $(FRANKA_CTR) not running -- start it: docker compose -f ~/franka_ros2/docker-compose.yml up -d"; exit 1; }; \
	  docker exec -it $(FRANKA_CTR) bash -lc '$(FRANKA_SRC) && ros2 launch mile_franka_controllers mile_bringup.launch.py robot_ip:=$(ROBOT_IP) load_gripper:=$(LOAD_GRIPPER)'; \
	fi

apriltag-up:                 ## launch camera + apriltag_ros + calibration static tf (foreground); MILE_REAL_STACK=fr3|multipanda (default multipanda); MILE_CAMERA=realsense|webcam (default: $(MILE_CAMERA))
	$(DC) exec -e MILE_REAL_STACK=$(MILE_REAL_STACK) -e MILE_CAMERA=$${MILE_CAMERA:-$(MILE_CAMERA)} -e MILE_CAMERA_CALIB sim bash -lc '$(ENVSH) && self=$$$$; pgrep -f "apriltag_realsense.launch.py|apriltag_webcam.launch.py|apriltag_node|realsense2_camera_node|usb_cam_node_exe|static_transform_publisher.*camera_to_base" | grep -vx $$self | xargs -r kill 2>/dev/null; sleep 2; if [ "$$MILE_CAMERA" = "webcam" ]; then ros2 launch mile_franka/launch/apriltag_webcam.launch.py; else ros2 launch mile_franka/launch/apriltag_realsense.launch.py; fi'

eval-real:                   ## Part 4: evaluate policy on the real FR3; MILE_REAL_STACK=fr3|multipanda (default multipanda); MILE_CAMERA=realsense|webcam (default: $(MILE_CAMERA)); MILE_APPLY_SIM_GAINS=1 to track against the lab sim
	$(DC) exec -e DISPLAY=$$DISPLAY -e MILE_REAL_STACK=$(MILE_REAL_STACK) -e MILE_CAMERA=$${MILE_CAMERA:-$(MILE_CAMERA)} -e MILE_CONTROLLER -e MILE_GRASP_ACTION -e MILE_APPLY_SIM_GAINS sim bash -lc '$(ENVSH) && echo "real stack=$(MILE_REAL_STACK) RMW=$$RMW_IMPLEMENTATION"; $(KILLCLIENTS); python3 scripts/eval_base_policy_real.py'

eval-expert-real:            ## evaluate the scripted expert policy (not a learned policy) on the real FR3; same prereqs as eval-real; MILE_REAL_STACK=fr3|multipanda (default multipanda); MILE_CAMERA=realsense|webcam (default: $(MILE_CAMERA)); MILE_APPLY_SIM_GAINS=1 to track against the lab sim; EPISODES/MAX_STEPS override defaults
	$(DC) exec -e DISPLAY=$$DISPLAY -e MILE_REAL_STACK=$(MILE_REAL_STACK) -e MILE_CAMERA=$${MILE_CAMERA:-$(MILE_CAMERA)} -e MILE_CONTROLLER -e MILE_GRASP_ACTION -e MILE_APPLY_SIM_GAINS sim bash -lc '$(ENVSH) && echo "real stack=$(MILE_REAL_STACK) RMW=$$RMW_IMPLEMENTATION"; $(KILLCLIENTS); python3 scripts/eval_expert_real.py --episodes $${EPISODES:-5} --max_steps $${MAX_STEPS:-600}'

# ── Debug / development ────────────────────────────────────────────────────────
# Diagnostic and hardware-check tools. Not needed for the tutorial flow.

spacemouse-check:            ## print live SpaceMouse deflection (Ctrl-C to stop)
	$(call RUN,python3 scripts/spacemouse_check.py)

joystick-check:              ## print live gamepad axes/buttons (Ctrl-C to stop)
	$(call RUN,python3 scripts/joystick_check.py)

vive-check:                   ## print live Vive controller position/buttons (Ctrl-C to stop)
	$(call RUN,python3 scripts/vive_check.py)

pose-test:                   ## run pose-layer unit tests (no ROS/hardware needed)
	$(call RUN,python3 -m pytest tests/test_calibration.py tests/test_apriltag_pose.py tests/test_ros_posestamped.py -v)

tune-cost:                   ## tune MILE intervention cost/scale from collected data; override DATASET/POLICY/MENTAL_MODEL
	$(call RUN,python3 scripts/tune_intervention_cost.py \
	  --dataset $${DATASET:-output_dir/franka/accumulated_dataset_round0.pkl} \
	  --policy $${POLICY:-trained_models/franka/base_policy} \
	  --mental_model $${MENTAL_MODEL:-trained_models/franka/base_policy} \
	  --cost_grid $${COST_GRID:-110:170:5} \
	  --scale_grid $${SCALE_GRID:-125,150,175,200})

real-home-smoke:             ## operator-gated real FR3 home + small Cartesian square; STEP=0.025 by default; MILE_REAL_STACK=fr3|multipanda (default multipanda)
	$(DC) exec -e MILE_REAL_STACK=$(MILE_REAL_STACK) -e MILE_APPLY_SIM_GAINS -e MILE_CONTROLLER -e MILE_SUBSTEP_M sim bash -lc '$(ENVSH) && $(KILLCLIENTS); python3 scripts/franka_real_home_smoke.py --step $${STEP:-0.1}'

close-gripper:               ## close the gripper; override GRIP_FORCE=.. CLOSE_WIDTH=..
	$(DC) exec sim bash -lc '$(ENVSH) && $(GRIP_NS); ros2 action send_goal $$ns/grasp franka_msgs/action/Grasp "{width: $(CLOSE_WIDTH), speed: 0.05, force: $(GRIP_FORCE), epsilon: {inner: 0.08, outer: 0.08}}"'

open-gripper:                ## open the gripper; override OPEN_WIDTH=..
	$(DC) exec sim bash -lc '$(ENVSH) && $(GRIP_NS); ros2 action send_goal $$ns/grasp franka_msgs/action/Grasp "{width: $(OPEN_WIDTH), speed: 0.05, force: $(GRIP_FORCE), epsilon: {inner: 0.08, outer: 0.08}}"'

view-tags:                   ## MJPEG stream with AprilTag overlay → http://localhost:8080 (needs apriltag-up); MILE_REAL_STACK=fr3|multipanda (default multipanda); MILE_CAMERA=realsense|webcam (default: $(MILE_CAMERA))
	$(DC) exec -e MILE_REAL_STACK=$(MILE_REAL_STACK) -e MILE_CAMERA=$${MILE_CAMERA:-$(MILE_CAMERA)} sim bash -lc '$(ENVSH) && python3 -u scripts/view_camera_tags.py'

view-twin:                   ## live MuJoCo digital twin of the real workspace (needs controller + apriltag-up; safe in parallel with mile-real/eval-real); MILE_REAL_STACK=fr3|multipanda (default multipanda); MILE_CAMERA=realsense|webcam (default: $(MILE_CAMERA))
	@if [ "$$DISPLAY" != "$${DISPLAY#localhost:}" ] || [ "$$DISPLAY" != "$${DISPLAY#/}" ]; then \
		docker exec -u root mile_sim bash -c \
			'which x11vnc >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -q x11vnc; }'; \
		MILE_REAL_STACK=$(MILE_REAL_STACK) MILE_CAMERA=$${MILE_CAMERA:-$(MILE_CAMERA)} \
			python3 scripts/view_twin_macos.py; \
	else \
		echo "Host prereq (once per login): xhost +local:root"; \
		$(DC) exec -e DISPLAY=$$DISPLAY -e MILE_REAL_STACK=$(MILE_REAL_STACK) -e MILE_CAMERA=$${MILE_CAMERA:-$(MILE_CAMERA)} sim \
			bash -lc '$(ENVSH) && python3 scripts/view_cubes_mujoco.py'; \
	fi

calibrate-camera:            ## eye-to-hand camera calibration → MJPEG preview at http://localhost:8080 (needs controller + apriltag-up); MILE_REAL_STACK=fr3|multipanda (default multipanda); MILE_CAMERA=realsense|webcam (default: $(MILE_CAMERA)); board: CHECKER_SIZE/CHECKER_SQUARE/CHARUCO_MARKER/ARUCO_DICT (same vars as calibrate-intrinsics)
	$(DC) exec -e MILE_REAL_STACK=$(MILE_REAL_STACK) -e MILE_CAMERA=$${MILE_CAMERA:-$(MILE_CAMERA)} -e MILE_CAMERA_CALIB -e PYTHONUNBUFFERED=1 sim bash -lc '$(ENVSH) && python3 -u scripts/calibrate_camera.py \
	  --squares_x $(word 1,$(subst x, ,$(CHECKER_SIZE))) --squares_y $(word 2,$(subst x, ,$(CHECKER_SIZE))) \
	  --square_length $(CHECKER_SQUARE) --marker_length $(CHARUCO_MARKER) \
	  --dict_name DICT_$$(echo $(ARUCO_DICT) | tr a-z A-Z)'

CHECKER_PATTERN ?= charuco             ## 'chessboard', 'circles', 'acircles', or 'charuco' (default here — this lab's board)
CHECKER_SIZE    ?= 11x8                ## chessboard: *internal* corners, cols x rows (not squares). charuco: squares_x x squares_y (squares, not corners). Override to match your board
CHECKER_SQUARE  ?= 0.015               ## square edge length in metres; override to match your board
CHARUCO_MARKER  ?= 0.011               ## ArUco marker edge length in metres; only used when CHECKER_PATTERN=charuco
ARUCO_DICT      ?= 4x4_50              ## ArUco dictionary: aruco_orig | {4x4,5x5,6x6,7x7}_{50,100,250,1000}; only used when CHECKER_PATTERN=charuco

calibrate-intrinsics:        ## webcam-only: intrinsic calibration GUI (needs apriltag-up MILE_CAMERA=webcam running + host X11: xhost +local:root); COMMIT in the GUI writes straight to config/webcam_intrinsics.yaml
	@echo "Host prereq (once per login): xhost +local:root"
	@echo "In the GUI: sweep the board through the frame until all bars are green, click CALIBRATE, then SAVE, then COMMIT."
	@echo "COMMIT writes straight to $${MILE_WEBCAM_INTRINSICS:-config/webcam_intrinsics.yaml} via usb_cam's set_camera_info service — no manual copy needed."
	$(DC) exec -e DISPLAY=$$DISPLAY sim bash -lc '$(ENVSH) && export PYTHONPATH=/usr/lib/python3/dist-packages:$$PYTHONPATH && ros2 run camera_calibration cameracalibrator --pattern $(CHECKER_PATTERN) --size $(CHECKER_SIZE) --square $(CHECKER_SQUARE) --charuco_marker_size $(CHARUCO_MARKER) --aruco_dict $(ARUCO_DICT) --ros-args --remap image:=/camera/image_raw --remap camera:=/camera'

franka-shell:                ## shell in the controller container; MILE_REAL_STACK=multipanda (default) → $(MULTIPANDA_CTR), =fr3 → $(FRANKA_CTR); override FRANKA_WS/MULTIPANDA_WS if the workspace path differs
	@if [ "$(strip $(MILE_REAL_STACK))" = "multipanda" ]; then \
	  docker exec -it $(MULTIPANDA_CTR) bash -lc '$(MULTIPANDA_SRC) && exec bash'; \
	else \
	  docker exec -it $(FRANKA_CTR) bash -lc '$(FRANKA_SRC) && exec bash'; \
	fi

# ── Data collection / offline training ────────────────────────────────────────
# Used when building or replacing the bundled trained models; not needed for the tutorial.

collect-mediocre:            ## collect mediocre demos → sim_demos_mediocre.npz
	$(call RUN,python3 scripts/franka_sim_rollout_record.py \
	    --episodes 100 --mediocre true --require_success true --max_attempts 150 \
	    --max_steps 500 --video_start_hold 0.5 --video_end_hold 0.5 \
	    --out_dir output_dir/franka/rollouts_mediocre_$(TS) \
	    --data output_dir/franka/sim_demos_$(TS).npz && \
	  cp output_dir/franka/sim_demos_$(TS).npz output_dir/franka/sim_demos_mediocre.npz)

collect-expert:              ## collect perfect (successful-only) demos → sim_demos_expert.npz
	$(call RUN,python3 scripts/franka_sim_rollout_record.py \
	    --episodes 100 --mediocre false --require_success true \
	    --max_steps 500 --video_start_hold 0.5 --video_end_hold 0.5 \
	    --out_dir output_dir/franka/rollouts_expert_$(TS) \
	    --data output_dir/franka/sim_demos_$(TS).npz && \
	  cp output_dir/franka/sim_demos_$(TS).npz output_dir/franka/sim_demos_expert.npz)

base-policy:                 ## BC-train the base policy offline from mediocre demos; override input with DEMOS=path.npz
	$(call RUN,python3 scripts/build_base_policy.py \
	  --demos $(DEMOS) --bc_batch_size 256 --bc_ent_weight 0.0 --eval_episodes 0 \
	  --save_path trained_models/franka/base_policy)

mile:                        ## full iterative MILE run in sim (config/franka_sim.yaml)
	$(call RUN,cd scripts && python3 train_mile.py --config ../config/franka_sim.yaml)

mile-real:                   ## full iterative MILE run on the real FR3 (needs controller + apriltag-up); MILE_REAL_STACK=fr3|multipanda (default multipanda); MILE_CAMERA=realsense|webcam (default: $(MILE_CAMERA)); MILE_APPLY_SIM_GAINS=1 to track against the lab sim
	$(DC) exec -e MILE_REAL_STACK=$(MILE_REAL_STACK) -e MILE_CAMERA=$${MILE_CAMERA:-$(MILE_CAMERA)} -e MILE_APPLY_SIM_GAINS sim bash -lc '$(ENVSH) && $(KILLCLIENTS); cd scripts && python3 train_mile.py --config ../config/franka_real.yaml'

fetch-artifacts:             ## verify bundled trained models are present (no download needed)
	@echo "Trained models are included in the repository (trained_models/). No download needed."
	@ls -l trained_models/initial_policy trained_models/expert_policy trained_models/gt_mental_model trained_models/warm_started_mental_model trained_models/franka/base_policy
