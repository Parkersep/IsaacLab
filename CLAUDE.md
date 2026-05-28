# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is Isaac Lab?

Isaac Lab is a GPU-accelerated robotics simulation framework built on NVIDIA Isaac Sim. It supports reinforcement learning, imitation learning, and motion planning with sim-to-real transfer. Currently on the `develop` branch targeting Isaac Sim 6.0.

## Architecture

### Package structure (`source/`)

| Package | Purpose |
|---|---|
| `isaaclab` | Core framework: environments, assets, managers, sensors, sim abstractions |
| `isaaclab_physx` | PhysX physics backend plugin (assets, sensors, scene data, renderers, cloner) |
| `isaaclab_newton` | Newton physics backend plugin (mirrors `isaaclab_physx` structure exactly) |
| `isaaclab_ov` | Omniverse-specific tiled camera renderers |
| `isaaclab_tasks` | Pre-built environments (30+ envs, both manager-based and direct styles) |
| `isaaclab_assets` | Robot/sensor config instances for specific hardware (not base classes) |
| `isaaclab_rl` | Thin wrappers for RL libraries (rsl_rl, rl_games, skrl, sb3, ray) |
| `isaaclab_mimic` | Imitation learning extension (Apache 2.0 licensed) |
| `isaaclab_teleop` | Teleoperation interfaces |
| `isaaclab_contrib` | Community contributions |
| `isaaclab_experimental` / `isaaclab_tasks_experimental` | Experimental features |
| `isaaclab_visualizers` | Visualization tools |

Each package has its own `pyproject.toml`, `config/extension.toml`, and `docs/CHANGELOG.rst`.

### Core framework (`source/isaaclab/isaaclab/`)

**Two environment paradigms:**
- **Manager-based** (`ManagerBasedEnv`, `ManagerBasedRLEnv`): Declarative composition via managers (Action, Observation, Reward, Termination, Event, Curriculum, Command, Recorder). Preferred for most tasks.
- **Direct** (`DirectRLEnv`, `DirectMARLEnv`): User subclasses implement step/reward logic directly. Lighter weight, supports multi-agent.

**Key subsystems:**
- **Assets** (`assets/`): `AssetBase` → `RigidObject`, `Articulation`, `RigidObjectCollection`. Config classes (`*Cfg`) are separate from runtime classes.
- **Scene** (`scene/`): `InteractiveScene` owns all assets, sensors, and terrain for an environment.
- **Sim** (`sim/`): `SimulationContext` wraps the physics engine. Includes `spawners/` (USD/URDF/MJCF), `converters/`, `schemas/`.
- **Sensors** (`sensors/`): `SensorBase` → camera, contact_sensor, frame_transformer, IMU, ray_caster.
- **Managers** (`managers/`): Orchestrate env behavior — observations, actions, rewards, terminations, events, curriculum, commands.

**Physics backend pattern:** `isaaclab_physx` and `isaaclab_newton` are drop-in plugins that override engine-specific parts (assets, sensors, scene_data_providers, renderers, cloner) while all manager/env/scene abstractions stay in core `isaaclab`.

### Scripts

- `scripts/reinforcement_learning/` — per-framework train scripts (rsl_rl, rl_games, skrl, sb3, ray)
- `scripts/tools/` — asset pipeline utilities (URDF/MJCF converters, data tools)
- `scripts/environments/` — standalone env demos and benchmarks

## Development guidelines

@AGENTS.md
