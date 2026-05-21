from __future__ import annotations

"""Episode-level routing for demo and intervention replay buffers."""

from collections.abc import Iterable
from typing import Literal


DemoInsertMode = Literal["none", "append", "replace_longest_if_better"]


def _strip_episode_metadata(transition: dict) -> dict:
    """Remove rollout-only bookkeeping keys before inserting into replay."""
    return {key: value for key, value in transition.items() if not key.startswith("_")}


def episode_success(episode: list[dict]) -> bool:
    """Read episode success from the final transition metadata."""
    if not episode:
        return False
    return bool(episode[-1].get("_success", False))


def intervention_indices(episode: Iterable[dict]) -> list[int]:
    """Return timesteps where the expert controller executed the action."""
    return [idx for idx, transition in enumerate(episode) if int(transition["interventions"]) != 0]


def add_episode_to_buffer(
    episode: list[dict],
    buffer,
    *,
    mode: DemoInsertMode = "append",
) -> dict[str, int | str | None]:
    """Insert a complete episode into a replay buffer with an episode-level policy."""
    clean_episode = [_strip_episode_metadata(transition) for transition in episode]
    if mode == "none":
        return {
            "status": "skipped_none",
            "added": 0,
            "removed": 0,
            "skipped": 1,
            "old_length": None,
            "new_length": len(clean_episode),
        }
    if hasattr(buffer, "insert_episode"):
        return buffer.insert_episode(clean_episode, mode=mode)
    if mode != "append":
        raise TypeError("Non-ReplayBuffer targets only support append mode.")
    for transition in clean_episode:
        buffer.add_transition(transition)
    return {
        "status": "appended",
        "added": len(clean_episode),
        "removed": 0,
        "skipped": 0,
        "old_length": None,
        "new_length": len(clean_episode),
    }


def route_episode_to_buffers(
    episode: list[dict],
    *,
    demo_buffer,
    intervention_buffer,
    include_failed_interventions: bool,
    demo_insert_mode: DemoInsertMode = "append",
) -> dict[str, int]:
    """Route a finished episode into demo and intervention buffers.

    Autonomous success episodes populate `demo_buffer` only if the learner
    completed the episode without expert actions. `demo_insert_mode` can keep
    appending, or replace the current longest demo episode when a shorter
    success episode arrives. Intervention episodes populate
    `intervention_buffer` from the first intervention onward. Failed expert
    suffixes are optional so we can ablate whether to imitate failed corrections
    later.
    """
    if not episode:
        return {
            "demo_added": 0,
            "demo_removed": 0,
            "demo_skipped": 0,
            "intervention_added": 0,
            "failed_intervention_seen": 0,
        }

    success = episode_success(episode)
    intervention_idxs = intervention_indices(episode)

    demo_added = 0
    demo_removed = 0
    demo_skipped = 0
    intervention_added = 0
    failed_intervention_seen = 0

    if success and not intervention_idxs:
        demo_result = add_episode_to_buffer(episode, demo_buffer, mode=demo_insert_mode)
        demo_added = int(demo_result["added"])
        demo_removed = int(demo_result["removed"])
        demo_skipped = int(demo_result["skipped"])

    if intervention_idxs:
        first_intervention_idx = intervention_idxs[0]
        if success or include_failed_interventions:
            for transition in episode[first_intervention_idx:]:
                intervention_buffer.add_transition(_strip_episode_metadata(transition))
                intervention_added += 1
        else:
            failed_intervention_seen = len(episode) - first_intervention_idx

    return {
        "demo_added": demo_added,
        "demo_removed": demo_removed,
        "demo_skipped": demo_skipped,
        "intervention_added": intervention_added,
        "failed_intervention_seen": failed_intervention_seen,
    }
