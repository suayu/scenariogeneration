from run_simulation import merge_episode_info


def test_merge_episode_info_preserves_required_fields_after_empty_nonterminal_step():
    previous = {
        "collision": False,
        "off_route": False,
        "completed": False,
        "progress": 0.0,
    }

    merged = merge_episode_info(previous, {})

    assert merged == previous


def test_merge_episode_info_applies_terminal_values_without_losing_audit_fields():
    previous = {
        "collision": False,
        "off_route": False,
        "completed": False,
        "progress": 0.0,
        "generation_failure": "profile_reachability_invalid",
    }

    merged = merge_episode_info(
        previous,
        {"collision": True, "progress": 0.4, "obstacle_collision_ids": [3]},
    )

    assert merged["collision"] is True
    assert merged["off_route"] is False
    assert merged["completed"] is False
    assert merged["progress"] == 0.4
    assert merged["generation_failure"] == "profile_reachability_invalid"
    assert merged["obstacle_collision_ids"] == [3]
