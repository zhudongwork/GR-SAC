"""Read legacy continuation artifacts without importing later experiment features."""
import copy


def normalize_legacy_config(values):
    values = copy.deepcopy(values)
    # These fields were recorded by the continuation runner but disabled in this SAC recipe.
    for key, disabled in (("residual_decision_horizon", 1), ("subtask_feature_dim", 0),
                          ("retain_failed_bootstrap", False)):
        if key in values and values.pop(key) != disabled:
            raise ValueError(f"{key} is outside the released SAC protocol")
    return values


def normalize_legacy_sac_config(values):
    values = copy.deepcopy(values)
    for key, disabled in (("target_critic_subset_size", None), ("n_steps", 1),
                          ("normalize_offset_features", False), ("subtask_feature_dim", 0)):
        if key in values and values.pop(key) != disabled:
            raise ValueError(f"{key} is outside the released SAC objective")
    return values
