import fnmatch
import inspect
import itertools
import logging
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Set, Tuple, Type, Union

import torch
import torch.nn as nn
from torch import Tensor


class SAM2Optimizer:
    """Wraps a torch optimizer with per-param-group LR/WD schedulers.

    SAM2 uses a "where"-based scheduler that tracks fractional training progress
    in [0, 1] rather than step-count-based schedulers.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, schedulers=None) -> None:
        self.optimizer = optimizer
        self.schedulers = schedulers
        self._validate_optimizer_schedulers()
        self.step_schedulers(0.0, 0)

    def _validate_optimizer_schedulers(self):
        if self.schedulers is None:
            return
        for _, set_of_schedulers in enumerate(self.schedulers):
            for option, _ in set_of_schedulers.items():
                assert option in self.optimizer.defaults, (
                    f"Optimizer option {option} not found in {self.optimizer}. "
                    f"Valid options are {self.optimizer.defaults.keys()}"
                )

    def step_schedulers(self, where: float, step: int) -> None:
        """Update param group options based on training progress.

        Args:
            where: Fractional training progress in [0, 1].
            step: Current global step count.
        """
        if self.schedulers is None:
            return
        for i, param_group in enumerate(self.optimizer.param_groups):
            for option, scheduler in self.schedulers[i].items():
                if "step" in inspect.signature(scheduler.__call__).parameters:
                    new_value = scheduler(step=step, where=where)
                elif (
                    hasattr(scheduler, "scheduler")
                    and "step" in inspect.signature(scheduler.scheduler.__call__).parameters
                ):
                    new_value = scheduler(step=step, where=where)
                else:
                    new_value = scheduler(where)
                param_group[option] = new_value

    def step(self, where: float, step: int, closure=None):
        self.step_schedulers(where, step)
        return self.optimizer.step(closure)

    def zero_grad(self, *args, **kwargs):
        return self.optimizer.zero_grad(*args, **kwargs)


class GradientClipper:
    """Gradient clipping utility for DDP models."""

    def __init__(self, max_norm: float = 1.0, norm_type: int = 2):
        assert isinstance(max_norm, (int, float)) or max_norm is None
        self.max_norm = float(max_norm) if max_norm is not None else None
        self.norm_type = norm_type

    def __call__(self, model: nn.Module):
        if self.max_norm is None:
            return
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.max_norm, norm_type=self.norm_type)


class ValueScaler:
    """Wraps a scheduler to scale its output by a constant multiplier."""

    def __init__(self, scheduler, mult_val: float):
        self.scheduler = scheduler
        self.mult_val = mult_val

    def __call__(self, *args, **kwargs):
        return self.scheduler(*args, **kwargs) * self.mult_val


def get_module_cls_to_param_names(
    model: nn.Module,
    param_allowlist: Optional[Set[str]] = None,
) -> Dict[Type, Set[str]]:
    """Map module classes to the names of parameters they directly own."""
    module_cls_to_params: Dict[Type, Set[str]] = {}
    for module_name, module in model.named_modules():
        module_cls = type(module)
        module_cls_to_params.setdefault(module_cls, set())
        for param_name, _ in module.named_parameters(recurse=False):
            full_name = f"{module_name}.{param_name}" if module_name else param_name
            if param_allowlist is None or full_name in param_allowlist:
                module_cls_to_params[module_cls].add(full_name)
    return module_cls_to_params


def set_default_parameters(scheduler_cfgs: list, all_parameter_names: Set[str]) -> None:
    """Assign unmatched parameters to the default (parameter_names=None) scheduler."""
    def _get_pn(cfg):
        return cfg["parameter_names"] if isinstance(cfg, dict) else cfg.parameter_names

    def _set_pn(cfg, val):
        if isinstance(cfg, dict):
            cfg["parameter_names"] = val
        else:
            cfg.parameter_names = val

    constraints = [_get_pn(cfg) for cfg in scheduler_cfgs if _get_pn(cfg) is not None]
    default_params = all_parameter_names - set.union(*constraints) if constraints else set(all_parameter_names)
    default_count = 0
    for cfg in scheduler_cfgs:
        if _get_pn(cfg) is None:
            _set_pn(cfg, default_params)
            default_count += 1
    assert default_count <= 1, "Only one scheduler per option can be default"
    if default_count == 0:
        scheduler_cfgs.append({"parameter_names": default_params})


def name_constraints_to_parameters(
    param_constraints: List[Set[str]],
    named_parameters: Dict[str, Tensor],
) -> List[torch.nn.Parameter]:
    """Return parameters matching the intersection of all constraint sets."""
    matching_names = set.intersection(*param_constraints)
    return [value for name, value in named_parameters.items() if name in matching_names]


def map_scheduler_cfgs_to_param_groups(
    all_scheduler_cfgs: Iterable[List[Dict]],
    named_parameters: Dict[str, Tensor],
) -> Tuple[List[Dict], List[Dict[str, List[torch.nn.Parameter]]]]:
    """Produce parameter groups from scheduler configs."""
    scheduler_cfgs_per_param_group = itertools.product(*all_scheduler_cfgs)
    schedulers = []
    param_groups = []
    for scheduler_cfgs in scheduler_cfgs_per_param_group:
        param_constraints = [cfg["parameter_names"] for cfg in scheduler_cfgs]
        matching_parameters = name_constraints_to_parameters(param_constraints, named_parameters)
        if len(matching_parameters) == 0:
            continue
        schedulers_for_group = {
            cfg["option"]: cfg["scheduler"] for cfg in scheduler_cfgs if "option" in cfg
        }
        schedulers.append(schedulers_for_group)
        param_groups.append({"params": matching_parameters})
    return schedulers, param_groups


def validate_param_group_params(param_groups: List[Dict], model: nn.Module):
    """Verify param groups are non-overlapping and cover all model parameters."""
    for pg in param_groups:
        assert len(pg["params"]) == len(set(pg["params"]))
    parameters = [set(pg["params"]) for pg in param_groups]
    model_parameters = {p for _, p in model.named_parameters()}
    for p1, p2 in itertools.permutations(parameters, 2):
        assert p1.isdisjoint(p2), "Param groups should be disjoint"
    assert set.union(*parameters) == model_parameters, (
        f"Param groups must include all model parameters. "
        f"Found {len(set.union(*parameters))} vs model has {len(model_parameters)}"
    )


def construct_optimizer(
    model: nn.Module,
    optimizer_conf: Any,
    options_conf: Optional[Mapping[str, List]] = None,
    param_group_modifiers_conf: Optional[List[Callable]] = None,
    param_allowlist: Optional[Set[str]] = None,
    validate_param_groups: bool = True,
) -> SAM2Optimizer:
    """Construct optimizer with per-param-group schedulers.

    This is the SAM2-style optimizer builder that supports complex per-parameter
    LR and weight decay scheduling via "where"-based progress tracking.

    Args:
        model: Model to optimize.
        optimizer_conf: Partial optimizer config (needs params to be filled).
        options_conf: Per-option scheduler configurations.
        param_group_modifiers_conf: Optional functions to modify scheduler configs.
        param_allowlist: Subset of parameters to optimize.
        validate_param_groups: Whether to validate group completeness.
    """
    if param_allowlist is None:
        param_allowlist = {name for name, _ in model.named_parameters()}

    named_parameters = {
        name: param for name, param in model.named_parameters() if name in param_allowlist
    }

    from hydra.utils import instantiate as hydra_instantiate

    if not options_conf:
        opt_fn = hydra_instantiate(optimizer_conf)
        optimizer = opt_fn(list(named_parameters.values())) if callable(opt_fn) and not isinstance(opt_fn, torch.optim.Optimizer) else opt_fn
        return SAM2Optimizer(optimizer)

    all_parameter_names = {name for name, _ in model.named_parameters() if name in param_allowlist}
    module_cls_to_all_param_names = get_module_cls_to_param_names(model, param_allowlist)

    scheduler_cfgs_per_option = _instantiate_options(options_conf)
    all_scheduler_cfgs = []
    for option, scheduler_cfgs in scheduler_cfgs_per_option.items():
        for config in scheduler_cfgs:
            config["option"] = option
            config["parameter_names"] = _unix_pattern_to_parameter_names(
                config, all_parameter_names, module_cls_to_all_param_names
            )
        set_default_parameters(scheduler_cfgs, all_parameter_names)
        all_scheduler_cfgs.append(scheduler_cfgs)

    if param_group_modifiers_conf:
        for modifier_cfg in param_group_modifiers_conf:
            modifier = hydra_instantiate(modifier_cfg)
            all_scheduler_cfgs = modifier(scheduler_cfgs=all_scheduler_cfgs, model=model)

    schedulers, param_groups = map_scheduler_cfgs_to_param_groups(all_scheduler_cfgs, named_parameters)
    if validate_param_groups:
        validate_param_group_params(param_groups, model)
    opt_fn = hydra_instantiate(optimizer_conf)
    optimizer = opt_fn(param_groups) if callable(opt_fn) and not isinstance(opt_fn, torch.optim.Optimizer) else opt_fn
    return SAM2Optimizer(optimizer, schedulers)


def _unix_pattern_to_parameter_names(
    scheduler_cfg,
    parameter_names: Set[str],
    module_cls_to_param_names: Dict[Type, Set[str]],
) -> Optional[Set[str]]:
    """Filter parameter names by unix patterns and/or module class names."""
    if isinstance(scheduler_cfg, dict):
        param_names_filter = scheduler_cfg.get("param_names", None)
        module_cls_filter = scheduler_cfg.get("module_cls_names", None)
    else:
        param_names_filter = getattr(scheduler_cfg, "param_names", None)
        module_cls_filter = getattr(scheduler_cfg, "module_cls_names", None)

    if param_names_filter is None and module_cls_filter is None:
        return None

    matched = set()
    if param_names_filter is not None:
        for pattern in param_names_filter:
            matches = set(fnmatch.filter(parameter_names, pattern))
            assert len(matches) >= 1, f"Pattern {pattern} doesn't match any parameter"
            logging.info(f"Matches for param_name [{pattern}]: {matches}")
            matched.update(matches)

    if module_cls_filter is not None:
        for cls_name in module_cls_filter:
            cls = _get_class(cls_name)
            if cls not in module_cls_to_param_names:
                raise AssertionError(f"module_cls_name {cls_name} doesn't match any module class")
            matching = module_cls_to_param_names[cls]
            assert len(matching) > 0, f"module_cls_name {cls_name} has no parameters"
            logging.info(f"Matches for module_cls_name [{cls_name}]: {matching}")
            matched.update(matching)

    return matched


def _get_class(class_name: str) -> Type:
    """Import a class by fully-qualified name."""
    parts = class_name.rsplit(".", 1)
    if len(parts) == 2:
        import importlib
        module = importlib.import_module(parts[0])
        return getattr(module, parts[1])
    raise ValueError(f"Cannot resolve class: {class_name}")


def _instantiate_optimizer(optimizer_conf, params) -> torch.optim.Optimizer:
    """Instantiate optimizer from config.

    Supports ConfigNode with _target_ (and _partial_) or a direct callable.
    """
    if hasattr(optimizer_conf, "to_dict"):
        d = optimizer_conf.to_dict()
        d.pop("_partial_", None)
        if "_target_" in d:
            target = d.pop("_target_")
            cls = target if isinstance(target, type) or callable(target) else _get_class(target)
            return cls(params, **d)
    if hasattr(optimizer_conf, "instantiate"):
        return optimizer_conf.instantiate(params)
    if callable(optimizer_conf):
        return optimizer_conf(params)
    raise ValueError(f"Cannot instantiate optimizer from {type(optimizer_conf)}")


def _instantiate_options(options_conf):
    """Instantiate scheduler options config.

    The options config is a dict of {option_name: list_of_scheduler_configs}.
    Each scheduler config has a 'scheduler' key with a _target_ that needs
    instantiation, plus optional 'param_names' and 'module_cls_names'.
    """
    from hydra.utils import instantiate as hydra_instantiate
    from omegaconf import OmegaConf

    if hasattr(options_conf, "items"):
        raw = dict(options_conf)
    elif hasattr(options_conf, "to_dict"):
        raw = options_conf.to_dict()
    else:
        raw = dict(options_conf)

    result = {}
    for option_name, scheduler_list in raw.items():
        instantiated_list = []
        for sched_cfg in scheduler_list:
            if hasattr(sched_cfg, "items"):
                sched_cfg_dict = dict(sched_cfg)
            else:
                sched_cfg_dict = sched_cfg

            entry = {}
            if "scheduler" in sched_cfg_dict:
                entry["scheduler"] = hydra_instantiate(sched_cfg_dict["scheduler"])
            entry["param_names"] = sched_cfg_dict.get("param_names", None)
            if entry["param_names"] is not None:
                entry["param_names"] = list(entry["param_names"])
            entry["module_cls_names"] = sched_cfg_dict.get("module_cls_names", None)
            if entry["module_cls_names"] is not None:
                entry["module_cls_names"] = list(entry["module_cls_names"])
            entry["parameter_names"] = None
            instantiated_list.append(entry)
        result[option_name] = instantiated_list
    return result


def _instantiate(conf):
    """Instantiate a config object, handling _partial_ for deferred construction."""
    if hasattr(conf, "to_dict"):
        d = conf.to_dict()
        is_partial = d.pop("_partial_", False)
        if "_target_" in d:
            target = d.pop("_target_")
            cls = target if isinstance(target, type) or callable(target) else _get_class(target)
            if is_partial:
                from functools import partial

                return partial(cls, **d)
            return cls(**d)
    if hasattr(conf, "instantiate"):
        return conf.instantiate()
    if callable(conf):
        return conf
    return conf


def layer_decay_param_modifier(
    scheduler_cfgs: List[List[Dict]],
    model: nn.Module,
    layer_decay_value: float,
    layer_decay_min: Optional[float] = None,
    apply_to: Optional[str] = None,
    overrides: List[Dict] = (),
) -> List[List[Dict]]:
    """Apply layer-wise learning rate decay (LARS-style).

    Args:
        scheduler_cfgs: Scheduler configs to modify.
        model: Model (must implement get_layer_id / get_num_layers on the target component).
        layer_decay_value: Decay factor per layer.
        layer_decay_min: Minimum decay value.
        apply_to: Dotted attribute path to the model component.
        overrides: Manual LR overrides for specific param patterns.
    """
    target = model
    if apply_to:
        for attr in apply_to.split("."):
            target = getattr(target, attr)

    num_layers = target.get_num_layers() + 1
    layer_decays = [layer_decay_value ** (num_layers - i) for i in range(num_layers + 1)]
    if layer_decay_min is not None:
        layer_decays = [max(val, layer_decay_min) for val in layer_decays]

    final_scheduler_cfgs = []
    for scheduler_cfg_group in scheduler_cfgs:
        curr_cfg_group = []
        for scheduler_cfg in scheduler_cfg_group:
            if scheduler_cfg["option"] != "lr":
                curr_cfg_group.append(scheduler_cfg)
                continue
            parameter_names = sorted(scheduler_cfg["parameter_names"])
            layer_cfg_groups = {}
            for param_name in parameter_names:
                layer_id = num_layers
                this_scale = layer_decays[layer_id]
                if apply_to and param_name.startswith(apply_to):
                    layer_id = target.get_layer_id(param_name)
                    this_scale = layer_decays[layer_id]
                    for override in overrides:
                        if fnmatch.fnmatchcase(param_name, override["pattern"]):
                            this_scale = float(override["value"])
                            layer_id = override["pattern"]
                            break

                if layer_id not in layer_cfg_groups:
                    layer_cfg_groups[layer_id] = {
                        "option": scheduler_cfg["option"],
                        "scheduler": ValueScaler(scheduler_cfg["scheduler"], this_scale),
                        "parameter_names": {param_name},
                    }
                else:
                    layer_cfg_groups[layer_id]["parameter_names"].add(param_name)

            for layer_cfg in layer_cfg_groups.values():
                curr_cfg_group.append(layer_cfg)
        final_scheduler_cfgs.append(curr_cfg_group)
    return final_scheduler_cfgs
