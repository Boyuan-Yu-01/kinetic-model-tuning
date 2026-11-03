"""Serialize mechanism YAML with round-trip scientific notation."""

import math

import yaml


class _MechanismDumper(yaml.SafeDumper):
    """Keep YAML scalar types while formatting finite floats."""


def _represent_float(dumper, value):
    if not math.isfinite(value):
        return dumper.represent_float(value)
    return dumper.represent_scalar('tag:yaml.org,2002:float', format(value, '.16e'))


_MechanismDumper.add_representer(float, _represent_float)


def dump_mechanism(data):
    """Return mechanism YAML with scientific notation for floating-point values."""
    return yaml.dump(data, Dumper=_MechanismDumper, sort_keys=False)
