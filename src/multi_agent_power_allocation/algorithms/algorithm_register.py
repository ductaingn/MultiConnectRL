import warnings
from enum import Enum
from typing import Tuple

from multi_agent_power_allocation.algorithms.high_level import (
    DQN,
    RAQL,
    SACRA,
    Algorithm,
    Random,
    SACRAVa,
)


class Algorithms(Enum):
    """
    High level algorithms
    """

    SACRA = SACRA
    SACRA_VA = SACRAVa
    DQN = DQN
    RAQL = RAQL
    RANDOM = Random


# Names of the algorithms in configs (`env_config.algorithm_list`) and checkpoints, as in the
# paper: name -> (algorithm, whether the baseline uses the full power budget)
ALGORITHM_NAMES = {
    "SACRA": (Algorithms.SACRA, False),
    "SACRA-Va": (Algorithms.SACRA_VA, False),
    "RAQL": (Algorithms.RAQL, False),
    "RAQL-FP": (Algorithms.RAQL, True),
    "DQN": (Algorithms.DQN, False),
    "DQN-FP": (Algorithms.DQN, True),
    "Random": (Algorithms.RANDOM, False),
}
# Former names, still accepted
LEGACY_ALGORITHM_NAMES = {"SACPA": "SACRA", "SACPF": "SACRA-Va", "SACRAVa": "SACRA-Va"}


def parse_algorithm_name(name: str) -> Tuple[Algorithms, bool]:
    """`(algorithm, full_power_budget)` of an algorithm name of a config or a checkpoint."""
    if name in LEGACY_ALGORITHM_NAMES:
        new_name = LEGACY_ALGORITHM_NAMES[name]
        warnings.warn(
            f"Algorithm name `{name}` is deprecated, use `{new_name}`",
            FutureWarning,
            stacklevel=2,
        )
        name = new_name
    try:
        return ALGORITHM_NAMES[name]
    except KeyError:
        raise ValueError(
            f"Unknown algorithm `{name}`, valid names: {list(ALGORITHM_NAMES)}"
        ) from None


def algorithm_name(policy: Algorithm) -> str:
    """Name of the algorithm of `policy`, as in the paper (e.g. `SACRA-Va`, `RAQL-FP`)."""
    full_power = bool(getattr(policy, "full_power_budget", False))
    for name, (algorithm, fp) in ALGORITHM_NAMES.items():
        if type(policy) is algorithm.value and fp == full_power:
            return name
    raise ValueError(f"No registered name for {type(policy).__name__}")
