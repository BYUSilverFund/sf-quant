"""Utils file for shared code between mvo_dynamic and mvo workers."""

import sf_quant.optimizer as sfo


# Registry of available constraints
CONSTRAINT_REGISTRY: dict[str, type] = {
    "ZeroBeta": sfo.constraints.ZeroBeta,
    "ZeroInvestment": sfo.constraints.ZeroInvestment,
    "UnitBeta": sfo.constraints.UnitBeta,
    "FullInvestment": sfo.constraints.FullInvestment,
    "LongOnly": sfo.constraints.LongOnly,
    "NoBuyingOnMargin": sfo.constraints.NoBuyingOnMargin
}


def get_constraints(constraint_names: list[str]) -> list:
    """Convert constraint names to constraint objects.

    Args:
        constraint_names: List of constraint class names.

    Returns:
        List of instantiated constraint objects.

    Raises:
        KeyError: If a constraint name is not in the registry.
    """
    constraints = []
    for name in constraint_names:
        if name not in CONSTRAINT_REGISTRY:
            available = ", ".join(CONSTRAINT_REGISTRY.keys())
            raise KeyError(f"Unknown constraint '{name}'. Available: {available}")
        constraints.append(CONSTRAINT_REGISTRY[name]())
    return constraints