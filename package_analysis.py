from __future__ import annotations
import argparse
import importlib
import inspect

white_list: list[str] = [
    "kornia.augmentation",
    "kornia.augmentation.auto",
    "kornia.augmentation.container",
    "kornia.color",
    "kornia.contrib",
    "kornia.enhance",
    "kornia.feature",
    "kornia.filters",
    "kornia.geometry",
    "kornia.geometry.calibration",
    "kornia.geometry.camera",
    "kornia.geometry.epipolar",
    "kornia.geometry.liegroup",
    "kornia.geometry.subpix",
    "kornia.geometry.transform",
    "kornia.io",
    "kornia.losses",
    "kornia.metrics",
    "kornia.morphology",
    "kornia.nerf",
    "kornia.tracking",
    "kornia.utils",
    "kornia.x",
]


def get_public_operators(module) -> list[str]:
    """List all public operators in a module.
    
    Args:
        module (module): The module to list operators from.
    
    Returns:
        list[str]: A list of public operators.
    """
    operators: list[str] = []
    public_names: list[str] = getattr(module, '__all__', [])

    if not public_names:
        public_names = [name for name, _ in inspect.getmembers(module)]

    name: str
    for name in public_names:
        if not name.startswith('_'):
            obj = getattr(module, name, None)
            if obj and (inspect.isfunction(obj) or inspect.isclass(obj)):
                operators.append(name)
    return operators


def list_package_operators() -> dict[str, list[str]]:
    """List all public operators in a package."""
    operators_per_module: dict[str, list[str]] = {}

    for module_name in white_list:
        try:
            module = importlib.import_module(module_name)
            operators: list[str] = get_public_operators(module)
            operators_per_module[module_name] = operators
            module_hierarchy: str = module_name.replace('.', ' -> ')
        except ImportError as e:
            print(f'Failed to import {module_name}: {e}')

    return operators_per_module


if __name__ == '__main__':

    # get all public operators in a package
    operators_per_module: dict[str, list[str]] = \
        list_package_operators()
    
    # count the number of operators
    num_operators: int = 0
    
    # print the result
    for module_name, operators in operators_per_module.items():
        module_hierarchy: str = module_name.replace('.', ' -> ')
        print(f'{module_hierarchy}')
        for operator in operators:
            print(f'    {operator}')
            num_operators += 1
    
    print(f'## Number of operators: {num_operators}')