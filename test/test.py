# Bare * is used to force the caller to use named arguments - so you cannot define a function with * as an argument when you have no following keyword arguments.
from __future__ import annotations

from typing import List, Callable


class T1:
    def __init__(
            self,
            choices: List[str] | None = None,
            *,
            value: List[str] | str | Callable | None = None,
            type: str = "value",
            label: str | None = None,
            info: str | None = None,
            every: float | None = None,
            show_label: bool = True,
            interactive: bool | None = None,
            visible: bool = True,
            elem_id: str | None = None,
            elem_classes: List[str] | str | None = None,
            **kwargs,
    ):
        print(choices)


args = {
    "choices": [1]
}
print(T1(**args))
print(T1(choices=[1]))
