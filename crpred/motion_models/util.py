from typing import List


def _difference(numbers: List[float]) -> List[float]:
    """
    calculate the difference between to two adjacent floats in a list
    """
    diff: List[float] = []
    for a, b in zip(numbers[0::], numbers[1::]):
        diff.append(b - a)

    return diff
