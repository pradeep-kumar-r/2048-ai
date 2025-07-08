import numpy as np
from typing import List


def deduplicate_array_sequence(sequence: List[np.ndarray]) -> List[np.ndarray]:
    seen = []
    result = []
    for item in sequence:
        item_tuple = tuple(map(tuple, item)) if hasattr(item, 'shape') else item
        if item_tuple not in seen:
            seen.append(item_tuple)
            result.append(item)
    return result