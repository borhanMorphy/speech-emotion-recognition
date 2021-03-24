from typing import List
import torch

def random_split(ds: torch.utils.data.Dataset, parts: List[float]) -> List[torch.utils.data.Dataset]:
    total_length = len(ds)

    assert len(parts) > 0, "parts must contain at least 1 float value"
    assert sum(parts) == 1, "sum of parts must be equal to 1 but found {}".format(sum(parts))

    lengths = list(map(lambda l: int(l*total_length), parts))

    left_over = total_length - sum(lengths)
    lengths[0] += left_over

    return torch.utils.data.random_split(ds, lengths)
    