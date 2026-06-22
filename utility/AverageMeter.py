import math
from collections.abc import Iterable

class AverageMeter(object):
    def __init__(self, items=None):
        self.items = items
        self.n_items = 1 if items is None else len(items)
        self.reset()

    def reset(self):
        self._val = [0.0] * self.n_items
        self._count = [0] * self.n_items

        # Welford state: mean and sum of squared deviations (M2)
        self._mean = [0.0] * self.n_items
        self._M2 = [0.0] * self.n_items

        # Store all raw values
        # - single metric: list[float]
        # - multi metric : list[list[float]]
        self.values_raw = [] if self.items is None else [[] for _ in range(self.n_items)]

    def _update_one(self, idx, v):
        v = float(v)
        self._val[idx] = v
        self._count[idx] += 1

        # store raw
        if self.items is None:
            self.values_raw.append(v)
        else:
            self.values_raw[idx].append(v)

        # Welford update
        n = self._count[idx]
        delta = v - self._mean[idx]
        self._mean[idx] += delta / n
        delta2 = v - self._mean[idx]
        self._M2[idx] += delta * delta2

    def _is_vector_like(self, x):
        if isinstance(x, (str, bytes, dict)):
            return False
        return isinstance(x, Iterable) and not isinstance(x, (int, float, bool))

    def update(self, values):
        """
        Update meter.
        - If items is None (single metric):
            * scalar -> one update
            * iterable -> treat as batch samples for the SAME metric
        - If items is not None (multi-metric):
            * iterable must match n_items -> update each metric by index
        """
        # numpy scalar / torch 0-d tensor
        try:
            if hasattr(values, "shape") and values.shape == ():
                self._update_one(0, values)
                return
        except Exception:
            pass

        if self._is_vector_like(values):
            # torch tensor -> list
            if hasattr(values, "detach"):
                values = values.detach().cpu().flatten().tolist()
            # numpy array -> list
            elif hasattr(values, "ravel"):
                values = values.ravel().tolist()

            if self.n_items == 1:
                for v in values:
                    self._update_one(0, v)
            else:
                if len(values) != self.n_items:
                    raise ValueError(
                        f"AverageMeter expected {self.n_items} values (items={self.items}), "
                        f"but got {len(values)}."
                    )
                for idx, v in enumerate(values):
                    self._update_one(idx, v)
        else:
            self._update_one(0, values)

    def val(self, idx=None):
        if idx is None:
            return self._val[0] if self.items is None else [self._val[i] for i in range(self.n_items)]
        return self._val[idx]

    def count(self, idx=None):
        if idx is None:
            return self._count[0] if self.items is None else [self._count[i] for i in range(self.n_items)]
        return self._count[idx]

    def avg(self, idx=None):
        if idx is None:
            return self._mean[0] if self.items is None else [self._mean[i] for i in range(self.n_items)]
        return self._mean[idx]

    def var(self, idx=None, ddof=1):
        def _var(i):
            n = self._count[i]
            denom = n - ddof
            if denom <= 0:
                return float("nan")
            return self._M2[i] / denom

        if idx is None:
            return _var(0) if self.items is None else [_var(i) for i in range(self.n_items)]
        return _var(idx)

    def std(self, idx=None, ddof=1):
        def _std(i):
            v = self.var(i, ddof=ddof)
            return math.sqrt(v) if v == v else float("nan")

        if idx is None:
            return _std(0) if self.items is None else [_std(i) for i in range(self.n_items)]
        return _std(idx)
