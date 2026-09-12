"""Read the non-tensor part of a ``torch.save`` checkpoint without torch.

Every ``.pt`` in this project is a zip archive whose ``data.pkl`` pickles a
plain dict.  Tensors are stubbed out, so this returns the ``metrics``,
``config``, ``tag`` and hyperparameter fields only.  Use it to build manifests
and to audit tables on a machine without torch; use ``torch.load`` to get the
embeddings themselves.
"""
from __future__ import annotations

import io
import pickle
import zipfile
from pathlib import Path
from typing import Any, Dict


class _Stub:
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, s):
        pass

    def __repr__(self):
        return "<tensor>"


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("torch") or module.startswith("geoopt"):
            return _Stub
        return super().find_class(module, name)

    def persistent_load(self, pid):
        return None


def _scalarise(obj: Any) -> Any:
    if isinstance(obj, _Stub):
        return "<tensor>"
    if isinstance(obj, dict):
        return {k: _scalarise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        if len(obj) > 20:
            return f"<list len={len(obj)}>"
        return [_scalarise(v) for v in obj]
    return obj


def peek(path: Path) -> Dict[str, Any]:
    """Return the checkpoint dict with tensors replaced by ``'<tensor>'``."""
    with zipfile.ZipFile(path) as zf:
        pk = [n for n in zf.namelist() if n.endswith("data.pkl")][0]
        obj = _Unpickler(io.BytesIO(zf.read(pk))).load()
    return _scalarise(obj)


# ----------------------------------------------------------------------------
# Tensor recovery without torch (float32/float64 CPU storages only)
# ----------------------------------------------------------------------------
_DTYPES = {"FloatStorage": "float32", "DoubleStorage": "float64", "HalfStorage": "float16",
           "LongStorage": "int64", "IntStorage": "int32", "BoolStorage": "bool"}


class _Tensor:
    def __init__(self, storage, offset, shape, stride):
        self.storage, self.offset, self.shape, self.stride = storage, offset, tuple(shape), tuple(stride)

    def array(self):
        import numpy as np
        key, dtype = self.storage
        arr = np.frombuffer(self._bytes[key], dtype=dtype)
        n = int(np.prod(self.shape)) if self.shape else 1
        return arr[self.offset:self.offset + n].reshape(self.shape).copy()


def _rebuild(storage, offset, shape, stride, *rest):
    return _Tensor(storage, offset, shape, stride)


class _TensorUnpickler(pickle.Unpickler):
    def __init__(self, f, storages):
        super().__init__(f)
        self._storages = storages

    def find_class(self, module, name):
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return _rebuild
        if module == "torch" and name in _DTYPES:
            return _DTYPES[name]
        if module.startswith("torch") or module.startswith("geoopt"):
            return _Stub
        if (module, name) == ("collections", "OrderedDict"):
            return dict
        return super().find_class(module, name)

    def persistent_load(self, pid):
        # ('storage', dtype, key, location, numel)
        return (pid[2], pid[1])


def load_arrays(path: Path) -> Dict[str, Any]:
    """Load a checkpoint into numpy arrays (top-level tensors and one level of nested dicts)."""
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        pk = [n for n in names if n.endswith("data.pkl")][0]
        prefix = pk[: -len("data.pkl")]
        raw = {n.split("/")[-1]: zf.read(n) for n in names if n.startswith(prefix + "data/")}
        obj = _TensorUnpickler(io.BytesIO(zf.read(pk)), raw).load()

    def conv(o):
        if isinstance(o, _Tensor):
            o._bytes = raw
            return o.array()
        if isinstance(o, dict):
            return {k: conv(v) for k, v in o.items()}
        if isinstance(o, list):
            return [conv(v) for v in o]
        return o
    return conv(obj)
