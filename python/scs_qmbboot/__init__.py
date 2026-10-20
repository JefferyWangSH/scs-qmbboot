'''
    Python interface to scs-qmbboot, a fork of the splitting conic solver SCS
    for quantum many-body bootstrap SDPs. Same calling convention as
    scs-python (import scs_qmbboot as scs):

        solver = scs.SCS(data, cone, linear_solver=None, **settings)
        sol = solver.solve(warm_start=True, x=None, y=None, s=None)
        sol['x'], sol['y'], sol['s'], sol['info']

    data: {'A': sparse (m, n), 'b': (m,), 'c': (n,)}; P (QPs) is not supported.
    cone: {'z', 'l', 'bu', 'bl', 'q', 's', 'cs', 'ep', 'ed', 'p'}, see scs.h.
    settings: the fields of ScsSettings except warm_start, defaults from the C
    library (scs_set_default_settings).
    linear_solver: 'mkl_gram' (Linux default), 'accelerate' (macOS default) or
    'qdldl'; see available_solvers().
'''

import os
import sys
import ctypes as C
import weakref
from importlib import metadata

import numpy as np
import scipy.sparse as sp

from . import _capi

try:
    __version__ = metadata.version('scs-qmbboot')
except metadata.PackageNotFoundError:
    __version__ = None

# SCS return integers correspond to one of these flags:
# (copied from include/scs.h)
INFEASIBLE_INACCURATE = -7  # SCS best guess infeasible
UNBOUNDED_INACCURATE = -6  # SCS best guess unbounded
SIGINT = -5  # interrupted by sig int
FAILED = -4  # SCS failed
INDETERMINATE = -3  # indeterminate (norm too small)
INFEASIBLE = -2  # primal infeasible, dual unbounded
UNBOUNDED = -1  # primal unbounded, dual infeasible
UNFINISHED = 0  # never returned, used as placeholder
SOLVED = 1  # problem solved to desired accuracy
SOLVED_INACCURATE = 2  # SCS best guess solved

DEFAULT_SOLVER = 'accelerate' if sys.platform == 'darwin' else 'mkl_gram'

SETTINGS = tuple(name for name, _ in _capi.ScsSettings._fields_ if name != 'warm_start')
_CONE_INTS = ('z', 'l', 'ep', 'ed')
_CONE_INT_ARRAYS = (('q', 'qsize'), ('s', 'ssize'), ('cs', 'cssize'))
_INT_MAX = np.iinfo(np.int32).max


def available_solvers() -> list[str]:
    return [name for name in _capi.LIBRARIES if os.path.exists(_capi.library_path(name))]


def _floats(v, size, what):
    a = np.ascontiguousarray(v, dtype=np.float64).reshape(-1)
    if size is not None and a.size != size:
        raise ValueError(f'{what} has length {a.size}, expected {size}')
    return a


def _ints(v, what):
    a = np.asarray(v).reshape(-1)
    if a.size and (not np.issubdtype(a.dtype, np.integer) or a.min() < 0 or a.max() > _INT_MAX):
        raise ValueError(f'{what} must be nonnegative integers')
    return np.ascontiguousarray(a, dtype=np.int32)


def _ptr(a, ctype):
    return a.ctypes.data_as(C.POINTER(ctype)) if a.size else None


def _info(info: _capi.ScsInfo) -> dict:
    out = {}
    for name, _ in info._fields_:
        v = getattr(info, name)
        if name == 'aa_stats':
            v = {f: getattr(v, f) for f, _ in v._fields_}
        elif isinstance(v, bytes):
            v = v.decode()
        out[name] = v
    return out


class SCS:
    def __init__(self, data: dict, cone: dict, linear_solver: str | None = None, **settings):
        self.linear_solver = linear_solver or DEFAULT_SOLVER
        lib = _capi.load(self.linear_solver)

        unknown = set(data) - {'A', 'b', 'c', 'P'}
        if unknown:
            raise ValueError(f'unknown data keys {sorted(unknown)}')
        if data.get('P') is not None:
            raise NotImplementedError('P (quadratic objective) is not supported')
        A = sp.csc_matrix(data['A'], dtype=np.float64)
        if not A.has_canonical_format:
            A = A.copy()
            A.sum_duplicates()
        m, n = A.shape
        if max(m, n, A.nnz) > _INT_MAX:
            raise ValueError('A is too large for 32-bit indices')
        Ax = np.ascontiguousarray(A.data, dtype=np.float64)
        Ai = np.ascontiguousarray(A.indices, dtype=np.int32)
        Ap = np.ascontiguousarray(A.indptr, dtype=np.int32)
        b = _floats(data['b'], m, 'b')
        c = _floats(data['c'], n, 'c')
        mat = _capi.ScsMatrix(_ptr(Ax, C.c_double), _ptr(Ai, C.c_int), _ptr(Ap, C.c_int), m, n)
        d = _capi.ScsData(m, n, C.pointer(mat), None, _ptr(b, C.c_double), _ptr(c, C.c_double))

        unknown = set(cone) - {'z', 'l', 'bu', 'bl', 'q', 's', 'cs', 'ep', 'ed', 'p'}
        if unknown:
            raise ValueError(f'unknown cone keys {sorted(unknown)}')
        k = _capi.ScsCone()
        keep = []  # arrays referenced by k until scs_init has copied them
        for name in _CONE_INTS:
            k_val = int(cone.get(name, 0))
            if k_val < 0:
                raise ValueError(f'cone {name!r} must be nonnegative')
            setattr(k, name, k_val)
        for name, size in _CONE_INT_ARRAYS:
            a = _ints(cone.get(name, []), f'cone {name!r}')
            keep.append(a)
            setattr(k, name, _ptr(a, C.c_int))
            setattr(k, size, a.size)
        p = _floats(cone.get('p', []), None, "cone 'p'")
        keep.append(p)
        k.p, k.psize = _ptr(p, C.c_double), p.size
        if 'bu' in cone or 'bl' in cone:
            bu = _floats(cone['bu'], None, "cone 'bu'")
            bl = _floats(cone['bl'], bu.size, "cone 'bl'")
            keep += [bu, bl]
            k.bu, k.bl, k.bsize = _ptr(bu, C.c_double), _ptr(bl, C.c_double), bu.size + 1

        stgs = _capi.ScsSettings()
        lib.scs_set_default_settings(C.byref(stgs))
        for name, value in settings.items():
            if name not in SETTINGS:
                raise TypeError(f'unknown setting {name!r}')
            if name.endswith('_filename'):
                value = None if value is None else os.fsencode(value)
            setattr(stgs, name, value)

        work = lib.scs_init(C.byref(d), C.byref(k), C.byref(stgs))
        if not work:
            raise ValueError('scs_init failed: invalid data, cone or settings (verbose=True prints the reason)')
        self._lib, self._work = lib, work
        self._finalizer = weakref.finalize(self, lib.scs_finish, work)
        self.m, self.n = m, n
        self._last = None

    def solve(self, warm_start: bool = True, x=None, y=None, s=None) -> dict:
        '''
            warm_start=True (default, as scs-python) starts from x, y, s; those
            not given are taken from the previous solve (zeros before the first
            one, which is the same as a cold start).
        '''
        if not self._finalizer.alive:
            raise RuntimeError('solver is closed')
        last = self._last or {}
        out = {}
        for name, v, size in (('x', x, self.n), ('y', y, self.m), ('s', s, self.m)):
            if warm_start and v is None:
                v = last.get(name)
            out[name] = np.zeros(size) if not warm_start or v is None else _floats(v, size, name).copy()
        sol = _capi.ScsSolution(*(_ptr(out[name], C.c_double) for name in ('x', 'y', 's')))
        info = _capi.ScsInfo()
        self._lib.scs_solve(self._work, C.byref(sol), C.byref(info), int(bool(warm_start)))
        out['info'] = _info(info)
        self._last = {name: out[name].copy() for name in ('x', 'y', 's')}
        return out

    def close(self):
        self._finalizer()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def solve(data: dict, cone: dict, linear_solver: str | None = None, **settings) -> dict:
    with SCS(data, cone, linear_solver, **settings) as solver:
        return solver.solve()
