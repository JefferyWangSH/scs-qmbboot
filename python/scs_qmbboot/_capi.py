'''
    ctypes mirror of include/scs.h (int indices, double floats) and loading of
    the solver libraries installed next to this file, one per linear system
    backend. Each library reports its struct sizes (python/abi.c); a library
    that does not match the mirror is refused.
'''

import ctypes as C
import os
import sys
from functools import cache

scs_int = C.c_int
scs_float = C.c_double
p_int = C.POINTER(scs_int)
p_float = C.POINTER(scs_float)


class ScsMatrix(C.Structure):
    _fields_ = [
        ('x', p_float),
        ('i', p_int),
        ('p', p_int),
        ('m', scs_int),
        ('n', scs_int),
    ]


class ScsSettings(C.Structure):
    _fields_ = [
        ('normalize', scs_int),
        ('scale', scs_float),
        ('adaptive_scale', scs_int),
        ('adaptive_diag_scale', scs_int),
        ('rho_x', scs_float),
        ('max_iters', scs_int),
        ('eps_abs', scs_float),
        ('eps_rel', scs_float),
        ('eps_infeas', scs_float),
        ('alpha', scs_float),
        ('time_limit_secs', scs_float),
        ('verbose', scs_int),
        ('warm_start', scs_int),
        ('acceleration_lookback', scs_int),
        ('acceleration_interval', scs_int),
        ('acceleration_type_1', scs_int),
        ('acceleration_regularization', scs_float),
        ('acceleration_relaxation', scs_float),
        ('write_data_filename', C.c_char_p),
        ('log_csv_filename', C.c_char_p),
    ]


class ScsData(C.Structure):
    _fields_ = [
        ('m', scs_int),
        ('n', scs_int),
        ('A', C.POINTER(ScsMatrix)),
        ('P', C.POINTER(ScsMatrix)),
        ('b', p_float),
        ('c', p_float),
    ]


class ScsCone(C.Structure):
    # without USE_SPECTRAL_CONES
    _fields_ = [
        ('z', scs_int),
        ('l', scs_int),
        ('bu', p_float),
        ('bl', p_float),
        ('bsize', scs_int),
        ('q', p_int),
        ('qsize', scs_int),
        ('s', p_int),
        ('ssize', scs_int),
        ('cs', p_int),
        ('cssize', scs_int),
        ('ep', scs_int),
        ('ed', scs_int),
        ('p', p_float),
        ('psize', scs_int),
    ]


class ScsSolution(C.Structure):
    _fields_ = [
        ('x', p_float),
        ('y', p_float),
        ('s', p_float),
    ]


class AaStats(C.Structure):
    _fields_ = [
        ('iter', scs_int),
        ('n_accept', scs_int),
        ('n_reject_lapack', scs_int),
        ('n_reject_rank0', scs_int),
        ('n_reject_nonfinite', scs_int),
        ('n_reject_weight_cap', scs_int),
        ('n_safeguard_reject', scs_int),
        ('last_rank', scs_int),
        ('last_aa_norm', scs_float),
        ('last_regularization', scs_float),
    ]


class ScsInfo(C.Structure):
    # without SPECTRAL_TIMING_FLAG
    _fields_ = [
        ('iter', scs_int),
        ('status', C.c_char * 128),
        ('lin_sys_solver', C.c_char * 128),
        ('status_val', scs_int),
        ('scale_updates', scs_int),
        ('pobj', scs_float),
        ('dobj', scs_float),
        ('res_pri', scs_float),
        ('res_dual', scs_float),
        ('gap', scs_float),
        ('res_infeas', scs_float),
        ('res_unbdd_a', scs_float),
        ('res_unbdd_p', scs_float),
        ('setup_time', scs_float),
        ('solve_time', scs_float),
        ('scale', scs_float),
        ('comp_slack', scs_float),
        ('rejected_accel_steps', scs_int),
        ('accepted_accel_steps', scs_int),
        ('aa_stats', AaStats),
        ('lin_sys_time', scs_float),
        ('cone_time', scs_float),
        ('accel_time', scs_float),
    ]


# sizes in the order of scs_qmbboot_abi (python/abi.c)
ABI = (scs_int, scs_float, ScsMatrix, ScsSettings, ScsData, ScsCone, ScsSolution, ScsInfo)

# linear_solver name -> library (CMake target)
LIBRARIES = {
    'mkl_gram': 'scsmklgram',
    'accelerate': 'scsaccel',
    'qdldl': 'scsdir',
}


def library_path(linear_solver: str) -> str:
    suffix = '.dylib' if sys.platform == 'darwin' else '.so'
    return os.path.join(os.path.dirname(__file__), f'lib{LIBRARIES[linear_solver]}{suffix}')


@cache
def load(linear_solver: str) -> C.CDLL:
    if linear_solver not in LIBRARIES:
        raise ValueError(f'unknown linear_solver {linear_solver!r}, expected one of {list(LIBRARIES)}')
    path = library_path(linear_solver)
    if not os.path.exists(path):
        raise ImportError(f'linear_solver {linear_solver!r} is not built in this installation ({path} missing)')
    lib = C.CDLL(path)

    lib.scs_qmbboot_abi.argtypes = [C.POINTER(C.c_size_t)]
    lib.scs_qmbboot_abi.restype = C.c_int
    sizes = (C.c_size_t * len(ABI))()
    count = lib.scs_qmbboot_abi(sizes)
    expected = [C.sizeof(t) for t in ABI]
    if count != len(ABI) or list(sizes) != expected:
        raise ImportError(
            f'{path} does not match the ctypes mirror of scs.h in {__file__}: '
            f'sizes {list(sizes)[:count]}, expected {expected}'
        )

    lib.scs_set_default_settings.argtypes = [C.POINTER(ScsSettings)]
    lib.scs_set_default_settings.restype = None
    lib.scs_init.argtypes = [C.POINTER(ScsData), C.POINTER(ScsCone), C.POINTER(ScsSettings)]
    lib.scs_init.restype = C.c_void_p
    lib.scs_solve.argtypes = [C.c_void_p, C.POINTER(ScsSolution), C.POINTER(ScsInfo), scs_int]
    lib.scs_solve.restype = scs_int
    lib.scs_finish.argtypes = [C.c_void_p]
    lib.scs_finish.restype = None
    lib.scs_version.argtypes = []
    lib.scs_version.restype = C.c_char_p
    return lib
