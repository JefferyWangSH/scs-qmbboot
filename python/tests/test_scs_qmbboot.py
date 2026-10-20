import numpy as np
import pytest
import scipy.sparse as sp

import scs_qmbboot as scs

SOLVERS = scs.available_solvers()
EPS = dict(eps_abs=1e-9, eps_rel=1e-9, verbose=False)


def svec(X):
    # lower triangle by column, off-diagonal entries scaled by sqrt(2)
    d = X.shape[0]
    return np.concatenate([[X[j, j]] + [np.sqrt(2) * X[i, j] for i in range(j + 1, d)] for j in range(d)])


def cvec(H):
    # as svec, off-diagonal entries as (sqrt(2) Re, sqrt(2) Im)
    d = H.shape[0]
    out = []
    for j in range(d):
        out.append(H[j, j].real)
        for i in range(j + 1, d):
            out += [np.sqrt(2) * H[i, j].real, np.sqrt(2) * H[i, j].imag]
    return np.array(out)


def lambda_max_problem(M, vec, cone_key):
    # min t  s.t.  t I - M in PSD:  s = b - A t  with  b = -vec(M), A = -vec(I)
    d = M.shape[0]
    A = sp.csc_matrix(-vec(np.eye(d)).reshape(-1, 1))
    return dict(A=A, b=-vec(M), c=np.array([1.0])), {cone_key: [d]}


def random_problem(seed, n=20, n_lp=15, d=8):
    # feasible and bounded with known optimum c'x0: complementary (s0, y0)
    rng = np.random.default_rng(seed)
    m = n_lp + d * (d + 1) // 2
    A = sp.random(m, n, density=0.4, random_state=rng, format='csc') + sp.eye(m, n, format='csc')
    s_lp, y_lp = rng.random(n_lp), rng.random(n_lp)
    s_lp[: n_lp // 2] = 0
    y_lp[n_lp // 2 :] = 0
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    lam = np.concatenate([np.zeros(d // 2), rng.random(d - d // 2) + 0.1])
    mu = np.concatenate([rng.random(d // 2) + 0.1, np.zeros(d - d // 2)])
    s0 = np.concatenate([s_lp, svec(Q @ np.diag(lam) @ Q.T)])
    y0 = np.concatenate([y_lp, svec(Q @ np.diag(mu) @ Q.T)])
    x0 = rng.standard_normal(n)
    data = dict(A=A, b=A @ x0 + s0, c=-A.T @ y0)
    return data, dict(l=n_lp, s=[d]), data['c'] @ x0


def test_abi_and_version():
    assert SOLVERS, 'no solver library installed'
    assert scs.DEFAULT_SOLVER in SOLVERS
    for solver in SOLVERS:
        # load() checks the struct sizes against the library (python/abi.c)
        c_version = scs._capi.load(solver).scs_version().decode()
        assert scs.__version__.startswith(c_version + '+qmbboot.')


@pytest.mark.parametrize('solver', SOLVERS)
def test_lp(solver):
    # min -x1 - x2  s.t.  x1 + 2 x2 <= 4,  3 x1 + x2 <= 6,  x >= 0
    A = sp.csc_matrix([[1.0, 2.0], [3.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    data = dict(A=A, b=np.array([4.0, 6.0, 0.0, 0.0]), c=np.array([-1.0, -1.0]))
    sol = scs.solve(data, dict(l=4), linear_solver=solver, **EPS)
    assert sol['info']['status_val'] == scs.SOLVED
    np.testing.assert_allclose(sol['x'], [1.6, 1.2], atol=1e-6)
    assert sol['info']['pobj'] == pytest.approx(-2.8, abs=1e-6)


@pytest.mark.parametrize('solver', SOLVERS)
def test_real_psd(solver):
    rng = np.random.default_rng(0)
    M = rng.standard_normal((5, 5))
    M = M + M.T
    data, cone = lambda_max_problem(M, svec, 's')
    sol = scs.solve(data, cone, linear_solver=solver, **EPS)
    assert sol['info']['pobj'] == pytest.approx(np.linalg.eigvalsh(M)[-1], abs=1e-6)


@pytest.mark.parametrize('solver', SOLVERS)
def test_complex_psd(solver):
    rng = np.random.default_rng(1)
    M = rng.standard_normal((5, 5)) + 1j * rng.standard_normal((5, 5))
    M = M + M.conj().T
    data, cone = lambda_max_problem(M, cvec, 'cs')
    sol = scs.solve(data, cone, linear_solver=solver, **EPS)
    assert sol['info']['pobj'] == pytest.approx(np.linalg.eigvalsh(M)[-1], abs=1e-6)


@pytest.mark.parametrize('solver', SOLVERS)
def test_random_problem_and_warm_start(solver):
    data, cone, opt = random_problem(2)
    with scs.SCS(data, cone, linear_solver=solver, eps_abs=1e-7, eps_rel=1e-7, verbose=False) as solver_:
        cold = solver_.solve(warm_start=False)
        assert cold['info']['status'] == 'solved'
        assert cold['info']['pobj'] == pytest.approx(opt, rel=1e-5)
        warm = solver_.solve(warm_start=True)
        assert warm['info']['status'] == 'solved'
        assert warm['info']['iter'] < cold['info']['iter']
        again = solver_.solve(warm_start=True, x=cold['x'], y=cold['y'], s=cold['s'])
        assert again['info']['iter'] < cold['info']['iter']


def test_solvers_agree():
    data, cone, _ = random_problem(3)
    sols = {s: scs.solve(data, cone, linear_solver=s, eps_abs=1e-7, eps_rel=1e-7, verbose=False) for s in SOLVERS}
    ref = sols[SOLVERS[-1]]
    for sol in sols.values():
        assert sol['info']['pobj'] == pytest.approx(ref['info']['pobj'], rel=1e-6)


@pytest.mark.parametrize('solver', SOLVERS)
def test_infeasible(solver):
    # x >= 1 and x <= 0
    data = dict(A=sp.csc_matrix([[-1.0], [1.0]]), b=np.array([-1.0, 0.0]), c=np.array([1.0]))
    sol = scs.solve(data, dict(l=2), linear_solver=solver, verbose=False)
    assert sol['info']['status_val'] in (scs.INFEASIBLE, scs.INFEASIBLE_INACCURATE)


def test_errors():
    data = dict(A=sp.csc_matrix([[1.0]]), b=np.array([1.0]), c=np.array([1.0]))
    with pytest.raises(TypeError, match='unknown setting'):
        scs.SCS(data, dict(l=1), warm_start=True)
    with pytest.raises(NotImplementedError):
        scs.SCS(dict(data, P=sp.csc_matrix([[1.0]])), dict(l=1))
    with pytest.raises(ValueError, match='unknown linear_solver'):
        scs.SCS(data, dict(l=1), linear_solver='mkl')
    with pytest.raises(ValueError, match='length'):
        scs.SCS(dict(data, b=np.ones(2)), dict(l=1))
    with pytest.raises(ValueError, match='scs_init failed'):
        scs.SCS(data, dict(l=2), verbose=False)
    missing = [s for s in ('mkl_gram', 'accelerate') if s not in SOLVERS]
    for s in missing:
        with pytest.raises(ImportError, match='not built'):
            scs.SCS(data, dict(l=1), linear_solver=s)
