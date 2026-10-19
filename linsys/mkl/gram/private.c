/*
 * Sparse direct linear system solver using the Gram matrix reduction, as the
 * dense direct solver (linsys/cpu/dense) but sparse, with MKL Pardiso:
 *
 *   [R_x + P   A'] [x]   [rx]
 *   [  A     -R_y] [y] = [ry]
 *
 * is reduced to:
 *
 *   G x = rx + A' R_y^{-1} ry,    where G = R_x + P + A' R_y^{-1} A
 *   y = R_y^{-1} (A x - ry)
 *
 * G is n x n sparse symmetric positive definite. Pardiso orders it itself
 * (parallel nested dissection), so factorizations and solves run in parallel;
 * on the quasi-definite KKT matrix (linsys/mkl/direct) nested dissection finds
 * no small separators, and a user fill-in ordering makes the solves serial.
 *
 * A row of A with k nonzeros puts a dense k x k block into G: this solver is
 * meant for problems whose rows of A are all short (e.g. the PSD cones of
 * bootstrap SDPs), not for problems with dense rows.
 */

#include "private.h"

#include <stdlib.h>

#define PARDISO_SYMBOLIC (11)
#define PARDISO_NUMERIC (22)
#define PARDISO_SOLVE (33)
#define PARDISO_CLEANUP (-1)

#ifdef DLONG
#define _PARDISO pardiso_64
#else
#define _PARDISO pardiso
#endif

/* Prototype for Pardiso, see linsys/mkl/direct/private.c. */
void _PARDISO(void **pt, const scs_int *maxfct, const scs_int *mnum,
              const scs_int *mtype, const scs_int *phase, const scs_int *n,
              const scs_float *a, const scs_int *ia, const scs_int *ja,
              scs_int *perm, const scs_int *nrhs, scs_int *iparm,
              const scs_int *msglvl, scs_float *b, scs_float *x,
              scs_int *error);
#if defined(BLAS64) && !defined(DLONG)
#error "MKL PARDISO requires DLONG when BLAS64 is set (pardiso_64 needs 64-bit ints)"
#endif

const char *scs_get_lin_sys_method(void) {
  return "sparse-direct-mkl-pardiso-gram";
}

/* Transpose of sparse CSC matrix A (m x n), i.e. A by rows */
static ScsMatrix *transpose(const ScsMatrix *A) {
  scs_int i, j, q, *z;
  ScsMatrix *C = SCS(cs_spalloc)(A->n, A->m, MAX(A->p[A->n], 1), 1, 0);
  z = (scs_int *)scs_calloc(A->m, sizeof(scs_int));
  if (!C || !z) {
    return SCS(cs_done)(C, z, SCS_NULL, 0);
  }
  for (i = 0; i < A->p[A->n]; ++i) {
    z[A->i[i]]++; /* row counts */
  }
  SCS(cumsum)(C->p, z, A->m); /* row pointers */
  for (j = 0; j < A->n; ++j) {
    for (i = A->p[j]; i < A->p[j + 1]; ++i) {
      q = z[A->i[i]]++;
      C->i[q] = j; /* place A(i,j) as entry C(j,i) */
      C->x[q] = A->x[i];
    }
  }
  return SCS(cs_done)(C, z, SCS_NULL, 1);
}

static void add_col(scs_int k, scs_int j, scs_int *mark, scs_int *cols,
                    scs_int *nz) {
  if (mark[k] != j) {
    mark[k] = j;
    if (cols) {
      cols[*nz] = k;
    }
    (*nz)++;
  }
}

/* Columns of row j of the upper triangle of G: j and every k > j that shares
 * an entry of P or a row of A with j, stored in cols if not NULL. Returns
 * their number. */
static scs_int gram_row(const ScsLinSysWork *p, scs_int j, scs_int *mark,
                        scs_int *cols) {
  const ScsMatrix *A = p->A, *At = p->At, *Pt = p->Pt;
  scs_int q, r, nz = 0;
  add_col(j, j, mark, cols, &nz);
  for (q = Pt ? Pt->p[j] : 0; Pt && q < Pt->p[j + 1]; ++q) {
    add_col(Pt->i[q], j, mark, cols, &nz);
  }
  for (q = A->p[j]; q < A->p[j + 1]; ++q) {
    for (r = At->p[A->i[q]]; r < At->p[A->i[q] + 1]; ++r) {
      if (At->i[r] > j) {
        add_col(At->i[r], j, mark, cols, &nz);
      }
    }
  }
  return nz;
}

static int cmp_int(const void *a, const void *b) {
  scs_int x = *(const scs_int *)a, y = *(const scs_int *)b;
  return (x > y) - (x < y);
}

/* Sparsity pattern of the upper triangle of G by rows, with sorted columns */
static ScsMatrix *gram_pattern(const ScsLinSysWork *p) {
  scs_int j, nz = 0;
  ScsMatrix *G = SCS_NULL;
  scs_int *mark = (scs_int *)scs_malloc(p->n * sizeof(scs_int));
  scs_int *cnt = (scs_int *)scs_calloc(p->n, sizeof(scs_int));
  if (mark && cnt) {
    for (j = 0; j < p->n; ++j) {
      mark[j] = -1;
    }
    for (j = 0; j < p->n; ++j) {
      cnt[j] = gram_row(p, j, mark, SCS_NULL);
      nz += cnt[j];
    }
    G = SCS(cs_spalloc)(p->n, p->n, nz, 1, 0);
  }
  if (G) {
    SCS(cumsum)(G->p, cnt, p->n); /* row pointers */
    for (j = 0; j < p->n; ++j) {
      mark[j] = -1;
    }
    for (j = 0; j < p->n; ++j) {
      gram_row(p, j, mark, &(G->i[G->p[j]]));
      qsort(&(G->i[G->p[j]]), G->p[j + 1] - G->p[j], sizeof(scs_int), cmp_int);
    }
  }
  return SCS(cs_done)(G, mark, cnt, G != SCS_NULL);
}

/* Form Gram matrix G = R_x + P + A' diag(r_y_inv) A on its pattern, each row
 * accumulated in a dense workspace of length n (one per thread). */
static scs_int form_gram(ScsLinSysWork *p, const scs_float *diag_r) {
  const ScsMatrix *A = p->A, *At = p->At, *Pt = p->Pt;
  ScsMatrix *G = p->G;
  scs_int i, failed = 0;

  for (i = 0; i < p->m; ++i) {
    p->r_y_inv[i] = 1.0 / diag_r[p->n + i];
  }
#ifdef _OPENMP
#pragma omp parallel reduction(| : failed)
#endif
  {
    scs_int j, q, r;
    scs_float a, *w = (scs_float *)scs_calloc(p->n, sizeof(scs_float));
    failed = !w;
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 64)
#endif
    for (j = 0; j < p->n; ++j) {
      if (!w) {
        continue;
      }
      w[j] = diag_r[j]; /* R_x */
      for (q = Pt ? Pt->p[j] : 0; Pt && q < Pt->p[j + 1]; ++q) {
        w[Pt->i[q]] += Pt->x[q]; /* P */
      }
      for (q = A->p[j]; q < A->p[j + 1]; ++q) { /* A' R_y^{-1} A */
        a = A->x[q] * p->r_y_inv[A->i[q]];
        for (r = At->p[A->i[q]]; r < At->p[A->i[q] + 1]; ++r) {
          if (At->i[r] >= j) {
            w[At->i[r]] += a * At->x[r];
          }
        }
      }
      for (q = G->p[j]; q < G->p[j + 1]; ++q) {
        G->x[q] = w[G->i[q]];
        w[G->i[q]] = 0.;
      }
    }
    scs_free(w);
  }
  return failed;
}

ScsLinSysWork *scs_init_lin_sys_work(const ScsMatrix *A, const ScsMatrix *P,
                                     const scs_float *diag_r) {
  ScsLinSysWork *p = (ScsLinSysWork *)scs_calloc(1, sizeof(ScsLinSysWork));
  if (!p) {
    return SCS_NULL;
  }

  p->n = A->n;
  p->m = A->m;
  p->A = A;

  p->At = transpose(A);
  p->Pt = P ? transpose(P) : SCS_NULL;
  p->r_y_inv = (scs_float *)scs_calloc(A->m, sizeof(scs_float));
  p->tmp_m = (scs_float *)scs_calloc(A->m, sizeof(scs_float));
  p->tmp_n = (scs_float *)scs_calloc(A->n, sizeof(scs_float));

  if (!p->At || (P && !p->Pt) || !p->r_y_inv || !p->tmp_m || !p->tmp_n) {
    scs_free_lin_sys_work(p);
    return SCS_NULL;
  }

  /* Form Gram matrix */
  p->G = gram_pattern(p);
  if (!p->G || form_gram(p, diag_r)) {
    scs_printf("Error in forming Gram matrix");
    scs_free_lin_sys_work(p);
    return SCS_NULL;
  }

  /* Set Pardiso variables */
  p->mtype = 2;          /* Real symmetric positive definite matrix */
  p->nrhs = 1;           /* Number of right hand sides */
  p->maxfct = 1;         /* Maximum number of numerical factorizations */
  p->mnum = 1;           /* Which factorization to use */
  p->error = 0;          /* Initialize error flag */
  p->msglvl = VERBOSITY; /* Printing information */

  /* For all iparm vars see MKL documentation */
  p->iparm[0] = 1;  /* Pardiso must inspect iparm */
  p->iparm[1] = 3;  /* Fill-in reordering from OpenMP */
  p->iparm[5] = 1;  /* Write solution into b */
  p->iparm[23] = 1; /* Two-level scheduling for parallel factorization */
  p->iparm[24] = 1; /* Parallel forward/backward solve */
  p->iparm[34] = 1; /* Use C-style indexing for indices */

#ifdef SFLOAT
  p->iparm[27] = 1; /* 1 is single precision, 0 is double */
#endif

  /* Permutation and symbolic factorization */
  p->phase = PARDISO_SYMBOLIC;
  _PARDISO(p->pt, &(p->maxfct), &(p->mnum), &(p->mtype), &(p->phase), &(p->n),
           p->G->x, p->G->p, p->G->i, SCS_NULL, &(p->nrhs), p->iparm,
           &(p->msglvl), SCS_NULL, SCS_NULL, &(p->error));

  if (p->error != 0) {
    scs_printf("Error during symbolic factorization: %d", (int)p->error);
    scs_free_lin_sys_work(p);
    return SCS_NULL;
  }

  /* Numerical factorization */
  p->phase = PARDISO_NUMERIC;
  _PARDISO(p->pt, &(p->maxfct), &(p->mnum), &(p->mtype), &(p->phase), &(p->n),
           p->G->x, p->G->p, p->G->i, SCS_NULL, &(p->nrhs), p->iparm,
           &(p->msglvl), SCS_NULL, SCS_NULL, &(p->error));

  if (p->error != 0) {
    scs_printf("Error during numerical factorization: %d", (int)p->error);
    scs_free_lin_sys_work(p);
    return SCS_NULL;
  }

  return p;
}

/*
 * Solve:
 *   x = G^{-1} (rx + A' R_y^{-1} ry)
 *   y = R_y^{-1} (A x - ry)
 *
 * Input b = [rx; ry], solution stored in b = [x; y].
 */
scs_int scs_solve_lin_sys(ScsLinSysWork *p, scs_float *b, const scs_float *s,
                          scs_float tol) {
  scs_int i;
  scs_float *rx = b;
  scs_float *ry = b + p->n;

  /* tmp_m = R_y^{-1} ry, ry = -ry */
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (i = 0; i < p->m; ++i) {
    p->tmp_m[i] = p->r_y_inv[i] * ry[i];
    ry[i] = -ry[i];
  }

  /* rx += A' * tmp_m   (rx = rx + A' R_y^{-1} ry) */
  SCS(accum_by_atrans)(p->A, p->tmp_m, rx);

  /* Solve G x = rx, result in rx (= b[:n]) */
  p->phase = PARDISO_SOLVE;
  _PARDISO(p->pt, &(p->maxfct), &(p->mnum), &(p->mtype), &(p->phase), &(p->n),
           p->G->x, p->G->p, p->G->i, SCS_NULL, &(p->nrhs), p->iparm,
           &(p->msglvl), rx, p->tmp_n, &(p->error));
  if (p->error != 0) {
    scs_printf("Error during linear system solution: %d", (int)p->error);
    return p->error;
  }

  /* ry = A x - ry, A x as A' x of A by rows */
  SCS(accum_by_atrans)(p->At, rx, ry);

  /* ry = R_y^{-1} (A x - ry) = y */
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (i = 0; i < p->m; ++i) {
    ry[i] *= p->r_y_inv[i];
  }

  return 0;
}

/* Update diagonal R entries: re-form Gram matrix and re-factorize */
scs_int scs_update_lin_sys_diag_r(ScsLinSysWork *p, const scs_float *diag_r) {
  if (form_gram(p, diag_r)) {
    scs_printf("Error in forming Gram matrix");
    return -1;
  }

  /* Perform numerical factorization */
  p->phase = PARDISO_NUMERIC;
  _PARDISO(p->pt, &(p->maxfct), &(p->mnum), &(p->mtype), &(p->phase), &(p->n),
           p->G->x, p->G->p, p->G->i, SCS_NULL, &(p->nrhs), p->iparm,
           &(p->msglvl), SCS_NULL, SCS_NULL, &(p->error));

  if (p->error != 0) {
    scs_printf("Error in PARDISO factorization when updating: %d.\n",
               (int)p->error);
    return p->error;
  }
  return 0;
}

void scs_free_lin_sys_work(ScsLinSysWork *p) {
  if (p) {
    /* Pardiso cleanup needs the index arrays of G, which exist whenever the
     * solver state does */
    if (p->G) {
      p->phase = PARDISO_CLEANUP;
      _PARDISO(p->pt, &(p->maxfct), &(p->mnum), &(p->mtype), &(p->phase),
               &(p->n), SCS_NULL, p->G->p, p->G->i, SCS_NULL, &(p->nrhs),
               p->iparm, &(p->msglvl), SCS_NULL, SCS_NULL, &(p->error));
      if (p->error != 0) {
        scs_printf("Error during MKL Pardiso cleanup: %d", (int)p->error);
      }
      SCS(cs_spfree)(p->G);
    }
    SCS(cs_spfree)(p->At);
    SCS(cs_spfree)(p->Pt);
    scs_free(p->r_y_inv);
    scs_free(p->tmp_m);
    scs_free(p->tmp_n);
    scs_free(p);
  }
}
