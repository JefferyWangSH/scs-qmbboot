#ifndef PRIV_H_GUARD
#define PRIV_H_GUARD

#ifdef __cplusplus
extern "C" {
#endif

#include "csparse.h"
#include "glbopts.h"
#include "linsys.h"
#include "scs_matrix.h"

struct SCS_LIN_SYS_WORK {
  scs_int n; /* number of QP variables */
  scs_int m; /* number of QP constraints */

  const ScsMatrix *A; /* does *not* own this memory */
  ScsMatrix *At;      /* A by rows (CSC of A') */
  ScsMatrix *Pt;      /* P by rows (CSC of P'), or NULL */
  ScsMatrix *G;       /* Gram matrix R_x + P + A' R_y^{-1} A, upper by rows */
  scs_float *r_y_inv; /* 1 / R_y diagonal, length m */
  scs_float *tmp_m;   /* workspace of length m */
  scs_float *tmp_n;   /* Pardiso solve workspace of length n */

  /* Pardiso variables */
  void *pt[64];      /* internal solver memory pointer pt */
  scs_int iparm[64]; /* Pardiso control parameters */
  scs_int mtype;     /* matrix type (2 for real symmetric positive definite) */
  scs_int nrhs;      /* number of right-hand sides (1) */
  scs_int maxfct;    /* maximum number of factors (1) */
  scs_int mnum;      /* indicates matrix for the solution phase (1) */
  scs_int phase;     /* control the execution phases of the solver */
  scs_int error;     /* the error indicator (0 for no error) */
  scs_int msglvl;    /* Message level information (0 for no output) */
};

#ifdef __cplusplus
}
#endif

#endif
