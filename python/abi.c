/*
 * Sizes of the types and structs of scs.h, compiled into each library of the
 * Python package so that python/scs_qmbboot/_capi.py can check its ctypes
 * mirror of scs.h against the library it loads.
 */

#include <stddef.h>

#include "scs.h"

/* Writes sizeof of scs_int, scs_float, ScsMatrix, ScsSettings, ScsData,
 * ScsCone, ScsSolution, ScsInfo to sizes[0..7]; returns the count. */
int scs_qmbboot_abi(size_t *sizes) {
  sizes[0] = sizeof(scs_int);
  sizes[1] = sizeof(scs_float);
  sizes[2] = sizeof(ScsMatrix);
  sizes[3] = sizeof(ScsSettings);
  sizes[4] = sizeof(ScsData);
  sizes[5] = sizeof(ScsCone);
  sizes[6] = sizeof(ScsSolution);
  sizes[7] = sizeof(ScsInfo);
  return 8;
}
