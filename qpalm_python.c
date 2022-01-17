#include "qpalm.h"
#include "ladel.h"

solver_sparse *python_allocate_sparse(size_t m, size_t n, size_t nzmax) {
  solver_common common, *c;
  c = &common;
  solver_sparse *M;
  M = ladel_sparse_alloc(m, n, nzmax, UNSYMMETRIC, TRUE, FALSE);
  return M;
}

QPALMSettings *python_allocate_settings(void) {
  return (QPALMSettings *) c_malloc(sizeof(QPALMSettings));
}

QPALMData *python_allocate_data(void) {
  return (QPALMData *) c_malloc(sizeof(QPALMData));
}

void python_free_settings(QPALMSettings *settings) {
  if (settings) c_free(settings);
}

void python_free_data(QPALMData *data) {
    solver_common common, *c;
    c = &common;
    if (data) {
      data->Q = ladel_sparse_free(data->Q);
      data->A = ladel_sparse_free(data->A);
      
      if (data->q) c_free(data->q);

      if (data->bmin) c_free(data->bmin);

      if (data->bmax) c_free(data->bmax);
      c_free(data);
    }
}