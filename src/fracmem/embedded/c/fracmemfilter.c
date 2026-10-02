/*
 * fracmemfilter.c -- see fracmemfilter.h for what this does and why.
 * Same math, same steps as fracmem/embedded/runtime.py, translated
 * line for line: an exact local window (the last L samples) plus p
 * leaky buckets standing in for everything older.
 */
#include "fracmemfilter.h"

void fracmemInit(FracmemFilter *f, int L, int p, float hPow, int caputo,
                  const float *w, const float *lam, const float *c)
{
    int i;

    f->L = L;
    f->p = p;
    f->hPow = hPow;
    f->caputo = caputo;

    for (i = 0; i < L; i++) {
        f->w[i] = w[i];
    }
    for (i = 0; i < p; i++) {
        f->lam[i] = lam[i];
        f->c[i] = c[i];
    }

    fracmemReset(f);
}

void fracmemReset(FracmemFilter *f)
{
    int i;

    for (i = 0; i < f->L; i++) {
        f->buf[i] = 0.0f;
    }
    for (i = 0; i < f->p; i++) {
        f->m[i] = 0.0f;
    }
    f->idx = 0;
    f->x0 = 0.0f;
    f->x0Set = 0;
}

float fracmemStep(FracmemFilter *f, float xk)
{
    float dropped, local, tail, mi;
    int i, j, pos;

    if (f->caputo) {
        if (!f->x0Set) {
            f->x0 = xk;
            f->x0Set = 1;
        }
        xk = xk - f->x0;
    }

    /* the sample about to fall out of the L-sample window, and the new
       sample taking its place */
    dropped = f->buf[f->idx];
    f->buf[f->idx] = xk;
    f->idx = f->idx + 1;
    if (f->idx >= f->L) {
        f->idx = 0;
    }

    /* exact local window: a plain weighted sum of the L most recent samples */
    local = 0.0f;
    pos = f->idx;
    for (j = 0; j < f->L; j++) {
        pos = pos - 1;
        if (pos < 0) {
            pos = f->L - 1;
        }
        local = local + f->w[j] * f->buf[pos];
    }
    local = local * f->hPow;

    /* leaky buckets: one multiply and one add per bucket, nothing else */
    tail = 0.0f;
    for (i = 0; i < f->p; i++) {
        mi = f->lam[i] * f->m[i] + dropped;
        f->m[i] = mi;
        tail = tail + f->c[i] * mi;
    }

    return local + tail;
}
