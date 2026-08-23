/*
 * fracmemfilter.h -- zero-dependency streaming fractional-derivative
 * filter for microcontrollers, plain C version.
 *
 * This is the C twin of fracmem/embedded/runtime.py's MicroFractionalFilter:
 * same recursion, same fixed memory, same O(L+p) work per sample. It only
 * APPLIES an already-fitted filter (the exact L local weights w, the
 * decay rates lam, and the combination weights c) -- fitting stays on a
 * laptop, in Python, where numpy/scipy are available. Use
 * fracmem.embedded.exportc to bake a fitted filter's numbers into a
 * ready-to-compile .c file that calls fracmemInit for you.
 *
 * No malloc anywhere: every array below has a fixed maximum size, so
 * this filter uses a fixed, known amount of RAM, however long the
 * device runs. If you need a bigger window or more buckets than the
 * defaults allow, raise fracmemMaxL / fracmemMaxP below and rebuild.
 */
#ifndef FRACMEMFILTER_H
#define FRACMEMFILTER_H

enum { fracmemMaxL = 64, fracmemMaxP = 32 };

typedef struct {
    int L;                    /* how many recent samples are handled exactly */
    int p;                    /* how many leaky buckets approximate everything older */
    float hPow;                /* h to the power -alpha, computed once */
    int caputo;                 /* 1 = subtract the first sample seen, 0 = don't */

    float w[fracmemMaxL];       /* the L exact GL weights */
    float lam[fracmemMaxP];     /* bucket decay (leak) rates, fixed forever */
    float c[fracmemMaxP];       /* bucket combination weights */

    float buf[fracmemMaxL];     /* circular buffer of the last L samples */
    float m[fracmemMaxP];       /* current stored value of each bucket */
    int idx;                     /* where in buf the next sample goes */

    float x0;                    /* first sample seen (only used if caputo) */
    int x0Set;                    /* 1 once x0 has been recorded */
} FracmemFilter;

/*
 * Wire already-fitted constants into f and clear its state to zero.
 * w has L entries, lam and c each have p entries. This function does
 * no fitting itself -- w, lam, c come from Python (CompressedFractionalFilter
 * .fit(...) or .fromAnalytic(...)); this just copies them in.
 */
void fracmemInit(FracmemFilter *f, int L, int p, float hPow, int caputo,
                  const float *w, const float *lam, const float *c);

/* Clear stored state back to zero. L, p, w, lam, c are left untouched. */
void fracmemReset(FracmemFilter *f);

/*
 * Feed one new sample, get back one derivative estimate. Exactly
 * L+p multiply-adds, no memory ever allocated or grown -- call this
 * once per sample, forever.
 */
float fracmemStep(FracmemFilter *f, float xk);

#endif
