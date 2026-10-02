/*
 * testmain.c -- small driver used only by tests/test_embedded_c.py to
 * exercise fracmemfilter.c the same way a real device would, and print
 * the results so Python can compare them against CompressedFractionalFilter.
 *
 * Reads everything from stdin, in this order, whitespace-separated:
 *   L p hPow caputo
 *   L values of w
 *   p values of lam
 *   p values of c
 *   n                (how many signal samples follow)
 *   n signal values
 * Prints one derivative estimate per line to stdout.
 */
#include <stdio.h>
#include "fracmemfilter.h"

int main(void)
{
    FracmemFilter f;
    float w[fracmemMaxL], lam[fracmemMaxP], c[fracmemMaxP];
    float hPow, xk, yk;
    int L, p, caputo, n, i;

    if (scanf("%d %d %f %d", &L, &p, &hPow, &caputo) != 4) {
        return 1;
    }
    for (i = 0; i < L; i++) {
        if (scanf("%f", &w[i]) != 1) return 1;
    }
    for (i = 0; i < p; i++) {
        if (scanf("%f", &lam[i]) != 1) return 1;
    }
    for (i = 0; i < p; i++) {
        if (scanf("%f", &c[i]) != 1) return 1;
    }

    fracmemInit(&f, L, p, hPow, caputo, w, lam, c);

    if (scanf("%d", &n) != 1) {
        return 1;
    }
    for (i = 0; i < n; i++) {
        if (scanf("%f", &xk) != 1) return 1;
        yk = fracmemStep(&f, xk);
        printf("%.8f\n", yk);
    }

    return 0;
}
