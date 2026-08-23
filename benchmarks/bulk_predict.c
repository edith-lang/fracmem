/*
 * Bulk-mode runner for a fracmem-exported filter: reads a raw float32
 * binary file, runs fracmemStep once per sample, writes the results to
 * another raw float32 binary file, and prints the compute-only elapsed
 * time (the per-sample loop alone, not file I/O) to stdout in seconds.
 *
 * Binary I/O on purpose: a text/scanf-per-sample harness would spend
 * most of its time in text parsing, not in the filter, at multi-
 * million-sample scale -- that would measure the C standard library's
 * text I/O, not fracmem.
 *
 * Compile together with fracmemfilter.c and one exported device_filter.c
 * (see export_c() in fracmem.embedded):
 *   cc -O2 -o bulk_predict bulk_predict.c fracmemfilter.c device_filter.c -I <c dir>
 * Usage:
 *   ./bulk_predict input.f32 output.f32
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "fracmemfilter.h"

extern FracmemFilter filt;
void filtSetup(void);

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s input.f32 output.f32\n", argv[0]);
        return 1;
    }

    FILE *fin = fopen(argv[1], "rb");
    if (!fin) { perror("fopen input"); return 1; }
    fseek(fin, 0, SEEK_END);
    long nbytes = ftell(fin);
    fseek(fin, 0, SEEK_SET);
    long n = nbytes / (long)sizeof(float);

    float *x = malloc((size_t)nbytes);
    float *y = malloc((size_t)nbytes);
    if (!x || !y) { fprintf(stderr, "out of memory\n"); return 1; }
    if (fread(x, sizeof(float), (size_t)n, fin) != (size_t)n) {
        fprintf(stderr, "short read\n"); return 1;
    }
    fclose(fin);

    filtSetup();

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (long k = 0; k < n; k++) {
        y[k] = fracmemStep(&filt, x[k]);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_nsec - t0.tv_nsec) / 1e9;

    FILE *fout = fopen(argv[2], "wb");
    if (!fout) { perror("fopen output"); return 1; }
    fwrite(y, sizeof(float), (size_t)n, fout);
    fclose(fout);

    printf("%.9f\n", elapsed);

    free(x);
    free(y);
    return 0;
}
