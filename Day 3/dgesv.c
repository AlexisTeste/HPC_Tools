#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
//#include "mkl_lapacke.h"

double *generate_matrix(int size)
{
    int i;
    double *matrix = (double *)malloc(sizeof(double) * size * size);
    srand(1);

    for (i = 0; i < size * size; i++)
    {
        matrix[i] = rand() % 100;
    }

    return matrix;
}

void print_matrix(const char *name, double *matrix, int size)
{
    int i, j;
    printf("matrix: %s \n", name);

    for (i = 0; i < size; i++)
    {
        for (j = 0; j < size; j++)
        {
            printf("%f ", matrix[i * size + j]);
        }
        printf("\n");
    }
}

int check_result(double *bref, double *b, int size)
{
    int i;
    for (i = 0; i < size * size; i++)
    {
        if (bref[i] != b[i])
            return 0;
    }
    return 1;
}

double *transpose(double *a, int n)
{
    double *mat;
    mat = (double *)malloc(sizeof(double) * n * n);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            mat[i * n + j] = a[j * n + i];
        }
    }
    return mat;
}

double *prodMat(double *a, double *b, int n)
{

    double *c = (double *)malloc(sizeof(double) * n * n);

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            c[i * n + j] = 0;
            for (int k = 0; k < n; k++)
            {
                c[i * n + j] += a[i * n + k] * b[k * n + j];
            }
        }
    }

    return c;
}

void QR(double *a, double *q, double *r, int n)
{

    double s;
    double *A = (double *)malloc(sizeof(double) * n * n);

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            q[i * n + j] = 0;
            r[i * n + j] = 0;
            A[i * n + j] = a[i * n + j];
        }
    }

    for (int k = 0; k < n; k++)
    {
        s = 0;
        for (int j = 0; j < n; j++)
        {
            s += A[j * n + k] * A[j * n + k];
        }
        r[k * n + k] = sqrt(s);

        for (int j = 0; j < n; j++)
        {
            q[j * n + k] = A[j * n + k] / r[k * n + k];
        }

        for (int i = k; i < n; i++)
        {
            s = 0;

            for (int j = 0; j < n; j++)
            {
                s += A[j * n + i] * q[j * n + k];
            }

            r[k * n + i] = s;

            for (int j = 0; j < n; j++)
            {
                A[j * n + i] -= r[k * n + i] * q[j * n + k];
            }
        }
    }
}

int my_dgesv(int n, int nrhs, double *a, int lda, int *ipiv, double *b, int ldb)
{

    int i, j, k;

    double *q = (double *)malloc(sizeof(double) * n * n);
    double *r = (double *)malloc(sizeof(double) * n * n);
    double *y = (double *)malloc(sizeof(double) * n * n);

    double *trQ = (double *)malloc(sizeof(double) * n * n);

    double s;

    QR(a, q, r, n);

    trQ = transpose(q, n);

    y = prodMat(trQ, b, n);

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            b[i*n+j] = 0;
        }
        
    }

    for (i = n - 1; i >= 0; i--)
    {
        for (j = n - 1; j >= 0; j--)
        {
            s = 0;
            for (k = i + 1; k < n; k++)
            {
                s += r[i * n + k] * b[k * n + j];
            }
            b[i * n + j] = (y[i * n + j] - s) / r[i * (n + 1)];
        }
    }

    return 1;
}

void main(int argc, char *argv[])
{

    int size = atoi(argv[1]);

    double *a, *aref;
    double *b, *bref;

    a = generate_matrix(size);
    aref = generate_matrix(size);
    b = generate_matrix(size);
    bref = generate_matrix(size);

    //print_matrix("A", a, size);
    //print_matrix("B", b, size);

    // Using MKL to solve the system
    int n = size, nrhs = size, lda = size, ldb = size, info;
    //double *ipiv = (double *)malloc(sizeof(double) * size);
    clock_t tStart; // = clock();
    // info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, n, nrhs, aref, lda, ipiv, bref, ldb);
    //printf("Time taken by MKL: %.2fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);

    tStart = clock();
    int *ipiv2 = (int *)malloc(sizeof(int) * size);
    my_dgesv(n, nrhs, a, lda, ipiv2, b, ldb);
    printf("Time taken by my implementation: %.2fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);

    // if (check_result(bref, b, size) == 1)
    //     printf("Result is ok!\n");
    // else
    //     printf("Result is wrong!\n");

    //print_matrix("X", b, size);
    //print_matrix("Xref", bref, size);
}
