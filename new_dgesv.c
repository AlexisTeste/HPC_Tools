#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
//#include "mkl_lapacke.h"
#include <omp.h>

double *generate_matrix(int size, double *matrix)
{
    int i;
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

double *transpose(double *a, int n, double *mat)
{
    int i,j;

    #pragma omp parallel for private (i,j)
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            mat[i * n + j] = a[j * n + i];
        }
    }
    return mat;
}

double *prodMat(double *a, double *b, int n, double *c)
{
    int i,j,k;

    #pragma omp parallel for private(i,j,k) schedule(static)
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

void QR(double *a, double *q, double *r, int n, double *A)
{

    double s;
    int i,j,k;

    #pragma omp parallel for private (i,j,k) schedule(static)
    for ( i = 0; i < n; i++)
    {
        for ( j = 0; j < n; j+=4)
        {
            q[i * n + j] = 0;
            r[i * n + j] = 0;
            A[i * n + j] = a[i * n + j];
            q[i * n + j+1] = 0;
            r[i * n + j+1] = 0;
            A[i * n + j+1] = a[i * n + j+1];
            q[i * n + j+2] = 0;
            r[i * n + j+2] = 0;
            A[i * n + j+2] = a[i * n + j+2];
            q[i * n + j+3] = 0;
            r[i * n + j+3] = 0;
            A[i * n + j+3] = a[i * n + j+3];
        }
    }
    
    for ( k = 0; k < n; k++)
    {
        s = 0;
        #pragma omp parallel for reduction(+:s)
        for (int j = 0; j < n; j++)
        {
            s += A[j * n + k] * A[j * n + k];
        }
        r[k * n + k] = sqrt(s);

        for ( j = 0; j < n; j++)
        {
            q[j * n + k] = A[j * n + k] / r[k * n + k];
        }
        #pragma omp parallel for private(j) schedule(dynamic) reduction(+:s)
        for ( i = k; i < n; i++)
        {
            s = 0;
            
            for ( j = 0; j < n; j++)
            {
                s += A[j * n + i] * q[j * n + k];
            }

            r[k * n + i] = s;

            for ( j = 0; j < n; j++)
            {
                A[j * n + i] -= r[k * n + i] * q[j * n + k];
            }
        }
    }
}

int my_dgesv(int n, int nrhs, double *a, int lda, int *ipiv, double *b, int ldb, double *q, double *r, double *y, double *trQ, double *A)
{

    int i, j, k;

    double s;

    QR(a, q, r, n, A);

    trQ = transpose(q, n, trQ);

    y = prodMat(trQ, b, n, y);
    #pragma omp parallel for private(j)
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j+=4)
        {
            b[i * n + j] = 0;
            b[i * n + j + 1] = 0;
            b[i * n + j + 2] = 0;
            b[i * n + j + 3] = 0;
        }
    }

    for (i = n - 1; i >= 0; i--)
    {
        #pragma omp parallel for private(k) schedule(dynamic) reduction(+:s)
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

    // Using MKL to solve the system
    int n = size, nrhs = size, lda = size, ldb = size, info;

    double *q = (double *)malloc(sizeof(double) * n * n);
    double *r = (double *)malloc(sizeof(double) * n * n);
    double *y = (double *)malloc(sizeof(double) * n * n);

    double *a = (double *)malloc(sizeof(double) * n * n);
    double *aref = (double *)malloc(sizeof(double) * n * n);
    double *b = (double *)malloc(sizeof(double) * n * n);
    double *bref = (double *)malloc(sizeof(double) * n * n);

    double *A = (double *)malloc(sizeof(double) * n * n);

    double *trQ = (double *)malloc(sizeof(double) * n * n);

    a = generate_matrix(size, a);
    aref = generate_matrix(size, aref);
    b = generate_matrix(size, b);
    bref = generate_matrix(size, bref);

    //print_matrix("A", a, size);
    //print_matrix("B", b, size);

    //double *ipiv = (double *)malloc(sizeof(double) * size);
    clock_t tStart; // = clock();
    // info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, n, nrhs, aref, lda, ipiv, bref, ldb);
    //printf("Time taken by MKL: %.2fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);

    tStart = clock();
    int *ipiv2 = (int *)malloc(sizeof(int) * size);
    my_dgesv(n, nrhs, a, lda, ipiv2, b, ldb, q, r, y, trQ, A);
    printf("Time taken by my implementation: %.2fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);

    // if (check_result(bref, b, size) == 1)
    //     printf("Result is ok!\n");
    // else
    //     printf("Result is wrong!\n");

    //print_matrix("X", b, size);
    //print_matrix("Xref", bref, size);
    free(ipiv2);
    free(q);
    free(r);
    free(y);
    free(a);
    free(aref);
    free(b);
    free(bref);
    free(A);
    free(trQ);
}
