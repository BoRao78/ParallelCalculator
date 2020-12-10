#define _POSIX_C_SOURCE 199309L
/* 
CS 5473 Introduction to Parallel Programming 
Project 2
Algorithm 2
Bo Rao  113417729   rao0004
Apr. 18, 2019 
*/
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define RANGELOW    100         // Low range of Matrix size 
#define RANGEHIGH   1000        // High range of Matrix size
#define RANGESTEP   100         // Range step
#define RAND_MAX    10000
#define MAXTEST     256

void generateMatrix(int m, int n, double *mat);
void distMatbyRow(double *sourMat, double *destMat, int m, int n, int count, int rank);
void distMatbyCol(double *sourMat, double *destMat, int m, int n, int count, int rank);
int matMultip(double *sourMatA, double *sourMatB, double *destMat, int m1, int n1, int m2, int n2);
void resMerge(double *resMat, double *resTemp, int m, int n, int count, int rank);
void matCopy(double *sourMat, double *destMat, int m, int n);
int matCmp(double *matA, double *matB, int m1, int n1, int m2, int n2);
int matReorderbyCol(double *mat, int m, int n, int m1, int n1);
int modulo(int x,int N);

void printMat(double *mat, int m, int n);


void MatrixMultiply(double *ttlMatA, double *ttlMatB, double *ttlRes, int n, int subn, int ttlNum, int subNum){
    int my_rank = omp_get_thread_num();
    int thread_count = omp_get_num_threads();
    double *subMatA = (double *)malloc(subNum * sizeof(double));
    double *subMatB = (double *)malloc(subNum * sizeof(double));
    double *subRes = &ttlRes[subNum * my_rank];
    double *ttlMatAdup = (double *)malloc(ttlNum * sizeof(double));
    matCopy(ttlMatA, ttlMatAdup, n, n);

    //distMatbyRow(ttlMatA, subMatA, n, n, thread_count, my_rank);
    distMatbyCol(ttlMatB, subMatB, n, n, thread_count, my_rank);

    //printf("debug (1) thread %d\n", my_rank);

    //Compute own columns of C
    long resNum = pow((long)subn, 2);
    double *resTemp = (double *)malloc(resNum * sizeof(double));
    for(int j = 0; j < thread_count; j++){
        //printf("debug (2) thread %d\n", my_rank);
        //subMatA = subMatAList[(my_rank - j) % thread_count];
        distMatbyRow(ttlMatAdup, subMatA, n, n, thread_count, modulo(my_rank - j, thread_count));
        //printf("debug (3) thread %d\n", my_rank);
        if(matMultip(subMatA, subMatB, resTemp, subn, n, n, subn) != 0){
            printf("Multiple error!\n");
            exit(-1);
        }
        //printf("debug (4) thread %d\n", my_rank);
        // merge res of each step to subRes
        resMerge(subRes, resTemp, n, subn, thread_count, modulo(my_rank - j, thread_count));        
    }
        
    free(resTemp);
    free(subMatA);free(subMatB);free(ttlMatAdup);
}


int main (int argc, char* argv[]){
    int n = 0, subn = 0;                            // Size of matrix
    long ttlNum = 0, subNum = 0;                    // total numbers in matrix
    double *ttlMatA = NULL;
    double *ttlMatB = NULL;                 
    double *ttlRes = NULL;

    double start, finish;               // Time start and finish
    double duration;                    // Time consumed
    double timeList[MAXTEST] = {0};
    int testCount = 0;

    int thread_count = strtol(argv[1], NULL, 10);

    //print the running information
    printf("(%c) Program running on %d threads.\n", 'a' + (int)log2(thread_count), thread_count);

    for(n = RANGELOW; n <= RANGEHIGH; n += RANGESTEP){
        subn = n/thread_count;
        ttlNum = pow((long)n, 2);
        subNum = ttlNum/thread_count;

        // Chceck if n divided by p
        printf("Matrix Size: %dx%d\n", n, n);
        if(n % thread_count != 0){
            printf("[SKIP]: n not divided by # of processes.\n");
            // Mark SKIP as running time = 0
            timeList[testCount] = 0.0;
            testCount++;
            continue;
        }

        // Generate Matrix A B & Res, Res = A*B
        ttlMatA = (double *)malloc(ttlNum * sizeof(double));
        ttlMatB = (double *)malloc(ttlNum * sizeof(double));
        ttlRes = (double *)malloc(ttlNum * sizeof(double));
        generateMatrix(n,n,ttlMatA);
        generateMatrix(n,n,ttlMatB);

        // //Dubug
        // printf("Generated Matrix A:  ");
        // printMat(ttlMatA, 10, 10);
        // printf("Generated Matrix B:  ");
        // printMat(ttlMatB, 10, 10);

        start = omp_get_wtime() * 1000;
#       pragma omp parallel num_threads(thread_count)
        MatrixMultiply(ttlMatA, ttlMatB, ttlRes, n, subn, ttlNum, subNum);

        //reorder result
        matReorderbyCol(ttlRes, n, n, n, subn);

        // Report run time
        finish = omp_get_wtime() * 1000;
        duration = finish - start;  //ms
        timeList[testCount] = duration;
        testCount++;
        printf( "%.6lf millisecond consumed, matrix size %dx%d\n", duration, n, n);

        // Check result
        double *resTemp = (double *)malloc(ttlNum * sizeof(double));
        matMultip(ttlMatA, ttlMatB,resTemp, n, n, n, n);
        if(matCmp(resTemp, ttlRes, n, n, n, n) == 0){
            printf("Matrix Multiplication Verified.\n");
        }
        else {
            printMat(ttlRes, n, n);
            printf("\n");
            printMat(resTemp, n, n);
            fprintf(stderr, "Error in Verifying the Multiplication.\n");
            exit(-1);
        }
        free(ttlMatA);free(ttlMatB);free(ttlRes);
        free(resTemp);    
    }

    // Print time list
    printf("Consumed Time list:\n[");
    for(int i = 0; i < testCount; i++)
        printf(" %.6lf",timeList[i]);
    printf("]\n");

    return 0;
}

// initialize mxn Matrix 
void generateMatrix(int m, int n, double *mat){
    struct timespec spec;
    clock_gettime(CLOCK_REALTIME, &spec);
    long ns = spec.tv_nsec;
    srand(ns);
    for(int i = 0; i < m * n; i++){
         mat[i] = ((rand() % RAND_MAX)/(double)RAND_MAX)*2 - 1;     //range [-1,1]
    }
}

void distMatbyRow(double *sourMat, double *destMat, int m, int n, int count, int rank){
    // start position of distributed rows
    int start = (m/count)*rank;
    int subm = m/count;
    for(int i = 0; i < subm; i++){
        for(int j = 0; j < n; j++){
            destMat[i * n + j] = sourMat[(i + start) * n + j];
        }
    }
}

void distMatbyCol(double *sourMat, double *destMat, int m, int n, int count, int rank){
    // start position of distributed columns
    int start = (m/count)*rank;
    int subn = n/count;
    for(int i = 0; i < m; i++){
        for(int j = 0; j < subn; j++){
            destMat[i * subn + j] = sourMat[i * n + (j + start)];
        }
    }
}

int matMultip(double *sourMatA, double *sourMatB, double *destMat, int m1, int n1, int m2, int n2){
    if(n1 != m2)
        return -1;
    double sumTemp = 0;
    for(int i = 0; i < m1; i++){
        for(int j = 0; j < n2; j++){
            sumTemp = 0;
            for(int a = 0; a < n1; a++){
                sumTemp += sourMatA[i * n1 + a] * sourMatB[a * n2 + j];
            }
            destMat[i * n2 + j] = sumTemp;
        }
    }
    return 0;
}

void resMerge(double *resMat, double *resTemp, int m, int n, int count, int rank){
    int subm = m/count;
    for(int i = 0; i < subm; i++){
        for(int j = 0; j < n; j++)
            resMat[(i + rank * subm) * n + j] = resTemp[i * n + j];
    }
}

void matCopy(double *sourMat, double *destMat, int m, int n){
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++)
            destMat[i * n + j] = sourMat[i * n + j];
    }
}

int matCmp(double *matA, double *matB, int m1, int n1, int m2, int n2){
    if(m1 != m2 || n1 != n2)
        return -1;
    int flag = 0;
    for(int i = 0; i < m1; i++){
        for(int j = 0; j < n1; j++){
            if(fabs(matA[i * n1 + j] - matB[i * n1 + j]) >= 1e-6)
                return -1;
        }
    }
    return flag;
}   

// Reorder result matrix
int matReorderbyCol(double *mat, int m, int n, int m1, int n1){
    if(m != m1 || n%n1 != 0)
        return -1;
    int partCount = n/n1;
    double *matTemp = (double *)malloc(m * n * sizeof(double));
    matCopy(mat, matTemp, m, n);
    for(int i = 0; i < m; i++){
        for(int j = 0; j < partCount; j++){
            for(int k = 0; k < n1; k++){
                mat[i * n + j * n1 + k] = matTemp[j * (m1 * n1) + i * n1 + k];
            }
        }
    }
    return 0;
}

// costimized modulo ( to make -1 % N = N - 1);
int modulo(int x,int N){
    return (x % N + N) % N;
}

// (For debug) Print matrix
void printMat(double *mat, int m, int n){
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            printf("%.3f ", mat[i * n + j]);
        }
        printf("\n");
    }
}
