#define _POSIX_C_SOURCE 199309L
/* 
CS 5473 Introduction to Parallel Programming 
Project 1
Algorithm 2
Bo Rao  113417729   rao0004
Mar. 10, 2019 
*/
#include <mpi.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#define RANGELOW    100       // Low range of Matrix size 
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


int main (int argc, char* argv[]){
    int my_rank, comm_sz, mpi_error_code;
    int n = 0, subn = 0;                            // Size of matrix
    long ttlNum = 0, subNum = 0;                    // total numbers in matrix
    double *ttlMatA = NULL;
    double *ttlMatB = NULL;                 
    double *subMatA = NULL;                         // Distributed local text 
    double *subMatB = NULL;
    double *ttlRes = NULL;
    double *subRes = NULL;

    double start, finish;               // Time start and finish
    double duration;                    // Time consumed
    double timeList[MAXTEST] = {0};
    int testCount = 0;
    mpi_error_code = MPI_Init(&argc, &argv); 
    MPI_Comm comm = MPI_COMM_WORLD;
    mpi_error_code = MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    mpi_error_code = MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    MPI_Barrier(comm); 

    //print the running information
    if(my_rank == 0)
        printf("(%c) Program running on %d processes.\n", 'a' + (int)log2(comm_sz), comm_sz);

    for(n = RANGELOW; n <= RANGEHIGH; n += RANGESTEP){
        subn = n/comm_sz;
        ttlNum = pow((long)n, 2);
        subNum = ttlNum/comm_sz;

        // Chceck if n divided by p
        if(my_rank == 0)
            printf("Matrix Size: %dx%d\n", n, n);
        if(n % comm_sz != 0){
            if(my_rank == 0){
                printf("[SKIP]: n not divided by # of processes.\n");
                // Mark SKIP as running time = 0
                timeList[testCount] = 0.0;
                testCount++;
            } 
            continue;
        }

        // (1) initialize and Distribute data
        if(my_rank == 0){

            // Generate Matrix A B & C, C = A*B
            ttlMatA = (double *)malloc(ttlNum * sizeof(double));
            ttlMatB = (double *)malloc(ttlNum * sizeof(double));
            generateMatrix(n,n,ttlMatA);
            generateMatrix(n,n,ttlMatB);

            // //Dubug
            // printf("Generated Matrix A:  ");
            // printMat(ttlMatA, n, n);
            // printf("Generated Matrix B:  ");
            // printMat(ttlMatB, n, n);

            ttlRes = (double *)malloc(ttlNum * sizeof(double));

            start = MPI_Wtime();
            // Distribute submatrix of A & B to self
            subMatA = (double *)malloc(subNum * sizeof(double));
            subMatB = (double *)malloc(subNum * sizeof(double));
            subRes = (double *)malloc(subNum * sizeof(double));
            distMatbyRow(ttlMatA, subMatA, n, n, comm_sz, 0);
            distMatbyCol(ttlMatB, subMatB, n, n, comm_sz, 0);

            // Distribute submatrix of A & B to other process
            double *subMatTemp = (double *)malloc(subNum * sizeof(double));
            for(int rank = 1; rank < comm_sz; rank++){
                distMatbyRow(ttlMatA, subMatTemp, n, n, comm_sz, rank);
                MPI_Send(subMatTemp, subNum, MPI_DOUBLE, rank, 0, comm);
                distMatbyCol(ttlMatB, subMatTemp, n, n, comm_sz, rank);
                MPI_Send(subMatTemp, subNum, MPI_DOUBLE, rank, 1, comm);
            }
            free(subMatTemp);
        }
        else{
            // receive submatrix of A & B to self
            subMatA = (double *)malloc(subNum * sizeof(double));
            subMatB = (double *)malloc(subNum * sizeof(double));
            subRes = (double *)malloc(subNum * sizeof(double));
            MPI_Recv(subMatA, subNum, MPI_DOUBLE, 0, 0, comm, MPI_STATUS_IGNORE);
            MPI_Recv(subMatB, subNum, MPI_DOUBLE, 0, 1, comm, MPI_STATUS_IGNORE);
        }


        // //Dubug
        // printf("process %d local Matrix A:  ", my_rank);
        // printMat(subMatA, subn, n);
        // printf("process %d local Matrix B:  ", my_rank);
        // printMat(subMatB, n, subn);

        // (2) Compute own columns of C
        long resNum = pow((long)subn, 2);
        double *resTemp = (double *)malloc(resNum * sizeof(double));
        double *subMatATemp = (double *)malloc(subNum * sizeof(double));
        int sendDest = 0;
        int RecvSour = 0;
        for(int j = 0; j < comm_sz; j++){
            if(matMultip(subMatA, subMatB, resTemp, subn, n, n, subn) != 0){
                printf("Multiple error!\n");
                exit(-1);
            }
            // merge res of each step to subRes
            resMerge(subRes, resTemp, n, subn, comm_sz, modulo(my_rank - j, comm_sz));
            matCopy(subMatA, subMatATemp, subn, n);
            sendDest = modulo(my_rank + 1, comm_sz);
            RecvSour = modulo(my_rank - 1, comm_sz);
            MPI_Sendrecv(subMatATemp, subNum, MPI_DOUBLE,sendDest, 2, subMatA, subNum, MPI_DOUBLE, RecvSour, 2, comm, MPI_STATUS_IGNORE);
        }
        free(resTemp); free(subMatATemp);


        // //Dubug
        // printf("process %d local Res Matrix:  \n", my_rank);
        // printMat(subRes, n, subn);

        // Gather the subRes to ttlRes (order is wrong)
        MPI_Gather(subRes, subNum, MPI_DOUBLE, ttlRes, subNum, MPI_DOUBLE, 0, comm);
        
        if(my_rank == 0){
            //reorder result
            matReorderbyCol(ttlRes, n, n, n, subn);

            // Report run time
            finish = MPI_Wtime();
            duration = (finish - start) * 1000;  //ms
            timeList[testCount] = duration;
            testCount++;
            printf( "%.6lf millisecond consumed, matrix size %dx%d\n", duration, n, n);

            // Check result
            resTemp = (double *)malloc(ttlNum * sizeof(double));
            matMultip(ttlMatA, ttlMatB,resTemp, n, n, n, n);
            if(matCmp(resTemp, ttlRes, n, n, n, n) == 0){
                 printf("Matrix Multiplication Verified.\n");
            }
            else {
                // printMat(ttlRes, n, n);
                // printMat(resTemp, n, n);
                fprintf(stderr, "Error in Verifying the Multiplication.\n");
                exit(-1);
            }
            free(resTemp);
        }
            
    }

    if(my_rank == 0){
        // Print time list
        printf("Consumed Time list:\n[");
        for(int i = 0; i < testCount; i++)
            printf(" %.6lf",timeList[i]);
        printf("]\n");
    }

    free(ttlMatA);free(ttlMatB);free(ttlRes);free(subMatA);free(subMatB);free(subRes);
    MPI_Finalize();
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
