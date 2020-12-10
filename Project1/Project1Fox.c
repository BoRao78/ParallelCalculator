#define _POSIX_C_SOURCE 199309L
/* 
CS 5473 Introduction to Parallel Programming 
Project 1
Fox's Algorithm
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
void getBlockCoord(int rank, int divide, int *blockX, int *blockY);
void getBlockRank(int *rank, int divide, int blockX, int blockY);
void getBlockbyCoord(double *sourMat, double *destBlock, int matLen, int blockLen, int blockX, int blockY);
void distBlock(double *sourMat, double *destBlock, int matLen, int blockLen, int divide, int rank);
void matZero(double *mat, int m, int n);
void matAdd(double *sourMat, double *destMat, int m, int n);
int matMultip(double *sourMatA, double *sourMatB, double *destMat, int m1, int n1, int m2, int n2);
void matCopy(double *sourMat, double *destMat, int m, int n);
int matCmp(double *matA, double *matB, int m1, int n1, int m2, int n2);
void matReorderbyBlock(double *mat, int n, int blockLen, int divide);

int modulo(int x,int N);

void printMat(double *mat, int m, int n);



int main (int argc, char* argv[]){
    int my_rank, comm_sz, mpi_error_code;
    int divide = 0;
    int n = 0, blockLen = 0;                            // Size of matrix
    long matSize = 0, blockSize = 0;

    double *ttlMatA = NULL;
    double *ttlMatB = NULL;                 
    double *blockA = NULL;              // Distributed local text 
    double *blockB = NULL;
    double *ttlRes = NULL;
    double *blockRes = NULL;

    double *blockTemp = NULL;

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

        matSize = (long)n * n;
        divide = (int)sqrt(comm_sz);
        blockLen = n/divide;
        blockSize = (long)blockLen * blockLen;

        // Chceck if n divided by p
        if(my_rank == 0)
            printf("Matrix Size: %dx%d\n", n, n);
        if(divide * divide != comm_sz || n % divide != 0){
            if(my_rank == 0){
                printf("[SKIP]: n not divided by sqrt(p) (p = # of processes).\n");
                // Mark SKIP as running time = 0
                timeList[testCount] = 0.0;
                testCount++;
            } 
            continue;
        }

        // (1) initialize and Distribute data
        if(my_rank == 0){

            // Generate Matrix A B & C, C = A*B
            ttlMatA = (double *)malloc(matSize * sizeof(double));
            ttlMatB = (double *)malloc(matSize * sizeof(double));
            generateMatrix(n,n,ttlMatA);
            generateMatrix(n,n,ttlMatB);
            ttlRes = (double *)malloc(matSize * sizeof(double));

            // //Dubug
            // printf("Generated Matrix A:  ");
            // printMat(ttlMatA, n, n);
            // printf("Generated Matrix B:  ");
            // printMat(ttlMatB, n, n);

            start = MPI_Wtime();
            // Distribute block of B to self
            blockA = (double *)malloc(blockSize * sizeof(double));
            blockB = (double *)malloc(blockSize * sizeof(double));
            blockRes = (double *)malloc(blockSize * sizeof(double));
            matZero(blockA, blockLen, blockLen);
            distBlock(ttlMatB, blockB, n, blockLen, divide, 0);

            // Distribute block of B to other process
            blockTemp = (double *)malloc(blockSize * sizeof(double));
            for(int rank = 1; rank < comm_sz; rank++){
                distBlock(ttlMatB, blockTemp, n, blockLen, divide, rank);
                MPI_Send(blockTemp, blockSize, MPI_DOUBLE, rank, 0, comm);
            }
            free(blockTemp);
        }
        else{
            // receive block of A & B to self
            blockA = (double *)malloc(blockSize * sizeof(double));
            blockB = (double *)malloc(blockSize * sizeof(double));
            blockRes = (double *)malloc(blockSize * sizeof(double));
            matZero(blockA, blockLen, blockLen);
            MPI_Recv(blockB, blockSize, MPI_DOUBLE, 0, 0, comm, MPI_STATUS_IGNORE);
        }

        //Dubug
        //printf("process %d block A:  ", my_rank);
        //printMat(blockA, blockLen, blockLen);
        // printf("process %d block B:  ", my_rank);
        // printMat(blockB, blockLen, blockLen);

        // (2) Compute own columns of C
        int sendDest = 0;
        int RecvSour = 0;
        matZero(blockRes, blockLen, blockLen);
        blockTemp = (double *)malloc(blockSize * sizeof(double));
        for(int j = 0; j < divide; j++){
            // Broadcast blocks of A
            for(int k = 0; k < divide; k++){
                if(my_rank == 0){
                    getBlockbyCoord(ttlMatA, blockTemp, n, blockLen, k, modulo(k + j, divide));
                    int blockX = my_rank / divide;
                    if(blockX == k)
                        matCopy(blockTemp, blockA, blockLen, blockLen);
                    for(int rank = 1; rank < comm_sz; rank++){
                            blockX = rank / divide;
                            if(blockX == k)
                                MPI_Send(blockTemp, blockSize, MPI_DOUBLE, rank, 1, comm);
                    }
                }
                else {
                    int blockX = my_rank / divide;
                    if(blockX == k)
                        MPI_Recv(blockA, blockSize, MPI_DOUBLE, 0, 1, comm, MPI_STATUS_IGNORE);
                }
            }

            // Compute blocks of result
            if(matMultip(blockA, blockB, blockTemp, blockLen, blockLen, blockLen, blockLen) != 0){
                printf("Multiple error!\n");
                exit(-1);
            }
            matAdd(blockTemp, blockRes, blockLen, blockLen);

            // Send and recv blocks of B
            int blockX = 0, blockY = 0;
            getBlockCoord(my_rank, divide, &blockX, &blockY);
            getBlockRank(&sendDest, divide, modulo(blockX - 1, divide), blockY);
            getBlockRank(&RecvSour, divide, modulo(blockX + 1, divide), blockY);
            matCopy(blockB, blockTemp, blockLen, blockLen);
            MPI_Sendrecv(blockTemp, blockSize, MPI_DOUBLE, sendDest, 2, blockB, blockSize, MPI_DOUBLE, RecvSour, 2, comm, MPI_STATUS_IGNORE);
        }
        free(blockTemp);

        // //Dubug
        // printf("process %d local Res Matrix:  \n", my_rank);
        // printMat(blockRes, blockLen, blockLen);

        // Gather the subRes to ttlRes (order is wrong)
        MPI_Gather(blockRes, blockSize, MPI_DOUBLE, ttlRes, blockSize, MPI_DOUBLE, 0, comm);
        
        if(my_rank == 0){
            //reorder result
            matReorderbyBlock(ttlRes, n, blockLen, divide);

            // Report run time
            finish = MPI_Wtime();
            duration = (finish - start) * 1000;  //ms
            timeList[testCount] = duration;
            testCount++;
            printf( "%.6lf millisecond consumed, matrix size %dx%d\n", duration, n, n);

            // Check result
            double *resTemp = (double *)malloc(matSize * sizeof(double));
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

    free(ttlMatA);free(ttlMatB);free(ttlRes);free(blockA);free(blockB);free(blockRes);
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

void getBlockCoord(int rank, int divide, int *blockX, int *blockY){
    *blockX = rank / divide;
    *blockY = rank % divide;
}

void getBlockRank(int *rank, int divide, int blockX, int blockY){
    *rank = blockX * divide + blockY;
}

void getBlockbyCoord(double *sourMat, double *destBlock, int matLen, int blockLen, int blockX, int blockY){
    for(int i = 0; i < blockLen; i++){
        for(int j = 0; j < blockLen; j++){
            destBlock[ i * blockLen + j] = sourMat[(blockX * blockLen + i) * matLen + blockY * blockLen + j];
        }
    }
}

void distBlock(double *sourMat, double *destBlock, int matLen, int blockLen, int divide, int rank){
    // The block cordinate
    int blockX = 0;
    int blockY = 0;
    getBlockCoord(rank, divide, &blockX, &blockY);
    getBlockbyCoord(sourMat, destBlock, matLen, blockLen, blockX, blockY);
}

void matZero(double *mat, int m, int n){
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++)
            mat[i * n + j] = 0.0;
    }
}

void matAdd(double *sourMat, double *destMat, int m, int n){
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++)
            destMat[i * n + j] += sourMat[i * n + j];
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

// Reorder result matrix by blocks
void matReorderbyBlock(double *mat, int n, int blockLen, int divide){
    double *matTemp = (double *)malloc(n * n * sizeof(double));
    matCopy(mat, matTemp, n, n);
    int blockCount = divide * divide;
    long blockSize = (long)blockLen * blockLen;
    int blockX = 0;
    int blockY = 0;
    
    for(int rank = 0; rank < blockCount; rank++){
        getBlockCoord(rank, divide, &blockX, &blockY);
        for(int i = 0; i < blockLen; i++){
            for(int j = 0; j < blockLen; j++){
                mat[(blockX * blockLen + i) * n + blockY * blockLen + j] = matTemp[rank * blockSize + i * blockLen + j];
            }
        }
    }
    free(matTemp);
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
