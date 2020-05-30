#ifndef HELPERS_HPP
#define HELPERS_HPP
#endif

// Mac OSX
#ifdef __MACH__
#include <mach/clock.h>
#include <mach/mach.h>
#endif

#define gpuErrChk(ans){ gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line,
    bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code),
            file, line);
        if (abort) getchar();
    }
}

typedef int Face;
typedef long long int64;
typedef unsigned long long uint64;

const int C = 4, // size of the combination
    THREADS = 1024, // num of threads (set to 2^10)
    MAXV = 200, // max number of vertices
    MAXS = MAXV*MAXV, // dimension of the matrix
    MAXF = 2*MAXV-4, // upper-bound of regions on a planar graph
    MAXE = MAXV*(MAXV-1)/2; // number of undirected edges

/*
    range       ---> Number of combinations of an instance
    graph       ---> Adjacency matrix itself
    length      ---> Adjacency matrix dimension
    resFaces    ---> Set of triangular faces for the output
    */
struct Graph
{
    int range;
    int graph[MAXS], length;
    int resFaces[6*MAXV];
};

/*
    tmpMax      ---> Max. value obtained for an instance
    numFaces    ---> Number of faces on a MPG
    remaining   ---> Number of remaining vertices
    F           ---> Set of triangular faces
    V           ---> Set of remaining vertices
    */
struct Params
{
    int *tmpMax;
    int *numFaces, *remaining;
    int *F, *V;
};

/*
    COMB        ---> Total number of combinations
    SIZE        ---> Number of vertices
    FACES       ---> Number of faces
    GPU_CNT     ---> Number of GPUs (default: 1)
    numComb     ---> Size of each combination binom(V, 4)
    resGraph    ---> Output graph having the best solution
    F           ---> Set of triangular faces of the best solution
    G           ---> Instance of Graph, which will be copied to the GPU
    */

int COMB;
int64 numComb[MAXV];
int SIZE, FACES, GPU_CNT = 1;
int resGraph[MAXS], F[6*MAXV];

Graph *G;

/*
    Prints elapsed time.
    */
void printElapsedTime(double start, double stop)
{
    double elapsed = stop - start;
    printf("Elapsed time: %.3lfs.\n", elapsed);
}

//-----------------------------------------------------------------------------

/*  
    Gets clock time.
    */
void current_utc_time(struct timespec *ts) 
{
    // OS X does not have clock_gettime, use clock_get_time
    #ifdef __MACH__
        clock_serv_t cclock;
        mach_timespec_t mts;
        host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);
        clock_get_time(cclock, &mts);
        mach_port_deallocate(mach_task_self(), cclock);
        ts->tv_sec = mts.tv_sec;
        ts->tv_nsec = mts.tv_nsec;
    #else
        clock_gettime(CLOCK_REALTIME, ts);
    #endif
}

//-----------------------------------------------------------------------------

double getTime()
{
    timespec ts;
    current_utc_time(&ts);
    return double(ts.tv_sec) + double(ts.tv_nsec) / 1e9;
}

//-----------------------------------------------------------------------------

void initValues()
{
    for (int i = 0; i < SIZE-1; ++i)
        for (int j = i+1; j < SIZE; ++j)
            resGraph[i*SIZE + j] = resGraph[j*SIZE + i] = -1;

    COMB = numComb[SIZE-1];
    FACES = 2*SIZE-4;
}

//-----------------------------------------------------------------------------

void readInput()
{
    int value;
    scanf("%d", &SIZE);
    if (SIZE > MAXV) {
        printf("ERROR: Number of vertices (%d) exceeds the maximum allowed (%d)\n", SIZE, MAXV);
        exit(EXIT_FAILURE);
    }

    G = (Graph*)malloc(sizeof(Graph));
    G->length = SIZE;

    for (int i = 0; i < SIZE-1; ++i)
        for (int j = i+1; j < SIZE; ++j)
        {
            scanf("%d", &value);
            G->graph[i*SIZE + j] = G->graph[j*SIZE + i] = value;
        }
    initValues();
}

//-----------------------------------------------------------------------------

/*
    Defines the number of combinations.
    */
void sizeDefinitions()
{
    for (int i = 4; i <= MAXV; ++i)
    {
        int resp = 1;
        for (int64 j = i-3; j <= i; ++j) resp *= j;
        resp /= 24LL;
        numComb[i-1] = resp;
    }
}
