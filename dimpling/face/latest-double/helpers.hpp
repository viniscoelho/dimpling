#ifndef HELPERS_HPP
    #define HELPERS_HPP
    typedef long long int64;
    typedef unsigned long long uint64;
#endif

#define EPS 1e-9

#define gpuErrChk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, char* file, int line,
    bool abort = true)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code),
            file, line);
        if (abort)
            getchar();
    }
}

const int C = 4; // size of the combination
const int THREADS = 1024; // num of threads (set to 2^10)
const int MAX = 200; //max number of vertices
const int MAX_S = MAX*MAX; // dimension of the matrix

/*  perm        ---> Number of permutations of an instance
    sz          ---> Adjacency matrix dimension (1D)
    graph       ---> Adjacency matrix itself
    F_ANS       ---> Set of triangular faces for the output
    */
struct Node{
    int perm;
    int sz;
    double graph[MAX_S];
    int F_ANS[6 * MAX];
};

/*  faces       ---> Number of triangular faces
    count       ---> Number of remaining vertices
    tmpMax      ---> Max value obtained for a seed
    F           ---> Set of triangular faces
    V           ---> Set of remaining vertices
    */
struct Params{
    int *faces, *count;
    double *tmpMax;
    int *F, *V;
};

/*  PERM        ---> Number of permutations
    SIZE        ---> Number of vertices
    GPU_CNT     ---> Number of GPUs
    bib         ---> Size of each permutation
    R           ---> Output graph having the best solution
    F           ---> Set of triangular faces of the best solution
    N           ---> Instance of Node, which will be copied to the GPU
    */

int64 PERM;
int SIZE, GPU_CNT = 1;
int64 bib[MAX];
double R[MAX_S];
int F[6 * MAX];

Node *N;

/*  Prints elapsed time.
    */
void printElapsedTime(double start, double stop){
    double elapsed = stop - start;
    printf("Elapsed time: %.3lfs.\n", elapsed);
}

/*  Gets clock time.
    */
double getTime(){
     timespec ts;
     clock_gettime(CLOCK_REALTIME, &ts);
     return double(ts.tv_sec) + double(ts.tv_nsec) / 1e9;
}

void initialize(){
    for (int i = 0; i < SIZE-1; i++){
        for (int j = i+1; j < SIZE; j++){
            R[i*SIZE + j] = R[j*SIZE + i] = 0.0;
        }
    }
}

void readInput(){
    double x;
    scanf("%d", &SIZE);
    PERM = bib[SIZE-1];

    N = (Node*)malloc(sizeof(Node));
    N->sz = SIZE;

    for (int i = 0; i < SIZE; i++){
        for (int j = 0; j < SIZE; j++){
            scanf("%lf", &x);
            if (i >= j) continue;
            N->graph[i*SIZE + j] = N->graph[j*SIZE + i] = x;
        }
    }
}

/*  Defines the number of permutations and blocks.
    */
void sizeDefinitions(){
    for (int i = 4; i <= MAX; i++){
        int64 resp = 1;
        for (int j = i-3; j <= i; j++) resp *= j;
        resp /= 24;
        bib[i-1] = resp;
    }
}

__device__ int SGN(double a)
{
    return ((a > EPS) ? (1) : ((a < -EPS) ? (-1) : (0)));
}
__device__ int CMP(double a, double b) { return SGN(a - b); }

int hostSGN(double a) { return ((a > EPS) ? (1) : ((a < -EPS) ? (-1) : (0))); }
int hostCMP(double a, double b) { return hostSGN(a - b); }

__device__ void AtomicMax(double* const address, const double value)
{
    if (*address >= value)
        return;

    uint64* const address_as_i = (uint64*)address;
    uint64 old = *address_as_i, assumed;

    do {
        assumed = old;
        if (__longlong_as_double(assumed) >= value)
            break;
        old = atomicCAS(address_as_i, assumed, __double_as_longlong(value));
    } while (assumed != old);
}
