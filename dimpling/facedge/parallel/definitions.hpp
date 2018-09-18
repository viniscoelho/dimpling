#ifndef DEFINITIONS_H
#define DEFINITIONS_H
#endif

// registers*num_threads*num_blocks < 32k or 64k

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
typedef int Face;

const int C = 4; // combination size
const int THREADS = 1024; // number of threads (default: 2^10)
const int MAXV = 150; // max. number of vertices
const int MAXS = MAXV*MAXV; // matrix dimension
const int MAXF = 2*MAXV-4; // number of regions on a planar graph
const int MAXE = MAXV*(MAXV-1)/2; // number of edges
const int NUM_F_EDGE = 2; //number of faces that each edge belongs to
const int NUM_E_FACE = 3; //number of edges that each face has
const int NUM_V_FACE = 3; //number of vertices that each face has

//-----------------------------------------------------------------------------

__device__ struct node
{
    int w, vertex, edge, face, extra;
    __device__ node (int w = 0, int vertex = 0, int edge = 0, int face = 0, int extra = 0)
        : w(w), vertex(vertex), edge(edge), face(face), extra(extra) {}
};

//-----------------------------------------------------------------------------

__device__ struct edge
{
    int u, v;
    __device__ edge (int u_t = 0, int v_t = 0)
    {
        if (u_t < v_t)
        {
            u = u_t;
            v = v_t;
        }
        else
        {
            u = v_t;
            v = u_t;
        }
    }

    __device__ bool operator==(const edge& at)
    {
        return (u == at.u && v == at.v) || (v == at.u && u == at.v);
    }
};

//-----------------------------------------------------------------------------

/*
    A hash based on buckets
    */
__device__ class HashBucket
{
private:
    //for 200 vertices: (200*200)/62
    //for 150 vertices: (150*150)/62
    uint64 h[220];

public:
    __device__ HashBucket ()
    {
        for (int i = 0; i < 220; ++i) h[i] = 0LL;
    }

    __device__ bool find(int value)
    {
        int pos = value/62;
        return h[pos] & (1LL << (value % 63));
    }

    __device__ void insert(int value)
    {
        h[value/62] |= (1LL << (value % 63));
    }
};

//-----------------------------------------------------------------------------

/*
    range       ---> Number of combinations of an instance
    graph       ---> Adjacency matrix itself
    length      ---> Adjacency matrix dimension
    resFaces   ---> Set of triangular faces for the output
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
    numEdges    ---> Number of edges on a MPG
    remaining   ---> Number of remaining vertices
    F           ---> Set of triangular faces
    V           ---> Set of remaining vertices
    E           ---> Set of edges
    edges_faces ---> Set having the faces that each edge belongs to
    */
struct Params
{
    int *tmpMax, *numFaces, *numEdges, *remaining;
    int *F, *V, *E;
    int *edges_faces;
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
int64 COMB;
int64 numComb[MAXV];
int SIZE, FACES, GPU_CNT = 1;
int resGraph[MAXS], F[6*MAXV];

Graph *G;
