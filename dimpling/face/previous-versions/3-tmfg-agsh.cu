#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#define C 4
#define THREADS 1024 // 2^10
#define MAX 85
#define MAX_S MAX* MAX
#define PERM_MAX (MAX * (MAX - 1) * (MAX - 2) * (MAX - 3)) / 24
#define pb push_back
#define mp make_pair

#define gpuErrChk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, char* file, int line, bool abort = true)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            getchar();
    }
}

using namespace std;

typedef long long int64;
typedef pair<int, int> ii;

/*
    sz          ---> Adjacency matrix dimension (1D)
    perm        ---> Number of permutations of an instance
    graph       ---> Adjacency matrix itself
    seeds       ---> Set of seeds
    faces       ---> Set of triangular faces for the output
*/
struct Node {
    int sz, perm;
    int graph[MAX_S], seeds[C * PERM_MAX], F_ANS[6 * MAX];
};

/*
    faces       ---> Number of triangular faces
    count       ---> Number of remaining vertices
    tmpMax      ---> Max value obtained for a seed
    F           ---> Set of triangular faces
    F           ---> Set of remaining vertices
*/
struct Params {
    int *faces, *count, *tmpMax;
    int *F, *V;
};

/*
    SIZE        ---> Number of vertices
    BLOCKS      ---> Number of blocks
    PERM        ---> Number of permutations
    R           ---> Output graph for a possible solution
    F           ---> Set of triangular faces of an instance
    qtd         ---> Number of possible 4-cliques
*/
clock_t start, stop;
int SIZE, BLOCKS, PERM, qtd = 0;
int R[MAX_S], F[8 * MAX], bib[MAX];
Node* N;

__device__ void initializeDevice(Params* devP, int sz, int t)
{
    devP->faces[t] = 0;
    devP->tmpMax[t] = -1;
    devP->count[t] = sz - 4;
}

/*
    Generates a list containing the vertices which are not on the planar graph
*/
__device__ void generateList(Node* devN, Params* devP, int t)
{
    int sz = devN->sz;
    int va = devN->seeds[t], vb = devN->seeds[t + devN->perm], vc = devN->seeds[t + 2 * devN->perm], vd = devN->seeds[t + 3 * devN->perm];
    for (int i = 0; i < sz; i++) {
        if (i == va || i == vb || i == vc || i == vd)
            devP->V[t + i * devN->perm] = -1;
        else
            devP->V[t + i * devN->perm] = i;
    }
}

/*
    Returns the weight of the planar graph so far
*/
__device__ void generateTriangularFaceList(Node* devN, Params* devP, int graph[], int t)
{
    int sz = devN->sz;
    int va = devN->seeds[t];
    int vb = devN->seeds[t + devN->perm];
    int vc = devN->seeds[t + 2 * devN->perm];
    int vd = devN->seeds[t + 3 * devN->perm];

    //generate first triangle of the output graph
    devP->F[t + (devP->faces[t] * 3) * devN->perm] = va;
    devP->F[t + (devP->faces[t] * 3 + 1) * devN->perm] = vb;
    devP->F[t + ((devP->faces[t]++) * 3 + 2) * devN->perm] = vc;
    int resp = graph[va * sz + vb] + graph[va * sz + vc] + graph[vb * sz + vc];

    //generate the next 3 possible faces
    devP->F[t + (devP->faces[t] * 3) * devN->perm] = va;
    devP->F[t + (devP->faces[t] * 3 + 1) * devN->perm] = vb;
    devP->F[t + ((devP->faces[t]++) * 3 + 2) * devN->perm] = vd;

    devP->F[t + (devP->faces[t] * 3) * devN->perm] = va;
    devP->F[t + (devP->faces[t] * 3 + 1) * devN->perm] = vc;
    devP->F[t + ((devP->faces[t]++) * 3 + 2) * devN->perm] = vd;

    devP->F[t + (devP->faces[t] * 3) * devN->perm] = vb;
    devP->F[t + (devP->faces[t] * 3 + 1) * devN->perm] = vc;
    devP->F[t + ((devP->faces[t]++) * 3 + 2) * devN->perm] = vd;
    resp += graph[va * sz + vd] + graph[vb * sz + vd] + graph[vc * sz + vd];
    devP->tmpMax[t] = resp;
}

/*
    Insert a new vertex, 3 new triangular faces and removes face 'f' from the set
*/
__device__ int operationT2(Node* devN, Params* devP, int graph[], int new_vertex, int f, int t)
{
    int sz = devN->sz, perm = devN->perm;
    //remove the chosen face and insert a new one
    int va = devP->F[t + (f * 3) * perm];
    int vb = devP->F[t + (f * 3 + 1) * perm];
    int vc = devP->F[t + (f * 3 + 2) * perm];

    devP->F[t + (f * 3) * perm] = new_vertex;
    devP->F[t + (f * 3 + 1) * perm] = va;
    devP->F[t + (f * 3 + 2) * perm] = vb;

    //and insert the other two possible faces
    devP->F[t + (devP->faces[t] * 3) * perm] = new_vertex;
    devP->F[t + (devP->faces[t] * 3 + 1) * perm] = va;
    devP->F[t + ((devP->faces[t]++) * 3 + 2) * perm] = vc;

    devP->F[t + (devP->faces[t] * 3) * perm] = new_vertex;
    devP->F[t + (devP->faces[t] * 3 + 1) * perm] = vb;
    devP->F[t + ((devP->faces[t]++) * 3 + 2) * perm] = vc;

    int resp = graph[va * sz + new_vertex] + graph[vb * sz + new_vertex] + graph[vc * sz + new_vertex];

    return resp;
}

/*
    Return the vertex with the maximum gain inserting within a face 'f'
*/
__device__ int maxGain(Node* devN, Params* devP, int graph[], int* f, int t)
{
    int sz = devN->sz, perm = devN->perm;
    int gain = -1, vertex = -1;
    //iterate through the remaining vertices
    for (int new_vertex = 0; new_vertex < sz; new_vertex++) {
        if (devP->V[t + new_vertex * perm] == -1)
            continue;
        //and test which has the maximum gain with its insetion
        //within all possible faces
        int faces = devP->faces[t];
        for (int i = 0; i < faces; i++) {
            int va = devP->F[t + (i * 3) * perm], vb = devP->F[t + (i * 3 + 1) * perm], vc = devP->F[t + (i * 3 + 2) * perm];
            int tmpGain = graph[va * sz + new_vertex] + graph[vb * sz + new_vertex] + graph[vc * sz + new_vertex];
            if (tmpGain > gain) {
                gain = tmpGain;
                *f = i;
                vertex = new_vertex;
            }
        }
    }
    return vertex;
}

__device__ void tmfg(Node* devN, Params* devP, int graph[], int t)
{
    while (devP->count[t]) {
        int f = -1;
        int vertex = maxGain(devN, devP, graph, &f, t);
        devP->V[t + vertex * devN->perm] = -1;
        devP->tmpMax[t] += operationT2(devN, devP, graph, vertex, f, t);
        devP->count[t]--;
    }
}

__global__ void tmfgParallel(Node* devN, Params devP, int* respMax, int* idx)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int sz = devN->sz, perm = devN->perm;
    extern __shared__ int graph[];

    for (int i = threadIdx.x; i < sz * sz; i += blockDim.x) {
        graph[i] = devN->graph[i];
        graph[i] = devN->graph[i];
    }
    __syncthreads();

    if (x < perm) {
        initializeDevice(&devP, sz, x);
        generateList(devN, &devP, x);
        generateTriangularFaceList(devN, &devP, graph, x);
        tmfg(devN, &devP, graph, x);

        __syncthreads();
        atomicMax(respMax, devP.tmpMax[x]);

        if (devP.tmpMax[x] == *respMax)
            *idx = x;
        __syncthreads();
    }
}

int tmfgPrepare()
{
    int resp = 0, idx = 0, *tmpResp, *tmpIdx;
    gpuErrChk(cudaMalloc((void**)&tmpResp, sizeof(int)));
    gpuErrChk(cudaMalloc((void**)&tmpIdx, sizeof(int)));
    gpuErrChk(cudaMemcpy(tmpResp, &resp, sizeof(int), cudaMemcpyHostToDevice));
    gpuErrChk(cudaMemcpy(tmpIdx, &idx, sizeof(int), cudaMemcpyHostToDevice));

    Node* devN;
    Params devP;

    cout << "Amount of memory: " << (3 * PERM + PERM * SIZE + 6 * SIZE * PERM * sizeof(int)) / 1000000 << "MB" << endl;

    gpuErrChk(cudaMalloc((void**)&devN, sizeof(Node)));
    gpuErrChk(cudaMemcpy(devN, N, sizeof(Node), cudaMemcpyHostToDevice));
    cout << "1 done." << endl;

    gpuErrChk(cudaMalloc((void**)&devP.faces, PERM * sizeof(int)));
    gpuErrChk(cudaMalloc((void**)&devP.count, PERM * sizeof(int)));
    gpuErrChk(cudaMalloc((void**)&devP.tmpMax, PERM * sizeof(int)));
    gpuErrChk(cudaMalloc((void**)&devP.F, PERM * 6 * SIZE * sizeof(int)));
    gpuErrChk(cudaMalloc((void**)&devP.V, PERM * SIZE * sizeof(int)));
    cout << "2 done." << endl;

    dim3 blocks(BLOCKS, 1);
    dim3 threads(THREADS, 1);

    cout << "Launching kernel..." << endl;
    tmfgParallel<<<blocks, threads, SIZE * SIZE * sizeof(int)>>>(devN, devP, tmpResp, tmpIdx);
    gpuErrChk(cudaDeviceSynchronize());
    cout << "Kernel finished." << endl;

    //copy back the maximum weight and the index of the graph
    //which gave this result
    gpuErrChk(cudaMemcpy(&resp, tmpResp, sizeof(int), cudaMemcpyDeviceToHost));
    cout << "1 done." << endl;
    gpuErrChk(cudaMemcpy(&idx, tmpIdx, sizeof(int), cudaMemcpyDeviceToHost));
    cout << "2 done." << endl;
    //gpuErrChk(cudaMemcpy(&F, devP.F[idx + ], (6*MAX)*sizeof(int), cudaMemcpyDeviceToHost));
    cout << "3 done." << endl;

    gpuErrChk(cudaFree(devN));
    gpuErrChk(cudaFree(devP.faces));
    gpuErrChk(cudaFree(devP.count));
    gpuErrChk(cudaFree(devP.tmpMax));
    gpuErrChk(cudaFree(devP.F));
    gpuErrChk(cudaFree(devP.V));
    cout << "Completed." << endl;

    return resp;
}

void printElapsedTime(clock_t start, clock_t stop)
{
    double elapsed = ((double)(stop - start)) / CLOCKS_PER_SEC;
    cout << fixed << setprecision(3) << "Elapsed time: " << elapsed << "s\n";
}

/*
    C           ---> Size of the combination
    index       ---> Current index in data[]
    data[]      ---> Temporary array to store a current combination
    i           ---> Index of current element in vertices[]
*/
void combineUntil(int index, vector<int>& data, int i)
{
    // Current cobination is ready, print it
    if (index == C) {
        for (int j = 0; j < C; j++) {
            N->seeds[qtd + j * PERM] = data[j];
        }
        qtd++;
        return;
    }

    // When there are no more elements to put in data[]
    if (i >= SIZE)
        return;
    //current is inserted; put next at a next location
    data[index] = i;
    combineUntil(index + 1, data, i + 1);
    //current is deleted; replace it with next
    combineUntil(index, data, i + 1);
}

/*
    Print all combinations of size 'C' using a temporary array 'data'
*/
void combine()
{
    vector<int> data(C);
    combineUntil(0, data, 0);
}

void initialize()
{
    for (int i = 0; i < SIZE; i++) {
        for (int j = i + 1; j < SIZE; j++) {
            R[i * SIZE + j] = R[j * SIZE + i] = -1;
        }
    }
}

void readInput()
{
    int x;
    cin >> SIZE;
    PERM = bib[SIZE - 1];
    BLOCKS = PERM / THREADS + 1;

    N = (Node*)malloc(sizeof(Node));
    N->sz = SIZE;
    N->perm = PERM;

    for (int i = 0; i < SIZE; i++) {
        for (int j = i + 1; j < SIZE; j++) {
            cin >> x;
            N->graph[i * SIZE + j] = x;
            N->graph[j * SIZE + i] = x;
        }
    }
}

/*
    Define the number of permutations and blocks
*/
void sizeDefinitions()
{
    for (int i = 6; i <= MAX; i++) {
        int resp = 1;
        for (int j = i - 3; j <= i; j++)
            resp *= j;
        resp /= 24;
        bib[i - 1] = resp;
    }
}

int main(int argv, char** argc)
{
    ios::sync_with_stdio(false);
    sizeDefinitions();
    //read the input, which is given by a size of a graph and its weighted edges.
    //the graph given is dense.
    readInput();
    initialize();
    //generate multiple 4-clique seeds, given the number of vertices
    combine();

    cudaSetDevice(3);

    start = clock();
    int respMax = tmfgPrepare();
    stop = clock();

    //reconstruct the graph given the regions of the graph
    // for ( int i = 0; i < 2*SIZE; i++ ){
    //     int va = F[i*3], vb = F[i*3 + 1], vc = F[i*3 + 2];
    //     if ( va == vb && vb == vc ) continue;
    //     R[va*SIZE + vb] = R[vb*SIZE + va] = N->graph[va*SIZE + vb];
    //     R[va*SIZE + vc] = R[vc*SIZE + va] = N->graph[va*SIZE + vc];
    //     R[vb*SIZE + vc] = R[vc*SIZE + vb] = N->graph[vb*SIZE + vc];
    // }

    // cout << "Printing generated graph: " << endl;
    // for ( int i = 0; i < SIZE; i++ ){
    //     for ( int j = i+1; j < SIZE; j++ ){
    //         cout << R[i*SIZE + j] << " ";
    //     }
    //     cout << endl;
    // }

    printElapsedTime(start, stop);
    cout << "Maximum weight found: " << respMax << endl;
    free(N);
    gpuErrChk(cudaDeviceReset());

    return 0;
}