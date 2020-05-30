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

//this version is prepared for at most 70 vertices
struct Node {
    int sz, qtd;
    int graph[MAX * MAX], TMP[6 * MAX], seeds[PERM_MAX * C];
};

struct Params {
    int faces, count, tmpMax;
    int F[6 * MAX], V[MAX];
};

/*
    SIZE        ---> Number of vertices
    C           ---> Size of the combination (Size of a seed clique)
    faces       ---> Quantity of triangular faces
    qtd         ---> Number of possible 4-cliques
    T           ---> Output graph for an instance
    R           ---> Output graph for an possible optimal solution
    F           ---> List containing triangular faces of an instance
    seeds       ---> Permutations of possible starting 4-cliques
    graph       ---> The graph itself
*/

clock_t start, stop;
int R[MAX][MAX], F[8 * MAX], bib[MAX];
int SIZE, BLOCKS, PERM, qtd = 0;
Node* N;

//generates a list containing the vertices which are not
//on the planar graph
__device__ void generateList(Node* devN, Params* devP, int t)
{
    int sz = devN->sz;
    int va = devN->seeds[t * 4], vb = devN->seeds[t * 4 + 1], vc = devN->seeds[t * 4 + 2], vd = devN->seeds[t * 4 + 3];
    for (int i = 0; i < sz; i++) {
        if (i == va || i == vb || i == vc || i == vd) {
            devP[t].V[i] = -1;
        } else
            devP[t].V[i] = i;
    }
}

//returns the weight of the planar graph so far
__device__ void generateTriangularFaceList(Node* devN, Params* devP, int t)
{
    int resp = 0, sz = devN->sz;
    int va = devN->seeds[t * 4], vb = devN->seeds[t * 4 + 1], vc = devN->seeds[t * 4 + 2], vd = devN->seeds[t * 4 + 3];

    //generate first triangle of the output graph
    devP[t].F[devP[t].faces * 3] = va, devP[t].F[devP[t].faces * 3 + 1] = vb, devP[t].F[(devP[t].faces++) * 3 + 2] = vc;
    resp = devN->graph[va * sz + vb] + devN->graph[va * sz + vc] + devN->graph[vb * sz + vc];

    //generate the next 3 possible faces
    devP[t].F[devP[t].faces * 3] = va, devP[t].F[devP[t].faces * 3 + 1] = vb, devP[t].F[(devP[t].faces++) * 3 + 2] = vd;
    devP[t].F[devP[t].faces * 3] = va, devP[t].F[devP[t].faces * 3 + 1] = vc, devP[t].F[(devP[t].faces++) * 3 + 2] = vd;
    devP[t].F[devP[t].faces * 3] = vb, devP[t].F[devP[t].faces * 3 + 1] = vc, devP[t].F[(devP[t].faces++) * 3 + 2] = vd;
    resp += devN->graph[va * sz + vd] + devN->graph[vb * sz + vd] + devN->graph[vc * sz + vd];
    devP[t].tmpMax = resp;
}

//insert a new vertex, 3 new triangular faces
//and removes face 'f' from the list
__device__ int operationT2(Node* devN, Params* devP, int new_vertex, int f, int t)
{
    //remove the chosen face and insert a new one
    int va = devP[t].F[f * 3], vb = devP[t].F[f * 3 + 1], vc = devP[t].F[f * 3 + 2];

    devP[t].F[f * 3] = new_vertex, devP[t].F[f * 3 + 1] = va, devP[t].F[f * 3 + 2] = vb;
    //and insert the other two possible faces
    devP[t].F[devP[t].faces * 3] = new_vertex, devP[t].F[devP[t].faces * 3 + 1] = va, devP[t].F[(devP[t].faces++) * 3 + 2] = vc;
    devP[t].F[devP[t].faces * 3] = new_vertex, devP[t].F[devP[t].faces * 3 + 1] = vb, devP[t].F[(devP[t].faces++) * 3 + 2] = vc;

    int sz = devN->sz;
    int resp = devN->graph[va * sz + new_vertex] + devN->graph[vb * sz + new_vertex] + devN->graph[vc * sz + new_vertex];

    return resp;
}

//return the vertex with the maximum gain inserting within a face 'f'
__device__ int maxGain(Node* devN, Params* devP, int* f, int t)
{
    int sz = devN->sz;
    int gain = -1, vertex = -1;
    //iterate through the remaining vertices
    for (int new_vertex = 0; new_vertex < sz; new_vertex++) {
        if (devP[t].V[new_vertex] == -1)
            continue;
        //and test which has the maximum gain with its insetion
        //within all possible faces
        int faces = devP[t].faces;
        for (int i = 0; i < faces; i++) {
            int va = devP[t].F[i * 3], vb = devP[t].F[i * 3 + 1], vc = devP[t].F[i * 3 + 2];
            int tmpGain = devN->graph[va * sz + new_vertex] + devN->graph[vb * sz + new_vertex] + devN->graph[vc * sz + new_vertex];
            if (tmpGain > gain) {
                gain = tmpGain;
                *f = i;
                vertex = new_vertex;
            }
        }
    }
    return vertex;
}

__device__ void tmfg(Node* devN, Params* devP, int t)
{
    while (devP[t].count) {
        int f = -1;
        int vertex = maxGain(devN, devP, &f, t);
        devP[t].V[vertex] = -1;
        devP[t].tmpMax += operationT2(devN, devP, vertex, f, t);
        devP[t].count--;
    }
}

__device__ void initializeDevice(Params* devP, int sz, int t)
{
    devP[t].faces = 0;
    devP[t].tmpMax = -1;
    devP[t].count = sz - 4;
}

__global__ void tmfgParallel(Node* devN, Params* devP, int* respMax, int* idx)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int sz = devN->sz;
    int perm = devN->qtd;
    __syncthreads();

    if (x < perm) {
        initializeDevice(devP, devN->sz, x);
        generateList(devN, devP, x);
        generateTriangularFaceList(devN, devP, x);
        tmfg(devN, devP, x);

        __syncthreads();
        atomicMax(respMax, devP[x].tmpMax);
        if (devP[x].tmpMax == *respMax)
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
    Params* devP;

    size_t sz = PERM * sizeof(Params);
    gpuErrChk(cudaMalloc((void**)&devP, sz));
    gpuErrChk(cudaMalloc((void**)&devN, sizeof(Node)));
    gpuErrChk(cudaMemcpy(devN, N, sizeof(Node), cudaMemcpyHostToDevice));

    dim3 blocks(BLOCKS, 1);
    dim3 threads(THREADS, 1);

    tmfgParallel<<<blocks, threads>>>(devN, devP, tmpResp, tmpIdx);
    gpuErrChk(cudaDeviceSynchronize());

    gpuErrChk(cudaMemcpy(&resp, tmpResp, sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrChk(cudaMemcpy(&idx, tmpIdx, sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrChk(cudaMemcpy(&F, devP[idx].F, (6 * MAX) * sizeof(int), cudaMemcpyDeviceToHost));

    gpuErrChk(cudaFree(devN));
    gpuErrChk(cudaFree(devP));
    return resp;
}

/*
    C      ---> Size of the combination
    index  ---> Current index in data[]
    data[] ---> Temporary array to store a current combination
    i      ---> Index of current element in vertices[]
*/
void combineUntil(int index, vector<int>& data, int i)
{
    // Current cobination is ready, print it
    if (index == C) {
        for (int j = 0; j < C; j++) {
            N->seeds[qtd * C + j] = data[j];
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

void combine()
{
    vector<int> data(C);
    //print all combinations of size 'r' using a temporary array 'data'
    combineUntil(0, data, 0);
}

//define the size of permutations and number of blocks
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

void initialize()
{
    for (int i = 0; i < SIZE; i++) {
        for (int j = i + 1; j < SIZE; j++) {
            R[i][j] = R[j][i] = -1;
        }
        R[i][i] = -1;
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
    N->qtd = PERM;

    for (int i = 0; i < SIZE; i++) {
        for (int j = i + 1; j < SIZE; j++) {
            cin >> x;
            N->graph[i * SIZE + j] = x;
            N->graph[j * SIZE + i] = x;
        }
        N->graph[i * SIZE + i] = -1;
    }
}

void printElapsedTime(clock_t start, clock_t stop)
{
    double elapsed = ((double)(stop - start)) / CLOCKS_PER_SEC;
    cout << fixed << setprecision(3) << "Elapsed time: " << elapsed << "s\n";
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

    start = clock();
    int respMax = tmfgPrepare();
    stop = clock();

    for (int i = 0; i < 2 * SIZE; i++) {
        int va = F[i * 3], vb = F[i * 3 + 1], vc = F[i * 3 + 2];
        if (va == vb && vb == vc)
            continue;
        R[va][vb] = R[vb][va] = N->graph[va * SIZE + vb];
        R[va][vc] = R[vc][va] = N->graph[va * SIZE + vc];
        R[vb][vc] = R[vc][vb] = N->graph[vb * SIZE + vc];
    }

    cout << "Printing generated graph: " << endl;
    for (int i = 0; i < SIZE; i++) {
        for (int j = i + 1; j < SIZE; j++) {
            cout << R[i][j] << " ";
        }
        cout << endl;
    }

    printElapsedTime(start, stop);
    cout << "Maximum weight found: " << respMax << endl;
    free(N);

    return 0;
}