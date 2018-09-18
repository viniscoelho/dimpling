#include <iostream>
#include <iomanip>
#include <algorithm>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <omp.h>

#define C 4
#define THREADS 1024 // 2^10
// #define MAX 85
#define MAX 110 // for multi-gpu only
#define MAX_S MAX*MAX
#define PERM_MAX (MAX*(MAX-1)*(MAX-2)*(MAX-3))/24
#define pb push_back
#define mp make_pair

#define gpuErrChk(ans){ gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort = true){
    if (code != cudaSuccess){
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) getchar();
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
struct Node{
    int sz, perm;
    int graph[MAX_S], seeds[C*PERM_MAX], F_ANS[6 * MAX];
};

/*
    faces       ---> Number of triangular faces
    count       ---> Number of remaining vertices
    tmpMax      ---> Max value obtained for a seed
    F           ---> Set of triangular faces
    F           ---> Set of remaining vertices
*/
struct Params{
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
int SIZE, PERM, GPU_CNT = 1;
int R[MAX_S], F[8 * MAX], bib[MAX];
int qtd = 0;
Node *N;

/*
    Generates a list containing the vertices which are not on the planar graph
*/
__device__ void generateList(Node* devN, Params* devP, int t, int offset){
    int sz = devN->sz, perm = devN->perm;

    int va = devN->seeds[(t + offset)* 4],
        vb = devN->seeds[(t + offset)* 4 + 1],
        vc = devN->seeds[(t + offset)* 4 + 2],
        vd = devN->seeds[(t + offset)* 4 + 3];
    for (int i = 0; i < sz; i++){
        if (i == va || i == vb || i == vc || i == vd) devP->V[t + i * perm] = -1;
        else devP->V[t + i * perm] = i;
    }
}

/*
    Returns the weight of the planar graph so far
*/
__device__ void generateTriangularFaceList(Node* devN, Params* devP, int graph[], int t, int offset){
    int sz = devN->sz, perm = devN->perm;

    int va = devN->seeds[(t + offset)* 4],
        vb = devN->seeds[(t + offset)* 4 + 1],
        vc = devN->seeds[(t + offset)* 4 + 2],
        vd = devN->seeds[(t + offset)* 4 + 3];

    //generate first triangle of the output graph
    devP->F[t + (devP->faces[t] * 3) * perm] = va;
    devP->F[t + (devP->faces[t] * 3 + 1) * perm] = vb;
    devP->F[t + ((devP->faces[t]++) * 3 + 2) * perm] = vc;

    //generate the next 3 possible faces
    devP->F[t + (devP->faces[t] * 3) * perm] = va;
    devP->F[t + (devP->faces[t] * 3 + 1) * perm] = vb;
    devP->F[t + ((devP->faces[t]++) * 3 + 2) * perm] = vd;

    devP->F[t + (devP->faces[t] * 3) * perm] = va;
    devP->F[t + (devP->faces[t] * 3 + 1) * perm] = vc;
    devP->F[t + ((devP->faces[t]++) * 3 + 2) * perm] = vd;

    devP->F[t + (devP->faces[t] * 3) * perm] = vb;
    devP->F[t + (devP->faces[t] * 3 + 1) * perm] = vc;
    devP->F[t + ((devP->faces[t]++) * 3 + 2) * perm] = vd;

    int resp = graph[va*sz + vb] + graph[va*sz + vc] + graph[vb*sz + vc];
    resp += graph[va*sz + vd] + graph[vb*sz + vd] + graph[vc*sz + vd];
    devP->tmpMax[t] = resp;
}

/*
    Insert a new vertex, 3 new triangular faces and removes face 'f' from the set
*/
__device__ int operationT2(Node* devN, Params* devP, int graph[], int new_vertex, int f, int t){
    int sz = devN->sz, perm = devN->perm;

    //remove the chosen face and insert a new one
    int va = devP->F[t + (f * 3) * perm],
        vb = devP->F[t + (f * 3 + 1) * perm],
        vc = devP->F[t + (f * 3 + 2) * perm];

    devP->F[t + (f * 3) * perm] = new_vertex,
    devP->F[t + (f * 3 + 1) * perm] = va,
    devP->F[t + (f * 3 + 2) * perm] = vb;
    
    //and insert the other two possible faces
    devP->F[t + (devP->faces[t] * 3) * perm] = new_vertex;
    devP->F[t + (devP->faces[t] * 3 + 1) * perm] = va;
    devP->F[t + ((devP->faces[t]++) * 3 + 2) * perm] = vc;

    devP->F[t + (devP->faces[t] * 3) * perm] = new_vertex;
    devP->F[t + (devP->faces[t] * 3 + 1) * perm] = vb;
    devP->F[t + ((devP->faces[t]++) * 3 + 2) * perm] = vc;

    int resp = graph[va*sz + new_vertex] + graph[vb*sz + new_vertex] + graph[vc*sz + new_vertex];

    return resp;
}

/*
    Return the vertex with the maximum gain inserting within a face 'f'
*/
__device__ int maxGain(Node* devN, Params* devP, int graph[], int* f, int t){
    int sz = devN->sz, perm = devN->perm;
    int gain = -1, vertex = -1;

    //iterate through the remaining vertices
    for (int new_vertex = 0; new_vertex < sz; new_vertex++){
        if (devP->V[t + new_vertex * perm] == -1) continue;
        //and test which has the maximum gain with its insetion
        //within all possible faces
        int faces = devP->faces[t];
        for (int i = 0; i < faces; i++){
            int va = devP->F[t + (i * 3) * perm], vb = devP->F[t + (i * 3 + 1) * perm], vc = devP->F[t + (i * 3 + 2) * perm];
            int tmpGain = graph[va*sz + new_vertex] + graph[vb*sz + new_vertex] + graph[vc*sz + new_vertex];
            if (tmpGain > gain){
                gain = tmpGain;
                *f = i;
                vertex = new_vertex;
            }
        }
    }
    return vertex;
}

__device__ void tmfg(Node* devN, Params* devP, int graph[], int t){
    int perm = devN->perm;
    while (devP->count[t]){
        int f = -1;
        int vertex = maxGain(devN, devP, graph, &f, t);
        devP->V[t + vertex * perm] = -1;
        devP->tmpMax[t] += operationT2(devN, devP, graph, vertex, f, t);
        devP->count[t]--;
    }
}

__device__ void copyGraph(Node *devN, Params *devP, int t){
    int faces = devP->faces[t], perm = devN->perm;
    for (int i = 0; i < faces; i++){
        int va = devP->F[t + (i * 3) * perm], vb = devP->F[t + (i * 3 + 1) * perm], vc = devP->F[t + (i * 3 + 2) * perm];
        devN->F_ANS[i * 3] = va, devN->F_ANS[i * 3 + 1] = vb, devN->F_ANS[i * 3 + 2] = vc;
    }
}

__device__ void initializeDevice(Params *devP, int sz, int t){
    devP->faces[t] = 0;
    devP->tmpMax[t] = -1;
    devP->count[t] = sz - 4;
}

__global__ void tmfgParallel(Node *devN, Params devP, int *respMax, int offset){
    int x = blockDim.x*blockIdx.x + threadIdx.x;
    int sz = devN->sz, perm = devN->perm;
    extern __shared__ int graph[];

    for (int i = threadIdx.x; i < sz*sz; i += blockDim.x)
        graph[i] = devN->graph[i];
    __syncthreads();

    if (x < perm){
        initializeDevice(&devP, sz, x);
        generateList(devN, &devP, x, offset);
        generateTriangularFaceList(devN, &devP, graph, x, offset);
        tmfg(devN, &devP, graph, x);
        atomicMax(respMax, devP.tmpMax[x]);
        __syncthreads();

        if (devP.tmpMax[x] == *respMax){
            copyGraph(devN, &devP, x);
        }
        __syncthreads();
    }
}

int tmfgPrepare(){
    int finalResp = -1, pos = -1;

#pragma omp parallel for num_threads(GPU_CNT)
    for (int gpu_id = 0; gpu_id < GPU_CNT; gpu_id++)
    {
        cudaSetDevice(gpu_id);
        int range = (int)ceil(PERM / (double)GPU_CNT);
        int perm = ((gpu_id + 1)*range > PERM ? PERM - gpu_id*range : range);
        int offset = gpu_id*range;
        N->perm = perm;

        int resp = -1, *tmpResp;
        gpuErrChk(cudaMalloc((void**)&tmpResp, sizeof(int)));
        gpuErrChk(cudaMemcpy(tmpResp, &resp, sizeof(int), cudaMemcpyHostToDevice));

        Node *devN;
        Params devP;

        size_t sz = range*sizeof(int)*3 + range*sizeof(int)*SIZE + range*sizeof(int)*(6*SIZE);

        printf("Using %d mbytes in Kernel %d\n", (sz + sizeof(Node)) / (1 << 20), gpu_id);
        fprintf(stderr, "Using %d mbytes in Kernel %d\n", (sz + sizeof(Node)) / (1 << 20), gpu_id);

        gpuErrChk(cudaMalloc((void**)&devN, sizeof(Node)));
        gpuErrChk(cudaMemcpy(devN, N, sizeof(Node), cudaMemcpyHostToDevice));

        gpuErrChk(cudaMalloc((void**)&devP.faces, perm*sizeof(int)));
        gpuErrChk(cudaMalloc((void**)&devP.count, perm*sizeof(int)));
        gpuErrChk(cudaMalloc((void**)&devP.tmpMax, perm*sizeof(int)));
        gpuErrChk(cudaMalloc((void**)&devP.F, 6*SIZE*perm*sizeof(int)));
        gpuErrChk(cudaMalloc((void**)&devP.V, SIZE*perm*sizeof(int)));

        dim3 blocks(perm / THREADS + 1, 1);
        dim3 threads(THREADS, 1);

        printf("Kernel %d launched with %d blocks, each w/ %d threads\n", gpu_id, range / THREADS + 1, THREADS);
        fprintf(stderr, "Kernel %d launched with %d blocks, each w/ %d threads\n", gpu_id, range / THREADS + 1, THREADS);
        tmfgParallel <<<blocks, threads, SIZE*SIZE*sizeof(int)>>>(devN, devP, tmpResp, offset);
        gpuErrChk(cudaDeviceSynchronize());

        //copy back the maximum weight and the set of faces
        //which gave this result
        gpuErrChk(cudaMemcpy(&resp, tmpResp, sizeof(int), cudaMemcpyDeviceToHost));
        printf("Kernel finished.\nLocal maximum found in Kernel %d: %d\n", gpu_id, resp);

        printf("Copying results...\n");
        #pragma omp critical
        {
            if (resp > finalResp){
                finalResp = resp;
                pos = gpu_id;
            }
        }
        // #pragma omp barrier

        if (pos == gpu_id){
            gpuErrChk(cudaMemcpy(&F, devN->F_ANS, 6 * MAX*sizeof(int), cudaMemcpyDeviceToHost));
        }

        printf("Freeing memory...\n");
        gpuErrChk(cudaFree(devN));
        gpuErrChk(cudaFree(devP.faces));
        gpuErrChk(cudaFree(devP.count));
        gpuErrChk(cudaFree(devP.tmpMax));
        gpuErrChk(cudaFree(devP.F));
        gpuErrChk(cudaFree(devP.V));

        gpuErrChk(cudaDeviceReset());

    }

    return finalResp;
}

void printElapsedTime(double start, double stop){
    double elapsed = stop - start;
    printf("Elapsed time: %.3lfs.\n", elapsed);
}

double getTime(){
    timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return double(ts.tv_sec) + double(ts.tv_nsec) / 1e9;
}

/*
    C           ---> Size of the combination
    index       ---> Current index in data[]
    data[]      ---> Temporary array to store a current combination
    i           ---> Index of current element in vertices[]
    */
void combineUntil(int index, vector<int>& data, int i){
    // Current cobination is ready, print it
    if ( index == C ){
        for ( int j = 0; j < C; j++ ){
            N->seeds[qtd*C + j] = data[j];
        }
        qtd++;
        return;
    }
 
    // When there are no more elements to put in data[]
    if ( i >= SIZE ) return;
 
    //current is inserted; put next at a next location
    data[index] = i;
    combineUntil(index+1, data, i+1);

    //current is deleted; replace it with next
    combineUntil(index, data, i+1);
}

void combine(){
    vector<int> data(C);
    //print all combinations of size 'r' using a temporary array 'data'
    combineUntil(0, data, 0);
}

void initialize(){
    for (int i = 0; i < SIZE; i++){
        for (int j = i + 1; j < SIZE; j++){
            R[i*SIZE + j] = R[j*SIZE + i] = -1;
        }
    }
}

void readInput(){
    int x;
    cin >> SIZE;
    PERM = bib[SIZE - 1];

    N = (Node*)malloc(sizeof(Node));
    N->sz = SIZE;

    for (int i = 0; i < SIZE; i++){
        for (int j = i + 1; j < SIZE; j++){
            cin >> x;
            N->graph[i*SIZE + j] = x;
            N->graph[j*SIZE + i] = x;
        }
    }
}

/*
    Define the number of permutations and blocks
    */
void sizeDefinitions(){
    for (int i = 6; i <= MAX; i++){
        int resp = 1;
        for (int j = i - 3; j <= i; j++) resp *= j;
        resp /= 24;
        bib[i - 1] = resp;
    }
}

int main(int argv, char** argc){
    ios::sync_with_stdio(false);
    sizeDefinitions();
    //read the input, which is given by a size of a graph and its weighted edges.
    //the given graph is dense.
    readInput();
    initialize();
    //given the number of vertices, generate multiple 4-clique seeds
    combine();

    if (argv == 2){
        cudaSetDevice(atoi(argc[1]));
    }
    else if (argv == 3){
        GPU_CNT = atoi(argc[2]);
        int d;
        cudaGetDeviceCount(&d);
        if (GPU_CNT > d) GPU_CNT = d;
    }
    //cout << gpuCount << endl;
    //cudaSetDevice(dv);

    double start = getTime();
    int respMax = tmfgPrepare();
    double stop = getTime();

    //reconstruct the graph given the regions of the graph
    for (int i = 0; i < 2 * SIZE; i++){
        int va = F[i * 3], vb = F[i * 3 + 1], vc = F[i * 3 + 2];
        if (va == vb && vb == vc) continue;
        R[va*SIZE + vb] = R[vb*SIZE + va] = N->graph[va*SIZE + vb];
        R[va*SIZE + vc] = R[vc*SIZE + va] = N->graph[va*SIZE + vc];
        R[vb*SIZE + vc] = R[vc*SIZE + vb] = N->graph[vb*SIZE + vc];
    }

    cout << "Printing generated graph: " << endl;
    for (int i = 0; i < SIZE; i++){
        for (int j = i + 1; j < SIZE; j++){
            cout << R[i*SIZE + j] << " ";
        }
        cout << endl;
    }

    printElapsedTime(start, stop);
    cout << "Maximum weight found: " << respMax << endl;
    free(N);


    return 0;
}