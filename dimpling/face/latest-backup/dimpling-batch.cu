#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <omp.h>
#define MB (1 << 20)

#include "default.hpp"
#include "combinadic.hpp"

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

using namespace std;

// Combinadic instance shared on the GPU
__shared__ Combination c;

//-----------------------------------------------------------------------------

/*
    Generates a list having vertices which are not on the planar graph.
    */
__device__ void generateVertexList(Graph *devG, Params *devP,
    int t, int offset)
{
    int len = devG->length;
    int range = devG->range;

    int *seeds = c.element(t + offset).getArray();

    int va = seeds[0], vb = seeds[1], vc = seeds[2], vd = seeds[3];
    for (int i = 0, pos = 0; i < len; ++i)
        if (i != va && i != vb && i != vc && i != vd)
            devP->V[t + (pos++) * range] = i;
}

//-----------------------------------------------------------------------------

/*
    Returns the initial solution weight of the planar graph.
    */
__device__ void generateFaceList(Graph *devG, Params *devP, int graph[],
    int t, int offset)
{
    int len = devG->length;
    int range = devG->range;
    int numFaces = devP->numFaces[t];

    int *seeds = c.element(t + offset).getArray();
    int va = seeds[0], vb = seeds[1], vc = seeds[2], vd = seeds[3];

    // Generate first face of the output graph
    devP->F[t + (numFaces * 3) * range] = va;
    devP->F[t + (numFaces * 3 + 1) * range] = vb;
    devP->F[t + ((numFaces++) * 3 + 2) * range] = vc;

    // Generate the next 3 possible faces
    devP->F[t + (numFaces * 3) * range] = va;
    devP->F[t + (numFaces * 3 + 1) * range] = vb;
    devP->F[t + ((numFaces++) * 3 + 2) * range] = vd;

    devP->F[t + (numFaces * 3) * range] = va;
    devP->F[t + (numFaces * 3 + 1) * range] = vc;
    devP->F[t + ((numFaces++) * 3 + 2) * range] = vd;

    devP->F[t + (numFaces * 3) * range] = vb;
    devP->F[t + (numFaces * 3 + 1) * range] = vc;
    devP->F[t + ((numFaces++) * 3 + 2) * range] = vd;

    int res = graph[va + len*vb] + graph[va + len*vc] + graph[vb + len*vc];
    res += graph[va + len*vd] + graph[vb + len*vd] + graph[vc + len*vd];

    devP->tmpMax[t] = res;
    devP->numFaces[t] = numFaces;
}

//-----------------------------------------------------------------------------

/*
    Inserts a new vertex, 3 new triangular faces
    and removes the face from the list.
    */
__device__ int faceDimple(Graph *devG, Params *devP, int graph[],
    int new_vertex, int f, int t)
{
    int len = devG->length;
    int range = devG->range;
    int numFaces = devP->numFaces[t];

    // Remove the chosen face and insert a new one on its place.
    int va = devP->F[t + (f * 3) * range],
        vb = devP->F[t + (f * 3 + 1) * range],
        vc = devP->F[t + (f * 3 + 2) * range];

    devP->F[t + (f * 3) * range] = new_vertex,
    devP->F[t + (f * 3 + 1) * range] = va,
    devP->F[t + (f * 3 + 2) * range] = vb;
    
    // Insert the other two new faces.
    devP->F[t + (numFaces * 3) * range] = new_vertex;
    devP->F[t + (numFaces * 3 + 1) * range] = va;
    devP->F[t + ((numFaces++) * 3 + 2) * range] = vc;

    devP->F[t + (numFaces * 3) * range] = new_vertex;
    devP->F[t + (numFaces * 3 + 1) * range] = vb;
    devP->F[t + ((numFaces++) * 3 + 2) * range] = vc;

    int res = graph[va + len*new_vertex] + graph[vb + len*new_vertex] +
        graph[vc + len*new_vertex];
    devP->numFaces[t] = numFaces;

    return res;
}

//-----------------------------------------------------------------------------

/*
    Return the vertex with the maximum gain inserting within a face 'f'.
    */
__device__ int maxGainFace(Graph *devG, Params *devP, int graph[], int *f, int t)
{
    int len = devG->length;
    int range = devG->range;
    int gain = -1, vertex = -1;

    // Iterate through the remaining vertices
    int remain = devP->remaining[t];
    int numFaces = devP->numFaces[t];
    for (int r = 0; r < remain; ++r)
    {
        int new_vertex = devP->V[t + r * range];
        // Test the dimple on each face
        for (int i = 0; i < numFaces; ++i)
        {
            int va = devP->F[t + (i * 3) * range],
                vb = devP->F[t + (i * 3 + 1) * range],
                vc = devP->F[t + (i * 3 + 2) * range];
            int tmpGain = graph[va + len*new_vertex] + graph[vb + len*new_vertex]
                + graph[vc + len*new_vertex];
            if (tmpGain > gain)
            {
                gain = tmpGain;
                *f = i;
                vertex = r;
            }
        }
    }
    return vertex;
}

//-----------------------------------------------------------------------------

__device__ void dimpling(Graph *devG, Params *devP, int graph[], int t)
{
    int range = devG->range;

    while (devP->remaining[t])
    {
        //Last position of the list of vertices
        int last = devP->remaining[t] - 1;
        int f = -1;
        //Index of the vertex which will be removed
        int vertex_idx = maxGainFace(devG, devP, graph, &f, t);
        //The vertex which will be removed.
        int vertex = devP->V[t + vertex_idx * range];
        devP->tmpMax[t] += faceDimple(devG, devP, graph, vertex, f, t);

        //Compress the list of vertices and remove the chosen vertex
        for (int i = vertex_idx; i <= last; ++i)
            devP->V[t + i * range] = devP->V[t + (i+1) * range];
        devP->remaining[t]--;
    }
}

//-----------------------------------------------------------------------------

__device__ void copyGraph(Graph *devG, Params *devP, int t)
{
    int numFaces = devP->numFaces[t];
    int range = devG->range;
    for (int i = 0; i < numFaces; ++i)
    {
        int va = devP->F[t + (i * 3) * range],
            vb = devP->F[t + (i * 3 + 1) * range],
            vc = devP->F[t + (i * 3 + 2) * range];
        devG->resFaces[i * 3] = va, devG->resFaces[i * 3 + 1] = vb,
            devG->resFaces[i * 3 + 2] = vc;
    }
}

//-----------------------------------------------------------------------------

__device__ void initializeDevice(Params *devP, int remaining, int t)
{
    devP->numFaces[t] = 0;
    devP->tmpMax[t] = -1LL;
    devP->remaining[t] = remaining - 4;
}

//-----------------------------------------------------------------------------

__global__ void solve(Graph * __restrict__ devG, Params devP, int *respMax,
    int offset, int range)
{
    devG->range = range;
    int x = blockDim.x*blockIdx.x + threadIdx.x;
    int len = devG->length;
    
    if (threadIdx.x == 0) c = Combination(len, 4);
    int *graph = devG->graph;
    __syncthreads();

    if (x < range)
    {
        initializeDevice(&devP, len, x);
        generateVertexList(devG, &devP, x, offset);
        generateFaceList(devG, &devP, graph, x, offset);
        dimpling(devG, &devP, graph, x);
        atomicMax(respMax, devP.tmpMax[x]);
        __syncthreads();

        if (devP.tmpMax[x] == *respMax)
            copyGraph(devG, &devP, x);
        __syncthreads();
    }
}

//-----------------------------------------------------------------------------

__global__ void solveShared(Graph * __restrict__ devG, Params devP,
    int *respMax, int offset, int range)
{
    devG->range = range;
    int x = blockDim.x*blockIdx.x + threadIdx.x;
    int len = devG->length;
    
    if (threadIdx.x == 0) c = Combination(len, 4);
    extern __shared__ int graph[];
    for (int i = threadIdx.x; i < len*len; i += blockDim.x)
        graph[i] = devG->graph[i];
    __syncthreads();

    if (x < range)
    {
        initializeDevice(&devP, len, x);
        generateVertexList(devG, &devP, x, offset);
        generateFaceList(devG, &devP, graph, x, offset);
        dimpling(devG, &devP, graph, x);
        atomicMax(respMax, devP.tmpMax[x]);
        __syncthreads();

        if (devP.tmpMax[x] == *respMax)
            copyGraph(devG, &devP, x);
        __syncthreads();
    }
}

//-----------------------------------------------------------------------------

int prepareEnvironment(int sharedOn)
{
    int finalResp = -1, pos = -1;

    #pragma omp parallel num_threads(GPU_CNT)
    {
        int gpu_id = omp_get_thread_num();
        cudaSetDevice(gpu_id);

        /*
            range   ---> Range of the seeds in the GPU
            comb    ---> Number of seeds divided between the GPUs
            offset  ---> Offset for each GPU
            */
        int range = (int)ceil(COMB / (double)GPU_CNT);
        int comb = ((gpu_id + 1)*range > COMB ? COMB - gpu_id*range : range);
        int offset = gpu_id*range;

        // Create a temporary result variable on the GPU and set its value to -1.
        int resp = -1, *tmpResp;
        gpuErrChk(cudaMalloc((void**)&tmpResp, sizeof(int)));
        gpuErrChk(cudaMemcpy(tmpResp, &resp, sizeof(int),
            cudaMemcpyHostToDevice));

        Graph *devG;
        Params devP;

        // Create an instance of Graph on the GPU and copy the values on the host
        // to it.
        gpuErrChk(cudaMalloc((void**)&devG, sizeof(Graph)));
        gpuErrChk(cudaMemcpy(devG, G, sizeof(Graph), cudaMemcpyHostToDevice));

        // Calculate the required amount of space to run the instance on the GPU.
        size_t sz_graph = sizeof(int)*MAXS + sizeof(int)*6*MAXV;
        size_t sz_prm = range*sizeof(int)*3 + range*sizeof(int)*(7*SIZE);

        fprintf(stderr, "Using %d MBytes on GPU %d\n",
            (sz_graph + sz_prm)/MB, gpu_id+1);
        
        /*
            BATCH_CNT   ---> Number of calls to the kernel for each GPU
            it_range    ---> Range of the seeds in a batch
            it_comb     ---> Number of seeds divided between batches
            it_offset   ---> Offset for each batch
            */
        size_t cuInfo = 0, cuTotal = 0;
        gpuErrChk(cudaMemGetInfo(&cuInfo, &cuTotal));
        cuInfo *= 0.95;
        printf("Free memory: %d MBytes\nTotal memory: %d mbytes\n",
            cuInfo/MB, cuTotal/MB);

        int BATCH_CNT = (int)ceil(sz_prm / (double)cuInfo);
        int it_range = (int)ceil(comb / (double)BATCH_CNT);
        int it_comb, it_offset;
        printf("Required num. of iterations: %d\n", BATCH_CNT);
        
        // Reserve the require amount of space for each variable on Params.
        gpuErrChk(cudaMalloc((void**)&devP.tmpMax, it_range*sizeof(int)));
        gpuErrChk(cudaMalloc((void**)&devP.numFaces, it_range*sizeof(int)));
        gpuErrChk(cudaMalloc((void**)&devP.remaining, it_range*sizeof(int)));
        gpuErrChk(cudaMalloc((void**)&devP.F, 6*SIZE*it_range*sizeof(int)));
        gpuErrChk(cudaMalloc((void**)&devP.V, SIZE*it_range*sizeof(int)));

        fprintf(stderr, "Kernel %d launched with %d blocks, each w/ %d threads\n",
            gpu_id+1, it_range/THREADS + 1, THREADS);

        for (int btch_id = 0; btch_id < BATCH_CNT; ++btch_id)
        {
            it_comb = ((btch_id + 1)*it_range > comb ? comb - btch_id*it_range : it_range);
            it_offset = btch_id*it_range + offset;
            
            dim3 blocks(it_comb/THREADS + 1, 1);
            dim3 threads(THREADS, 1);
            
            // Won't use shared memory with instances over MAXV vertives
            if (SIZE > 100 || !sharedOn)
            {
                solve<<<blocks, threads>>>(devG, devP, tmpResp,
                    it_offset, it_comb);
                gpuErrChk(cudaDeviceSynchronize());
            }
            else
            {
                solveShared<<<blocks, threads, SIZE*SIZE*sizeof(int)>>>(devG,
                    devP, tmpResp, it_offset, it_comb);
                gpuErrChk(cudaDeviceSynchronize());
            }

            // Copy the maximum weight found.
            gpuErrChk(cudaMemcpy(&resp, tmpResp, sizeof(int),
                cudaMemcpyDeviceToHost));

            // The result obtained by each GPU will only be copied if its value
            // is higher than the best one.
            printf("Iteration num. %d completed!\n"
                "Kernel %d finished with local maximum %d\n"
                "Copying results...\n", btch_id+1, gpu_id+1, resp);

            // Warning: possible deadlock if another application uses too much
            // resource from a GPU
            // #pragma omp barrier
            // {
                #pragma omp critical
                {
                    if (resp > finalResp)
                    {
                        finalResp = resp;
                        pos = gpu_id;
                    }
                }
            // }

            if (pos == gpu_id)
                gpuErrChk(cudaMemcpy(&F, devG->resFaces, 6*SIZE*sizeof(int),
                    cudaMemcpyDeviceToHost));
        }

        printf("Freeing memory...\n");
        gpuErrChk(cudaFree(devG));
        gpuErrChk(cudaFree(devP.numFaces));
        gpuErrChk(cudaFree(devP.remaining));
        gpuErrChk(cudaFree(devP.tmpMax));
        gpuErrChk(cudaFree(devP.F));
        gpuErrChk(cudaFree(devP.V));

        gpuErrChk(cudaDeviceReset());
    }

    return finalResp;
}

//-----------------------------------------------------------------------------

int main(int argv, char** argc)
{
    int sharedOn = 0;

    // Turn shared memory on and set which GPU to use.
    if (argv == 2)
    {
        sharedOn = 1;
        cudaSetDevice(atoi(argc[1]));
    }
    // Turn shared memory on/off and set how many GPUs to use.
    else if (argv == 3)
    {
        sharedOn = atoi(argc[1]);
        GPU_CNT = atoi(argc[2]);
        int d;
        cudaGetDeviceCount(&d);
        if (GPU_CNT > d) GPU_CNT = d;
    }
    else
    {
        printf("ERROR! Minimum num. of arguments: 1\n");
        printf("Try:\nsingle gpu - ./a.out gpu_id\n");
        printf("multi-gpu  - ./a.out sharedOnOff num_gpus\n");
        return 0;
    }

    sizeDefinitions();
    // Read the input, which is given by the size of a graph and its weighted
    // edges. The given graph should be a complete graph.
    readInput();

    double start = getTime();
    int respMax = prepareEnvironment(sharedOn);
    double stop = getTime();

    // Construct the solution given the graph faces
    for (int i = 0; i <= FACES; ++i)
    {
        int va = F[i * 3], vb = F[i * 3 + 1], vc = F[i * 3 + 2];
        if (va == vb && vb == vc) continue;
        resGraph[va*SIZE + vb] = resGraph[vb*SIZE + va] = G->graph[va*SIZE + vb];
        resGraph[va*SIZE + vc] = resGraph[vc*SIZE + va] = G->graph[va*SIZE + vc];
        resGraph[vb*SIZE + vc] = resGraph[vc*SIZE + vb] = G->graph[vb*SIZE + vc];
    }

    printf("Printing generated graph:\n");
    for (int i = 0; i < SIZE; ++i)
    {
        for (int j = i + 1; j < SIZE; ++j)
            printf("%d ", (resGraph[i*SIZE + j] == -1 ? 0 : resGraph[i*SIZE + j]));
        printf("\n");
    }

    printElapsedTime(start, stop);
    printf("Maximum weight found: %d\n", respMax);
    free(G);

    return 0;
}
