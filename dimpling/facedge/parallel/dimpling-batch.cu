#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <omp.h>
#define MB (1 << 20)
#define GB (1 << 30)

#include "combinadic.hpp"
#include "default.hpp"
#include "definitions.hpp"

using namespace std;

#include "functions.hpp"

//-----------------------------------------------------------------------------

__global__ void solve(Graph* __restrict__ devG, Params devP, int* respMax,
    int offset, int range)
{
    devG->range = range;
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    int len = devG->length;

    if (threadIdx.x == 0)
        c = Combination(len, 4);
    int* graph = devG->graph;
    __syncthreads();

    if (t < range) {
        initializeDevice(devG, &devP, len, t);
        generateVertexList(devG, &devP, t, offset);
        generateFaceList(devG, &devP, graph, t, offset);
        dimpling(devG, &devP, graph, t);
        atomicMax(respMax, devP.tmpMax[t]);
        __syncthreads();

        if (devP.tmpMax[t] == *respMax)
            copyGraph(devG, &devP, t);
        __syncthreads();
    }
}

//-----------------------------------------------------------------------------

__global__ void solveShared(Graph* __restrict__ devG, Params devP,
    int* respMax, int offset, int range)
{
    devG->range = range;
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    int len = devG->length;

    if (threadIdx.x == 0)
        c = Combination(len, 4);
    extern __shared__ int graph[];
    for (int i = threadIdx.x; i < len * len; i += blockDim.x)
        graph[i] = devG->graph[i];

    __syncthreads();

    if (t < range) {
        initializeDevice(devG, &devP, len, t);
        generateVertexList(devG, &devP, t, offset);
        generateFaceList(devG, &devP, graph, t, offset);
        dimpling(devG, &devP, graph, t);
        atomicMax(respMax, devP.tmpMax[t]);
        __syncthreads();

        if (devP.tmpMax[t] == *respMax)
            copyGraph(devG, &devP, t);
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
        int range = ceil(COMB / (double)GPU_CNT);
        int comb = ((gpu_id + 1) * range > COMB ? COMB - gpu_id * range : range);
        int offset = gpu_id * range;

        // Create a temporary result variable on the GPU and set its value to -1.
        int resp = -1, *tmpResp;
        gpuErrChk(cudaMalloc((void**)&tmpResp, sizeof(int)));
        gpuErrChk(cudaMemcpy(tmpResp, &resp, sizeof(int), cudaMemcpyHostToDevice));

        Graph* devG;
        Params devP;

        // Create an instance of Graph on the GPU and copy the values on the host
        // to it.
        gpuErrChk(cudaMalloc((void**)&devG, sizeof(Graph)));
        gpuErrChk(cudaMemcpy(devG, G, sizeof(Graph), cudaMemcpyHostToDevice));

        // Calculate the required amount of space to run the instance on the GPU.
        size_t sz_graph = MAXS * sizeof(int) + 6 * MAXV * sizeof(int);
        size_t sz_prm = 4 * range * sizeof(int) + (10 * MAXV) * range * sizeof(int)
            + (2 * MAXV * MAXV) * range * sizeof(int);

        printf("Using %d MBytes on GPU %d\n", (sz_graph + sz_prm) / MB, gpu_id + 1);

        /*
            BATCH_CNT   ---> Number of calls to the kernel for each GPU
            it_range    ---> Range of the seeds in the batch
            it_comb     ---> Number of seeds divided between the batches
            it_offset   ---> Offset for each batch
            */
        size_t cuInfo = 0, cuTotal = 0;
        gpuErrChk(cudaMemGetInfo(&cuInfo, &cuTotal));
        cuInfo *= 0.95;
        printf("Free memory: %d MBytes\nTotal memory: %d MBytes\n",
            cuInfo / MB, cuTotal / MB);

        int BATCH_CNT = ceil(sz_prm / (double)cuInfo);
        int it_range = ceil(comb / (double)BATCH_CNT);
        int it_comb, it_offset;
        printf("Required num. of iterations: %d\n", BATCH_CNT);

        // Reserve the require amount of space for each variable on Params.
        gpuErrChk(cudaMalloc((void**)&devP.tmpMax, it_range * sizeof(int)));
        gpuErrChk(cudaMalloc((void**)&devP.numFaces, it_range * sizeof(int)));
        gpuErrChk(cudaMalloc((void**)&devP.numEdges, it_range * sizeof(int)));
        gpuErrChk(cudaMalloc((void**)&devP.remaining, it_range * sizeof(int)));
        gpuErrChk(cudaMalloc((void**)&devP.F, 6 * SIZE * it_range * sizeof(int)));
        gpuErrChk(cudaMalloc((void**)&devP.V, SIZE * it_range * sizeof(int)));
        gpuErrChk(cudaMalloc((void**)&devP.E, 3 * SIZE * it_range * sizeof(int)));
        gpuErrChk(cudaMalloc((void**)&devP.edges_faces, 2 * SIZE * SIZE * it_range * sizeof(int)));

        fprintf(stderr, "Kernel %d launched with %d blocks, each w/ %d threads\n",
            gpu_id + 1, it_range / THREADS + 1, THREADS);

        for (int btch_id = 0; btch_id < BATCH_CNT; ++btch_id) {
            // Clear edge_faces on each step
            gpuErrChk(cudaMemset(devP.edges_faces, -1, 2 * SIZE * SIZE * it_range * sizeof(int)));
            it_comb = ((btch_id + 1) * it_range > comb ? comb - btch_id * it_range : it_range);
            it_offset = btch_id * it_range + offset;

            dim3 blocks(it_comb / THREADS + 1, 1);
            dim3 threads(THREADS, 1);

            // Won't use shared memory with instances over MAXV vertives
            if (SIZE > MAXV || !sharedOn) {
                solve<<<blocks, threads>>>(devG, devP, tmpResp, it_offset, it_comb);
                gpuErrChk(cudaDeviceSynchronize());
            } else {
                solveShared<<<blocks, threads, SIZE * SIZE * sizeof(int)>>>(devG,
                    devP, tmpResp, it_offset, it_comb);
                gpuErrChk(cudaDeviceSynchronize());
            }

            // Copy the maximum weight found.
            gpuErrChk(cudaMemcpy(&resp, tmpResp, sizeof(int), cudaMemcpyDeviceToHost));

            // The result obtained by each GPU will only be copied if its value
            // is higher than the best one.
            printf("Iteration num. %d completed!\n"
                   "Kernel %d finished with local maximum %d\n"
                   "Copying results...\n",
                btch_id + 1, gpu_id + 1, resp);

            // Warning: possible deadlock if another application uses too much
            // resource from a GPU
            // #pragma omp barrier
            // {
#pragma omp critical
            {
                if (resp > finalResp) {
                    finalResp = resp;
                    pos = gpu_id;
                }
            }
            // }

            if (pos == gpu_id)
                gpuErrChk(cudaMemcpy(&F, devG->resFaces, 6 * SIZE * sizeof(int),
                    cudaMemcpyDeviceToHost));
        }

        printf("Freeing memory...\n");
        gpuErrChk(cudaFree(devG));
        gpuErrChk(cudaFree(devP.numFaces));
        gpuErrChk(cudaFree(devP.numEdges));
        gpuErrChk(cudaFree(devP.remaining));
        gpuErrChk(cudaFree(devP.tmpMax));
        gpuErrChk(cudaFree(devP.F));
        gpuErrChk(cudaFree(devP.V));
        gpuErrChk(cudaFree(devP.E));
        gpuErrChk(cudaFree(devP.edges_faces));

        gpuErrChk(cudaDeviceReset());
    }

    return finalResp;
}

//-----------------------------------------------------------------------------

int main(int argv, char** argc)
{
    int sharedOn = 0;
    // Turn shared memory on and set which GPU to use.
    if (argv == 2) {
        sharedOn = 1;
        cudaSetDevice(atoi(argc[1]));
    }
    // Turn shared memory on/off and set how many GPUs to use.
    else if (argv == 3) {
        sharedOn = atoi(argc[1]);
        GPU_CNT = atoi(argc[2]);
        int d;
        cudaGetDeviceCount(&d);
        if (GPU_CNT > d)
            GPU_CNT = d;
    } else {
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
    for (int i = 0; i <= FACES; ++i) {
        int va = F[i * 3], vb = F[i * 3 + 1], vc = F[i * 3 + 2];
        if (va == vb && vb == vc)
            continue;
        resGraph[va * SIZE + vb] = resGraph[vb * SIZE + va] = G->graph[va * SIZE + vb];
        resGraph[va * SIZE + vc] = resGraph[vc * SIZE + va] = G->graph[va * SIZE + vc];
        resGraph[vb * SIZE + vc] = resGraph[vc * SIZE + vb] = G->graph[vb * SIZE + vc];
    }

    printf("Printing generated graph:\n");
    for (int i = 0; i < SIZE; ++i) {
        for (int j = i + 1; j < SIZE; ++j)
            printf("%d ", (resGraph[i * SIZE + j] == -1 ? 0 : resGraph[i * SIZE + j]));
        printf("\n");
    }

    printElapsedTime(start, stop);
    printf("Maximum weight found: %d\n", respMax);
    free(G);

    return 0;
}
