#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <vector>

#include "helpers.hpp"
#include "combinadic.hpp"

using namespace std;

/*
    Shared Combinadic instance on the GPU
    */
__shared__ Combination c;

__device__ void generateList(Node* devN, Params* devP, int t, int offset)
{
    int sz = devN->sz;
    int perm = devN->perm;

    int* seeds = c.element(t + offset).getArray();

    int va = seeds[0], vb = seeds[1], vc = seeds[2], vd = seeds[3];
    for (int i = 0, pos = 0; i < sz; i++) {
        if (i != va && i != vb && i != vc && i != vd)
            devP->V[t + (pos++) * perm] = i;
    }
}

/*
    Returns the weight of the initial planar subgraph.
    */
__device__ void generateTriangularFaceList(Node* devN, Params* devP, double graph[],
    int t, int offset)
{
    int sz = devN->sz;
    int perm = devN->perm;

    int* seeds = c.element(t + offset).getArray();

    int va = seeds[0], vb = seeds[1], vc = seeds[2], vd = seeds[3];

    // Generate first triangle of the output graph
    devP->F[t + (devP->faces[t] * 3) * perm] = va;
    devP->F[t + (devP->faces[t] * 3 + 1) * perm] = vb;
    devP->F[t + ((devP->faces[t]++) * 3 + 2) * perm] = vc;

    // Generate the next 3 possible faces
    devP->F[t + (devP->faces[t] * 3) * perm] = va;
    devP->F[t + (devP->faces[t] * 3 + 1) * perm] = vb;
    devP->F[t + ((devP->faces[t]++) * 3 + 2) * perm] = vd;

    devP->F[t + (devP->faces[t] * 3) * perm] = va;
    devP->F[t + (devP->faces[t] * 3 + 1) * perm] = vc;
    devP->F[t + ((devP->faces[t]++) * 3 + 2) * perm] = vd;

    devP->F[t + (devP->faces[t] * 3) * perm] = vb;
    devP->F[t + (devP->faces[t] * 3 + 1) * perm] = vc;
    devP->F[t + ((devP->faces[t]++) * 3 + 2) * perm] = vd;

    double resp = graph[va + sz * vb] + graph[va + sz * vc] + graph[vb + sz * vc];
    resp += graph[va + sz * vd] + graph[vb + sz * vd] + graph[vc + sz * vd];

    devP->tmpMax[t] = resp;
}

/*
    Inserts a new vertex, 3 new triangular faces and removes face 'f'
    from the set.
    */
__device__ double operationT2(Node* devN, Params* devP, double graph[],
    int new_vertex, int f, int t)
{
    int sz = devN->sz;
    int perm = devN->perm;

    // Remove the chosen face and insert a new one
    int va = devP->F[t + (f * 3) * perm],
        vb = devP->F[t + (f * 3 + 1) * perm],
        vc = devP->F[t + (f * 3 + 2) * perm];

    devP->F[t + (f * 3) * perm] = new_vertex,
                          devP->F[t + (f * 3 + 1) * perm] = va,
                          devP->F[t + (f * 3 + 2) * perm] = vb;

    // and insert the other two possible faces
    devP->F[t + (devP->faces[t] * 3) * perm] = new_vertex;
    devP->F[t + (devP->faces[t] * 3 + 1) * perm] = va;
    devP->F[t + ((devP->faces[t]++) * 3 + 2) * perm] = vc;

    devP->F[t + (devP->faces[t] * 3) * perm] = new_vertex;
    devP->F[t + (devP->faces[t] * 3 + 1) * perm] = vb;
    devP->F[t + ((devP->faces[t]++) * 3 + 2) * perm] = vc;

    double resp = graph[va + sz * new_vertex] + graph[vb + sz * new_vertex] + graph[vc + sz * new_vertex];

    return resp;
}

/*
    Returns the vertex with the maximum gain inserting within a face 'f'.
    */
__device__ int maxGain(Node* devN, Params* devP, double graph[], int* f, int t)
{
    int sz = devN->sz;
    int perm = devN->perm;
    double gain = -1.0;
    int vertex = -1;

    // Iterate through the remaining vertices
    int remain = devP->count[t];
    for (int r = 0; r < remain; r++) {
        int new_vertex = devP->V[t + r * perm];
        // and test which has the maximum gain with its insertion
        // within all possible faces
        int faces = devP->faces[t];
        for (int i = 0; i < faces; i++) {
            int va = devP->F[t + (i * 3) * perm],
                vb = devP->F[t + (i * 3 + 1) * perm],
                vc = devP->F[t + (i * 3 + 2) * perm];
            double tmpGain = graph[va + sz * new_vertex] + graph[vb + sz * new_vertex]
                + graph[vc + sz * new_vertex];
            if (CMP(tmpGain, gain) == 1) {
                gain = tmpGain;
                *f = i;
                vertex = r;
            }
        }
    }
    return vertex;
}

__device__ void dimpling(Node* devN, Params* devP, double graph[], int t)
{
    int perm = devN->perm;

    while (devP->count[t]) {
        int last = devP->count[t] - 1;
        int f = -1;
        int vertex_idx = maxGain(devN, devP, graph, &f, t);
        int vertex = devP->V[t + vertex_idx * perm];
        devP->tmpMax[t] += operationT2(devN, devP, graph, vertex, f, t);

        for (int i = vertex_idx; i <= last; i++)
            devP->V[t + i * perm] = devP->V[t + (i + 1) * perm];
        devP->count[t]--;
    }
}

__device__ void copyGraph(Node* devN, Params* devP, int t)
{
    int faces = devP->faces[t];
    int perm = devN->perm;
    for (int i = 0; i < faces; i++) {
        int va = devP->F[t + (i * 3) * perm],
            vb = devP->F[t + (i * 3 + 1) * perm],
            vc = devP->F[t + (i * 3 + 2) * perm];
        devN->F_ANS[i * 3] = va, devN->F_ANS[i * 3 + 1] = vb,
                        devN->F_ANS[i * 3 + 2] = vc;
    }
}

__device__ void initializeDevice(Params* devP, int sz, int t)
{
    devP->faces[t] = 0;
    devP->tmpMax[t] = 0.0;
    devP->count[t] = sz - 4;
}

__global__ void dimplingKernel(Node* __restrict__ devN, Params devP,
    double* respMax, int offset, int perm)
{
    devN->perm = perm;
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int sz = devN->sz;

    if (threadIdx.x == 0)
        c = Combination(sz, 4);
    double* graph = devN->graph;
    __syncthreads();

    if (x < perm) {
        initializeDevice(&devP, sz, x);
        generateList(devN, &devP, x, offset);
        generateTriangularFaceList(devN, &devP, graph, x, offset);
        dimpling(devN, &devP, graph, x);
        AtomicMax(respMax, devP.tmpMax[x]);
        __syncthreads();

        if (devP.tmpMax[x] == *respMax) {
            copyGraph(devN, &devP, x);
        }
        __syncthreads();
    }
}

__global__ void dimplingKernelShared(Node* __restrict__ devN, Params devP,
    double* respMax, int offset, int perm)
{
    devN->perm = perm;
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int sz = devN->sz;

    if (threadIdx.x == 0)
        c = Combination(sz, 4);
    extern __shared__ double graph[];
    for (int i = threadIdx.x; i < sz * sz; i += blockDim.x)
        graph[i] = devN->graph[i];
    __syncthreads();

    if (x < perm) {
        initializeDevice(&devP, sz, x);
        generateList(devN, &devP, x, offset);
        generateTriangularFaceList(devN, &devP, graph, x, offset);
        dimpling(devN, &devP, graph, x);
        AtomicMax(respMax, devP.tmpMax[x]);
        __syncthreads();

        if (devP.tmpMax[x] == *respMax) {
            copyGraph(devN, &devP, x);
        }
        __syncthreads();
    }
}

double dimplingPrepare(int sharedOn)
{
    double finalResp = 0.0;
    int pos = -1;

#pragma omp parallel num_threads(GPU_CNT)
    {
        int gpu_id = omp_get_thread_num();
        cudaSetDevice(gpu_id);

        /*  range        ---> Range of the seeds in the GPU
            perm         ---> Number of seeds divided between the GPUs
            offset       ---> Offset for each GPU
            */
        int64 range = (int)ceil(PERM / (double)GPU_CNT);
        int64 perm = ((gpu_id + 1) * range > PERM ? PERM - gpu_id * range : range);
        int64 offset = gpu_id * range;

        // Create a temporary result variable on the GPU and set its value to -1
        double resp = 0.0, *tmpResp;
        gpuErrChk(cudaMalloc((void**)&tmpResp, sizeof(double)));
        gpuErrChk(cudaMemcpy(tmpResp, &resp, sizeof(double), cudaMemcpyHostToDevice));

        Node* devN;
        Params devP;

        // Create an instance of Node on the GPU and copy the values on the host
        // to it
        gpuErrChk(cudaMalloc((void**)&devN, sizeof(Node)));
        gpuErrChk(cudaMemcpy(devN, N, sizeof(Node), cudaMemcpyHostToDevice));

        // Calculate the required amount of space to run the instance on the GPU
        size_t sz_node = sizeof(double) * MAX_S + sizeof(int) * 6 * MAX + sizeof(int) * 2;
        size_t sz_prm = range * sizeof(int) * 2 + range * sizeof(double) + range * sizeof(int) * (7 * SIZE);

        if ((sz_graph + sz_prm) / MB > (1 << 10)) {
            printf("Using %d GBytes on GPU %d\n", (sz_graph + sz_prm) / GB, gpu_id + 1);
        } else {
            printf("Using %d MBytes on GPU %d\n", (sz_graph + sz_prm) / MB, gpu_id + 1);
        }

        size_t cuInfo = 0, cuTotal = 0;
        gpuErrChk(cudaMemGetInfo(&cuInfo, &cuTotal));
        cuInfo *= 0.95;
        printf("Free memory: %d MBytes\n"
            "Total memory: %d MBytes\n",
            cuInfo / MB, cuTotal / MB);

        /*  BATCH_CNT       ---> Number of calls to the kernel for each GPU
            it_range        ---> Range of the seeds in the batch
            it_perm         ---> Number of seeds divided between the batches
            it_offset       ---> Offset for each batch
            */
        int BATCH_CNT = (int)ceil(sz_prm / (double)cuInfo);
        int it_range = (int)ceil(perm / (double)BATCH_CNT);
        int it_perm, it_offset;
        printf("Required num. of iterations: %d\n", BATCH_CNT);

        /*  Reserve the require amount of space for each variable on Params.
            */
        gpuErrChk(cudaMalloc((void**)&devP.faces, it_range * sizeof(int)));
        gpuErrChk(cudaMalloc((void**)&devP.count, it_range * sizeof(int)));
        gpuErrChk(cudaMalloc((void**)&devP.tmpMax, it_range * sizeof(double)));
        gpuErrChk(cudaMalloc((void**)&devP.F, 6 * SIZE * it_range * sizeof(int)));
        gpuErrChk(cudaMalloc((void**)&devP.V, SIZE * it_range * sizeof(int)));

        fprintf(stderr, "Kernel %d launched with %d blocks, each w/ %d threads\n",
            gpu_id + 1, it_range / THREADS + 1, THREADS);

        for (int btch_id = 0; btch_id < BATCH_CNT; btch_id++) {
            it_perm = ((btch_id + 1) * it_range > perm ? perm - btch_id * it_range : it_range);
            it_offset = btch_id * it_range + offset;

            dim3 blocks(it_perm / THREADS + 1, 1);
            dim3 threads(THREADS, 1);

            if (SIZE > 75 || !sharedOn) {
                dimplingKernel<<<blocks, threads>>>(devN, devP, tmpResp, it_offset, it_perm);
                gpuErrChk(cudaDeviceSynchronize());
            } else {
                dimplingKernelShared<<<blocks, threads, SIZE * SIZE * sizeof(double)>>>(devN, devP,
                    tmpResp, it_offset, it_perm);
                gpuErrChk(cudaDeviceSynchronize());
            }

            // Copy the maximum weight found
            gpuErrChk(cudaMemcpy(&resp, tmpResp, sizeof(double), cudaMemcpyDeviceToHost));

            // The result obtained by each GPU will only be copied if its value is higher
            // than the current one
            printf("Kernel finished with local maximum %.3lf. Copying results...\n", resp);

            // Warning: possible deadlock if another application uses too much
            // resource from a GPU
            // #pragma omp barrier
#pragma omp critical
            {
                if (hostCMP(resp, finalResp) == 1) {
                    finalResp = resp;
                    pos = gpu_id;
                }
            }

            if (pos == gpu_id)
                gpuErrChk(cudaMemcpy(&F, devN->F_ANS, 6 * MAX * sizeof(int),
                    cudaMemcpyDeviceToHost));
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

int main(int argv, char** argc)
{
    sizeDefinitions();
    // Read the input, which is given by the size of a graph and its weighted
    // edges. The given graph should be a complete graph
    readInput();
    initialize();

    int sharedOn = 0;
    if (argv == 2) {
        sharedOn = 1;
        cudaSetDevice(atoi(argc[1]));
    } else if (argv == 3) {
        sharedOn = atoi(argc[1]);
        GPU_CNT = atoi(argc[2]);
        int d;
        cudaGetDeviceCount(&d);
        if (GPU_CNT > d)
            GPU_CNT = d;
    } else {
        printf("ERROR! Minimum num. of arguments: 1\n"
            "Usage:\n"
            "single gpu - ./a.out gpu_id\n"
            "multi-gpu  - ./a.out sharedOnOff num_gpus\n"
            "examples:\n"
            "  ./a.out 0\n"
            "  ./a.out 1 1\n"
            "  ./a.out 1 4\n");
        return 0;
    }

    double start = getTime();
    double respMax = dimplingPrepare(sharedOn);
    double stop = getTime();

    // Reconstruct the graph given the faces of the graph
    for (int i = 0; i < 2 * SIZE; i++) {
        int va = F[i * 3], vb = F[i * 3 + 1], vc = F[i * 3 + 2];
        // Outbounds verification
        if (va == vb && vb == vc)
            continue;
        R[va * SIZE + vb] = R[vb * SIZE + va] = N->graph[va * SIZE + vb];
        R[va * SIZE + vc] = R[vc * SIZE + va] = N->graph[va * SIZE + vc];
        R[vb * SIZE + vc] = R[vc * SIZE + vb] = N->graph[vb * SIZE + vc];
    }

    printf("Printing generated graph:\n");
    for (int i = 0; i < SIZE; i++) {
        for (int j = i + 1; j < SIZE; j++) {
            printf("%lf ", (R[i * SIZE + j] == -1 ? 0 : R[i * SIZE + j]));
        }
        printf("\n");
    }

    printElapsedTime(start, stop);
    printf("Maximum weight found: %.3lf\n", respMax);
    free(N);

    return 0;
}
