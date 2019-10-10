#include <bits/stdc++.h>
#define pb push_back
#define mp make_pair
#define fi first
#define se second

using namespace std;

typedef int Face;
typedef long long int64;
typedef pair<int, int> ii;

const double EPS = 1e-9;

const int C = 4; // Combination size
const int MAXV = 100; // Max. number of vertices
const int MAXF = 2 * MAXV - 4; // Num. of faces on a planar graph

struct node {
    int w, vertex, face;
    node(int w = 0, int vertex = 0, int face = 0)
        : w(w)
        , vertex(vertex)
        , face(face)
    {
    }
};

int sgn(double a) { return ((a > EPS) ? (1) : ((a < -EPS) ? (-1) : (0))); }
int cmp(double a, double b = 0.0) { return sgn(a - b); }

#include "combinadic.h"

//-----------------------------------------------------------------------------
// OSX
#ifdef __MACH__
#include <mach/clock.h>
#include <mach/mach.h>
#endif
//-----------------------------------------------------------------------------
/*
    Print elapsed time.
    */
void printElapsedTime(double start, double stop)
{
    double elapsed = stop - start;
    printf("Elapsed time: %.3lfs.\n", elapsed);
}
//-----------------------------------------------------------------------------
/*  
    Get clock time.
    */
void current_utc_time(struct timespec* ts)
{
#ifdef __MACH__ // OS X does not have clock_gettime, use clock_get_time
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
/* 
    V       ---> Number of vertices
    F       ---> Number of faces
    COMB    ---> Number of combinations
    graph   ---> The graph itself
    faces   ---> List containing triangular faces
    numComb ---> Number of combinations binom(V, 4)
*/
Combination c;
Face faces[MAXF][3];
int graph[MAXV][MAXV], R[MAXV][MAXV], V, F, COMB;
int64 numComb[MAXV];
//-----------------------------------------------------------------------------
/*
    Defines the number of combinations.
    */
void sizeDefinitions()
{
    for (int64 i = 4LL; i <= MAXV; ++i) {
        int64 resp = 1LL;
        for (int64 j = i - 3; j <= i; ++j)
            resp *= j;
        resp /= 24LL;
        numComb[i - 1] = resp;
    }
}
//-----------------------------------------------------------------------------
void readInput()
{
    scanf("%d", &V);
    for (int i = 0; i < V; ++i) {
        for (int j = i + 1; j < V; ++j) {
            scanf("%d", &graph[i][j]);
            graph[j][i] = graph[i][j];
        }
        graph[i][i] = -1;
    }

    COMB = numComb[V - 1];
    F = 2 * V - 4;
    c = Combination(V, 4);
}
//-----------------------------------------------------------------------------
/*
    Generates a list having vertices which are not on the planar graph.
    */
void generateVertexList(int idx, set<int>& vertices)
{
    vector<int> seeds = c.element(idx).getArray();
    for (int i = 0; i < V; ++i) {
        if (i != seeds[0] && i != seeds[1] && i != seeds[2] && i != seeds[3])
            vertices.insert(i);
    }
}
//-----------------------------------------------------------------------------
/*
    Returns the initial solution weight of the planar graph.
    */
int generateFaceList(int idx, Face tmpFaces[][3], int* numFaces)
{
    vector<int> seeds = c.element(idx).getArray();
    int res = 0;
    for (int i = 0; i < C - 1; ++i)
        for (int j = i + 1; j < C; ++j) {
            int va = seeds[i], vb = seeds[j];
            res += graph[va][vb];
        }

    for (int i = 0; i < C - 2; ++i)
        for (int j = i + 1; j < C - 1; ++j)
            for (int k = j + 1; k < C; ++k, (*numFaces)++) {
                //Vertices of a face
                int va = seeds[i], vb = seeds[j], vc = seeds[k];
                tmpFaces[*numFaces][0] = va, tmpFaces[*numFaces][1] = vb,
                tmpFaces[*numFaces][2] = vc;
            }

    return res;
}
//-----------------------------------------------------------------------------
/*
    Inserts a new vertex, 3 new triangular faces
    and removes the face from the list.
    */
void faceDimple(int new_vertex, int face, int tmpFaces[][3], int* numFaces)
{
    //Remove the chosen face and insert a new one on its place.
    int va = tmpFaces[face][0], vb = tmpFaces[face][1], vc = tmpFaces[face][2];

    tmpFaces[face][0] = new_vertex, tmpFaces[face][1] = va,
    tmpFaces[face][2] = vb;

    //Insert the other two new faces.
    tmpFaces[*numFaces][0] = new_vertex, tmpFaces[*numFaces][1] = va,
    tmpFaces[(*numFaces)++][2] = vc;
    tmpFaces[*numFaces][0] = new_vertex, tmpFaces[*numFaces][1] = vb,
    tmpFaces[(*numFaces)++][2] = vc;
}
//-----------------------------------------------------------------------------
/*
    Returns the vertex with the maximum gain inserting within a face.
    */
node maxGainFace(set<int>& vertices, Face tmpFaces[][3], int* numFaces)
{
    node gains(-1, -1, -1);
    // Iterate through the remaining vertices.
    for (int new_vertex : vertices) {
        // Test which one has the maximum gain with its insertion
        // within all possible faces.
        for (int face = 0; face < *numFaces; ++face) {
            int va = tmpFaces[face][0], vb = tmpFaces[face][1],
                vc = tmpFaces[face][2];
            int gain = graph[va][new_vertex] + graph[vb][new_vertex]
                + graph[vc][new_vertex];

            if (gain > gains.w)
                gains = node(gain, new_vertex, face);
        }
    }
    return gains;
}
//-----------------------------------------------------------------------------
int solve(set<int>& vertices, int tmpMax, Face tmpFaces[][3], int* numFaces)
{
    int maxValue = tmpMax;

    while (!vertices.empty()) {
        //first -> vertex; second -> face;
        node gain = maxGainFace(vertices, tmpFaces, numFaces);
        vertices.erase(gain.vertex);
        maxValue += gain.w;
        faceDimple(gain.vertex, gain.face, tmpFaces, numFaces);
    }
    return maxValue;
}
//-----------------------------------------------------------------------------
int main(int argv, char** argc)
{
    double start, stop;

    // Read the input, which is given by the size of a graph and its weighted
    // edges. The given graph should be complete.
    sizeDefinitions();
    readInput();

    int respMax = -1;

    start = getTime();
#pragma omp parallel for
    for (int i = 0; i < COMB; i++) {
        //List of faces for solution i
        Face tmpFaces[MAXF][3];
        int numFaces = 0;
        set<int> vertices;

        //A list with the remaining vertices
        generateVertexList(i, vertices);
        //Get the weight of the initial solution
        int tmpMax = generateFaceList(i, tmpFaces, &numFaces);
        int ans = solve(vertices, tmpMax, tmpFaces, &numFaces);

#pragma omp critical
        {
            if (ans >= respMax) {
                respMax = ans;
                for (int j = 0; j < numFaces; ++j)
                    for (int k = 0; k < 3; ++k)
                        faces[j][k] = tmpFaces[j][k];
            }
        }
    }
    stop = getTime();

    printf("Printing generated graph:\n");
    // Construct the solution given the graph faces
    for (int i = 0; i < F; ++i) {
        int va = faces[i][0], vb = faces[i][1], vc = faces[i][2];
        if (va == vb && vb == vc)
            continue;
        R[va][vb] = R[vb][va] = graph[va][vb];
        R[va][vc] = R[vc][va] = graph[va][vc];
        R[vb][vc] = R[vc][vb] = graph[vb][vc];
    }
    // Print the graph
    for (int i = 0; i < V; ++i) {
        for (int j = i + 1; j < V; ++j)
            printf("%d ", (R[i][j] == -1 ? 0 : R[i][j]));
        printf("\n");
    }

    printElapsedTime(start, stop);
    printf("Maximum weight found: %d\n", respMax);

    return 0;
}
