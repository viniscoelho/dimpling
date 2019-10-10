#include <bits/stdc++.h>
#define pb push_back
#define mp make_pair
#define fi first
#define se second

using namespace std;

typedef int Face;
typedef long long int64;
typedef pair<int, int> ii;
typedef vector<int> vi;

const double MINUTE = 60.0;
const double HOUR = 3600.0;
const double EPS = 1e-9;
const int INF = 0x3f3f3f3f;
const int MOD = 1000000007;

const int C = 4; //size of the combination
const int MAXV = 110; //max number of vertices
const int MAXF = 2 * MAXV - 4; //upper-bound of regions on a planar graph
const int MAXE = MAXV * (MAXV - 1) / 2; //number of undirected edges

int sgn(double a) { return ((a > EPS) ? (1) : ((a < -EPS) ? (-1) : (0))); }
int cmp(double a, double b = 0.0) { return sgn(a - b); }

struct node {
    int w, vertex, edge, face, extra;
    node(int w = 0, int vertex = 0, int edge = 0, int face = 0, int extra = 0)
        : w(w)
        , vertex(vertex)
        , edge(edge)
        , face(face)
        , extra(extra)
    {
    }
};

struct edge {
    int u, v;
    edge(int u_t = 0, int v_t = 0)
    {
        u = min(u_t, v_t);
        v = max(v_t, u_t);
    };

    bool operator==(const edge& at)
    {
        return (u == at.u && v == at.v) || (v == at.u && u == at.v);
    }
};

#include "combinadic.h"

//-----------------------------------------------------------------------------
// Mac
#ifdef __MACH__
#include <mach/clock.h>
#include <mach/mach.h>
#endif
//-----------------------------------------------------------------------------
/*
    Prints the elapsed time.
    */
void printElapsedTime(double start, double stop)
{
    double elapsed = stop - start;
    if (cmp(elapsed, MINUTE) == -1)
        printf("Elapsed time: %.3lfs.\n", elapsed);
    else if (cmp(elapsed, HOUR) == 1)
        printf("Elapsed time: %.3lfhs.\n", elapsed / HOUR);
    else if (cmp(elapsed, MINUTE) == 1)
        printf("Elapsed time: %.3lfmin.\n", elapsed / MINUTE);
}
//-----------------------------------------------------------------------------
/*  
    Gets the clock time.
    */
void getCurrentTime(struct timespec* ts)
{
#ifdef __MACH__ //OS X does not have clock_gettime, use clock_get_time
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
    getCurrentTime(&ts);
    return double(ts.tv_sec) + double(ts.tv_nsec) / 1e9;
}
//-----------------------------------------------------------------------------
/* 
    V       ---> Number of vertices
    F       ---> Number of faces
    COMB    ---> Number of combinations
    graph   ---> The graph itself
    faces   ---> A list containing triangular faces
    numComb ---> Number of combinations binom(V, 4)
*/
Combination c;
Face faces[MAXF][3];
int graph[MAXV][MAXV], R[MAXV][MAXV], V, F, COMB;
int64 numComb[MAXV];
//Index to Edge
vector<edge> idx_to_edge;
//Edge to Index
map<ii, int> edge_to_idx;

//Degree of each vertex
int vertex_degree[MAXV];
int min_degree;
int max_degree;
int avg_degree;

set<ii> degrees;
//-----------------------------------------------------------------------------
/*
    Defines the number of combinations.
    */
void defineCombinations()
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
            R[i][j] = R[j][i] = -1;
        }
        graph[i][i] = -1;
        R[i][i] = -1;
    }

    int count = 0;
    //Map the edges
    for (int i = 0; i < V - 1; ++i)
        for (int j = i + 1; j < V; ++j, ++count) {
            edge_to_idx[mp(i, j)] = count;
            idx_to_edge.pb(edge(i, j));
        }

    COMB = numComb[V - 1];
    F = 2 * V - 4;
    c = Combination(V, 4);
}
//-----------------------------------------------------------------------------
/*
    Creates a list having vertices which are not on the planar graph.
    */
void createVertexList(int idx, set<int>& vertices)
{
    vector<int> seeds = c.element(idx).getArray();
    for (int i = 0; i < V; ++i)
        if (i != seeds[0] && i != seeds[1] && i != seeds[2] && i != seeds[3])
            vertices.insert(i);
}
//-----------------------------------------------------------------------------
/*
    Returns the initial solution weight of the planar graph.
    */
int createFaceList(int idx, set<int>& edges, vector<vi>& edges_faces,
    Face tmpFaces[][3], int* numFaces)
{
    vector<int> seeds = c.element(idx).getArray();
    int res = 0;
    for (int i = 0; i < C - 1; ++i)
        for (int j = i + 1; j < C; j++) {
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

                //Edges
                int ea = edge_to_idx[mp(va, vb)], eb = edge_to_idx[mp(va, vc)],
                    ec = edge_to_idx[mp(vb, vc)];
                //Insert all the edges on a list
                edges.insert(ea);
                edges.insert(eb);
                edges.insert(ec);
                //Faces which each edge belongs
                edges_faces[ea].pb(*numFaces);
                edges_faces[eb].pb(*numFaces);
                edges_faces[ec].pb(*numFaces);
            }

    return res;
}
//-----------------------------------------------------------------------------
/*
    Check wether this edge already belongs to two faces.
    If remove_edge == true, this edge will be removed
    from both faces it belongs to.
    */
void addEdgeToFace(edge at, int face, vector<vi>& edges_faces,
    bool remove_edge = false)
{
    int e = edge_to_idx[mp(at.u, at.v)];
    //Check wether this edge already belongs to two faces.
    if (remove_edge) {
        for (int i = 0; i < edges_faces[e].size(); ++i)
            if (edges_faces[e][i] == face) {
                swap(edges_faces[e][i], edges_faces[e].back());
                edges_faces[e].pop_back();
                break;
            }
    }
    //Edge 'e' belongs to this face now.
    else
        edges_faces[e].pb(face);
}
//-----------------------------------------------------------------------------
/*
    Inserts a new vertex, 3 new triangular faces
    and removes the 'dimpled' face from the list.
    */
void faceDimple(int new_vertex, int face, set<int>& edges, vector<vi>& edges_faces,
    int tmpFaces[][3], int* numFaces)
{
    vector<edge> used_edges;
    set<int> used;
    //Update face_edges
    for (int i = 0; i < 2; ++i) {
        for (int j = i + 1; j < 3; ++j) {
            int u = tmpFaces[face][i], v = tmpFaces[face][j];
            used_edges.pb(edge(u, v));
            used.insert(u);
            used.insert(v);
            //Remove this edge from this face.
            addEdgeToFace(edge(u, v), face, edges_faces, true);
        }
    }

    //Update edges: add(v, new_vertex), for each v in used
    for (int v : used) {
        edge tmp_e = edge(v, new_vertex);
        int e = edge_to_idx[mp(tmp_e.u, tmp_e.v)];
        edges.insert(e);
    }

    vector<int> new_faces;
    new_faces.pb(face);
    new_faces.pb((*numFaces)++);
    new_faces.pb((*numFaces)++);

    for (int i = 0; i < new_faces.size(); ++i) {
        int f = new_faces[i];
        int va = used_edges[i].u, vb = used_edges[i].v;
        addEdgeToFace(used_edges[i], f, edges_faces);
        addEdgeToFace(edge(va, new_vertex), f, edges_faces);
        addEdgeToFace(edge(vb, new_vertex), f, edges_faces);

        tmpFaces[f][0] = new_vertex, tmpFaces[f][1] = va,
        tmpFaces[f][2] = vb;
    }
}
//-----------------------------------------------------------------------------
/*
    Inserts a new vertex and creates 4 new triangular faces.
    This process removes two faces and an edge, and then, adds 4 edges.
    */
void edgeDimple(int new_vertex, int edge_idx, int face, int extra, set<int>& edges,
    vector<vi>& edges_faces, int tmpFaces[][3], int* numFaces)
{
    //The edge removed
    edge r_edge = idx_to_edge[edge_idx];

    vector<edge> used_edges;
    set<int> used;
    //Update face_edges
    for (int i = 0; i < 2; ++i) {
        for (int j = i + 1; j < 3; ++j) {
            int u = tmpFaces[face][i], v = tmpFaces[face][j];
            used.insert(u);
            used.insert(v);
            if (r_edge == edge(u, v)) {
                //Remove this edge from this face
                addEdgeToFace(edge(u, v), face, edges_faces, true);
            } else {
                used_edges.pb(edge(u, v));
                //Remove this edge from this face
                addEdgeToFace(edge(u, v), face, edges_faces, true);
            }
        }
    }
    for (int i = 0; i < 2; ++i) {
        for (int j = i + 1; j < 3; ++j) {
            int u = tmpFaces[extra][i], v = tmpFaces[extra][j];
            used.insert(u);
            used.insert(v);
            if (r_edge == edge(u, v)) {
                //Remove this edge from this face
                addEdgeToFace(edge(u, v), extra, edges_faces, true);
            } else {
                used_edges.pb(edge(u, v));
                //Remove this edge from this face
                addEdgeToFace(edge(u, v), extra, edges_faces, true);
            }
        }
    }
    //Update edges: add(v, new_vertex), for each v in used
    for (int v : used) {
        edge tmp_e = edge(v, new_vertex);
        int e = edge_to_idx[mp(tmp_e.u, tmp_e.v)];
        edges.insert(e);
    }

    vector<int> new_faces;
    new_faces.pb(face);
    new_faces.pb(extra);
    new_faces.pb((*numFaces)++);
    new_faces.pb((*numFaces)++);

    for (int i = 0; i < new_faces.size(); ++i) {
        int f = new_faces[i];
        int va = used_edges[i].u, vb = used_edges[i].v;
        addEdgeToFace(used_edges[i], f, edges_faces);
        addEdgeToFace(edge(va, new_vertex), f, edges_faces);
        addEdgeToFace(edge(vb, new_vertex), f, edges_faces);

        tmpFaces[f][0] = new_vertex, tmpFaces[f][1] = va,
        tmpFaces[f][2] = vb;
    }
}
//-----------------------------------------------------------------------------
/*
    Returns a vertex having the maximum gain inserting within a face.
    */
node maxGainFace(set<int>& vertices, Face tmpFaces[][3], int* numFaces)
{
    node gains(-1, -1, -1, -1);
    //Iterate through the remaining vertices.
    for (int new_vertex : vertices) {
        //Test the dimple on each face
        for (int face = 0; face < *numFaces; ++face) {
            int gain = 0;
            for (int k = 0; k < 3; ++k) {
                int u = tmpFaces[face][k];
                gain += graph[u][new_vertex];
            }

            if (gain > gains.w)
                gains = node(gain, new_vertex, -1, face);
        }
    }
    return gains;
}
//-----------------------------------------------------------------------------
/*
    Returns an edge which the removal has the maximum gain when inserting
    a vertex into it.
    */
node maxGainEdge(set<int>& vertices, set<int>& edges, vector<vi>& edges_faces,
    Face tmpFaces[][3])
{
    node gains(-1, -1, -1, -1, -1);
    //Iterate through the remaining vertices
    for (int new_vertex : vertices) {
        //Test the dimple on each edge
        for (int e : edges) {
            int gain = 0;
            edge r = idx_to_edge[e];
            //Check these faces
            vector<int> faces_v = edges_faces[e];
            set<int> used;
            //2 faces for each vertex
            for (int f : faces_v) {
                //3 vertices for each face
                for (int k = 0; k < 3; ++k) {
                    int u = tmpFaces[f][k];
                    //If I have not used this vertex yet
                    if (!used.count(u)) {
                        used.insert(u);
                        gain += graph[u][new_vertex];
                    }
                }
            }

            gain -= graph[r.u][r.v];
            if (gain > gains.w)
                gains = node(gain, new_vertex, e, faces_v[0], faces_v[1]);
        }
    }
    return gains;
}
//-----------------------------------------------------------------------------
int solve(set<int>& vertices, set<int>& edges, vector<vi>& edges_faces,
    int tmpMax, Face tmpFaces[][3], int* numFaces)
{
    int maxValue = tmpMax;

    while (!vertices.empty()) {
        node gain_f = maxGainFace(vertices, tmpFaces, numFaces);
        node gain_e = maxGainEdge(vertices, edges, edges_faces, tmpFaces);

        //Choose face dimple if the gain is the same
        //since edge dimple is slower
        if (gain_f.w >= gain_e.w) {
            vertices.erase(gain_f.vertex);
            maxValue += gain_f.w;
            faceDimple(gain_f.vertex, gain_f.face, edges, edges_faces,
                tmpFaces, numFaces);
        } else {
            vertices.erase(gain_e.vertex);
            edges.erase(gain_e.edge);
            maxValue += gain_e.w;
            edgeDimple(gain_e.vertex, gain_e.edge, gain_e.face, gain_e.extra,
                edges, edges_faces, tmpFaces, numFaces);
        }
    }
    return maxValue;
}
//-----------------------------------------------------------------------------
int main(int argv, char** argc)
{
    double start, stop;
    //Read the input, which is composed by the size of a graph and its weighted
    //edges. The given graph is complete.
    defineCombinations();
    readInput();

    int respMax = -1;
    min_degree = MAXV;

    start = getTime();
#pragma omp parallel for
    for (int i = 0; i < COMB; i++)
    // for (int i = 0; i < 1; i++)
    {
        //List of faces for solution i
        Face tmpFaces[MAXF][3];
        int numFaces = 0;
        set<int> vertices, edges;
        //Which faces an edge belongs?
        vector<vi> edges_faces(MAXE);

        //A list with the remaining vertices
        createVertexList(i, vertices);
        //Get the weight of the initial solution
        int tmpMax = createFaceList(i, edges, edges_faces, tmpFaces, &numFaces);
        int ans = solve(vertices, edges, edges_faces, tmpMax, tmpFaces, &numFaces);

#pragma omp critical
        {
            //For analysis only
            for (int j = 0; j < numFaces; ++j)
                for (int k = 0; k < 3; ++k)
                    //increase the degree of a vertex
                    //each time it appears
                    vertex_degree[tmpFaces[j][k]]++;

            for (int i = 0; i < V; i++) {
                min_degree = min(min_degree, vertex_degree[i]);
                max_degree = max(max_degree, vertex_degree[i]);
                avg_degree += vertex_degree[i];
                //clear it to be used in another iteration
                vertex_degree[i] = 0;
            }

            degrees.insert(mp(0, min_degree));
            degrees.insert(mp(1, max_degree));
            degrees.insert(mp(2, avg_degree / V));
            min_degree = MAXV;
            avg_degree = 0;
            //Analysis code ends here

            if (ans >= respMax) {
                respMax = ans;
                for (int j = 0; j < numFaces; ++j)
                    for (int k = 0; k < 3; ++k)
                        faces[j][k] = tmpFaces[j][k];
            }
        }
    }
    stop = getTime();

    //Construct the solution given the graph faces
    for (int i = 0; i < F; ++i) {
        int va = faces[i][0], vb = faces[i][1], vc = faces[i][2];
        if (va == vb && vb == vc)
            continue;
        R[va][vb] = R[vb][va] = graph[va][vb];
        R[va][vc] = R[vc][va] = graph[va][vc];
        R[vb][vc] = R[vc][vb] = graph[vb][vc];
    }
    //Print the graph
    printf("Printing generated graph:\n");
    for (int i = 0; i < V; ++i) {
        for (int j = i + 1; j < V; ++j)
            printf("%d ", (R[i][j] == -1 ? -1 : R[i][j]));
        printf("\n");
    }

    printElapsedTime(start, stop);
    printf("Maximum weight found: %d\n\n", respMax);

    for (auto d : degrees)
        printf("Degree type %d -- degree value %d\n", d.first, d.second);
    return 0;
}
