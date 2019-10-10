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
        // cout << "Constructor: " << u << " " << v << "\n";
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
// int edge_to_idx[MAXE];
map<ii, int> edge_to_idx;
//-----------------------------------------------------------------------------
/*
    Define the number of combinations.
    */
void sizeDefinitions()
{
    for (int64 i = 4LL; i <= MAXV; i++) {
        int64 resp = 1LL;
        for (int64 j = i - 3; j <= i; j++)
            resp *= j;
        resp /= 24LL;
        numComb[i - 1] = resp;
    }
}
//-----------------------------------------------------------------------------
void readInput()
{
    cin >> V;
    for (int i = 0; i < V; i++) {
        for (int j = i + 1; j < V; j++) {
            cin >> graph[i][j];
            graph[j][i] = graph[i][j];
            R[i][j] = R[j][i] = -1;
        }
        graph[i][i] = -1;
        R[i][i] = -1;
    }

    int count = 0;
    //Map the edges
    for (int i = 0; i < V - 1; i++)
        for (int j = i + 1; j < V; j++, count++) {
            edge_to_idx[mp(i, j)] = count;
            // edge_to_idx[mp(j, i)] = count;
            idx_to_edge.pb(edge(i, j));
            // idx_to_edge.pb(edge(j, i));
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
    for (int i = 0; i < V; i++) {
        if (i != seeds[0] && i != seeds[1] && i != seeds[2] && i != seeds[3])
            vertices.insert(i);
    }
}
//-----------------------------------------------------------------------------
/*
    Returns the initial solution weight of the planar graph.
    */
int generateFaceList(int idx, set<int>& edges, vector<vi>& edges_faces,
    Face tmpFaces[][3], int* numFaces)
{
    vector<int> seeds = c.element(idx).getArray();
    int res = 0;
    for (int i = 0; i < C - 1; i++)
        for (int j = i + 1; j < C; j++) {
            int va = seeds[i], vb = seeds[j];
            res += graph[va][vb];
        }

    for (int i = 0; i < C - 2; i++)
        for (int j = i + 1; j < C - 1; j++)
            for (int k = j + 1; k < C; k++, (*numFaces)++) {
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
    Insert a new vertex, 3 new triangular faces
    and removes the face from the list.
    */
void faceDimple(int new_vertex, int face, int tmpFaces[][3], int* numFaces)
{
    int va = tmpFaces[face][0], vb = tmpFaces[face][1], vc = tmpFaces[face][2];

    //Remove the chosen face and insert a new one on its place.
    tmpFaces[face][0] = new_vertex, tmpFaces[face][1] = va,
    tmpFaces[face][2] = vb;

    //Insert the other two new faces.
    tmpFaces[*numFaces][0] = new_vertex, tmpFaces[*numFaces][1] = va,
    tmpFaces[(*numFaces)++][2] = vc;
    tmpFaces[*numFaces][0] = new_vertex, tmpFaces[*numFaces][1] = vb,
    tmpFaces[(*numFaces)++][2] = vc;
}
//-----------------------------------------------------------------------------
void addEdgeFace(edge at, int face, vector<vi>& edges_faces,
    bool check = false)
{
    int e = edge_to_idx[mp(at.u, at.v)];
    //Check wether this edge already belongs to two faces.
    if (check) {
        for (int i = 0; i < edges_faces[e].size(); i++)
            if (edges_faces[e][i] == face) {
                swap(edges_faces[e][i], edges_faces[e].back());
                edges_faces[e].pop_back();
                break;
            }
    }
    //Edge 'e' belongs to this face now.
    else {
        edges_faces[e].pb(face);
    }
}
//-----------------------------------------------------------------------------
/*
    Inserts a new vertex and 4 new triangular faces
    and removes two faces and an edge.
    */
void edgeDimple(int new_vertex, int edge_idx, int face, int extra, set<int>& edges,
    vector<vi>& edges_faces, int tmpFaces[][3], int* numFaces)
{
    //The removed edge.
    edge r_edge = idx_to_edge[edge_idx];

    vector<edge> used_edges;
    set<int> used;
    //Update face_edges
    for (int i = 0; i < 2; i++) {
        for (int j = i + 1; j < 3; j++) {
            int u = tmpFaces[face][i], v = tmpFaces[face][j];
            used.insert(u);
            used.insert(v);
            if (r_edge == edge(u, v)) {
                //Remove this edge from this face.
                addEdgeFace(edge(u, v), face, edges_faces, true);
            } else {
                used_edges.pb(edge(u, v));
                //Remove this edge from this face.
                addEdgeFace(edge(u, v), face, edges_faces, true);
            }
        }
    }
    for (int i = 0; i < 2; i++) {
        for (int j = i + 1; j < 3; j++) {
            int u = tmpFaces[extra][i], v = tmpFaces[extra][j];
            used.insert(u);
            used.insert(v);
            if (r_edge == edge(u, v)) {
                //Remove this edge from this face.
                addEdgeFace(edge(u, v), extra, edges_faces, true);
            } else {
                used_edges.pb(edge(u, v));
                //Remove this edge from this face.
                addEdgeFace(edge(u, v), extra, edges_faces, true);
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

    for (int i = 0; i < new_faces.size(); i++) {
        int f = new_faces[i];
        int va = used_edges[i].u, vb = used_edges[i].v;
        addEdgeFace(used_edges[i], f, edges_faces);
        addEdgeFace(edge(va, new_vertex), f, edges_faces);
        addEdgeFace(edge(vb, new_vertex), f, edges_faces);

        tmpFaces[f][0] = new_vertex, tmpFaces[f][1] = va,
        tmpFaces[f][2] = vb;
    }
}
//-----------------------------------------------------------------------------
/*
    Return the vertex with the maximum gain
    inserting within a face.
    */
node maxGainFace(set<int>& vertices, Face tmpFaces[][3], int* numFaces)
{
    node gains(-1, -1, -1, -1);
    //Iterate through the remaining vertices.
    for (int new_vertex : vertices) {
        //Test the dimple on each face
        for (int face = 0; face < *numFaces; face++) {
            int gain = 0;
            for (int k = 0; k < 3; k++) {
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
    Return the edge with the removal has the maximum gain when inserting
    a vertex into it.
    */
node maxGainEdge(set<int>& vertices, set<int>& edges, vector<vi>& edges_faces,
    Face tmpFaces[][3])
{
    node gains(-1, -1, -1, -1, -1);
    //Iterate through the remaining vertices
    // cout << "maxGainEdge: \n";
    for (int new_vertex : vertices) {
        //Test the dimple on each edge
        for (int e : edges) {
            int gain = 0;
            edge r = idx_to_edge[e];
            //Check these faces
            vector<int> faces_v = edges_faces[e];
            // cout << "Edge: " << r.u << " " << r.v << "\n";
            if (edges_faces[e].size() != 2)
                cout << "Num. of faces: " << edges_faces[e].size() << "\n";

            set<int> used;
            //2 faces for each vertex
            for (int f : faces_v) {
                //3 vertices for each face
                for (int k = 0; k < 3; k++) {
                    int u = tmpFaces[f][k];
                    // cout << u << " ";
                    //If I have not used this vertex yet
                    if (!used.count(u)) {
                        used.insert(u);
                        gain += graph[u][new_vertex];
                    }
                }
                // cout << "\n";
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
        // face_node gain = maxGainFace(vertices, tmpFaces, numFaces);
        node gain = maxGainEdge(vertices, edges, edges_faces, tmpFaces);

        vertices.erase(gain.vertex);
        edges.erase(gain.edge);
        maxValue += gain.w;
        // faceDimple(gain.vertex, gain.face, tmpFaces, numFaces);
        edgeDimple(gain.vertex, gain.edge, gain.face, gain.extra, edges,
            edges_faces, tmpFaces, numFaces);
    }
    return maxValue;
}
//-----------------------------------------------------------------------------
int main(int argv, char** argc)
{
    // ios::sync_with_stdio(false);
    double start, stop;

    //Read the input, which is given by the size of a graph and its weighted
    //edges. The given graph is complete.
    sizeDefinitions();
    readInput();

    int respMax = -1;

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
        generateVertexList(i, vertices);
        //Get the weight of the initial solution
        int tmpMax = generateFaceList(i, edges, edges_faces, tmpFaces, &numFaces);
        int ans = solve(vertices, edges, edges_faces, tmpMax, tmpFaces, &numFaces);

#pragma omp critical
        {
            if (ans >= respMax) {
                respMax = ans;
                for (int j = 0; j < numFaces; j++)
                    for (int k = 0; k < 3; k++)
                        faces[j][k] = tmpFaces[j][k];
            }
        }
    }
    stop = getTime();

    cout << "Printing generated graph: " << endl;
    //Construct the solution given the graph faces
    for (int i = 0; i < F; i++) {
        int va = faces[i][0], vb = faces[i][1], vc = faces[i][2];
        if (va == vb && vb == vc)
            continue;
        R[va][vb] = R[vb][va] = graph[va][vb];
        R[va][vc] = R[vc][va] = graph[va][vc];
        R[vb][vc] = R[vc][vb] = graph[vb][vc];
    }
    //Print the graph
    for (int i = 0; i < V; i++) {
        for (int j = i + 1; j < V; j++)
            cout << (R[i][j] == -1 ? -1 : R[i][j]) << " ";
        cout << endl;
    }
    //Print the vertices of each face
    // cout << F << endl;
    // for (int i = 0; i < F; i++)
    // {
    //     sort(faces[i], faces[i]+3);
    //     cout << faces[i][0] << " " << faces[i][1] << " " << faces[i][2] << endl;
    // }

    printElapsedTime(start, stop);
    cout << "Maximum weight found: " << respMax << endl;

    return 0;
}
