#include <bits/stdc++.h>
#define pb push_back
#define mp make_pair
#define fi first
#define se second

using namespace std;

#include "definitions.h"
#include "combinadic.h"
#include "default.h"
#include "functions.h"

//-----------------------------------------------------------------------------
void readInput()
{
    scanf("%d", &V);
    for (int i = 0; i < V; ++i)
    {
        for (int j = i+1; j < V; ++j)
        {
            scanf("%d", &graph[i][j]);
            graph[j][i] = graph[i][j];
            resGraph[i][j] = resGraph[j][i] = -1;
        }
        graph[i][i] = -1;
        resGraph[i][i] = -1;
    }

    COMB = numComb[V-1];
    F = 2*V-4;
    c = Combination(V, 4);
}
//-----------------------------------------------------------------------------
/*
    Generates a list having vertices which are not on the planar graph.
    */
void generateVertexList(int t, vector<int>& vertices)
{
    vector<int> seeds = c.element(t).getArray();
    for (int i = 0; i < V; ++i)
        if (i != seeds[0] && i != seeds[1] && i != seeds[2] && i != seeds[3])
            vertices.pb(i);
}
//-----------------------------------------------------------------------------
/*
    Returns the initial solution weight of the planar graph.
    */
int generateFaceList(int t, vector<int>& edges, vector<vi>& edges_faces,
    Face tmpFaces[][NUM_E_FACE], int *numFaces)
{
    vector<int> seeds = c.element(t).getArray();

    //A hash table
    hash_b h;

    int res = 0;
    for (int i = 0; i < C-1; ++i)
        for (int j = i+1; j < C; ++j)
        {
            int va = seeds[i], vb = seeds[j];
            res += graph[va][vb];
        }

    for (int i = 0; i < C-2; ++i)
        for (int j = i+1; j < C-1; ++j)
            for (int k = j+1; k < C; ++k, ++(*numFaces))
            {
                //Vertices of a face
                int va = seeds[i], vb = seeds[j], vc = seeds[k];
                tmpFaces[*numFaces][0] = va, tmpFaces[*numFaces][1] = vb,
                    tmpFaces[*numFaces][2] = vc;

                //Edges
                int ea = edge_to_idx(edge(va, vb)),
                    eb = edge_to_idx(edge(va, vc)),
                    ec = edge_to_idx(edge(vb, vc));
                //Insert all the edges on a list
                if (!h.find(ea))
                {
                    edges.pb(ea);
                    h.insert(ea);
                }
                if (!h.find(eb))
                {
                    edges.pb(eb);
                    h.insert(eb);
                }
                if (!h.find(ec))
                {
                    edges.pb(ec);
                    h.insert(ec);
                }
                //Faces which each edge belongs to
                edges_faces[ea].pb(*numFaces);
                edges_faces[eb].pb(*numFaces);
                edges_faces[ec].pb(*numFaces);
            }
    return res;
}
//-----------------------------------------------------------------------------
int solve(vector<int>& vertices, vector<int>& edges, vector<vi>& edges_faces,
    int tmpMax, Face tmpFaces[][NUM_E_FACE], int *numFaces)
{
    int maxValue = tmpMax;

    while (!vertices.empty())
    {
        node gain_f = maxGainFace(vertices, tmpFaces, numFaces);
        node gain_e = maxGainEdge(vertices, edges, edges_faces, tmpFaces);

        if (gain_f.w >= gain_e.w)
        {
            int new_vertex = vertices[gain_f.vertex];
            swap(vertices[gain_f.vertex], vertices.back());
            vertices.pop_back();

            maxValue += gain_f.w;
            faceDimple(new_vertex, gain_f.face, edges, edges_faces,
                tmpFaces, numFaces);
        }
        else
        {
            int new_vertex = vertices[gain_e.vertex];
            swap(vertices[gain_e.vertex], vertices.back());
            vertices.pop_back();

            int removed_edge = edges[gain_e.edge];
            swap(edges[gain_e.edge], edges.back());
            edges.pop_back();

            maxValue += gain_e.w;
            edgeDimple(new_vertex, removed_edge, gain_e.face, gain_e.extra,
                edges, edges_faces, tmpFaces, numFaces);
        }
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
    for (int i = 0; i < COMB; ++i)
    {
        //List of faces for solution i
        Face tmpFaces[MAXF][NUM_E_FACE];
        int numFaces = 0;
        vector<int> vertices, edges;
        //Which faces an edge belongs?
        vector<vi> edges_faces(MAXE);

        //A list with the remaining vertices
        generateVertexList(i, vertices);
        //Get the weight of the initial solution
        int tmpMax = generateFaceList(i, edges, edges_faces, tmpFaces, &numFaces);
        int ans = solve(vertices, edges, edges_faces, tmpMax, tmpFaces, &numFaces);

        #pragma omp critical
        {
            if (ans >= respMax)
            {
                respMax = ans;
                for (int j = 0; j < numFaces; ++j)
                    for (int k = 0; k < NUM_E_FACE; ++k)
                        faces[j][k] = tmpFaces[j][k];
            }
        }
    }
    stop = getTime();
    
    // printf("Printing generated graph:\n");
    //Construct the solution given the graph faces
    for (int i = 0; i < F; ++i)
    {
        int va = faces[i][0], vb = faces[i][1], vc = faces[i][2];
        if (va == vb && vb == vc) continue;
        resGraph[va][vb] = resGraph[vb][va] = graph[va][vb];
        resGraph[va][vc] = resGraph[vc][va] = graph[va][vc];
        resGraph[vb][vc] = resGraph[vc][vb] = graph[vb][vc];
    }
    //Print the graph
    printf("%d\n", V);
    for (int i = 0; i < V; ++i)
    {
        for (int j = i+1; j < V; ++j)
            printf("%d ", (resGraph[i][j] == -1 ? -1 : resGraph[i][j]));
        printf("\n");
    }
    //Print the vertices of each face
    // printf("%d\n", F);
    // for (int i = 0; i < F; i++)
    // {
    //     sort(faces[i], faces[i]+3);
    //     printf("%d %d %d\n", faces[i][0], faces[i][1], faces[i][2]);
    // }

    printElapsedTime(start, stop);
    printf("Maximum weight found: %d\n", respMax);

    return 0;
}
