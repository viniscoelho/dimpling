#ifndef FUNCTIONS_H
    #define FUNCTIONS_H
#endif

void addEdgeToFace(edge at, int face, vector<vi>& edges_faces,
    bool check = false)
{
    int e = edge_to_idx(at);
    //Check wether this edge already belongs to two faces.
    if (check)
    {
        for (int i = 0; i < edges_faces[e].size(); ++i)
            if (edges_faces[e][i] == face)
            {
                swap(edges_faces[e][i], edges_faces[e].back());
                edges_faces[e].pop_back();
                break;
            }
    }
    //Edge 'e' belongs to this face now.
    else edges_faces[e].pb(face);
}
//-----------------------------------------------------------------------------
/*
    Inserts a new vertex and 4 new triangular faces
    and removes two faces and an edge.
    */
void edgeDimple(int new_vertex, int edge_idx, int face, int extra,
    vector<int>& edges, vector<vi>& edges_faces, int tmpFaces[][NUM_E_FACE],
    int *numFaces)
{
    //The removed edge.
    edge r_edge = idx_to_edge(edge_idx);

    vector<edge> used_edges;
    set<int> used;
    //Update face_edges
    for (int i = 0; i < NUM_F_EDGE; ++i)
    {
        for (int j = i+1; j < NUM_E_FACE; ++j)
        {
            int u = tmpFaces[face][i], v = tmpFaces[face][j];
            used.insert(u); used.insert(v);
            if (r_edge == edge(u, v))
            {
                //Remove this edge from this face.
                addEdgeToFace(edge(u, v), face, edges_faces, true);
            }
            //If this is not the removed edge, add it to a list.
            else
            {
                used_edges.pb(edge(u, v));
                //Remove this edge from this face.
                addEdgeToFace(edge(u, v), face, edges_faces, true);
            }
        }
    }
    for (int i = 0; i < NUM_F_EDGE; ++i)
    {
        for (int j = i+1; j < NUM_E_FACE; ++j)
        {
            int u = tmpFaces[extra][i], v = tmpFaces[extra][j];
            used.insert(u); used.insert(v);
            if (r_edge == edge(u, v))
            {
                //Remove this edge from this face.
                addEdgeToFace(edge(u, v), extra, edges_faces, true);
            }
            //If this is not the removed edge, add it to a list.
            else
            {
                used_edges.pb(edge(u, v));
                //Remove this edge from this face.
                addEdgeToFace(edge(u, v), extra, edges_faces, true);
            }
        }
    }
    //Update edges: add(v, new_vertex), for each v in used
    for (int v : used)
    {
        int e = edge_to_idx(edge(v, new_vertex));
        edges.pb(e);
    }

    vector<int> new_faces;
    new_faces.pb(face); new_faces.pb(extra);
    new_faces.pb((*numFaces)++); new_faces.pb((*numFaces)++);

    for (int i = 0; i < new_faces.size(); ++i)
    {
        int f = new_faces[i];
        //Connect the vertices from this edge to the new vertex.
        int va = used_edges[i].u, vb = used_edges[i].v;
        //Add this edge to this face as well.
        addEdgeToFace(used_edges[i], f, edges_faces);
        addEdgeToFace(edge(va, new_vertex), f, edges_faces);
        addEdgeToFace(edge(vb, new_vertex), f, edges_faces);

        tmpFaces[f][0] = new_vertex, tmpFaces[f][1] = va,
            tmpFaces[f][2] = vb;
    }
}
//-----------------------------------------------------------------------------
/*
    Return the edge with the removal has the maximum gain when inserting
    a vertex into it.
    */
node maxGainEdge(vector<int>& vertices, vector<int>& edges,
    vector<vi>& edges_faces, Face tmpFaces[][NUM_E_FACE])
{
    // cout << "maxGainEdge: \n";
    node gains(-1, -1, -1, -1, -1);
    //Iterate through the remaining vertices
    int vertex_pos = 0;
    for (int new_vertex : vertices)
    {
        //Test the dimple on each edge
        int edge_pos = 0;
        for (int e : edges)
        {
            int gain = 0;
            edge r = idx_to_edge(e);
            //Check these faces
            vector<int> faces_v = edges_faces[e];
            set<int> used;
            //2 faces for each vertex
            for (int f : faces_v)
            {
                //3 vertices for each face
                for (int k = 0; k < NUM_E_FACE; ++k)
                {
                    int u = tmpFaces[f][k];
                    //If I have not used this vertex yet
                    if (!used.count(u))
                    {
                        used.insert(u);
                        gain += graph[u][new_vertex];
                    }
                }
            }

            gain -= graph[r.u][r.v];
            //This way, I don't have to worry about which order they are stored.
            if (gain > gains.w || (gain == gains.w && new_vertex < vertices[gains.vertex]))
                gains = node(gain, vertex_pos, edge_pos, faces_v[0], faces_v[1]);
            edge_pos++;
        }
        vertex_pos++;
    }
    return gains;
}
//-----------------------------------------------------------------------------
/*
    Insert a new vertex, 3 new triangular faces
    and removes the face from the list.
    */
void faceDimple(int new_vertex, int face, vector<int>& edges, vector<vi>& edges_faces,
    int tmpFaces[][NUM_E_FACE], int *numFaces)
{
    vector<edge> used_edges;
    set<int> used;
    //Update face_edges
    for (int i = 0; i < NUM_F_EDGE; ++i)
    {
        for (int j = i+1; j < NUM_E_FACE; ++j)
        {
            int u = tmpFaces[face][i], v = tmpFaces[face][j];
            used_edges.pb(edge(u, v));
            used.insert(u); used.insert(v);
            //Remove this edge from this face.
            addEdgeToFace(edge(u, v), face, edges_faces, true);
        }
    }

    //Update edges: add(v, new_vertex), for each v in used
    for (int v : used)
    {
        int e = edge_to_idx(edge(v, new_vertex));
        edges.pb(e);
    }

    vector<int> new_faces;
    new_faces.pb(face);
    new_faces.pb((*numFaces)++); new_faces.pb((*numFaces)++);

    for (int i = 0; i < new_faces.size(); ++i)
    {
        int f = new_faces[i];
        //Connect the vertices from this edge to the new vertex.
        int va = used_edges[i].u, vb = used_edges[i].v;
        //Add this edge to face 'f' as well.
        addEdgeToFace(used_edges[i], f, edges_faces);
        addEdgeToFace(edge(va, new_vertex), f, edges_faces);
        addEdgeToFace(edge(vb, new_vertex), f, edges_faces);

        tmpFaces[f][0] = new_vertex, tmpFaces[f][1] = va,
            tmpFaces[f][2] = vb;
    }
}
//-----------------------------------------------------------------------------
/*
    Return the vertex having the maximum gain
    inserting within a face.
    */
node maxGainFace(vector<int>& vertices, Face tmpFaces[][NUM_E_FACE],
    int *numFaces)
{
    node gains(-1, -1, -1, -1);
    //Iterate through the remaining vertices.
    int vertex_pos = 0;
    for (int new_vertex : vertices)
    {
        //Test the dimple on each face
        for (int face = 0; face < *numFaces; ++face)
        {
            int gain = 0;
            for (int k = 0; k < NUM_E_FACE; ++k)
            {
                int u = tmpFaces[face][k];
                gain += graph[u][new_vertex];
            }
            //This way, I don't have to worry about which order they are stored.
            if (gain > gains.w || (gain == gains.w && new_vertex < vertices[gains.vertex]))
                gains = node(gain, vertex_pos, -1, face);
        }
        vertex_pos++;
    }
    return gains;
}
//-----------------------------------------------------------------------------
