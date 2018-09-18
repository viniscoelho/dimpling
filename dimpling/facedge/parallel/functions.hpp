#ifndef FUNCTIONS_H
#define FUNCTIONS_H
#endif

// Combinadic instance shared on the GPU
__shared__ Combination c;

/*
    Returns the index of an edge.
    */
__device__ int edge_to_idx(edge e, int SIZE)
{
    return e.u*SIZE + e.v;
}

//-----------------------------------------------------------------------------

/*
    Returns an edge given an index.
    */
__device__ edge idx_to_edge(int index, int SIZE)
{
    return edge(index/SIZE, index%SIZE);
}

//-----------------------------------------------------------------------------

/*
    t       ---> thread index
    offset  ---> given offset for each gpu
    Generates a list of vertices which are not on the initial planar graph.
    */
__device__ void generateVertexList(Graph* devG, Params* devP, int t, int offset)
{
    int len = devG->length;
    int range = devG->range;

    //Get the seed corresponding to the given index
    int *seeds = c.element(t + offset).getArray();

    int va = seeds[0], vb = seeds[1], vc = seeds[2], vd = seeds[3];
    for (int i = 0, pos = 0; i < len; ++i)
        if (i != va && i != vb && i != vc && i != vd)
            devP->V[t + (pos++) * range] = i;
}

//-----------------------------------------------------------------------------

/*
    t       ---> thread index
    offset  ---> given offset for each gpu
    Returns the initial solution weight for the planar graph and
    initializes necessary structures, such as the edges indexes,
    and defines which edges belongs to a face.
    */
__device__ void generateFaceList(Graph* devG, Params* devP, int graph[], int t,
    int offset)
{
    int len = devG->length;
    int range = devG->range;

    int *seeds = c.element(t + offset).getArray();
    HashBucket edges;

    int va = seeds[0], vb = seeds[1], vc = seeds[2], vd = seeds[3];

    // Sum the edges to get the solution weight value.
    int res = graph[va + len*vb] + graph[va + len*vc] + graph[vb + len*vc];
    res += graph[va + len*vd] + graph[vb + len*vd] + graph[vc + len*vd];

    for (int i = 0; i < C-2; ++i)
    {
        for (int j = i+1; j < C-1; ++j)
        {
            for (int k = j+1; k < C; ++k, ++devP->numFaces[t])
            {
                int numFaces = devP->numFaces[t];
                int numEdges = devP->numEdges[t];
                // Vertices of a face
                va = seeds[i], vb = seeds[j], vc = seeds[k];
                devP->F[t + (numFaces * 3) * range] = va,
                devP->F[t + (numFaces * 3 + 1) * range] = vb,
                devP->F[t + (numFaces * 3 + 2) * range] = vc;

                // Insert all the edges on a list
                int edge_a = edge_to_idx(edge(va, vb), len),
                    edge_b = edge_to_idx(edge(va, vc), len),
                    edge_c = edge_to_idx(edge(vb, vc), len);

                // Check whether this edge was already inserted
                if (!edges.find(edge_a))
                {
                    edges.insert(edge_a);
                    devP->E[t + (numEdges++) * range] = edge_a;
                }
                if (!edges.find(edge_b))
                {
                    edges.insert(edge_b);
                    devP->E[t + (numEdges++) * range] = edge_b;
                }
                if (!edges.find(edge_c))
                {
                    edges.insert(edge_c);
                    devP->E[t + (numEdges++) * range] = edge_c;
                }
                devP->numEdges[t] = numEdges;

                // Faces that each edge belongs to
                // Does it belong to a face already?
                if (devP->edges_faces[t + (edge_a * 2) * range] == -1)
                    devP->edges_faces[t + (edge_a * 2) * range] = numFaces;
                // If yes, just "push back"
                else devP->edges_faces[t + (edge_a * 2 + 1) * range] = numFaces;

                if (devP->edges_faces[t + (edge_b * 2) * range] == -1)
                    devP->edges_faces[t + (edge_b * 2) * range] = numFaces;
                else devP->edges_faces[t + (edge_b * 2 + 1) * range] = numFaces;

                if (devP->edges_faces[t + (edge_c * 2) * range] == -1)
                    devP->edges_faces[t + (edge_c * 2) * range] = numFaces;
                else devP->edges_faces[t + (edge_c * 2 + 1) * range] = numFaces;
            }
        }
    }

    devP->tmpMax[t] = res;
}

//-----------------------------------------------------------------------------

__device__ void addEdgeToFace(Graph* devG, Params* devP, edge cur_edge,
    int face, int t, bool check = false)
{
    int len = devG->length;
    int range = devG->range;
    int edge_idx = edge_to_idx(cur_edge, len);
    // Check whether cur_edge already belongs to two faces.
    if (check)
    {
        for (int i = 0; i < NUM_F_EDGE; ++i)
            if (devP->edges_faces[t + (edge_idx * 2 + i) * range] == face)
            {
                // Make sure that the 'back of the array' is empty.
                devP->edges_faces[t + (edge_idx * 2 + i) * range] =
                    devP->edges_faces[t + (edge_idx * 2 + 1) * range];
                devP->edges_faces[t + (edge_idx * 2 + 1) * range] = -1;
            }
    }
    // cur_edge belongs to this face now.
    else
    {
        if (devP->edges_faces[t + (edge_idx * 2) * range] == -1)
            devP->edges_faces[t + (edge_idx * 2) * range] = face;
        else devP->edges_faces[t + (edge_idx * 2 + 1) * range] = face;
    }
}

//-----------------------------------------------------------------------------

/*
    Inserts a new vertex and 4 new triangular faces
    and removes two faces and an edge.
    */
__device__ void edgeDimple(Graph* devG, Params* devP, int new_vertex,
    int edge_idx, int face, int extra, int t)
{
    int len = devG->length;
    int range = devG->range;
    int numEdges = devP->numEdges[t];

    //The removed edge.
    edge r_edge = idx_to_edge(edge_idx, len);

    HashBucket h;
    edge used_edges[6];
    int used[6], num_used = 0, num_edge = 0;
    //Update face_edges
    for (int i = 0; i < NUM_F_EDGE; ++i)
        for (int j = i+1; j < NUM_E_FACE; ++j)
        {
            int u = devP->F[t + (face * 3 + i) * range],
                v = devP->F[t + (face * 3 + j) * range];
            if (!h.find(u)){
                used[num_used++] = u;
                h.insert(u);
            }
            if (!h.find(v)){
                used[num_used++] = v;
                h.insert(v);
            }
            if (r_edge == edge(u, v))
            {
                //Remove this edge from this face.
                addEdgeToFace(devG, devP, edge(u, v), face, t, true);
            }
            else
            {
                used_edges[num_edge++] = edge(u, v);
                //Remove this edge from this face.
                addEdgeToFace(devG, devP, edge(u, v), face, t, true);
            }
        }

    for (int i = 0; i < NUM_F_EDGE; ++i)
        for (int j = i+1; j < NUM_E_FACE; ++j)
        {
            int u = devP->F[t + (extra * 3 + i) * range],
                v = devP->F[t + (extra * 3 + j) * range];
            if (!h.find(u)){
                used[num_used++] = u;
                h.insert(u);
            }
            if (!h.find(v)){
                used[num_used++] = v;
                h.insert(v);
            }
            if (r_edge == edge(u, v))
            {
                //Remove this edge from this face.
                addEdgeToFace(devG, devP, edge(u, v), extra, t, true);
            }
            else
            {
                used_edges[num_edge++] = edge(u, v);
                //Remove this edge from this face.
                addEdgeToFace(devG, devP, edge(u, v), extra, t, true);
            }
        }

    //Update edges: add(v, new_vertex), for each v in used
    for (int i = 0; i < num_used; ++i)
    {
        int e = edge_to_idx(edge(used[i], new_vertex), len);
        devP->E[t + (numEdges++) * range] = e;
    }
    devP->numEdges[t] = numEdges;

    int new_faces[4], num_faces = 0;
    new_faces[num_faces++] = face; new_faces[num_faces++] = extra;
    new_faces[num_faces++] = devP->numFaces[t]++;
    new_faces[num_faces++] = devP->numFaces[t]++;

    for (int i = 0; i < num_faces; ++i)
    {
        int f = new_faces[i];
        int va = used_edges[i].u, vb = used_edges[i].v;

        addEdgeToFace(devG, devP, used_edges[i], f, t);
        addEdgeToFace(devG, devP, edge(va, new_vertex), f, t);
        addEdgeToFace(devG, devP, edge(vb, new_vertex), f, t);

        devP->F[t + (f * 3) * range] = new_vertex;
        devP->F[t + (f * 3 + 1) * range] = va;
        devP->F[t + (f * 3 + 2) * range] = vb;
    }
}

//-----------------------------------------------------------------------------

/*
    Returns the max gain and the removed edge index
    having the maximum gain inserting a vertex within it.
    */
__device__ node maxGainEdge(Graph* devG, Params* devP, int graph[], int t)
{
    int len = devG->length;
    int range = devG->range;

    node gains(-1, -1, -1, -1, -1);
    //Iterate through the remaining vertices.
    int remain = devP->remaining[t];
    int num_edges = devP->numEdges[t];
    for (int v_i = 0; v_i < remain; ++v_i)
    {
        int new_vertex = devP->V[t + v_i * range];
        //Test the dimple on each edge
        for (int e_i = 0; e_i < num_edges; ++e_i)
        {
            int gain = 0, cur_edge = devP->E[t + e_i * range];
            edge r = idx_to_edge(cur_edge, len);
            //Check these faces
            int faces_v[2];
            faces_v[0] = devP->edges_faces[t + (cur_edge * 2) * range],
            faces_v[1] = devP->edges_faces[t + (cur_edge * 2 + 1) * range];

            HashBucket used;
            //2 faces for each vertex
            for (int f = 0; f < NUM_F_EDGE; ++f)
            {
                //3 vertices for each face
                for (int k = 0; k < NUM_E_FACE; ++k)
                {
                    int u = devP->F[t + (faces_v[f] * 3 + k) * range];
                    //If I have not used this vertex yet
                    if (!used.find(u))
                    {
                        used.insert(u);
                        gain += graph[u*len + new_vertex];
                    }
                }
            }

            gain -= graph[r.u*len + r.v];
            //This way, I don't have to worry about which order they are stored.
            if (gain > gains.w)
                gains = node(gain, v_i, e_i, faces_v[0], faces_v[1]);
        }
    }
    return gains;
}

//-----------------------------------------------------------------------------

/*
    Inserts a new vertex, 3 new triangular faces
    and removes the face from the list.
    */
__device__ void faceDimple(Graph *devG, Params *devP, int new_vertex,
    int face, int t)
{
    int len = devG->length;
    int range = devG->range;
    int numEdges = devP->numEdges[t];

    edge used_edges[6];
    int used[6], num_used = 0, num_edge = 0;
    HashBucket h;
    //Update face_edges
    for (int i = 0; i < NUM_F_EDGE; ++i)
    {
        for (int j = i+1; j < NUM_E_FACE; ++j)
        {
            int u = devP->F[t + (face * 3 + i) * range],
                v = devP->F[t + (face * 3 + j) * range];
            if (!h.find(u))
            {
                used[num_used++] = u;
                h.insert(u);
            }
            if (!h.find(v))
            {
                used[num_used++] = v;
                h.insert(v);
            }
            //Remove this edge from this face.
            used_edges[num_edge++] = edge(u, v);
            addEdgeToFace(devG, devP, edge(u, v), face, t, true);
        }
    }

    //Update edges: add(v, new_vertex), for each v in used
    for (int i = 0; i < num_used; ++i)
    {
        int e = edge_to_idx(edge(used[i], new_vertex), len);
        devP->E[t + (numEdges++) * range] = e;
    }
    devP->numEdges[t] = numEdges;

    int new_faces[3], num_faces = 0;
    new_faces[num_faces++] = face;
    new_faces[num_faces++] = devP->numFaces[t]++;
    new_faces[num_faces++] = devP->numFaces[t]++;

    for (int i = 0; i < num_faces; ++i)
    {
        int f = new_faces[i];
        int va = used_edges[i].u, vb = used_edges[i].v;

        addEdgeToFace(devG, devP, used_edges[i], f, t);
        addEdgeToFace(devG, devP, edge(va, new_vertex), f, t);
        addEdgeToFace(devG, devP, edge(vb, new_vertex), f, t);

        devP->F[t + (f * 3) * range] = new_vertex;
        devP->F[t + (f * 3 + 1) * range] = va;
        devP->F[t + (f * 3 + 2) * range] = vb;
    }
}

//-----------------------------------------------------------------------------

/*
    Returns the max gain and the vertex index having the maximum gain
    inserting within a face.
    */
__device__ node maxGainFace(Graph *devG, Params *devP, int graph[], int t)
{
    int len = devG->length;
    int range = devG->range;

    node gains(-1, -1, -1, -1, -1);
    // Iterate through the remaining vertices.
    int remain = devP->remaining[t];
    int num_faces = devP->numFaces[t];
    for (int v_i = 0; v_i < remain; ++v_i)
    {
        int new_vertex = devP->V[t + v_i * range];
        // Test the dimple on each face
        for (int face = 0; face < num_faces; ++face)
        {
            int gain = 0;
            for (int k = 0; k < NUM_E_FACE; ++k)
            {
                int u = devP->F[t + (face * 3 + k) * range];
                gain += graph[u*len + new_vertex];
            }
            if (gain > gains.w)
                gains = node(gain, v_i, -1, face);
        }
    }
    return gains;
}

//-----------------------------------------------------------------------------

__device__ void dimpling(Graph *devG, Params *devP, int graph[], int t)
{
    int range = devG->range;

    while (devP->remaining[t])
    {
        //Last position of the list of vertices.
        int last_vertex = devP->remaining[t] - 1;
        int last_edge = devP->numEdges[t] - 1;

        node gain_f = maxGainFace(devG, devP, graph, t);
        node gain_e = maxGainEdge(devG, devP, graph, t);

        if (gain_f.w >= gain_e.w)
        {
            int new_vertex = devP->V[t + gain_f.vertex * range];
            //Compress the list of vertices and remove the chosen vertex.
            for (int i = gain_f.vertex; i <= last_vertex; ++i)
                devP->V[t + i * range] = devP->V[t + (i+1) * range];

            devP->tmpMax[t] += gain_f.w;
            faceDimple(devG, devP, new_vertex, gain_f.face, t);
        }
        else
        {
            int new_vertex = devP->V[t + gain_e.vertex * range];
            //Compress the list of vertices and remove the chosen vertex.
            for (int i = gain_e.vertex; i <= last_vertex; ++i)
                devP->V[t + i * range] = devP->V[t + (i+1) * range];

            int removed_edge = devP->E[t + gain_e.edge * range];
            //Compress the list of edges and remove the chosen edge.
            for (int i = gain_e.edge; i <= last_edge; ++i)
                devP->E[t + i * range] = devP->E[t + (i+1) * range];
            devP->numEdges[t]--;

            devP->tmpMax[t] += gain_e.w;
            edgeDimple(devG, devP, new_vertex, removed_edge, gain_e.face,
                gain_e.extra, t);
        }
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

__device__ void initializeDevice(Graph *devG, Params *devP, int remaining, int t)
{
    devP->numFaces[t] = 0;
    devP->numEdges[t] = 0;
    devP->tmpMax[t] = -1;
    devP->remaining[t] = remaining - 4;
}
