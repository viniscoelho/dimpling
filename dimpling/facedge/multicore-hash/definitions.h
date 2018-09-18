#ifndef DEFINITIONS_H
    #define DEFINITIONS_H
#endif

typedef int Face;
typedef long long int64;
typedef unsigned long long uint64;
typedef pair<int, int> ii;
typedef vector<int> vi;

const double EPS = 1e-9;
const int INF = 0x3f3f3f3f;
const int MOD = 1000000007;

const int C = 4; //combination size
const int MAXV = 150; //max. number of vertices
const int MAXF = 2*MAXV-4; //number of regions on a planar graph
const int MAXE = MAXV*MAXV; //number of undirected edges
const int NUM_F_EDGE = 2; //number of faces that each edge belongs to
const int NUM_E_FACE = 3; //number of edges that each face has

int sgn(double a){ return ((a > EPS) ? (1) : ((a < -EPS) ? (-1) : (0))); }
int cmp(double a, double b = 0.0){ return sgn(a - b); }

struct node
{
    int w, vertex, edge, face, extra;
    node (int w = 0, int vertex = 0, int edge = 0, int face = 0, int extra = 0)
        : w(w), vertex(vertex), edge(edge), face(face), extra(extra) {}
};

struct edge
{
    int u, v;
    edge (int u_t = 0, int v_t = 0)
    {
        u = min(u_t, v_t);
        v = max(v_t, u_t);
    }

    bool operator==(const edge& at)
    {
        return (u == at.u && v == at.v) || (v == at.u && u == at.v);
    }
};

struct hash_b
{
    //for 200 vertices: 200*200 / 62
    uint64 h[650];
    
    hash_b ()
    {
        for (int i = 0; i < 650; ++i) h[i] = 0LL;
    }

    bool find(int value)
    {
        int pos = value/62;
        return h[pos] & (1LL << (value % 63));
    }

    void insert(int value)
    {
        h[value/62] |= (1LL << (value % 63));
    }
};
