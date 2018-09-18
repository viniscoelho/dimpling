//-----------------------------------------------------------------------------
#include <utility>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <vector>
#include <queue>
#include <set>
#include <map>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <omp.h>
//-----------------------------------------------------------------------------
#include <ilcplex/ilocplex.h>
//-----------------------------------------------------------------------------
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/planar_face_traversal.hpp>
#include <boost/graph/boyer_myrvold_planar_test.hpp>
//-----------------------------------------------------------------------------
#define pb push_back
#define mp make_pair
#define MAXSTR 1024
#define MAX 100
#define EPS 1e-6

using namespace std;
using namespace boost;

#include "combinadic.h"

typedef pair<int, int> ii;
typedef adjacency_list< vecS, vecS, undirectedS,
    property<vertex_index_t, int>,
    property<edge_index_t, int> > Graph;
typedef vector< graph_traits<Graph>::edge_descriptor > vec_t;

bool found = false;
char errmsg[MAXSTR];

int nNodes = 0;
int nCols  = 0;
int cbSize = 4;

vector<vector<int>> graphIn;
map<vector<int>, int> colsId;

int sgn(double a) { return ((a > EPS) ? (1) : ((a < -EPS) ? (-1) : (0))); }
int cmp(double a, double b = 0.0) { return sgn(a - b); }

/*
    Print elapsed time.
    */
void printElapsedTime(double start, double stop)
{
    double elapsed = stop - start;
    printf("Elapsed time: %.3lfs.\n", elapsed);
}
//-----------------------------------------------------------------------------
// Mac
#ifdef __MACH__
#include <mach/clock.h>
#include <mach/mach.h>
#endif
//-----------------------------------------------------------------------------
/*  Get clock time.
    */
void current_utc_time(struct timespec *ts) 
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

double getTime()
{
    timespec ts;
    current_utc_time(&ts);
    return double(ts.tv_sec) + double(ts.tv_nsec) / 1e9;
}
//-----------------------------------------------------------------------------
int faces_count = 0;
map< vector<int>, int > hasFace;
// Some planar face traversal visitors that will 
// print the vertices and edges on the faces
struct output_visitor : public planar_face_traversal_visitor
{
    vector<int> faceVertices;
    void begin_face(){ printf("New face: "); }
    void end_face(){
        printf("\n");
        sort(faceVertices.begin(), faceVertices.end());
        hasFace[faceVertices] = 1;
        faceVertices.clear();
        faces_count++;
    }
};

struct vertex_output_visitor : public output_visitor
{
    template <typename Vertex> 
    void next_vertex(Vertex v) 
    { 
        std::cout << v + 1 << " ";
        faceVertices.pb((int)v);
    }
};

struct edge_output_visitor : public output_visitor
{
    template <typename Edge> 
    void next_edge(Edge e)
    { 
        std::cout << e << " "; 
    }
};
//-----------------------------------------------------------------------------
CPXENVptr env = NULL;
CPXLPptr  lp  = NULL;
//-----------------------------------------------------------------------------
void readData()
{
    printf("---------- Input graph ----------\n");
    //- - - - - - - - - - - - - - - - - - -
    scanf("%d", &nNodes);
    printf("%d\n", nNodes);
    // cbSize = nNodes-1;
    //- - - - - - - - - - - - - - - - - - -
    
    graphIn.resize( nNodes + 1 );
    for (int i = 1; i <= nNodes; i++)
    {
        graphIn[i].resize( nNodes + 1 );
    }
    
    //- - - - - - - - - - - - - - - - - - -
    for (int i = 1; i < nNodes; i++)
    {
        for (int j = i + 1; j <= nNodes; j++)
        {
            scanf("%d", &graphIn[i][j]);
            graphIn[j][i] = graphIn[i][j];
            printf("%d ", graphIn[i][j]);
        }
        printf("\n");
    }
    printf("---------------------------------\n\n");
    //- - - - - - - - - - - - - - - - - - -
}
//-----------------------------------------------------------------------------
int findColID( int i, int j, int k )
{
    map< vector<int>, int >::iterator itm;
    
    vector< int > v(3);
    v[0] = i;
    v[1] = j;
    v[2] = k;
    
    itm = colsId.find( v );
    if (itm == colsId.end())
    {
        return -1;
        // cout << "Cannot find colId x_" << j << '(' << h << ')' << '_' << k << '\n';
        // abort();
    }
    else
    {
        return itm->second;
    }
}
//-----------------------------------------------------------------------------
void buildModel()
{
    int status = 0;
    
    //- - - - - - - - - - - - - - - - - - -
    CPXchgobjsen( env, lp, CPX_MAX );
    
    nCols  = nNodes * (nNodes - 1) / 2;
    printf("nCols = %d\n", nCols);
    nCols += nNodes * (nNodes * nNodes - 3 * nNodes + 2) / 6;
    printf("nCols = %d\n", nCols);
    //- - - - - - - - - - - - - - - - - - -
    
    //- - - - - - - - - - - - - - - - - - -
    char    sense[2];
    int     rmatbeg[2];
    double  rhs[2];
    
    char*   ctype    = new char[nCols + 1];
    char**  cname    = new char*[nCols + 1];
    char**  rname    = new char*[1];
    rname[0] = new char[20];
    int*    rmatind  = new int[nCols + 1];
    double* rmatval  = new double[nCols + 1];
    double* obj      = new double[nCols + 1];
    double* lb       = new double[nCols + 1];
    double* ub       = new double[nCols + 1];
    //- - - - - - - - - - - - - - - - - - -
    
    //- - - - - - - - - - - - - - - - - - -
    int col = 0;
    for (int i = 1; i < nNodes; i++)
    {
        for (int j = i + 1; j <= nNodes; j++, col++)
        {
            vector<int> v(3);
            v[0] = i;
            v[1] = j;
            v[2] = 0;
            colsId[v] = col;
            
            cname[col] = new char[20];
            sprintf(cname[col], "x_%d_%d", i, j);
            lb[col]    = 0.0;
            ub[col]    = 1.0;
            ctype[col] = 'B';
            obj[col]   = graphIn[i][j];
        }
    }
    //- - - - - - - - - - - - - - - - - - -
    printf("col = %d\n", col);
    
    //- - - - - - - - - - - - - - - - - - -
    for (int i = 1; i < nNodes - 1; i++)
    {
        for (int j = i + 1; j < nNodes; j++)
        {
            for (int k = j + 1; k <= nNodes; k++, col++)
            {
                vector<int> v(3);
                v[0] = i;
                v[1] = j;
                v[2] = k;
                colsId[v] = col;
                
                cname[col] = new char[20];
                sprintf(cname[col], "f_%d_%d_%d", i, j, k);
                lb[col]    = 0.0;
                ub[col]    = 1.0;
                ctype[col] = 'B';
                obj[col]   = 0.0;
            }
        }
    }
    //- - - - - - - - - - - - - - - - - - -
    printf("col = %d\n", col);
    
    //- - - - - - - - - - - - - - - - - - -
    if (status = CPXnewcols( env, lp, nCols, obj, lb, ub, ctype, (char **) cname ))
    {
        printf("CPXnewcols: Could not add new columns, error %d\n", status);
        abort();
    }
    //- - - - - - - - - - - - - - - - - - -
    printf("CCCCCC\n");
    
    //R7 - - - - - - - - - - - - - - - - - - -
    rhs[0]     = 3.0 * (nNodes - 2);
    sense[0]   = 'E';
    rmatbeg[0] = 0;
    
    int p = 0;
    for (int i = 1; i < nNodes; i++)
    {
        for (int j = i + 1; j <= nNodes; j++, p++)
        {
            rmatind[p] = p;
            rmatval[p] = 1.0;
        }
    }
    sprintf(rname[0], "R7");
    
    if (status = CPXaddrows( env, lp,
                            (int) 0, (int) 1, p,
                            (const double *) rhs, (const char *) sense,
                            (const int *) rmatbeg, (const int *) rmatind,
                            (const double *) rmatval,
                            NULL, (char **) rname ))
    {
        printf("CPXaddrows: Could not add new rows (%s)\n", rname);
        abort();
    }
    printf("777777\n");
    //R7 - - - - - - - - - - - - - - - - - - -
    
    //R8 - - - - - - - - - - - - - - - - - - -
    rhs[0]   = 0.0;
    sense[0] = 'E';
    rmatbeg[0] = 0;
    
    for (int i = 1; i < nNodes; i++)
    {
        for (int j = i + 1; j <= nNodes; j++)
        {
            p = 0;
            for (int k = 1; k < i; k++, p++)
            {
                rmatind[p] = findColID( k, i, j );;
                rmatval[p] = 1.0;
            }
            for (int k = i + 1; k < j; k++, p++)
            {
                rmatind[p] = findColID( i, k, j );;
                rmatval[p] = 1.0;
            }
            for (int k = j + 1; k <= nNodes; k++, p++)
            {
                rmatind[p] = findColID( i, j, k );;
                rmatval[p] = 1.0;
            }
            rmatind[p] = findColID( i, j, 0 );;
            rmatval[p] = -2.0;
            p++;
            
            sprintf(rname[0], "R8_%d_%d", i, j);
            
            if (status = CPXaddrows( env, lp,
                                    (int) 0, (int) 1, p,
                                    (const double *) rhs, (const char *) sense,
                                    (const int *) rmatbeg, (const int *) rmatind,
                                    (const double *) rmatval,
                                    NULL, (char **) rname ))
            {
                printf("CPXaddrows: Could not add new rows (%s)\n", rname);
                abort();
            }
        }
    }
    printf("888888\n");
    //R8 - - - - - - - - - - - - - - - - - - -
    
    //R9 - - - - - - - - - - - - - - - - - - -
    rhs[0]     = 3.0;
    sense[0]   = 'G';
    rmatbeg[0] = 0;
    
    for (int i = 1; i <= nNodes; i++)
    {
        p = 0;
        for (int j = 1; j < i; j++, p++)
        {
            rmatind[p] = findColID( j, i, 0 );
            rmatval[p] = 1.0;
        }
        for (int j = nNodes; j > i; j--, p++)
        {
            rmatind[p] = findColID( i, j, 0 );
            rmatval[p] = 1.0;
        }
        sprintf(rname[0], "R9_%d", i);
        
        if (status = CPXaddrows( env, lp,
                                (int) 0, (int) 1, p,
                                (const double *) rhs, (const char *) sense,
                                (const int *) rmatbeg, (const int *) rmatind,
                                (const double *) rmatval,
                                NULL, (char **) rname ))
        {
            printf("CPXaddrows: Could not add new rows (%s)\n", rname);
            abort();
        }
    }
    printf("999999\n");
    //R9  - - - - - - - - - - - - - - - - - - -
    
    //R10 - - - - - - - - - - - - - - - - - - -
    rhs[0]     = 0.0;
    sense[0]   = 'L';
    rmatbeg[0] = 0;
    
    for (int i = 1; i < nNodes - 1; i++)
    {
        for (int j = i + 1; j < nNodes; j++)
        {
            for (int k = j + 1; k <= nNodes; k++)
            {
                p = 0;
                rmatind[p] = findColID( i, j, k );;
                rmatval[p] = 1.0;
                p++;
                rmatind[p] = findColID( i, j, 0 );;
                rmatval[p] = -1.0;
                p++;
                
                sprintf(rname[0], "R10_%d_%d_%d", i, j, k);
                
                if (status = CPXaddrows( env, lp,
                                        (int) 0, (int) 1, p,
                                        (const double *) rhs, (const char *) sense,
                                        (const int *) rmatbeg, (const int *) rmatind,
                                        (const double *) rmatval,
                                        NULL, (char **) rname ))
                {
                    printf("CPXaddrows: Could not add new rows (%s)\n", rname);
                    abort();
                }
            }
        }
    }
    printf("101010\n");
    //R10 - - - - - - - - - - - - - - - - - - -
    
    //R11 - - - - - - - - - - - - - - - - - - -
    rhs[0]     = 0.0;
    sense[0]   = 'L';
    rmatbeg[0] = 0;
    
    for (int i = 1; i < nNodes - 1; i++)
    {
        for (int j = i + 1; j < nNodes; j++)
        {
            for (int k = j + 1; k <= nNodes; k++)
            {
                p = 0;
                rmatind[p] = findColID( i, j, k );;
                rmatval[p] = 1.0;
                p++;
                rmatind[p] = findColID( j, k, 0 );;
                rmatval[p] = -1.0;
                p++;
                
                sprintf(rname[0], "R11_%d_%d_%d", i, j, k);
                
                if (status = CPXaddrows( env, lp,
                                        (int) 0, (int) 1, p,
                                        (const double *) rhs, (const char *) sense,
                                        (const int *) rmatbeg, (const int *) rmatind,
                                        (const double *) rmatval,
                                        NULL, (char **) rname ))
                {
                    printf("CPXaddrows: Could not add new rows (%s)\n", rname);
                    abort();
                }
            }
        }
    }
    printf("111111\n");
    //R11 - - - - - - - - - - - - - - - - - - -
    
    //R12 - - - - - - - - - - - - - - - - - - -
    rhs[0]     = 0.0;
    sense[0]   = 'L';
    rmatbeg[0] = 0;
    
    for (int i = 1; i < nNodes - 1; i++)
    {
        for (int j = i + 1; j < nNodes; j++)
        {
            for (int k = j + 1; k <= nNodes; k++)
            {
                p = 0;
                rmatind[p] = findColID( i, j, k );;
                rmatval[p] = 1.0;
                p++;
                rmatind[p] = findColID( i, k, 0 );;
                rmatval[p] = -1.0;
                p++;
                
                sprintf(rname[0], "R12_%d_%d_%d", i, j, k);
                
                if (status = CPXaddrows( env, lp,
                                        (int) 0, (int) 1, p,
                                        (const double *) rhs, (const char *) sense,
                                        (const int *) rmatbeg, (const int *) rmatind,
                                        (const double *) rmatval,
                                        NULL, (char **) rname ))
                {
                    printf("CPXaddrows: Could not add new rows (%s)\n", rname);
                    abort();
                }
            }
        }
    }
    printf("121212\n");
    //R12 - - - - - - - - - - - - - - - - - - -
    
    //- - - - - - - - - - - - - - - - - - -
    delete[] rmatind;
    
    delete[] rmatval;
    delete[] obj;
    delete[] lb;
    delete[] ub;
    delete[] ctype;
    
    for (int i = 0; i <= nCols; i++)
    {
        delete[] cname[i];
    }
    delete[] cname;
    
    delete[] rname[0];
    delete[] rname;
    //- - - - - - - - - - - - - - - - - - -
}
//-----------------------------------------------------------------------------
void openCplex()
{
    int status = 0;
    
    if (!(env = CPXopenCPLEX( &status )))
    {
        printf("Could not open CPLEX environment.\n");
        CPXgeterrorstring( env, status, errmsg );
        printf("%s\n", errmsg);
        abort();
    }
    
    if (status = CPXsetintparam( env, CPX_PARAM_SCRIND, CPX_ON ))
    {
        printf("PARAM_SCRIND: Failure to turn on screen indicator, error %d\n",
            status);
        abort();
    }
    
    if (status = CPXsetintparam (env, CPX_PARAM_SIMDISPLAY, 2))
    {
        printf("PARAM_SIMDISPLAY: Failed to turn up simplex display level.\n");
        abort();
    }
    
    if (status = CPXsetintparam( env, CPX_PARAM_DATACHECK, CPX_ON ))
    {
        printf("PARAM_DATACHECK: Failure to turn on data checking, error %d\n",
            status);
        abort();
    }
    
    if (!(lp = CPXcreateprob( env, &status, "PLANAR" )))
    {
        printf("Failed to create LP PLANAR.\n");
        abort();
    }
}
//-----------------------------------------------------------------------------
void writeSolutions()
{
    int    solStat  = 0;
    double objValue = 0.0;
    
    double* x = new double[nCols + 1];
    if (x == NULL)
    {
        printf("Could not allocate memory for solution.\n");
        abort();
    }
    
    int status = CPXsolution( env, lp, &solStat, &objValue, x, NULL, NULL, NULL);
    if (status)
    {
        printf("Failed to obtain cplex solution.\n");
        abort();
    }
    
    printf("\nSolution status = %d\n", solStat);
    printf("Solution value  = %.0lf\n\n", objValue);
    
    //- - - - - - - - - - - - - - - - - - -
    int col = 0;
    for (int i = 1; i < nNodes; i++)
    {
        for (int j = i + 1; j <= nNodes; j++, col++)
        {
            if (cmp(x[col]) == 1)
            {
                printf("x(%d,%d) = %0.lf\n", i, j, x[col]);
            }
        }
    }
    
    for (int i = 1; i < nNodes - 1; i++)
    {
        for (int j = i + 1; j < nNodes; j++)
        {
            for (int k = j + 1; k <= nNodes; k++, col++)
            {
                if (cmp(x[col]) == 1)
                {
                    printf("f(%d,%d, %d) = %0.lf\n", i, j, k, x[col]);
                }
            }
        }
    }
    //- - - - - - - - - - - - - - - - - - -
    delete[] x;
    //- - - - - - - - - - - - - - - - - - -
}
//-----------------------------------------------------------------------------
void addCut( vector<int>& S )
{
    int status = 0;
    static int r13 = 1;
    
    //- - - - - - - - - - - - - - - - - - -
    char    sense[2];
    int     rmatbeg[2];
    double  rhs[2];
    
    char**  rname    = new char*[1];
    rname[0]         = new char[20];
    int*    rmatind  = new int[nCols + 1];
    double* rmatval  = new double[nCols + 1];
    //- - - - - - - - - - - - - - - - - - -
    
    //- - - - - - - - - - - - - - - - - - -
    rhs[0]   = 2 * (S.size() - 2) - 1;
    sense[0] = 'L';
    rmatbeg[0] = 0;
    
    int p = 0;
    for (int i = 1; i < S.size() - 1; i++)
    {
        for (int j = i + 1; j < S.size(); j++)
        {
            for (int k = j + 1; k <= S.size(); k++)
            {
                vector<int> tmpVertices;
                tmpVertices.pb(S[i-1]); tmpVertices.pb(S[j-1]); tmpVertices.pb(S[k-1]);
                if (hasFace.count(tmpVertices))
                {
                    rmatind[p] = findColID( S[i-1]+1, S[j-1]+1, S[k-1]+1 );
                    rmatval[p] = 1.0;
                    p++;
                }
            }
        }
    }
    
    sprintf(rname[0], "R13_%d", r13++);
    
    if (status = CPXaddrows( env, lp,
                            (int) 0, (int) 1, p,
                            (const double *) rhs, (const char *) sense,
                            (const int *) rmatbeg, (const int *) rmatind,
                            (const double *) rmatval,
                            NULL, (char **) rname ))
    {
        printf("CPXaddrows: Could not add new rows (%s)\n", rname);
        abort();
    }
    //- - - - - - - - - - - - - - - - - - -
    
    //- - - - - - - - - - - - - - - - - - -
    delete[] rmatind;  
    delete[] rmatval;
    delete[] rname[0];
    delete[] rname;
    //- - - - - - - - - - - - - - - - - - -
    
}
//-----------------------------------------------------------------------------
//vector<ii> getEdges()
map<ii, bool> getEdges()
{
    int    solStat  = 0;
    double objValue = 0.0;
    
    double* x = new double[nCols + 1];
    if (x == NULL)
    {
        printf("Could not allocate memory for solution.\n");
        abort();
    }
    
    int status = CPXsolution(env, lp, &solStat, &objValue, x, NULL, NULL, NULL);
    if (status)
    {
        printf("Failed to obtain cplex solution.\n");
        abort();
    }
    
    // vector<ii> resp;
    map<ii, bool> resp;
    //- - - - - - - - - - - - - - - - - - -
    int col = 0;
    for (int i = 1; i < nNodes; i++)
    {
        for (int j = i + 1; j <= nNodes; j++, col++)
        {
            if (cmp(x[col]) == 1)
            {
                // resp.pb(mp(i, j));
                resp[mp(i, j)] = true;
            }
        }
    }
    delete [] x;
    return resp;
}
//-----------------------------------------------------------------------------
//Solution obtained from the relaxed model.
map<ii, bool> sol;
//Was this combination of edges used as a restriction already?
map<vector<ii>, int> cutFound;
//A list having MPGs used as restrictions.
map<int64, vector<ii>> edgesListM;    //smallest
vector<vector<ii>> edgesListV;        //smpg

/*
    Add a set of edges that generates a cut.
    edgesIdx    ---> Set of edges
    t           ---> Iteration number
*/
void findCut(vector<ii> edgesIdx, int t)
{
    //get num of vertices
    //m = 3n - 6; n = (m+6)/3
    int graphSize = (edgesIdx.size()+6)/3;
    vector<int> S;
    set<int> V;

    //create the graph which will generate a restriction 13
    Graph tmp_planar(graphSize);
    for (int i = 0; i < edgesIdx.size(); ++i)
    {
        int u = edgesIdx[i].first, v = edgesIdx[i].second;
        if (!V.count(u))
        {
            S.pb(u);
            V.insert(u);
        }
        if (!V.count(v))
        {
            S.pb(v);
            V.insert(v);
        }
        add_edge(u, v, tmp_planar);
    }
    //Necessary.
    sort(S.begin(), S.end());

    //Initialize the interior edge index; necessary for face traversal.
    property_map<Graph, edge_index_t>::type e_index = get(edge_index, tmp_planar);
    graph_traits<Graph>::edges_size_type edge_count = 0;
    graph_traits<Graph>::edge_iterator ei, ei_end;
    for (tie(ei, ei_end) = edges(tmp_planar); ei != ei_end; ++ei)
        put(e_index, *ei, edge_count++);

    vector<vec_t> embedding(num_vertices(tmp_planar));
    if (boyer_myrvold_planarity_test(tmp_planar, &embedding[0]))
    {
        //Clear the face set.
        hasFace.clear();

        vertex_output_visitor v_vis;
        printf("---------- Adding cut ----------\n");
        planar_face_traversal(tmp_planar, &embedding[0], v_vis);
        printf("Number of faces = %d\n", faces_count);
        printf("Number of edges = %d\n", num_edges(tmp_planar));
        printf("--------------------------------\n");
        faces_count = 0;

        cutFound[edgesIdx]++;
        addCut(S);
    }
}
//-----------------------------------------------------------------------------
/*
    n       ---> Size of the input array
    k       ---> Size of the combination
    app     ---> Approach option
*/
void combine(int n, int k, int app)
{
    Combination c(n, k);
    int64 ub = c.choose(n, k);
    #pragma omp parallel for shared(edgesListM, edgesListV)
    for (int64 i = 0; i < ub; i++)
    {
        Graph tmp_planar(k);

        vector<int> S = c.element(i).getArray();
        vector<ii> edges;
        
        for (int vertex = 0; vertex < S.size()-1; vertex++)
        {
            int u = S[vertex];
            for (int adj = vertex+1; adj < S.size(); adj++)
            {
                int v = S[adj];
                //add an edge if (u, v) belongs to the solution found
                if (!sol.count(mp(u+1, v+1))) continue;
                edges.pb(mp(u, v));
                add_edge(u, v, tmp_planar);
            }
        }
        //Was this restriction inserted already?
        if (cutFound.count(edges)) continue;
        
        //Cannot be maximal
        if (num_edges(tmp_planar) != 3*S.size()-6) continue;

        //Is it planar?
        if (boyer_myrvold_planarity_test(tmp_planar))
        {
            #pragma omp critical
            {
                if (app) edgesListM[i] = edges;
                else edgesListV.pb(edges);
            }
        }
    }
}
//-----------------------------------------------------------------------------
void bcSmallest(int app)
{
    // mipBasis();
    int t = 0;
    while (true)
    {
        if (CPXmipopt( env, lp ))
        {
            printf("CPXmipopt: Failed to optimize LP.\n");
            abort();
        }
        char name[20];
        sprintf(name, "planar-%d.lp", t);
        CPXwriteprob(env, lp, name, NULL);

        sol = getEdges();
        map<ii, bool>::iterator ed;
        writeSolutions();

        //Build the solution graph.
        Graph planar;
        for (ed = sol.begin(); ed != sol.end(); ed++)
        {
            ii at = ed->first;
            int a = at.first-1, b = at.second-1;
            add_edge(a, b, planar);
        }

        if (boyer_myrvold_planarity_test(planar))
        {
            printf("Maximal Planar Subgraph found!\n");
            break;
        }
        else
        {
            //S initial size must be at least 4.
            edgesListM.clear();
            cbSize = 4;
            found = true;
            while (cbSize < nNodes)
            {
                double st = getTime();
                printf("Combination size = %d\n", cbSize);
                combine(nNodes, cbSize, app);
                double fn = getTime();
                printElapsedTime(st, fn);
                
                if (edgesListM.size()) break;
                cbSize++;
            }
            if (edgesListM.size() == 0)
            {
                printf("There is no MPG for restriction 13.\n");
                return;
            }
            //Use the first that was found.
            findCut(edgesListM.begin()->second, t);
        }
        // char m_name[20];
        // sprintf(m_name, "MIPS-%d", t);
        // CPXwritemipstarts(env, lp, m_name, 0, 0);
        t++;
    }
}
//-----------------------------------------------------------------------------
void bcSMPG(int app)
{
    // mipBasis();
    int t = 0;
    while (true)
    {
        if (CPXmipopt( env, lp ))
        {
            printf("CPXmipopt: Failed to optimize LP.\n");
            abort();
        }
        char name[20];
        sprintf(name, "planar-%d.lp", t);
        CPXwriteprob(env, lp, name, NULL);

        sol = getEdges();
        map<ii, bool>::iterator ed;
        writeSolutions();

        //Build the solution graph.
        Graph planar;
        for (ed = sol.begin(); ed != sol.end(); ed++)
        {
            ii at = ed->first;
            int a = at.first-1, b = at.second-1;
            add_edge(a, b, planar);
        }

        if (boyer_myrvold_planarity_test(planar))
        {
            printf("Maximal Planar Subgraph found!\n");
            break;
        }
        else
        {
            edgesListV.clear();
            cbSize = 4;
            found = true;
            while (cbSize < nNodes)
            {
                double st = getTime();
                printf("Combination size = %d\n", cbSize);
                combine(nNodes, cbSize, app);
                double fn = getTime();
                printElapsedTime(st, fn);
                cbSize++;
            }
            if (edgesListV.size() == 0)
            {
                printf("There is no MPG for restriction 13.\n");
                return;
            }

            //Having all MPGs generated, we have to get the smallest one
            //which is not a subgraph of any other.
            int idx = edgesListV.size()-1;
            vector<ii>::iterator vit;
            for (int i = 0; i < edgesListV.size()-1; i++)
            {
                found = true;
                for (int j = i+1; j < edgesListV.size(); j++)
                {
                    if (edgesListV[i].size() == edgesListV[j].size()) continue;

                    vector<ii> tmp(edgesListV[j].size());
                    vit = set_intersection(edgesListV[i].begin(), edgesListV[i].end(),
                        edgesListV[j].begin(), edgesListV[j].end(), tmp.begin());

                    //If the smallest subgraph is a subgraph of any other,
                    //then it cannot be a restriction.
                    int sz = (int)(vit-tmp.begin());
                    if (sz == edgesListV[i].size())
                    {
                        found = false;
                        break;
                    }
                }
                if (found)
                {
                    idx = i;
                    break;
                }
            }
            //Use the smallest as the best cut.
            findCut(edgesListV[idx], t);
        }
        t++;
    }
}
//-----------------------------------------------------------------------------
int main(int argv, char** argc)
{
    // ios::sync_with_stdio(false);
    if (argv < 2)
    {
        printf("ERROR! Required num. of arguments: 2\n");
        printf("Try:\nsmallest - ./a.out 1\nsmpg     - ./a.out 0\n");
        return 0;
    }
    
    int status;
    
    readData();
    openCplex();
    
    if (argv == 2)
    {
        buildModel();
    }
    else if (argv == 3)
    {
        if (status = CPXreadcopyprob(env, lp, argc[2], "LP"))
        {
            printf("Error! Could not read file!\n");
            exit(1);
        }
    }

    int option = atoi(argc[1]);
    
    if (option) bcSmallest(option);
    else bcSMPG(option);

    if (status = CPXwriteprob(env, lp, "planar.lp", NULL))
    {
        printf("CPXwriteprob: Failed to write LP to disk, error %d\n",
            status);
    }
    
    writeSolutions();
    
    if (status = CPXfreeprob(env, &lp))
    {
        printf("CPXfreeprob failed, error code %d\n", status);
    }
    
    if (status = CPXcloseCPLEX(&env))
    {
        CPXgeterrorstring(env, status, errmsg);
        printf("CPXcloseCPLEX: Could not close CPLEX environment.\n");
        printf("%s\n", errmsg);
    }
    //- - - - - - - - - - - - - - - - - - -
    
    return 0;
}
//-----------------------------------------------------------------------------
void mipBasis()
{
    int status = 0;

    // int cstat[nCols + 1];
    // int rstat[nRows + 1];

    int nFaces, x, nzcnt = 0, effortlevel = 0;
    int beg = 0;

    vector<int> varindices;
    vector<double> values;
    for (int i = 1; i < nNodes; i++)
    {
        for (int j = i + 1; j <= nNodes; j++)
        {
            scanf("%d", &x);
            vector<int> tmp;
            tmp.pb(i); tmp.pb(j); tmp.pb(0);

            varindices.pb(colsId[tmp]);
            values.pb(x ? 1.0 : 0.0);
            nzcnt++;
        }
    }

    map< vector<int>, int > sFaces;
    scanf("%d", &nFaces);
    for (int i = 0; i < nFaces; i++)
    {
        vector<int> sF;
        for (int j = 0; j < 3; j++)
        {
            scanf("%d", &x);
            sF.pb(x+1);
        }
        sort(sF.begin(), sF.end());
        sFaces[sF] = 1;
    }

    for (int i = 1; i < nNodes - 1; i++)
    {
        for (int j = i + 1; j < nNodes; j++)
        {
            for (int k = j + 1; k <= nNodes; k++)
            {
                vector<int> tmp;
                tmp.pb(i); tmp.pb(j); tmp.pb(k);
                
                varindices.pb(colsId[tmp]);
                values.pb(sFaces.count(tmp) ? 1.0 : 0.0);
                nzcnt++;
            }
        }
    }
    printf("Result: %d\n", nzcnt);

    int arr_varindices[nzcnt];
    double arr_values[nzcnt];
    copy(varindices.begin(), varindices.end(), arr_varindices);
    copy(values.begin(), values.end(), arr_values);

    /* Now copy the mip start */

    CPXsetintparam( env, CPX_PARAM_ADVIND, 1 );

    if (status = CPXaddmipstarts (env, lp, 1, nzcnt, &beg, arr_varindices,
        arr_values, &effortlevel, NULL))
    {
        printf("CPXaddmipstarts: Could not add mip start.\n");
        abort();
    }

    CPXwritemipstarts(env, lp, "MIPS", 0, 0);
}
//-----------------------------------------------------------------------------
