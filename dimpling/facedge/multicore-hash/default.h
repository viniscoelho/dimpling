#ifndef DEFAULT_H
    #define DEFAULT_H
#endif

//-----------------------------------------------------------------------------
// Mac OSX
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
    printf("Elapsed time: %.5lfs.\n", elapsed);
}
//-----------------------------------------------------------------------------
/*  
    Get clock time.
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
Face faces[MAXF][NUM_E_FACE];
int graph[MAXV][MAXV], resGraph[MAXV][MAXV], V, F, COMB;
int64 numComb[MAXV];
//-----------------------------------------------------------------------------
/*
    Define the number of combinations.
    */
void sizeDefinitions()
{
    for (int64 i = 4LL; i <= MAXV; ++i)
    {
        int64 resp = 1LL;
        for (int64 j = i-3; j <= i; ++j) resp *= j;
        resp /= 24LL;
        numComb[i-1] = resp;
    }
}
//-----------------------------------------------------------------------------
/*
    Returns the index of an edge.
    */
int edge_to_idx(edge e)
{
    return e.u*V + e.v;
}
//-----------------------------------------------------------------------------
/*
    Returns an edge given an index.
    */
edge idx_to_edge(int index)
{
    return edge(index/V, index%V);
}
//-----------------------------------------------------------------------------
