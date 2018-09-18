#ifndef DEFAULT_H
#define DEFAULT_H
#endif

// Mac OSX
#ifdef __MACH__
#include <mach/clock.h>
#include <mach/mach.h>
#endif

const double EPS = 1e-9;
int sgn(double a) { return ((a > EPS) ? (1) : ((a < -EPS) ? (-1) : (0))); }
int cmp(double a, double b = 0.0) { return sgn(a - b); }

/*
    Print elapsed time.
    */
void printElapsedTime(double start, double stop)
{
    double elapsed = stop - start;
    printf("Elapsed time: %.3lfs.\n", elapsed);
    if (cmp(elapsed, 60.0) == 1) printf("Elapsed time: %.3lfmin.\n", elapsed/60.0);
    if (cmp(elapsed, 3600.0) == 1) printf("Elapsed time: %.3lfhs.\n", elapsed/3600.0);
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

void initValues()
{
    for (int i = 0; i < SIZE-1; ++i)
        for (int j = i+1; j < SIZE; ++j)
            resGraph[i*SIZE + j] = resGraph[j*SIZE + i] = -1;
    
    COMB = numComb[SIZE-1];
    FACES = 2*SIZE-4;
}

//-----------------------------------------------------------------------------

void readInput()
{
    int value;
    scanf("%d", &SIZE);
    COMB = numComb[SIZE-1];

    G = (Graph*)malloc(sizeof(Graph));
    G->length = SIZE;

    for (int i = 0; i < SIZE-1; ++i)
    {
        for (int j = i+1; j < SIZE; ++j)
        {
            scanf("%d", &value);
            G->graph[i*SIZE + j] = G->graph[j*SIZE + i] = value;
        }
    }
    initValues();
}

//-----------------------------------------------------------------------------

/*
    Define the number of combinations.
    */
void sizeDefinitions()
{
    for (int i = 4; i <= MAXV; ++i)
    {
        int64 resp = 1;
        for (int j = i-3; j <= i; ++j) resp *= j;
        resp /= 24;
        numComb[i-1] = resp;
    }
}
