#include <bits/stdc++.h>

using namespace std;

/*
    Code used to generate a single random graph.
    The weights of each edge varies from 0 to 199
*/
int main(int argv, char** argc){
    ios::sync_with_stdio(false);
    int vertices = atoi(argc[1]), n = atoi(argc[2]);
    srand(time(NULL)+n);

    ostringstream fileName;
    fileName << "graph-" << n << ".in";
    ofstream graphStream(fileName.str());

    graphStream << vertices << "\n";
    for (int i = 0; i < vertices; i++){
        for (int j = i+1; j < vertices; j++){
            graphStream << rand() % 200 << " ";
        }
        graphStream << "\n";
    }
    graphStream.close();
    return 0;
}