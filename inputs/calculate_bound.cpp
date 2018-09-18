#include <bits/stdc++.h>

using namespace std;

int main() {
	int n;
	cin >> n;
	vector<double> w;
	for (int i = 0; i < n; i++){
		for (int j = 0; j < n; j++){
			double v;
			cin >> v;
			if (i >= j) continue;
			w.push_back(v);
		}
	}
	sort(w.begin(), w.end(), greater<double>());
	cout << fixed << setprecision(6) <<  w[0] << " " << w.back() << "\n";
	double ans = 0.0;
	for (int i = 0; i < 3*n-6; i++) ans += w[i];
	cout << fixed << setprecision(6) << ans << "\n";
	return 0;
}
