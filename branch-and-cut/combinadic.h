#ifndef COMBINADIC_H
#define COMBINADIC_H
	typedef long long int64;
#endif

class Combination
{
	private:
		int n, k;
		vector<int> data;
		int64 largestV(int, int, int64);

	public:
		Combination(int n, int k)
		{
		    if (n < 0 || k < 0) // normally n >= k
		    {
		    	cout << "Negative parameter in constructor\n";
		    	return;
		    }

		    this->n = n;
		    this->k = k;
		    data.clear();
		    for (int i = 0; i < k; i++)
		    	data.push_back(i);
	  	}// Combination(n, k)

		Combination(int n, int k, const vector<int>& a) // Combination from a[]
		{
			if (k != a.size())
			{
				cout << "Array length does not equal k\n";
				return;
			}

			this->n = n;
			this->k = k;
			data.clear();
			for (int i = 0; i < a.size(); i++)
				data.pb(a[i]);

			if (!isValid())
			{
				cout << "Bad value from array!\n";
				return;
			}
		}// Combination(n, k, a)

		static int64 choose(int n, int k)
		{
			if (n < 0 || k < 0)
			{
				cout << "Invalid negative parameter in choose()\n";
				return -1;
			}
			if (n < k) return 0;  // special case
			if (n == k) return 1;

			int64 delta, iMax;

			if (k < n-k) // ex: choose(100, 3)
			{
				delta = n-k;
				iMax = k;
			}
			else         // ex: choose(100, 97)
			{
				delta = k;
				iMax = n-k;
			}

			int64 ans = delta + 1;

			for (int64 i = 2; i <= iMax; ++i)
				ans = (ans * (delta + i)) / i; 

			return ans;
		} // Choose()

		bool isValid();
		Combination successor();
		Combination element(int64);
		vector<int> getArray();

};// Combination class

// return largest value v where v < a and choose(v, b) <= x
int64 Combination::largestV(int a, int b, int64 x)
{
	int v = a - 1;
	       
	while (choose(v, b) > x) --v;

	return v;
}// LargestV()

bool Combination::isValid()
{
	if (data.size() != k) return false; // corrupted

	for (int i = 0; i < k; i++)
	{
		if (data[i] < 0 || data[i] > n - 1)
			return false; // value out of range

		for (int j = i+1; j < k; ++j)
			if (data[i] >= data[j])
				return false; // duplicate or not lexicographic
	}

	return true;
}// IsValid()

Combination Combination::successor()
{
	if (data[0] == n - k) return Combination(0, 0);

	Combination ans(n, k);

	int i;
	for (i = 0; i < k; i++)
		ans.data[i] = data[i];

	for (i = k - 1; i > 0 && ans.data[i] == (n - k + i); --i);

	++ans.data[i];

	for (int j = i; j < k - 1; j++)
		ans.data[j+1] = ans.data[j] + 1;

	return ans;
}// Successor()

// return the mth lexicographic element of combination C(n, k)
Combination Combination::element(int64 m) 
{
	vector<int> ans(k);

	int a = n;
	int b = k;
	int64 x = (choose(n, k) - 1) - m; // x is the "dual" of m

	for (int i = 0; i < k; ++i)
	{
		ans[i] = largestV(a, b, x); // largest value v, where v < a and vCb < x    
		x = x - choose(ans[i], b);
		a = ans[i];
		b = b-1;
	}

	for (int i = 0; i < k; ++i)
	{
		ans[i] = (n-1) - ans[i];
	}

	return Combination(n, k, ans);
}// Element()

vector<int> Combination::getArray()
{
	return data;
} // getArray()

/*
int main(){
	Combination c(6, 4);
	cout << "With n=5 and k=3 there are " << c.choose(40, 20) << " combination elements.\n";

	cout << "The elements are:\n";
	for (int i = 0 ; i < c.choose(6, 4); ++i)
	{
		cout << i << ": " << c.toString();
		c = c.successor();
	}

	c = Combination(35, 12);
	cout << "\nEnter an index: ";
	int m;
	cin >> m;
	cout << "That combination element is: " << c.element(m).toString() << endl;
	return 0;
}
*/


