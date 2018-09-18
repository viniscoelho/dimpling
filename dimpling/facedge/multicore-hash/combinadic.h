#ifndef COMBINADIC_H
#define COMBINADIC_H
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
        // usually, n >= k
        if (n < 0 || k < 0) return;

        this->n = n;
        this->k = k;
        data.clear();
        for (int i = 0; i < k; ++i) data.push_back(i);
    }

    Combination(int n, int k, vector<int>& a) // Combination from a[]
    {
        if (k != a.size()) return;

        this->n = n;
        this->k = k;
        data.clear();
        for (int i = 0; i < a.size(); ++i) data.pb(a[i]);

        if (!isValid()) return;
    }
    Combination(){
        
    }

    static int64 choose(int n, int k)
    {
        if (n < 0 || k < 0) return -1;
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
    }

    bool isValid();
    int64 largestValue(int, int, int64);
    Combination successor();
    Combination element(int64);
    string toString();
    vector<int> getArray();
};
//-----------------------------------------------------------------------------
/*
    Returns if it is a valid combination
    */
bool Combination::isValid()
{
    if (data.size() != k) return false; // corrupted

    for (int i = 0; i < k; ++i)
    {
        if (data[i] < 0 || data[i] > n - 1)
            return false; // value out of range

        for (int j = i+1; j < k; ++j)
            if (data[i] >= data[j])
                return false; // duplicate or not lexicographic
    }

    return true;
}
//-----------------------------------------------------------------------------
/* 
    Return the largest value 'v' where 'v < a' and 'choose(v, b) <= x'
    */
int64 Combination::largestValue(int a, int b, int64 x)
{
    int v = a - 1;    
    while (choose(v, b) > x) --v;

    return v;
}
//-----------------------------------------------------------------------------
/*
    Returns the next combination
    */
Combination Combination::successor()
{
    if (data[0] == n - k) return Combination(0, 0);

    Combination ans(n, k);

    int i;
    for (i = 0; i < k; ++i)
        ans.data[i] = data[i];

    for (i = k - 1; i > 0 && ans.data[i] == (n - k + i); --i);

    ++ans.data[i];

    for (int j = i; j < k - 1; ++j)
        ans.data[j+1] = ans.data[j] + 1;

    return ans;
}
//-----------------------------------------------------------------------------
/*
    Returns the m-th lexicographic element of combination C(n, k)
    */
Combination Combination::element(int64 m) 
{
    vector<int> ans(k);

    int a = n;
    int b = k;
    int64 x = (choose(n, k) - 1) - m; // x is the "dual" of m

    for (int i = 0; i < k; ++i)
    {
        ans[i] = largestValue(a, b, x); // largest value v, where v < a and vCb < x    
        x = x - choose(ans[i], b);
        a = ans[i];
        b = b-1;
    }

    for (int i = 0; i < k; ++i)
        ans[i] = (n-1) - ans[i];

    return Combination(n, k, ans);
}
//-----------------------------------------------------------------------------
string Combination::toString()
{
    string s = "{ ";
    for (int i = 0; i < k; ++i)
        s += to_string(data[i]) + " ";
    s += "}\n";
    return s;
}
//-----------------------------------------------------------------------------
/*
    Returns the combined array
    */
vector<int> Combination::getArray()
{
    return data;
}
