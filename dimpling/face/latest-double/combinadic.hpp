#ifndef COMBINADIC_HPP
#define COMBINADIC_HPP
#endif

__device__ class Combination
{
private:
    int n, k;
    int data[4];
    int64 largestV(int, int, int64);

public:
    __device__ Combination(int n, int k)
    {
        // n >= k
        if (n < 0 || k < 0)
        {
            printf("Negative parameter in constructor\n");
            return;
        }

        this->n = n;
        this->k = k;
        for (int i = 0; i < k; ++i) data[i] = i;
    }

    __device__ Combination(int n, int k, int* a)
    {
        this->n = n;
        this->k = k;

        for (int i = 0; i < k; ++i) data[i] = a[i];

        if (!isValid())
        {
            printf("Bad value from array!\n");
            return;
        }
    }
    
    __device__ Combination(){ }

    __device__ static int64 choose(int n, int k)
    {
        if (n < 0 || k < 0)
        {
            printf("Invalid negative parameter in choose()\n");
            return -1;
        }
        // special case
        if (n < k) return 0;
        if (n == k) return 1;

        int64 delta, iMax;

        // ex: choose(100, 3)
        if (k < n-k)
        {
            delta = n-k;
            iMax = k;
        }
        // ex: choose(100, 97)
        else
        {
            delta = k;
            iMax = n-k;
        }

        int64 ans = delta + 1;

        for (int64 i = 2; i <= iMax; ++i)
            ans = (ans * (delta + i)) / i;

        return ans;
    }

    __device__ bool isValid();
    __device__ Combination successor();
    __device__ Combination element(int64);
    __device__ int* getArray();

};

//-----------------------------------------------------------------------------

/*
    Returns if it is a valid combination.
    */
__device__ bool Combination::isValid()
{
    for (int i = 0; i < k; ++i)
    {
        if (data[i] < 0 || data[i] > n - 1)
            // value out of range
            return false;

        for (int j = i+1; j < k; ++j)
            if (data[i] >= data[j])
                // duplicate or not lexicographic
                return false;
    }

    return true;
}

//-----------------------------------------------------------------------------

/*
    Return the largest value 'v' where 'v < a' and 'choose(v, b) <= x'.
    */
__device__ int64 Combination::largestV(int a, int b, int64 x)
{
    int v = a - 1;
    while (choose(v, b) > x) --v;
    return v;
}

//-----------------------------------------------------------------------------

/*
    Returns the next combination.
    */
__device__ Combination Combination::successor()
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
    Returns the m-th lexicographic element of combination C(n, k).
    */
__device__ Combination Combination::element(int64 m)
{
    int ans[4];

    int a = n;
    int b = k;
    // x is the "dual" of m
    int64 x = (choose(n, k) - 1) - m;

    for (int i = 0; i < k; ++i)
    {
        // largest value v, where v < a and vCb < x
        ans[i] = largestV(a, b, x);
        x = x - choose(ans[i], b);
        a = ans[i];
        b = b-1;
    }

    for (int i = 0; i < k; ++i)
        ans[i] = (n-1) - ans[i];

    return Combination(n, k, ans);
}

//-----------------------------------------------------------------------------

/*
    Returns the combined array.
    */
__device__ int* Combination::getArray()
{
    return data;
}
