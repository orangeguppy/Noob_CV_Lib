#include <algorithm>
#include <vector>

// Find the median for histogram (this is for the median filtering function)
int findMedian(int* histogram, int n, int range) {
    int count = 0;
    for (int i = 0 ; i < range ; i++) {
        count += histogram[i];
        if (count > n / 2) {
            return i;
        }
    }
    return 0;
}