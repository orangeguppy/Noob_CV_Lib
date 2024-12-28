#ifndef MEDIAN_FILTER_H
#define MEDIAN_FILTER_H
#include <queue>

class MedianFilter {
public:
    MedianFilter();
    void insertValue(float value); // Insert an element into the two-heap structure
    void removeVal(float value);
    float getMedian();

private:
    void balanceHeaps(); // Balance the min heap and the max heap
    template <typename T, typename Container, typename Compare>
    void removeElement(std::priority_queue<T, Container, Compare>& heap, T element); // This is a helper function for removeValue
    std::priority_queue<float> maxHeap; // Max-heap for the lower half, it is a max-heap by default
    std::priority_queue<float, std::vector<float>, std::greater<float>> minHeap; // Min-heap for the upper half
};
#endif