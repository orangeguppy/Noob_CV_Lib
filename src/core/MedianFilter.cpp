#include <iostream>
#include <queue>
#include "core/MedianFilter.h"

MedianFilter::MedianFilter() {
    // Nothing happens here as the queues and everything are already declared as empty
}

void MedianFilter::insertValue(float value) {
    // If the new value is less than the median, or if maxHeap is empty
    if (maxHeap.empty() || value < maxHeap.top()) {
        maxHeap.push(value);
    } else {
        minHeap.push(value);
    }
    // The min and max heaps may not be balanced after adding a new value, so balance them
    balanceHeaps();
}

void MedianFilter::removeVal(float value) {
    // If the element is inside maxHeap
    if (value <= maxHeap.top()) {
        if (maxHeap.empty()) {
            std::cerr << "Attempted to remove from an empty data structure." << std::endl;
            return;
        }
        removeElement(maxHeap, value);
    }
    else {
        if (minHeap.empty()) {
            std::cerr << "Attempted to remove from an empty data structure." << std::endl;
            return;
        }
        removeElement(minHeap, value);
    }
    balanceHeaps();
}

// This is inefficient, I think its an ok tradeoff because its needed to maintain a sliding window
// Need to use templates because priority queues with differtent comparators by default are not considered the same type
template <typename T, typename Container, typename Compare>
void MedianFilter::removeElement(std::priority_queue<T, Container, Compare>& heap, T element) {
    std::vector<T> temp;
    // Store all elements that are removed while lookling for the element
    while (!heap.empty() && heap.top() != element) {
        temp.push_back(heap.top());
        heap.pop();
    }
    // We've found the element, remove it!
    if (!heap.empty()) {
        heap.pop();
    }

    // Reinsert the elements that were removed earlier on
    for (float num : temp) {
        heap.push(num);
    }
}

void MedianFilter::balanceHeaps() {
    // If maxHeap > minHeap + 1
    if (maxHeap.size() > minHeap.size() + 1) {
        minHeap.push(maxHeap.top());
        maxHeap.pop();
    } else if (maxHeap.size() < minHeap.size()) {
        maxHeap.push(minHeap.top());
        minHeap.pop();
    }
}

float MedianFilter::getMedian() {
    balanceHeaps();
    // If maxHeap has more elements than minHeap, the top element of maxHeap is the median
    if (maxHeap.size() > minHeap.size()) {
        return maxHeap.top();
    } else {
        return minHeap.top();
    }
}