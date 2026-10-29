// core.h -- generic data structures and algorithms (nothing music-specific).
// Works on any set of N-dimensional points.
//
// Distances are compared as squared Euclidean distance (no sqrt while
// searching); euclideanDistance() is only used when printing results.

#ifndef CORE_H
#define CORE_H

#include <cstddef>
#include <stdexcept>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <string>
#include <cstdint>

namespace core {

// Vector of doubles. Kept simple on purpose: no templates, no operator
// overloading. dim() comes from the size of the vector, and get()/set()
// check bounds.
class VecN {
public:
    VecN() = default;

    // n coordinates, all set to init
    explicit VecN(std::size_t n, double init = 0.0) : coords_(n, init) {}

    explicit VecN(const std::vector<double>& coords) : coords_(coords) {}

    std::size_t dim() const { return coords_.size(); }

    double get(std::size_t i) const {
        if (i >= coords_.size()) {
            throw std::out_of_range("VecN::get: index out of range");
        }
        return coords_[i];
    }

    void set(std::size_t i, double value) {
        if (i >= coords_.size()) {
            throw std::out_of_range("VecN::set: index out of range");
        }
        coords_[i] = value;
    }

private:
    std::vector<double> coords_;
};

// throws if the two vectors have different lengths
inline void ensureSameDim(const VecN& a, const VecN& b) {
    if (a.dim() != b.dim()) {
        throw std::runtime_error("dimension mismatch: "
                                 + std::to_string(a.dim()) + " vs "
                                 + std::to_string(b.dim()));
    }
}

// All the vectors, addressed by index. The tree only stores indices into
// this, so the data isn't copied. The tree keeps a pointer to the dataset,
// so don't change or destroy it while the tree is in use.
class KDataset {
public:
    // first point sets the dimension, later ones must match it
    void add(const VecN& point) {
        if (points_.empty()) {
            dim_ = point.dim();
        } else if (point.dim() != dim_) {
            throw std::invalid_argument(
                "KDataset::add: point has " + std::to_string(point.dim()) +
                " coordinates but the dataset holds " + std::to_string(dim_) +
                "-dimensional vectors");
        }
        points_.push_back(point);
    }

    void clear() {
        points_.clear();
        dim_ = 0;
    }

    std::size_t size() const { return points_.size(); }

    std::size_t dim() const { return dim_; }

    const VecN& point(std::size_t i) const {
        if (i >= points_.size()) {
            throw std::out_of_range("KDataset::point: index out of range");
        }
        return points_[i];
    }

private:
    std::vector<VecN> points_;
    std::size_t dim_ = 0;
};

// Squared Euclidean distance. Same ordering as the real distance, so no
// sqrt needed for comparisons.
inline double squaredDistance(const VecN& a, const VecN& b) {
    ensureSameDim(a, b);
    double sum = 0.0;
    for (std::size_t i = 0; i < a.dim(); ++i) {
        double diff = a.get(i) - b.get(i);
        sum += diff * diff;
    }
    return sum;
}

inline double euclideanDistance(const VecN& a, const VecN& b) {
    return std::sqrt(squaredDistance(a, b));
}

// One search result: squared distance and dataset index.
struct Candidate {
    double distSq = 0.0;
    int index = -1;
};

// closest first; ties go to the smaller index so results are repeatable
inline bool candidateAscending(const Candidate& a, const Candidate& b) {
    if (a.distSq != b.distSq) return a.distSq < b.distSq;
    return a.index < b.index;
}

// Inverse order on distSq only: feeding this to push_heap/pop_heap turns
// the vector into a max-heap on distance, so the WORST kept candidate sits
// at the front and is the first one to evict.
inline bool fartherFirst(const Candidate& a, const Candidate& b) {
    return a.distSq < b.distSq;
}

// Keeps the K best candidates. It's a max-heap, so the worst one of the K
// is on top and is the one to throw out when something better turns up.
// worstDistSq() is what the search uses for pruning (infinity until the
// heap is full). Sorting happens once at the end in drainSorted().
class BoundedMaxHeap {
public:
    explicit BoundedMaxHeap(int capacity)
        : capacity_(capacity) {
        capacity_ = capacity < 0 ? 0 : capacity;
        if (capacity_ > 0) heap_.reserve(static_cast<std::size_t>(capacity_) * 2u);
    }

    double worstDistSq() const {
        if (static_cast<int>(heap_.size()) < capacity_ || heap_.empty()) {
            return std::numeric_limits<double>::infinity();
        }
        return heap_.front().distSq;
    }

    // if full, only replaces the worst when strictly closer (ties keep the old one)
    void offer(double distSq, int index) {
        if (capacity_ <= 0) return;
        if (static_cast<int>(heap_.size()) < capacity_) {
            heap_.push_back(Candidate{distSq, index});
            std::push_heap(heap_.begin(), heap_.end(), fartherFirst);
        } else if (distSq < heap_.front().distSq) {
            std::pop_heap(heap_.begin(), heap_.end(), fartherFirst);
            heap_.back() = Candidate{distSq, index};
            std::push_heap(heap_.begin(), heap_.end(), fartherFirst);
        }
    }

    bool full() const { return static_cast<int>(heap_.size()) >= capacity_; }
    std::size_t size() const { return heap_.size(); }
    int capacity() const { return capacity_; }

    // returns everything sorted closest first, heap is empty afterwards
    std::vector<Candidate> drainSorted();

private:
    std::vector<Candidate> heap_;
    int capacity_ = 0;
};

// Top-down merge sort, O(n log n), stable. Only used on the final top-K.
template <typename T, typename Compare>
static void mergeSortRun(std::vector<T>& v, Compare less,
                         std::vector<T>& scratch,
                         std::size_t lo, std::size_t hi) {
    if (hi - lo <= 1) return;
    std::size_t mid = lo + (hi - lo) / 2;
    mergeSortRun(v, less, scratch, lo, mid);
    mergeSortRun(v, less, scratch, mid, hi);
    // merge the two halves into scratch, then copy back
    std::size_t i = lo, j = mid, k = lo;
    while (i < mid && j < hi) {
        if (less(v[j], v[i])) scratch[k++] = v[j++];   // strict < keeps it stable
        else                  scratch[k++] = v[i++];
    }
    while (i < mid) scratch[k++] = v[i++];
    while (j < hi)  scratch[k++] = v[j++];
    for (std::size_t t = lo; t < hi; ++t) v[t] = scratch[t];
}

template <typename T, typename Compare>
void mergeSort(std::vector<T>& values, Compare less) {
    if (values.size() <= 1) return;
    std::vector<T> scratch(values.size());
    mergeSortRun(values, less, scratch, 0, values.size());
}

// defined here because it needs mergeSort
inline std::vector<Candidate> BoundedMaxHeap::drainSorted() {
    std::vector<Candidate> result(std::move(heap_));
    mergeSort(result, candidateAscending);
    return result;
}

// K-d tree over a KDataset. Each node holds one point (by index). The
// split axis cycles with depth (depth % dims), and the median point on that
// axis becomes the node; nth_element does the partitioning so each level
// avoids a full sort. This keeps the tree balanced. Duplicate values are
// fine, a point just lands on one side. Doesn't own the dataset.
class KDTree {
public:
    KDTree() = default;

    ~KDTree() { destroyTree(); }
    KDTree(const KDTree&) = delete;
    KDTree& operator=(const KDTree&) = delete;

    // builds over data (which must outlive the tree); false if data is empty
    bool build(const KDataset& data) {
        destroyTree();
        if (data.size() == 0 || data.dim() == 0) return false;
        data_ = &data;
        dims_ = data.dim();
        nodeCount_ = 0;
        maxDepth_ = 0;
        std::vector<int> indices;
        indices.reserve(data.size());
        for (std::size_t i = 0; i < data.size(); ++i) indices.push_back(static_cast<int>(i));
        root_ = buildNode(data, indices, 0);
        return root_ != nullptr;
    }

    void clear() { destroyTree(); }

    bool empty() const { return root_ == nullptr; }

    std::size_t nodeCount() const { return nodeCount_; }
    int maxDepth() const { return maxDepth_; }
    std::size_t dim() const { return dims_; }

    // nodes visited in the last search, to compare against brute force (N)
    long long lastNodesVisited() const { return nodesVisited_; }

    // k nearest points to query, closest first. excludeIndex is skipped
    // (-1 for none). At each node: offer its point to the heap, go down the
    // near side, then only go down the far side if the splitting plane is
    // closer than the worst candidate we have so far.
    std::vector<Candidate> searchKNearest(const VecN& query, int k,
                                          int excludeIndex = -1) const {
        nodesVisited_ = 0;
        std::vector<Candidate> results;
        if (k <= 0 || root_ == nullptr || data_ == nullptr || data_->size() == 0) {
            return results;
        }
        if (query.dim() != dims_) {
            throw std::runtime_error("KDTree::searchKNearest: query dimensionality ("
                                     + std::to_string(query.dim()) +
                                     ") does not match tree dimensionality (" +
                                     std::to_string(dims_) + ")");
        }
        // cap k so a huge k doesn't allocate a huge heap
        if (k > static_cast<int>(data_->size())) k = static_cast<int>(data_->size());
        BoundedMaxHeap heap(k);
        searchNode(root_, query, excludeIndex, heap);
        return heap.drainSorted();
    }

private:
    struct Node {
        int pointIndex = -1;
        int splitDim = 0;
        Node* left = nullptr;
        Node* right = nullptr;
    };

    Node* buildNode(const KDataset& data, std::vector<int>& indices, int depth) {
        if (indices.empty()) return nullptr;
        ++nodeCount_;
        if (depth > maxDepth_) maxDepth_ = depth;

        Node* node = new Node();
        node->splitDim = static_cast<int>(depth % dims_);
        const int axis = node->splitDim;
        // Median on the split axis becomes the node; nth_element partitions
        // around it in O(n) instead of sorting the whole slice.
        auto alongAxis = [&](int a, int b) {
            return data.point(static_cast<std::size_t>(a))
                       .get(static_cast<std::size_t>(axis)) <
                   data.point(static_cast<std::size_t>(b))
                       .get(static_cast<std::size_t>(axis));
        };
        std::size_t mid = indices.size() / 2;
        std::nth_element(indices.begin(), indices.begin() + mid, indices.end(),
                         alongAxis);
        node->pointIndex = indices[mid];
        std::vector<int> leftIndices(indices.begin(), indices.begin() + mid);
        std::vector<int> rightIndices(indices.begin() + mid + 1, indices.end());
        node->left  = buildNode(data, leftIndices, depth + 1);
        node->right = buildNode(data, rightIndices, depth + 1);
        return node;
    }

    void searchNode(const Node* node, const VecN& query, int excludeIndex,
                    BoundedMaxHeap& heap) const {
        if (node == nullptr) return;
        ++nodesVisited_;
        const VecN& nodePoint = data_->point(static_cast<std::size_t>(node->pointIndex));
        if (node->pointIndex != excludeIndex) {
            heap.offer(squaredDistance(query, nodePoint), node->pointIndex);
        }
        double axisDiff = query.get(static_cast<std::size_t>(node->splitDim)) -
                          nodePoint.get(static_cast<std::size_t>(node->splitDim));
        const Node* nearChild = (axisDiff <= 0.0) ? node->left : node->right;
        const Node* farChild  = (axisDiff <= 0.0) ? node->right : node->left;
        searchNode(nearChild, query, excludeIndex, heap);
        // far side only if the plane is closer than our current worst
        if (heap.worstDistSq() > axisDiff * axisDiff) {
            searchNode(farChild, query, excludeIndex, heap);
        }
    }

    void destroyTree() {
        destroyNode(root_);
        root_ = nullptr;
        data_ = nullptr;
        dims_ = 0;
        nodeCount_ = 0;
        maxDepth_ = 0;
        nodesVisited_ = 0;
    }

    static void destroyNode(Node* node) {
        if (node == nullptr) return;
        destroyNode(node->left);
        destroyNode(node->right);
        delete node;
    }

    Node* root_ = nullptr;
    const KDataset* data_ = nullptr;
    std::size_t dims_ = 0;
    std::size_t nodeCount_ = 0;
    int maxDepth_ = 0;
    mutable long long nodesVisited_ = 0;
};

// Linear scan over every point, same heap as the tree. Used as a baseline
// to check the tree's answers and compare speed.
inline std::vector<Candidate> bruteForceKNearest(const KDataset& data,
                                                 const VecN& query,
                                                 int k,
                                                 int excludeIndex = -1) {
    std::vector<Candidate> results;
    if (k <= 0 || data.size() == 0) return results;
    if (query.dim() != data.dim()) {
        throw std::runtime_error("bruteForceKNearest: query dimensionality ("
                                 + std::to_string(query.dim()) +
                                 ") does not match dataset dimensionality (" +
                                 std::to_string(data.dim()) + ")");
    }
    if (k > static_cast<int>(data.size())) k = static_cast<int>(data.size());
    BoundedMaxHeap heap(k);
    for (std::size_t i = 0; i < data.size(); ++i) {
        if (static_cast<int>(i) == excludeIndex) continue;
        heap.offer(squaredDistance(query, data.point(i)), static_cast<int>(i));
    }
    return heap.drainSorted();
}

// String key -> int hash table with separate chaining (FNV-1a hash).
// Doubles the buckets when the load factor goes over 1. Duplicate keys are
// allowed (same track can appear twice in a library); find() returns the
// first one, findAll() returns all of them.
class HashTable {
public:
    explicit HashTable(std::size_t initialBuckets = 64) {
        if (initialBuckets < 8) initialBuckets = 8;
        buckets_.assign(initialBuckets, {});
    }

    void insert(const std::string& key, int value) {
        if (static_cast<double>(entries_ + 1) >
            maxLoadFactor_ * static_cast<double>(buckets_.size())) {
            rehash(buckets_.size() * 2);
        }
        buckets_[bucketFor(key)].emplace_back(key, value);
        ++entries_;
    }

    bool find(const std::string& key, int& valueOut) const {
        for (const auto& kv : buckets_[bucketFor(key)]) {
            if (kv.first == key) { valueOut = kv.second; return true; }
        }
        return false;
    }

    std::vector<int> findAll(const std::string& key) const {
        std::vector<int> out;
        for (const auto& kv : buckets_[bucketFor(key)]) {
            if (kv.first == key) out.push_back(kv.second);
        }
        return out;
    }

    bool contains(const std::string& key) const {
        int tmp;
        return find(key, tmp);
    }

    std::size_t size() const { return entries_; }
    std::size_t bucketCount() const { return buckets_.size(); }

private:
    using Chain = std::vector<std::pair<std::string, int>>;

    std::size_t bucketFor(const std::string& key) const {
        return fnv1a(key) % buckets_.size();
    }

    static std::uint64_t fnv1a(const std::string& key) {
        std::uint64_t hash = 14695981039346656037ULL;
        for (unsigned char c : key) {
            hash ^= c;
            hash *= 1099511628211ULL;
        }
        return hash;
    }

    void rehash(std::size_t newBucketCount) {
        if (newBucketCount < 8) newBucketCount = 8;
        std::vector<Chain> next(newBucketCount);
        for (const Chain& ch : buckets_) {
            for (const auto& kv : ch) {
                next[fnv1a(kv.first) % newBucketCount].push_back(kv);
            }
        }
        std::swap(buckets_, next);
    }

    std::vector<Chain> buckets_;
    std::size_t entries_ = 0;
    double maxLoadFactor_ = 1.0;
};

} // namespace core

#endif // CORE_H
