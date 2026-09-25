#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <queue>
#include <algorithm>

namespace py = pybind11;

// A struct to hold score and index
struct ScoreIdx {
    float score;
    int index;
    bool operator<(const ScoreIdx& other) const {
        return score > other.score; // Min-heap based on score
    }
};

void merge_topk(
    py::array_t<int> indptr,
    py::array_t<int> indices,
    py::array_t<float> data,
    int start_c,
    int k,
    py::array_t<float> global_top_scores,
    py::array_t<int> global_top_indices
) {
    auto buf_indptr = indptr.unchecked<1>();
    auto buf_indices = indices.unchecked<1>();
    auto buf_data = data.unchecked<1>();
    auto buf_global_scores = global_top_scores.mutable_unchecked<2>();
    auto buf_global_indices = global_top_indices.mutable_unchecked<2>();

    int num_queries = buf_global_scores.shape(0);

    // Parallelize across queries using OpenMP if compiled with it
    #pragma omp parallel for
    for (int i = 0; i < num_queries; ++i) {
        int row_start = buf_indptr(i);
        int row_end = buf_indptr(i + 1);
        int num_non_zero = row_end - row_start;

        if (num_non_zero == 0) continue;

        // We use a min-heap to keep track of the top K elements in this chunk
        // combined with the existing top K elements in the global array.
        std::vector<ScoreIdx> heap;
        heap.reserve(k * 2);

        // Add valid existing global top-K to heap
        for (int j = 0; j < k; ++j) {
            float s = buf_global_scores(i, j);
            int idx = buf_global_indices(i, j);
            if (idx != -1) {
                heap.push_back({s, idx});
            }
        }

        // Add chunk items to heap
        for (int j = row_start; j < row_end; ++j) {
            float s = buf_data(j);
            int idx = buf_indices(j) + start_c;
            
            if (heap.size() < (size_t)k) {
                heap.push_back({s, idx});
                if (heap.size() == (size_t)k) {
                    std::make_heap(heap.begin(), heap.end());
                }
            } else {
                if (s > heap.front().score) {
                    std::pop_heap(heap.begin(), heap.end());
                    heap.back() = {s, idx};
                    std::push_heap(heap.begin(), heap.end());
                }
            }
        }

        // If we never reached k elements, just sort what we have
        if (heap.size() < (size_t)k) {
            std::sort(heap.begin(), heap.end(), [](const ScoreIdx& a, const ScoreIdx& b){
                return a.score > b.score;
            });
        } else {
            // Sort the heap (it acts as a min-heap, so sorting it gives descending if we reverse or just pop)
            std::sort_heap(heap.begin(), heap.end()); 
            std::reverse(heap.begin(), heap.end()); // Make it descending
        }

        // Write back to global arrays
        for (size_t j = 0; j < heap.size() && j < (size_t)k; ++j) {
            buf_global_scores(i, j) = heap[j].score;
            buf_global_indices(i, j) = heap[j].index;
        }
    }
}

PYBIND11_MODULE(fast_topk, m) {
    m.doc() = "C++ extension for extremely fast Top-K sparse matrix extraction";
    m.def("merge_topk", &merge_topk, "Merges local Top-K from sparse matrix into global Top-K",
          py::arg("indptr"), py::arg("indices"), py::arg("data"), 
          py::arg("start_c"), py::arg("k"), 
          py::arg("global_top_scores"), py::arg("global_top_indices"));
}
