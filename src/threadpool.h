#pragma once
#include <atomic>
#include <functional>
#include <thread>
#include <vector>

namespace mk {

// Fork-join pool with spinning workers. Decode-step tasks are ~100µs; a
// condvar wake costs several µs per worker per matmul, so workers spin on an
// atomic generation counter instead.
class ThreadPool {
public:
    explicit ThreadPool(int n_threads);
    ~ThreadPool();

    int size() const { return n_threads_; }

    // Runs fn(thread_idx, n_threads) on all threads (including the caller as
    // index 0) and waits for completion.
    void parallel(const std::function<void(int, int)>& fn);

    // Convenience: split [0, n) into contiguous chunks per thread.
    void parallel_for(int n, const std::function<void(int begin, int end)>& fn);

private:
    void worker(int idx);
    int n_threads_;
    std::vector<std::thread> threads_;
    std::atomic<uint64_t> generation_{0};
    std::atomic<int> pending_{0};
    std::atomic<bool> stop_{false};
    const std::function<void(int, int)>* job_ = nullptr;
};

} // namespace mk
