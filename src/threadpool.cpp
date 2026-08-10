#include "threadpool.h"

namespace mk {

// A count below 1 would reach parallel_for as a zero or negative divisor.
ThreadPool::ThreadPool(int n_threads) : n_threads_(n_threads < 1 ? 1 : n_threads) {
    for (int i = 1; i < n_threads_; ++i)
        threads_.emplace_back([this, i] { worker(i); });
}

ThreadPool::~ThreadPool() {
    stop_.store(true, std::memory_order_release);
    generation_.fetch_add(1, std::memory_order_release);
    for (auto& t : threads_) t.join();
}

void ThreadPool::worker(int idx) {
    uint64_t seen = 0;
    for (;;) {
        while (generation_.load(std::memory_order_acquire) == seen) {
            if (stop_.load(std::memory_order_acquire)) return;
            __builtin_ia32_pause();
        }
        seen = generation_.load(std::memory_order_acquire);
        if (stop_.load(std::memory_order_acquire)) return;
        (*job_)(idx, n_threads_);
        pending_.fetch_sub(1, std::memory_order_acq_rel);
    }
}

void ThreadPool::parallel(const std::function<void(int, int)>& fn) {
    if (n_threads_ == 1) { fn(0, 1); return; }
    job_ = &fn;
    pending_.store(n_threads_ - 1, std::memory_order_release);
    generation_.fetch_add(1, std::memory_order_release);
    fn(0, n_threads_);
    while (pending_.load(std::memory_order_acquire) != 0)
        __builtin_ia32_pause();
}

void ThreadPool::parallel_for(int n, const std::function<void(int, int)>& fn) {
    parallel([&](int idx, int nt) {
        int chunk = (n + nt - 1) / nt;
        int begin = idx * chunk;
        int end = begin + chunk > n ? n : begin + chunk;
        if (begin < end) fn(begin, end);
    });
}

} // namespace mk
