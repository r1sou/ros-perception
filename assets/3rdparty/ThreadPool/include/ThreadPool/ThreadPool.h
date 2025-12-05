// #ifndef THREAD_POOL_H
// #define THREAD_POOL_H

// #include <vector>
// #include <queue>
// #include <memory>
// #include <thread>
// #include <mutex>
// #include <condition_variable>
// #include <future>
// #include <functional>
// #include <stdexcept>

// class ThreadPool {
// public:
//     ThreadPool():stop(false){};
//     ThreadPool(size_t);
//     ThreadPool(const ThreadPool&) = delete;
//     ThreadPool& operator=(const ThreadPool&) = delete;
//     void init(size_t threads);
//     template<class F, class... Args>
//     auto enqueue(F&& f, Args&&... args) 
//         -> std::future<typename std::result_of<F(Args...)>::type>;
//     ~ThreadPool();
// private:
//     // need to keep track of threads so we can join them
//     std::vector< std::thread > workers;
//     // the task queue
//     std::queue< std::function<void()> > tasks;
//     int max_queue_size = 0;
    
//     // synchronization
//     std::mutex queue_mutex;
//     std::condition_variable condition;
//     bool stop;

// };

// inline void ThreadPool::init(size_t threads){
//     for(size_t i = 0;i<threads;++i)
//         workers.emplace_back(
//             [this]
//             {
//                 for(;;)
//                 {
//                     std::function<void()> task;

//                     {
//                         std::unique_lock<std::mutex> lock(this->queue_mutex);
//                         this->condition.wait(lock,
//                             [this]{ return this->stop || !this->tasks.empty(); });
//                         if(this->stop && this->tasks.empty())
//                             return;
//                         task = std::move(this->tasks.front());
//                         this->tasks.pop();
//                     }

//                     task();
//                 }
//             }
//         );
//     max_queue_size = threads;
// }

// // the constructor just launches some amount of workers
// inline ThreadPool::ThreadPool(size_t threads)
//     :   stop(false)
// {
//     init(threads);
// }

// // add new work item to the pool
// template<class F, class... Args>
// auto ThreadPool::enqueue(F&& f, Args&&... args) 
//     -> std::future<typename std::result_of<F(Args...)>::type>
// {
//     using return_type = typename std::result_of<F(Args...)>::type;

//     auto task = std::make_shared< std::packaged_task<return_type()> >(
//             std::bind(std::forward<F>(f), std::forward<Args>(args)...)
//         );
        
//     std::future<return_type> res = task->get_future();
//     {
//         std::unique_lock<std::mutex> lock(queue_mutex);

//         // don't allow enqueueing after stopping the pool
//         if(stop)
//             throw std::runtime_error("enqueue on stopped ThreadPool");

//         tasks.emplace([task](){ (*task)(); });
//     }
//     condition.notify_one();
//     return res;
// }

// // the destructor joins all threads
// inline ThreadPool::~ThreadPool()
// {
//     {
//         std::unique_lock<std::mutex> lock(queue_mutex);
//         stop = true;
//     }
//     condition.notify_all();
//     for(std::thread &worker: workers)
//         worker.join();
// }

// #endif



#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <vector>
#include <queue>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <stdexcept>
#include <atomic>

class ThreadPool {
public:
    explicit ThreadPool(size_t threads = 0, size_t queue_capacity_multiplier = 4)
        : stop(false)
    {
        size_t t = threads > 0 ? threads : std::thread::hardware_concurrency();
        max_queue_size = t * queue_capacity_multiplier;
        init(t);
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    void init(size_t threads, size_t queue_capacity_multiplier = 4) {

        max_queue_size = threads * queue_capacity_multiplier;
        for (size_t i = 0; i < threads; ++i) {
            workers.emplace_back([this] {
                for (;;) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->condition.wait(lock,
                            [this] { return this->stop || !this->tasks.empty(); });

                        if (this->stop && this->tasks.empty())
                            return;

                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }
                    task();
                }
            });
        }
    }

    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args)
        -> std::future<typename std::result_of<F(Args...)>::type>
    {
        using return_type = typename std::result_of<F(Args...)>::type;

        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );

        std::future<return_type> res = task->get_future();

        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if (stop)
                throw std::runtime_error("enqueue on stopped ThreadPool");
            if (tasks.size() >= max_queue_size) {
                throw std::runtime_error("ThreadPool task queue is full");
            }
            tasks.emplace([task]() { (*task)(); });
        }

        condition.notify_one();
        return res;
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for (std::thread& worker : workers)
            worker.join();
    }

    size_t queue_size() const {
        std::lock_guard<std::mutex> lock(queue_mutex);
        return tasks.size();
    }

    size_t queue_capacity() const { return max_queue_size; }
    size_t thread_count() const { return workers.size(); }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;

    size_t max_queue_size{0};

    mutable std::mutex queue_mutex;
    std::condition_variable condition;
    std::atomic<bool> stop{false};
};

#endif