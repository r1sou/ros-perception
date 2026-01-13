#pragma once

#include "common/common.h"
#include "ThreadPool/BS_thread_pool.hpp"

class CommonThreadPool{
public:
    CommonThreadPool(int num_threads = 4){
        thread_pool_ = std::make_unique<BS::thread_pool<>>(num_threads);

    }
    ~CommonThreadPool(){
        if(thread_pool_){
            thread_pool_->wait();
            thread_pool_->reset();
        }
    }
public:
    std::unique_ptr<BS::thread_pool<>>  thread_pool_ = nullptr;
};