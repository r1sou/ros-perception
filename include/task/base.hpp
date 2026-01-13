#pragma once 

#include "model/engine.hpp"

class BasicTask {
public:
    BasicTask() = default;
    ~BasicTask() = default;
public:
    virtual void init_task(nlohmann::json task_config = nlohmann::json()) = 0; 
    virtual void reset() = 0; 
    virtual void run(std::shared_ptr<InferenceData_t> infer_data, std::shared_ptr<Engine> engine) = 0; 
    virtual void release() = 0; 
};
