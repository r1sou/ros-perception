#pragma once

#include "task/base.hpp"

class Obstacle : public BasicTask {
public:
    Obstacle() = default;
    ~Obstacle(){
        release();
    }
public:
    void init_task(nlohmann::json task_config = nlohmann::json()) override{}
    void reset() override{}
    void run(std::shared_ptr<InferenceData_t> infer_data, std::shared_ptr<Engine> engine) override{}
    void release() override{}
};
