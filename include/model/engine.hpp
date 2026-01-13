#pragma once

#include "model/yolo/yolo.h"
#include "model/stereonet/stereonet.h"
#include "model/bytetrack/BYTETracker.h"

#include "common/threadpool.h"

class Engine{
public:
    Engine(){
        // preprocess_pool_ = std::make_unique<CommonThreadPool>();
        // postprocess_pool_ = std::make_unique<CommonThreadPool>();
    }
    ~Engine(){
        hbDNNRelease(packed_dnn_handle_);
    }
public:
    void configuration(nlohmann::json config, std::string project_root){
        for(auto model_config : config["models"]){
            if(model_config["model"] == "yolo" && model_config["launch"].get<bool>()){
                std::string model_path = project_root + "/" + model_config["model_path"].get<std::string>();
                auto model = std::make_shared<YOLO>();
                model->configuration(packed_dnn_handle_, model_path, model_config["params"]);
                models_.push_back(model);

                model_tasks_.push_back(std::unordered_set<std::string>(model_config["tasks"].begin(), model_config["tasks"].end()));
            }
            else if(model_config["model"] == "bytetrack" && model_config["launch"].get<bool>()){
                auto model = std::make_shared<BYTETracker>();
                model->configuration_config(model_config["params"]);
                models_.push_back(model);

                model_tasks_.push_back(std::unordered_set<std::string>(model_config["tasks"].begin(), model_config["tasks"].end()));

                track_model_index = models_.size() - 1;
            }
            else if(model_config["model"] == "stereonet" && model_config["launch"].get<bool>()){
                std::string model_path = project_root + "/" + model_config["model_path"].get<std::string>();
                auto model = std::make_shared<StereoNet>();
                model->configuration(packed_dnn_handle_, model_path, model_config["params"]);
                models_.push_back(model);

                model_tasks_.push_back(std::unordered_set<std::string>(model_config["tasks"].begin(), model_config["tasks"].end()));
            }
        }
        config_ = config;
    }
    void reset_track(){
        if(track_model_index != -1){
            models_[track_model_index]->reset();
        }

    }
    void inferenceV1(std::shared_ptr<InferenceData_t> infer_data, std::string task = "recognize"){
        for(int i = 0; i < models_.size(); i++){
            if(model_tasks_[i].find(task) == model_tasks_[i].end()){
                continue;
            }
            {
                // ScopeTimer t("model: " + config_["models"][i]["model"].get<std::string>() + " preprocess");
                models_[i]->preprocess(infer_data, i);
            }
            {
                // ScopeTimer t("model: " + config_["models"][i]["model"].get<std::string>() + " inference");
                models_[i]->inference();
            }
            {
                // ScopeTimer t("model: " + config_["models"][i]["model"].get<std::string>() + " postprocess");
                models_[i]->postprocess(infer_data, i);
            }
        }
    }
    void inference(std::shared_ptr<InferenceData_t> infer_data, std::string task = "recognize"){
        infer_data->output.detect_output.resize(models_.size());
        for(int i = 0; i < models_.size(); i++){
            infer_data->output.detect_output[i].detect_labels = config_.value("detect", nlohmann::json());
        }
        infer_data->output.track_output.track_labels = config_.value("track", nlohmann::json());
        
        if(infer_data->output.track_output.track_labels.size() >= 2){
            RCLCPP_INFO_STREAM(
                rclcpp::get_logger("inference"), 
                "expect track labels size < 2, but get " << infer_data->output.track_output.track_labels.size()
            );
            return;
        }
        inferenceV1(infer_data, task);
    }
private:
    CommonThreadPool preprocess_pool_, postprocess_pool_;
private:
    int track_model_index = -1;
private:
    nlohmann::json config_;
    hbPackedDNNHandle_t packed_dnn_handle_;
    std::vector<std::shared_ptr<BasicModel>> models_;
    std::vector<std::unordered_set<std::string>> model_tasks_;
};