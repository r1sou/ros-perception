#pragma once

#include "model/bytetrack/STrack.h"

#include "model/base/base.hpp"

struct Object
{
	std::vector<float> rect; // x1 y1 x2 y2
    float prob;
	Object() = default;
	Object(std::vector<float> rect, float prob) : rect(rect), prob(prob) {}
};

class BYTETracker: public BasicModel{
public:
	BYTETracker() = default;
	~BYTETracker(){
		RCLCPP_INFO_STREAM(rclcpp::get_logger("ByteTrack"), "tensor release");
	}
public:
	void configuration_config(nlohmann::json config) override{
		if (!config.size()){
			return;
		}
		model_config_.track_thresh = config.value("track_thresh", model_config_.track_thresh);
		model_config_.high_thresh = config.value("high_thresh", model_config_.high_thresh);
		model_config_.track_buffer = config.value("track_buffer", model_config_.track_buffer);
		model_config_.match_thresh = config.value("match_thresh", model_config_.match_thresh);
	}
    void preprocess(std::shared_ptr<InferenceData_t> infer_data, int index) override{
	}
	void inference() override{
	}
    void postprocess(std::shared_ptr<InferenceData_t> infer_data, int index) override{
		std::vector<Object> objects;
		auto track_labels = infer_data->output.track_output.track_labels;
		for(auto &detect_output: infer_data->output.detect_output){
			for(int i = 0; i < detect_output.bboxes.size(); i++){
				std::string name = detect_output.names[i];
				if(!track_labels.size() && !track_labels.contains(name)){
					continue;
				}
				objects.push_back(Object(detect_output.bboxes[i], detect_output.scores[i]));
			}
		}
		auto output_stracks = update(objects);
		for(auto &strack: output_stracks){
			std::vector<float> tlwh = strack.tlwh;
			std::vector<float> bbox = {tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]};
			infer_data->output.track_output.bboxes.push_back(bbox);
			infer_data->output.track_output.track_ids.push_back(strack.track_id);
			infer_data->output.track_output.track_names.push_back(fmt::format("person: {}", strack.track_id));
			infer_data->output.track_output.box_areas.push_back(tlwh[2] * tlwh[3]);
		}
	}
	void reset() override{
		frame_id = 0;
		tracked_stracks.clear();
		lost_stracks.clear();
		removed_stracks.clear();
	}
public:
	std::vector<STrack> update(const std::vector<Object> &objects);
	cv::Scalar get_color(int idx);
public:
	std::vector<STrack*> joint_stracks(std::vector<STrack*> &tlista, std::vector<STrack> &tlistb);
	std::vector<STrack> joint_stracks(std::vector<STrack> &tlista, std::vector<STrack> &tlistb);
	std::vector<STrack> sub_stracks(std::vector<STrack> &tlista, std::vector<STrack> &tlistb);
	void remove_duplicate_stracks(std::vector<STrack> &resa, std::vector<STrack> &resb, std::vector<STrack> &stracksa, std::vector<STrack> &stracksb);
	void linear_assignment(
		std::vector<std::vector<float> > &cost_matrix, int cost_matrix_size, int cost_matrix_size_size, float thresh,
		std::vector<std::vector<int> > &matches, std::vector<int> &unmatched_a, std::vector<int> &unmatched_b
	);
	std::vector<std::vector<float> > iou_distance(std::vector<STrack*> &atracks, std::vector<STrack> &btracks, int &dist_size, int &dist_size_size);
	std::vector<std::vector<float> > iou_distance(std::vector<STrack> &atracks, std::vector<STrack> &btracks);
	std::vector<std::vector<float> > ious(std::vector<std::vector<float> > &atlbrs, std::vector<std::vector<float> > &btlbrs);
	double lapjv(
		const std::vector<std::vector<float> > &cost, std::vector<int> &rowsol, std::vector<int> &colsol, 
		bool extend_cost = false, float cost_limit = LONG_MAX, bool return_cost = true
	);
public:
    TRACK_CONFIG model_config_;
public:
	int frame_id;
	std::vector<STrack> tracked_stracks;
	std::vector<STrack> lost_stracks;
	std::vector<STrack> removed_stracks;
	byte_kalman::KalmanFilter kalman_filter;
};