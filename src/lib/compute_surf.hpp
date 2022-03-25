#ifndef SIFT_COMPUTE_SURF_HPP_
#define SIFT_COMPUTE_SURF_HPP_

#include <string>
#include <tuple>
#include <vector>

#include <opencv2/core/mat.hpp>

std::tuple<std::vector<cv::KeyPoint>, cv::Mat> ComputeSurf(const std::string& file_name, cv::Mat& img);

std::vector<cv::DMatch> KeypointMatcher(cv::Mat& descriptors_1, cv::Mat& descriptors_2 ) ;



bool GetOutliersUsingHomogrophy
(       
        std::vector<cv::KeyPoint> obj_keypoints,
        std::vector<cv::KeyPoint> scene_keypoints, 
        std::vector<cv::DMatch> &matches,
        cv::Mat& homography
    
);
#endif