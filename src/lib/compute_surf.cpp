#include "compute_surf.hpp"

#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

#include <opencv2/xfeatures2d.hpp>
#include "opencv2/features2d.hpp"

#include <opencv2/imgcodecs.hpp>
#include <iostream>

using std::tuple;
using std::string;
using std::vector;

using namespace cv;
using namespace cv::xfeatures2d;

tuple<vector<KeyPoint>, Mat> ComputeSurf(const string& file_name, Mat& img){


    img = cv::imread(file_name,  cv::IMREAD_GRAYSCALE);

    int min_hessian = 400;

    Ptr<SURF> detector = SURF::create();
    detector->setHessianThreshold(min_hessian);

    vector<KeyPoint> keypoints_1;
    Mat descriptors;

    detector->detectAndCompute( img, Mat(), keypoints_1, descriptors );

    std::cout << "Number of SIFTs: " << descriptors.rows << "\n"
            << "Size of each SIFT: " << descriptors.cols << "\n";
  
    return std::make_tuple(keypoints_1, descriptors);


}


vector<DMatch> KeypointMatcher(Mat& descriptors_1,Mat& descriptors_2 ) {
        
    FlannBasedMatcher matcher;
    
    std::vector< DMatch > matches;

    matcher.match( descriptors_1, descriptors_2, matches );

    double max_dist = 0; double min_dist = 100;

    for( int i = 0; i < descriptors_1.rows; i++ )
    { double dist = matches[i].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
    }

    std::vector<DMatch> best_matches;

    for( int i = 0; i < descriptors_1.rows; i++ )
    { if( matches[i].distance <= 2*min_dist )
        { best_matches.push_back( matches[i]); }
    }

    std::cout << "Number of Mathces: " << size(best_matches) << "\n";  

    return best_matches;
}


bool GetOutliersUsingHomogrophy
(       
        vector<KeyPoint> obj_keypoints,
        vector<KeyPoint> scene_keypoints, 
        vector<DMatch> &matches,
        Mat& homography
    
)
{

    if (size(matches) < 8 )
    {
        return false;
    }

     vector<Point2f> obj_points(matches.size());
     vector<Point2f> scene_points(matches.size());

     for(size_t i=0; i < matches.size(); i++)
     {
         obj_points[i] = obj_keypoints[matches[i].trainIdx].pt;
         scene_points[i] = scene_keypoints[matches[i].queryIdx].pt;

     }       

    Mat inliers_mask;

    const double ransac_reproj_threshold = 2.5f;
    

    homography= findHomography(    obj_points,
                                    scene_points,
                                    RANSAC,
                                    ransac_reproj_threshold, 
                                    inliers_mask);


    vector<DMatch> inlier_points;
    
    for (size_t i=0; i< matches.size(); i++)
    {
        if (inliers_mask.at<uchar>(i))
            inlier_points.push_back(matches[i]);
    }
    
    //matches.swap(inlier_points);

    std::cout << "Number of Mathcess From Homography: " << size(matches) << "\n";  

        
    return matches.size() > 8;
    
}  

