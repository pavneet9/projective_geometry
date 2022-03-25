
#include <string>
#include <vector>
#include <lib/compute_surf.hpp> 
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/xfeatures2d.hpp"

using std::string;
using std::vector;

using namespace cv;
using namespace cv::xfeatures2d;

int main(){

    string train_file_name = "../../img/train.png";
    string query_file_name = "../../img/query.png";

    Mat img_object;
    Mat img_scene;

    auto [keypoints_1, descriptors_1] = ComputeSurf(train_file_name, img_object);
    auto [keypoints_2, descriptors_2] = ComputeSurf(query_file_name, img_scene);
    
    auto best_matches =  KeypointMatcher(descriptors_1, descriptors_2);


    // further refine matches using homogrophy
    Mat homography;

    GetOutliersUsingHomogrophy
    (       
            keypoints_1,
            keypoints_2, 
            best_matches,
            homography        
    );

    Mat img_matches;

    drawMatches( img_object, keypoints_1, img_scene, keypoints_2,
            best_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
            std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

    imwrite("../../img/out.png", img_matches);
    return 0;



}