#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// PanoramicImage Class
class Tracker{

// Methods

public:

    // Constructor
    Tracker(vector<Mat> images_f, String dataset_path);
    
    // Match objects
    void match(vector<vector<KeyPoint>> &list_keypoints_dataset, vector<Mat> &list_descriptors_dataset, vector<KeyPoint> &keypoints_frame, Mat &descriptors_frame, double ratio);
    
    // adsada
    vector<vector<KeyPoint>> findGoodKeyPoints(vector<vector<DMatch>> allgood_matches, vector<vector<KeyPoint>> all_keypoints, vector<KeyPoint> keypoints_frame);

// Data

protected:

    // Vector of images of the given video
    vector<Mat> images_frame;
    
    // Vector of images of given dataset
    vector<Mat> images_dataset;
};
