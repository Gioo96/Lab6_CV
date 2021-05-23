#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/video/tracking.hpp>
#include <iostream>


using namespace cv;
using namespace std;

// PanoramicImage Class
class Tracking{

// Methods

public:

    // Constructor
    Tracking(vector<Mat> images_f, String dataset_path);
    
    // Visualize good keypoints
    vector<vector<Point2f>> visualizeGoodKeypoints(vector<vector<KeyPoint>> &list_keypoints_dataset, vector<Mat> &list_descriptors_dataset, vector<KeyPoint> &keypoints_frame, Mat &descriptors_frame, vector<Mat> &H, Mat &img_keypoints);
    
    // Draw the rectangle
    vector<vector<Point2f>> drawRect(vector<Mat> H, Mat img_keypoints, vector<vector<Point2f>> tot_corners);
    
    //Track the objects
    void trackObjects(vector<vector<Point2f>> allcoords_keypoints, vector<vector<Point2f>> vec_corners);

// Data

public:

    // Vector of images of the given video
    vector<Mat> images_frame;
    
    // Vector of images of given dataset
    vector<Mat> images_dataset;
};
