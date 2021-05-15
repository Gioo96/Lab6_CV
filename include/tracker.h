#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>
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
    void match(vector<vector<KeyPoint>> &list_keypoints_dataset, vector<Mat> &list_descriptors_dataset, vector<KeyPoint> &keypoints_frame, Mat &descriptors_frame);

// Data

protected:

    // Vector of images of the given video
    vector<Mat> images_frame;
    
    // Vector of images of given dataset
    vector<Mat> images_dataset;
};
