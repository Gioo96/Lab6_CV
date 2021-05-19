#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/calib3d.hpp>
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
    void visualizeGoodKeypoints(vector<vector<KeyPoint>> &list_keypoints_dataset, vector<Mat> &list_descriptors_dataset, vector<KeyPoint> &keypoints_frame, Mat &descriptors_frame, vector<Mat> &H);
    
    // Draw the rectangle
    void drawRect(vector<Mat> H);

// Data

protected:

    // Vector of images of the given video
    vector<Mat> images_frame;
    
    // Vector of images of given dataset
    vector<Mat> images_dataset;
};
