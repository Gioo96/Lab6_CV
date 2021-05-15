#include "tracker.h"

using namespace cv;
using namespace std;

// PanoramicImage Class
// constructor
Tracker::Tracker(vector<Mat> images_f, String dataset_path) {

    images_frame = images_f;
    vector<String> fn;
    glob(dataset_path, fn, false);
    if (!fn.size()) {
        
        cout<<"Error loading the dataset"<<endl;
        return;
    }
    else {
        
        size_t count = fn.size(); // Number of images
        Size size(1500,1000);
    
        for (size_t i = 0; i < count; i++) {
            
            images_dataset.push_back(imread(fn[i], IMREAD_COLOR));
            resize(images_dataset.at(i),images_dataset.at(i) ,size);
        }
    }
}

void Tracker::match(vector<vector<KeyPoint>> &list_keypoints_dataset, vector<Mat> &list_descriptors_dataset, vector<KeyPoint> &keypoints_frame, Mat &descriptors_frame) {
    
    // COMPUTE KEYPOINTS & DESCRIPTORS
    Ptr<SIFT> sift = SIFT::create();
    
    // Keypoints & descriptors foreach image of the dataset
    for (size_t i = 0; i < images_dataset.size(); i ++) {
        
        vector<KeyPoint> keypoints_d;
        Mat descriptors_d;
        sift->detectAndCompute(images_dataset.at(i), Mat(), keypoints_d, descriptors_d);
        list_keypoints_dataset.push_back(keypoints_d);
        list_descriptors_dataset.push_back(descriptors_d);
    }
    
    // Keypoints & descriptors of the first video frame
    sift->detectAndCompute(images_frame.at(0), Mat(), keypoints_frame, descriptors_frame);
    
    // GET MATCHES
    
}
