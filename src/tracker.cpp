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

void Tracker::match(vector<vector<KeyPoint>> &list_keypoints_dataset, vector<Mat> &list_descriptors_dataset, vector<KeyPoint> &keypoints_frame, Mat &descriptors_frame, double ratio) {
    
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
    vector<vector<DMatch>> allgood_matches;
    Ptr<BFMatcher> matcher = BFMatcher::create(NORM_L2, true);
    for (int i = 0; i < list_descriptors_dataset.size()-1; i ++) {
        
        // Get all matches of first frame and dataset img
        vector<DMatch> matches;
        matcher->match(list_descriptors_dataset.at(i), descriptors_frame, matches);
    
        // Get minimum distance between descriptors
        double min_distance = 200;
        for (int j = 0; j < matches.size(); j++) {

            if (matches.at(j).distance < min_distance && matches.at(j).distance > 0) {

                min_distance = matches.at(j).distance;
            }
        }
        
        // Refine matches
        vector<DMatch> good_matches;
        for (int j = 0; j < matches.size(); j++) {

            if (matches.at(j).distance < ratio*min_distance) {

                good_matches.push_back(matches.at(j));
            }
        }
        allgood_matches.push_back(good_matches);
        
    }
}
vector<vector<KeyPoint>> Tracker::a(vector<vector<DMatch>> allgood_matches, vector<vector<KeyPoint>> list_keypoints_dataset, vector<KeyPoint> keypoints_frame) {
    
    // Find valid matches (RANSAC)
    vector<vector<int>> mask_all;
    vector<vector<Point2f>> src_vec, dst_vec;
    for (int i = 0; i < allgood_matches.size(); i++) {
        vector<Point2f> src,dst;
        vector<int> mask; // mask will contain 0 if the match is wrong
        for (int j = 0; j < allgood_matches.at(i).size(); j++) {
            
            // Good matches between the dataset image and first framed img
            src.push_back(list_keypoints_dataset.at(i).at(allgood_matches.at(i).at(j).queryIdx).pt);
            // Good matches in the second image (2 consecutive images are considered)
            dst.push_back(keypoints_frame.at(allgood_matches.at(i).at(j).trainIdx).pt);
        }
        src_vec.push_back(src);
        dst_vec.push_back(dst);
        findHomography(src, dst, mask, RANSAC);
        mask_all.push_back(mask);
    }
    
    vector<vector<KeyPoint>> allgood_keypoints;
    for (int i=0; i<mask_all.size(); i++) {
        
        vector<KeyPoint> good_keypoints;
        for (int j=0; j<mask_all.at(i).size(); j++) {
            
            if (mask_all.at(i).at(j) != 0) {
                
                good_keypoints.push_back();
            }
            
        }
    }
    
    // Develop part4,5
}
