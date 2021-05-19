#include "tracker.h"

using namespace cv;
using namespace std;

// PanoramicImage Class
// constructor
Tracking::Tracking(vector<Mat> images_f, String dataset_path) {

    images_frame = images_f;
        
    vector<String> fn;
    glob(dataset_path, fn, false);
    if (!fn.size()) {
        
        cout<<"Error loading the dataset"<<endl;
        return;
    }
    else {
        
        size_t count = fn.size(); // Number of images
        for (size_t i = 0; i < count; i++) {
            
            images_dataset.push_back(imread(fn[i], IMREAD_COLOR));
        }
    }
}

vector<vector<KeyPoint>> Tracking::getGoodKeypoints(vector<vector<KeyPoint>> &list_keypoints_dataset, vector<Mat> &list_descriptors_dataset, vector<KeyPoint> &keypoints_frame, Mat &descriptors_frame, double ratio) {
    
    // COMPUTE KEYPOINTS & DESCRIPTORS
    Ptr<SIFT> sift = SIFT::create();
    
    // Keypoints & descriptors foreach image of the dataset
    for (int i = 0; i < images_dataset.size(); i ++) {
        
        vector<KeyPoint> keypoints_d;
        Mat descriptors_d;
        sift->detectAndCompute(images_dataset.at(i), Mat(), keypoints_d, descriptors_d);
        list_keypoints_dataset.push_back(keypoints_d);
        list_descriptors_dataset.push_back(descriptors_d);
    }
    
    // Keypoints & descriptors of the first video frame
    sift->detectAndCompute(images_frame.at(0), Mat(), keypoints_frame, descriptors_frame);
    
    // GET GOOD KEYPOINTS
    vector<vector<KeyPoint>> allgood_keypoints;
    Ptr<BFMatcher> matcher = BFMatcher::create(NORM_L2, true);
    for (int i = 0; i < images_dataset.size(); i ++) {
        
        // Get all matches of first video frame and image 'i' of the dataset
        vector<DMatch> matches;
        matcher->match(descriptors_frame, list_descriptors_dataset.at(i), matches);
        
        // Get minimum distance between descriptors
        double min_distance = 200;
        for (int j = 0; j < matches.size(); j++) {

            //if (matches.at(j).distance < min_distance && matches.at(j).distance > 0) {
            if (matches.at(j).distance < min_distance) {

                min_distance = matches.at(j).distance;
            }
        }
        
        // Refine matches
        vector<DMatch> refined_matches;
        for (int j = 0; j < matches.size(); j++) {

            if (matches.at(j).distance < ratio * min_distance) {

                refined_matches.push_back(matches.at(j));
            }
        }
        cout<<refined_matches.size()<<endl;
        
        // Good matches (RANSAC)
        vector<Point2f> src,dst;
        vector<int> mask; // mask will contain 0 if the match is wrong
        for (int j = 0; j < refined_matches.size(); j++) {
            
            // Refined matches between the dataset image and first framed img (Pixel cords)
            src.push_back(keypoints_frame.at(refined_matches.at(j).queryIdx).pt);
            dst.push_back(list_keypoints_dataset.at(i).at(refined_matches.at(j).trainIdx).pt);
        }
        
        try {
            findHomography(src, dst, mask, RANSAC);
            if (refined_matches.size() < 4) {
                throw "Not enough matches have been found!";
            }
        }
        catch (const char* msg) {
            
            cout<<msg<<endl;
        }

        vector<DMatch> good_matches;
        for (int j=0; j<mask.size(); j++) {
    
            if (mask.at(j) != 0) {
    
                good_matches.push_back(refined_matches.at(j));
            }
        }
        
        // Good keypoints
        vector<KeyPoint> good_keypoints;
        for (int j=0; j<good_matches.size(); j++) {
    
            good_keypoints.push_back(keypoints_frame.at(good_matches.at(j).queryIdx));
        }
        
        allgood_keypoints.push_back(good_keypoints);
        
    }
    return allgood_keypoints;
}

vector<vector<KeyPoint>> Tracking::visualize_matches(vector<vector<DMatch>> allgood_matches, vector<vector<KeyPoint>> list_keypoints_dataset, vector<KeyPoint> keypoints_frame) {
    
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
    
//    vector<vector<KeyPoint>> allgood_keypoints;
//    for (int i=0; i<mask_all.size(); i++) {
//
//        vector<KeyPoint> good_keypoints;
//        for (int j=0; j<mask_all.at(i).size(); j++) {
//            
//            if (mask_all.at(i).at(j) != 0) {
//
//                good_keypoints.push_back();
//            }
//
//        }
//    }
    
    // Develop part4,5
}
