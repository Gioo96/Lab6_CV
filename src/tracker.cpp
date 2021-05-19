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

void Tracking::visualizeGoodKeypoints(vector<vector<KeyPoint>> &list_keypoints_dataset, vector<Mat> &list_descriptors_dataset, vector<KeyPoint> &keypoints_frame, Mat &descriptors_frame, vector<Mat> &H) {
    
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

            if (matches.at(j).distance < 3 * min_distance) {

                refined_matches.push_back(matches.at(j));
            }
        }
        
        // Good matches (RANSAC)
        vector<Point2f> scene, object;
        for (int j = 0; j < refined_matches.size(); j++) {
            
            // Refined matches between the dataset image and first framed img (Pixel cords)
            scene.push_back(keypoints_frame.at(refined_matches.at(j).queryIdx).pt);
            object.push_back(list_keypoints_dataset.at(i).at(refined_matches.at(j).trainIdx).pt);
        }
        
        Mat H_single;
        vector<int> mask; // mask will contain 0 if the match is wrong
        H_single = findHomography(object, scene, mask, RANSAC);
        H.push_back(H_single);

        // Good matches
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
    
    // Visualize good keypoints
    vector<Scalar> color = {Scalar(0,0,255), Scalar(255,0,0), Scalar(0,255,0), Scalar(50,50,50)};
    Mat img_keypoints = images_frame.at(0);
    for (int i=0; i<images_dataset.size(); i++) {
        
        drawKeypoints(img_keypoints, allgood_keypoints.at(i), img_keypoints, color.at(i));
    }
    imshow("Visualize good keypoints", img_keypoints);
    waitKey(0);
}

void Tracking::drawRect(vector<Mat> H) {
 
    // Corners of each image of the dataset
    vector<Point2f> obj_corners(4);
    
    obj_corners[0] = Point2f(0, 0);
    obj_corners[1] = Point2f(static_cast<float>(images_dataset.at(0).cols), 0);
    obj_corners[2] = Point2f(static_cast<float>(images_dataset.at(0).cols), static_cast<float>(images_dataset.at(0).rows));
    obj_corners[3] = Point2f(0, static_cast<float>(images_dataset.at(0).rows));
         
    vector<Scalar> color = {Scalar(0,0,255), Scalar(255,0,0), Scalar(0,255,0), Scalar(50,50,50)};
    for (int i=0; i<H.size(); i++) {
    
        // Compute corrispondent pixel in frame image
        vector<Point2f> scene_corners(4);
        perspectiveTransform(obj_corners, scene_corners, H.at(i));
    
        // Draw lines
        line(images_frame.at(0), scene_corners[0], scene_corners[1], color.at(i), 3);
        line(images_frame.at(0), scene_corners[1], scene_corners[2], color.at(i), 3);
        line(images_frame.at(0), scene_corners[2], scene_corners[3], color.at(i), 3);
        line(images_frame.at(0), scene_corners[3], scene_corners[0], color.at(i), 3);
        imshow("Object detection n. " + to_string(i+1), images_frame.at(0));
        waitKey(0);
    }
}
