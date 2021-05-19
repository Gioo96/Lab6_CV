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
            //resize(images_dataset.at(i),images_dataset.at(i) ,size);
        }
    }
}

void Tracker::match(vector<vector<KeyPoint>> &list_keypoints_dataset, vector<Mat> &list_descriptors_dataset, vector<KeyPoint> &keypoints_frame, Mat &descriptors_frame, double ratio) {
    
    // COMPUTE KEYPOINTS & DESCRIPTORS
    Ptr<SIFT> sift = SIFT::create();
    
    Scalar color;
    std::vector<Point2f> obj_corners(4);
    std::vector<Point2f> scene_corners(4);
    
    vector<Mat> img_dataset_keypoints(images_dataset.size());
    // Keypoints & descriptors foreach image of the dataset
    for (size_t i = 0; i < images_dataset.size(); i ++) {
        
        vector<KeyPoint> keypoints_d;
        Mat descriptors_d;
        sift->detectAndCompute(images_dataset.at(i), Mat(), keypoints_d, descriptors_d);
        list_keypoints_dataset.push_back(keypoints_d);
        list_descriptors_dataset.push_back(descriptors_d);
        
        // Show the keypoints figured out for each dataset image
        img_dataset_keypoints.at(i) = images_dataset.at(i).clone();
        
        // Set different color for each object
        if (i==0){
            
            color = Scalar(255,0,0);
        }
        if (i==1){
            
            color = Scalar(0,255,0);
        }
        if (i==2){
            
            color = Scalar(0,0,255);
        }
        if (i==3){
            
            color = Scalar(0,255,255);
        }
        
        drawKeypoints(images_dataset.at(i), keypoints_d, img_dataset_keypoints.at(i), color);
        string string_image = "Keypoints of dataset image "+to_string(i);
        imshow(string_image, img_dataset_keypoints.at(i));
        waitKey(0);
    }
    
    
    // Keypoints & descriptors of the first video frame
    Mat first_image_frame_keypoints = images_frame.at(0).clone();
    sift->detectAndCompute(images_frame.at(0), Mat(), keypoints_frame, descriptors_frame);
    
    // Show the keypoints figured out in the first frame
    drawKeypoints(images_frame.at(0), keypoints_frame, first_image_frame_keypoints, Scalar(0,255,0));
    cout <<keypoints_frame.size() << endl;
    imshow("Image with all keypoints", first_image_frame_keypoints);
    waitKey(0);
    
    // GET MATCHES
    vector<vector<DMatch>> allgood_matches;
    Ptr<BFMatcher> matcher = BFMatcher::create(NORM_L2, true);
    vector<vector<int>> mask_all;
    vector<vector<KeyPoint>> allgood_keypoints;

    for (int i = 0; i < list_descriptors_dataset.size(); i ++) {
        
        // Set different color for each object
        if (i==0){
            color = Scalar(255,0,0);
        }
        if (i==1){
            color = Scalar(0,255,0);
        }
        if (i==2){
            color = Scalar(0,0,255);
        }
        if (i==3){
            color = Scalar(0,255,255);
        }
        
        // Get all matches of first frame and dataset img
        vector<DMatch> matches;
        matcher->match(list_descriptors_dataset.at(i), descriptors_frame, matches);     // Matches contains all matches
    
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
        
        // Visualize the matches
        vector<Mat> img_matches(list_descriptors_dataset);
        drawMatches(images_dataset.at(i), list_keypoints_dataset.at(i), images_frame.at(0), keypoints_frame, good_matches, img_matches.at(i), Scalar::all(-1), Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        imshow("Image matches of object number "+to_string(i), img_matches.at(i));
        waitKey(0);
        
        //allgood_matches.push_back(good_matches);
        
        vector<Point2f> obj,scene;
        vector<KeyPoint> keypoints_dataset;
        vector<KeyPoint> keypoints_frame_good;
        
        vector<int> mask; // mask will contain 0 if the match is wrong
        for (int j = 0; j < good_matches.size(); j++) {
            
            // Good matches between the dataset image and first framed img
            obj.push_back(list_keypoints_dataset.at(i).at(good_matches.at(j).queryIdx).pt);
            keypoints_dataset.push_back(list_keypoints_dataset.at(i).at(good_matches.at(j).queryIdx));
            
            // Good matches in the second image (2 consecutive images are considered)
            scene.push_back(keypoints_frame.at(good_matches.at(j).trainIdx).pt);
            keypoints_frame_good.push_back(keypoints_frame.at(good_matches.at(j).trainIdx));
        }
        
        // Show the keypoints figured out in the first frame considering matches
        Mat second_image_frame_keypoints = images_frame.at(0).clone();
        drawKeypoints(images_frame.at(0), keypoints_frame_good, second_image_frame_keypoints, color);
        cout <<"Good keypoints considering matches "<<keypoints_frame_good.size() << endl;
    
        imshow("Image with all good keypoints", second_image_frame_keypoints);
        waitKey(0);

        Mat H = findHomography(obj, scene, mask, RANSAC);
        
        vector<Point2f> obj_corners(4);
        
        obj_corners[0] = Point2f(0, 0);
        obj_corners[1] = Point2f((float)images_dataset.at(i).cols, 0);
        obj_corners[2] = Point2f((float)images_dataset.at(i).cols, (float)images_dataset.at(i).rows);
        obj_corners[3] = Point2f(0, (float)images_dataset.at(i).rows);
        vector<Point2f> scene_corners(4);
        
        cout << "That's OK" << endl;
        
        perspectiveTransform(obj_corners, scene_corners, H);
        
        line( img_matches.at(i), scene_corners[0] + Point2f((float)images_dataset.at(i).cols, 0),
              scene_corners[1] + Point2f((float)images_dataset.at(i).cols, 0), Scalar(0, 255, 0), 4 );
        line( img_matches.at(i), scene_corners[1] + Point2f((float)images_dataset.at(i).cols, 0),
              scene_corners[2] + Point2f((float)images_dataset.at(i).cols, 0), Scalar( 0, 255, 0), 4 );
        line( img_matches.at(i), scene_corners[2] + Point2f((float)images_dataset.at(i).cols, 0),
              scene_corners[3] + Point2f((float)images_dataset.at(i).cols, 0), Scalar( 0, 255, 0), 4 );
        line( img_matches.at(i), scene_corners[3] + Point2f((float)images_dataset.at(i).cols, 0),
              scene_corners[0] + Point2f((float)images_dataset.at(i).cols, 0), Scalar( 0, 255, 0), 4 );

        imshow("Good Matches & Object detection of image "+to_string(i), img_matches.at(i));
        waitKey(0);
        
        vector<KeyPoint> good_keypoints;
        for (int j=0; j<mask.size(); j++) {
    
            if (mask.at(j) != 0) {
    
                good_keypoints.push_back(keypoints_frame.at(good_matches.at(j).trainIdx));
            }
        }
        allgood_keypoints.push_back(good_keypoints);
        
        /*Mat third_image_frame_keypoints = images_frame.at(0).clone();
        drawKeypoints(images_frame.at(0), good_keypoints, third_image_frame_keypoints, color);
        cout <<"Final keypoints "<<good_keypoints.size()<<endl;
        imshow("Image with final keypoints", third_image_frame_keypoints);
        waitKey(0);*/
        
    }
    
}

