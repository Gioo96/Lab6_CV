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

vector<vector<Point2f>> Tracking::visualizeGoodKeypoints(vector<vector<KeyPoint>> &list_keypoints_dataset, vector<Mat> &list_descriptors_dataset, vector<KeyPoint> &keypoints_frame, Mat &descriptors_frame, vector<Mat> &H, Mat &img_keypoints) {
    
    // COMPUTE KEYPOINTS & DESCRIPTORS
    Ptr<SIFT> sift = SIFT::create();
    
    vector<Mat> gray_images_dataset(images_dataset.size());
    vector<Mat> detected_edges(images_dataset.size());
    vector<Mat> dst(images_dataset.size());
    int lowThreshold = 0;
    const int ratio_canny = 3;
    const int kernel_size = 3;
    vector<vector<Point2f>> allcoords_keypoints;
    
    // Keypoints & descriptors foreach image of the dataset
    for (int i = 0; i < images_dataset.size(); i ++) {
        
        vector<KeyPoint> keypoints_d;
        Mat descriptors_d;
        sift->detectAndCompute(images_dataset.at(i), Mat(), keypoints_d, descriptors_d);
        list_keypoints_dataset.push_back(keypoints_d);
        list_descriptors_dataset.push_back(descriptors_d);
        
        // Try to find the four corners for each image of dataset
        cvtColor(images_dataset.at(i), gray_images_dataset.at(i), COLOR_BGR2GRAY);
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
        vector<Point2f> coords_keypoints;
        
        
        for (int j=0; j<good_matches.size(); j++) {
    
            good_keypoints.push_back(keypoints_frame.at(good_matches.at(j).queryIdx));
            coords_keypoints.push_back(good_keypoints.at(j).pt);
            
            //cout <<" Punto"+to_string(j)<< good_keypoints.at(j).pt << endl;
        }
        
        allcoords_keypoints.push_back(coords_keypoints);
        allgood_keypoints.push_back(good_keypoints);
    }
    
    // Visualize good keypoints
    vector<Scalar> color = {Scalar(0,0,255), Scalar(255,0,0), Scalar(0,255,0), Scalar(50,50,50)};
    
    img_keypoints = images_frame.at(0).clone();

    for (int i=0; i<images_dataset.size(); i++) {
        
        drawKeypoints(img_keypoints, allgood_keypoints.at(i), img_keypoints, color.at(i));
    }
    
    imshow("Visualize good keypoints", img_keypoints);
    waitKey(0);
    
    return allcoords_keypoints;
}

void Tracking::drawRect(vector<Mat> H, Mat img_keypoints) {
 
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
        line(img_keypoints, scene_corners[0], scene_corners[1], color.at(i), 3);
        line(img_keypoints, scene_corners[1], scene_corners[2], color.at(i), 3);
        line(img_keypoints, scene_corners[2], scene_corners[3], color.at(i), 3);
        line(img_keypoints, scene_corners[3], scene_corners[0], color.at(i), 3);
        //imshow("Object detection n. " + to_string(i+1), images_frame.at(0));
        //waitKey(0);
    }
    imshow("Object detection on first frame", img_keypoints);
    waitKey(0);
}

void Tracking::trackObjects(vector<vector<Point2f>> allcoords_keypoints){
    int size_video = images_frame.size();
    vector<Mat> final_frame(images_frame.size());
    vector<Scalar> color = {Scalar(0,0,255), Scalar(255,0,0), Scalar(0,255,0), Scalar(50,50,50)};
    
    Mat gray, prevGray, image, first_image_gray;
    //vector<vector<Point2f>> points_eachframe;

    cvtColor(images_frame.at(0), first_image_gray, COLOR_BGR2GRAY);
    TermCriteria termcrit(TermCriteria::MAX_ITER|TermCriteria::EPS,20,0.03);

    Size subPixWinSize(10,10), winSize(10,10);
    int win_size = 10;
    
    vector<vector<Point2f>> allPrevPoints;
    for (int i = 0; i < images_dataset.size(); i++){
        vector<Point2f> prevPoints(allcoords_keypoints.at(i).size());
        prevPoints = allcoords_keypoints.at(i);
        allPrevPoints.push_back(prevPoints);
    }
    
    for (int i = 1; i < images_frame.size(); i++){
        vector<uchar> status;
        vector<float> err;
        
        //Copy each frame on image and save on gray the single channel image
        images_frame.at(i).copyTo(image);
        cvtColor(image, gray, COLOR_BGR2GRAY);
     
        
        vector<vector<Point2f>> allNextPoints;
        
        for (int j = 0; j < images_dataset.size(); j++){
            
            vector<Point2f> nextPoints;
            
            cornerSubPix(first_image_gray, allPrevPoints.at(j), Size(win_size, win_size), Size(-1,-1), termcrit);
            calcOpticalFlowPyrLK(first_image_gray, gray, allPrevPoints.at(j), nextPoints, status, err, Size( win_size*2+1, win_size*2+1 ), 5, termcrit);
            
            allNextPoints.push_back(nextPoints);
        
            for (int k = 0; k < allPrevPoints.at(j).size(); k++){
                
                //circle(first_image_gray, (allPrevPoints.at(j)).at(k), 3, color.at(j));
                circle(image, (allNextPoints.at(j)).at(k), 3, color.at(j));
            }
            
            for (int h = 0; h < allPrevPoints.at(j).size(); h++){
                
                if (!status[h]){
                    
                    allPrevPoints.at(j)[h] = allNextPoints.at(j)[h];
                }
            }
        }
        
        first_image_gray = gray;
        
        final_frame.at(i) = image;
        imshow("Video", final_frame.at(i));
        waitKey(1);
    }
    
}
