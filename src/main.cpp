#include "tracker.cpp"


using namespace std;
using namespace cv;

int main(int argc, const char * argv[]) {

    String path = "/Users/gioel/Documents/Control\ System\ Engineering/Computer\ Vision/Lab_6/data/video.mov";
    // vector of frames
    vector<Mat> frames;
    VideoCapture cap(path);
    if (cap.isOpened()) {
        for(;;) {
            cv::Mat frame;
            cap.read(frame);
            if (frame.empty()) {
                cout<< "Blank frame grabbed\n";
                break;
            }
            
            frames.push_back(frame);
        }
    }
    else {
        
        cout<<"Video has not been captured!"<<endl;
        return -1;
    }

    // Load dataset and video frames
    String dataset_path = "/Users/gioel/Documents/Control System Engineering/Computer Vision/Lab_6/data/objects/*.png";
    Tracking track(frames, dataset_path);
    
    vector<vector<KeyPoint>> list_keypoints_dataset;
    vector<Mat> list_descriptors_dataset;
    vector<KeyPoint> keypoints_frame;
    Mat descriptors_frame;
    vector<Mat> H;
    track.visualizeGoodKeypoints(list_keypoints_dataset, list_descriptors_dataset, keypoints_frame, descriptors_frame, H);
    
    
    vector<Point2f> obj_corners(4);

    obj_corners[0] = Point2f(0, 0);
    obj_corners[1] = Point2f((float)track.images_dataset.at(0).cols, 0);
    obj_corners[2] = Point2f((float)track.images_dataset.at(0).cols, (float)track.images_dataset.at(0).rows);
    obj_corners[3] = Point2f(0, (float)track.images_dataset.at(0).rows);
    vector<Point2f> scene_corners(4);

    cout << "That's OK" << endl;

    //perspectiveTransform(obj_corners, scene_corners, H);

//    line( track.images_frame.at(0), scene_corners[0] + Point2f((float)images_dataset.at(i).cols, 0),
//          scene_corners[1] + Point2f((float)images_dataset.at(i).cols, 0), Scalar(0, 255, 0), 4 );
//    line( img_matches.at(i), scene_corners[1] + Point2f((float)images_dataset.at(i).cols, 0),
//          scene_corners[2] + Point2f((float)images_dataset.at(i).cols, 0), Scalar( 0, 255, 0), 4 );
//    line( img_matches.at(i), scene_corners[2] + Point2f((float)images_dataset.at(i).cols, 0),
//          scene_corners[3] + Point2f((float)images_dataset.at(i).cols, 0), Scalar( 0, 255, 0), 4 );
//    line( img_matches.at(i), scene_corners[3] + Point2f((float)images_dataset.at(i).cols, 0),
//          scene_corners[0] + Point2f((float)images_dataset.at(i).cols, 0), Scalar( 0, 255, 0), 4 );
//
//    imshow("Good Matches & Object detection of image "+to_string(i), img_matches.at(i));
//    waitKey(0);
//
    return 0;
}
