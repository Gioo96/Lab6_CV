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
    String dataset_path = "/Users/gioel/Documents/Control\ System\ Engineering/Computer\ Vision/Lab_6/data/objects/*.png";
    Tracking track(frames, dataset_path);
    
    vector<vector<KeyPoint>> list_keypoints_dataset;
    vector<Mat> list_descriptors_dataset;
    vector<KeyPoint> keypoints_frame;
    Mat descriptors_frame;
    vector<vector<KeyPoint>> allgood_keypoints;
    allgood_keypoints = track.getGoodKeypoints(list_keypoints_dataset, list_descriptors_dataset, keypoints_frame, descriptors_frame, 3);
    
    // Visualize good keypoints
    vector<Scalar> color = {Scalar(0,0,255), Scalar(255,0,0), Scalar(0,255,0), Scalar(50,50,50)};
    Mat img_keypoints = track.images_frame.at(0);
    for (int i=0; i<track.images_dataset.size(); i++) {
        
        drawKeypoints(img_keypoints, allgood_keypoints.at(i), img_keypoints, color.at(i));
    }
    imshow("Good keypoints", img_keypoints);
    waitKey(0);
    
    return 0;
}
