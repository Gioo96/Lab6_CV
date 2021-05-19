#include "tracker.cpp"


using namespace std;
using namespace cv;

int main(int argc, const char * argv[]) {

    String path = "../data/video.mov";
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
    String dataset_path = "../data/objects/*.png";
    
    // Create track object of class tracker
    Tracker track(frames, dataset_path);

    // Initialization of required variables
    vector<vector<KeyPoint>> list_keypoints_dataset;
    vector<Mat> list_descriptors_dataset;
    vector<KeyPoint> keypoints_frame;
    Mat descriptors_frame;
    double ratio = 3;
    
    track.match(list_keypoints_dataset, list_descriptors_dataset, keypoints_frame, descriptors_frame, ratio);
 
     return 0;
}
