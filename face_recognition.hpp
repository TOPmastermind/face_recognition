#include "opencv2/opencv.hpp"
#include "opencv2/dnn.hpp"
#include "opencv2/face.hpp"

class FaceRecognition
{
    public:
        FaceRecognition();

        /* recognize correct face in input image */
        void recognize(cv::Mat input_image);

    private:
        cv::dnn::Net network;                                                                   // deep neural network for face detection
        std::string detection_model_file_path = "../model_files/opencv_face_detector_uint8.pb"; // model file path used in DNN
        std::string config_file_path = "../model_files/opencv_face_detector.pbtxt";             // config file path used in DNN
        std::string recognition_model_file_path = "../model_files/recognition_model.xml";       // face recognition model file path
        cv::Ptr<cv::face::LBPHFaceRecognizer> face_recognizer;                                  // face recognizer of OpenCV

        /* brighten image in low light condition */
        void brightenImage(cv::Mat input_image, cv::Mat &output_image);

        /* detect faces in input image, output is face regions (x1, y1, x2, y2) */
        void detectFace(cv::Mat input_image, std::vector<cv::Vec4i> &detection_output);
};