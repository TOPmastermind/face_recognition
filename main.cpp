#include "face_recognition.hpp"

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        printf("Usage: ./main [camera_index]. Example: ./main 0\n");
    }
    else
    {
        FaceRecognition face_recognition;
        cv::VideoCapture camera_input;
        
        if (camera_input.open(atoi(argv[1])))
        {
            while (camera_input.isOpened())
            {   
                cv::Mat image, output_image;
                camera_input.read(image);
                face_recognition.recognize(image);
            }
        }
        else
        {
            printf("Could not open camera\n");
        }
    }
    
    return 0;
}