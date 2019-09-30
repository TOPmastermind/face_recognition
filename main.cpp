#include "opencv2/opencv.hpp"
#include "opencv2/videoio.hpp"

int main(int argc, char *argv[])
{
    cv::VideoCapture camera_input(0);
    while(camera_input.isOpened())
    {   
        cv::Mat image, output_image;
        camera_input.read(image);
        cv::cvtColor(image, output_image, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(output_image, output_image);
        
        cv::imshow("Camera", image);
        cv::imshow("Output image", output_image);
        cv::waitKey(1);
    }
}