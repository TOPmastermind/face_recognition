#include "face_recognition.hpp"

FaceRecognition::FaceRecognition()
{
    this->network = cv::dnn::readNetFromTensorflow(this->detection_model_file_path, this->config_file_path);
    this->face_recognizer = cv::face::LBPHFaceRecognizer::create(1, 8, 12, 12);
    this->face_recognizer->read(this->recognition_model_file_path);
}

void FaceRecognition::brightenImage(cv::Mat input_image, cv::Mat &output_image)
{
    cv::cvtColor(input_image, output_image, cv::COLOR_BGR2YCrCb);
    std::vector<cv::Mat> channels;
    cv::split(output_image, channels);
    cv::equalizeHist(channels[0], channels[0]);
    cv::merge(channels, output_image);
    cv::cvtColor(output_image, output_image, cv::COLOR_YCrCb2BGR);
}

void FaceRecognition::detectFace(cv::Mat input_image, std::vector<cv::Vec4i> &detection_output)
{
    detection_output.clear();
    cv::Mat input_blob = cv::dnn::blobFromImage(input_image, 1, cv::Size(300, 300), cv::Scalar(104, 177, 123), 1);
    this->network.setInput(input_blob, "data");
    cv::Mat detection = this->network.forward("detection_out");
    cv::Mat detection_matrix(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
    for (int i = 0; i < detection_matrix.rows; i++)
    {
        float confidence = detection_matrix.at<float>(i, 2);
        if (confidence > 0.6)
        {
            int x1 = static_cast<int>(detection_matrix.at<float>(i, 3) * input_image.cols);
            int y1 = static_cast<int>(detection_matrix.at<float>(i, 4) * input_image.rows);
            int x2 = static_cast<int>(detection_matrix.at<float>(i, 5) * input_image.cols);
            int y2 = static_cast<int>(detection_matrix.at<float>(i, 6) * input_image.rows);
            detection_output.push_back(cv::Vec4i(x1, y1, x2, y2));
        }
    }
}

void FaceRecognition::recognize(cv::Mat input_image)
{
    cv::Mat bright_image;
    this->brightenImage(input_image, bright_image);
    std::vector<cv::Vec4i> detection_output;
    this->detectFace(bright_image, detection_output);

    cv::Mat output_image = input_image.clone();
    for (int i = 0; i < detection_output.size(); i++)
    {
        cv::Vec4i face_region = detection_output[i];

        cv::Rect region(cv::Point(face_region[0], face_region[1]), cv::Point(face_region[2], face_region[3]));
        cv::Mat face_image = bright_image(region);
        cv::resize(face_image, face_image, cv::Size(300, 300));
        cv::cvtColor(face_image, face_image, cv::COLOR_BGR2GRAY);
        int label;
        double confidence;
        this->face_recognizer->predict(face_image, label, confidence);
        if (confidence < 105)
        {
            cv::rectangle(output_image, cv::Point(face_region[0], face_region[1]), cv::Point(face_region[2], face_region[3]), cv::Scalar(0, 255, 0), 2, 3);
        }
    }

    cv::imshow("Output", output_image);
    cv::waitKey(1);
}