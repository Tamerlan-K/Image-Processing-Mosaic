#include <opencv2/opencv.hpp>
#include <sstream>


int importDB(cv::String DBDirectory, size_t DB_size, std::vector<cv::Mat> images, cv::Size cropSize)
{
	if (DBDirectory.size() < 1) return -1;

    for (int i = 1; i <= DB_size; i++) {
        std::stringstream ss;
        ss << DBDirectory << "/img (" << i << ").jpg";
        cv::String filename = ss.str();
        cv::Mat img = cv::imread(filename);
        if (img.empty())
            break;
        cv::Mat smoll;
        cv::resize(img, smoll, cropSize, 0, 0, cv::INTER_LINEAR);
        images.push_back(smoll);
    }
}
