#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

using namespace std;
using namespace cv;

constexpr auto DB_SIZE_MAX = 2000;
double matrix[1000][1000];

double myMap(double Ilow, double Ihigh, double Olow, double Ohigh, double input);
Mat mosaicRectangular(Mat target, int boxHeight, int boxWidth, string DBDirectory);
Mat mosaicTriangular(Mat target, int boxHeight, int boxWidth, string DBDirectory);
Mat mosaicHexagonal(Mat target, int hexSize, string DBDirectory);

int main(int argc, char** argv)
{
    int rows = 1000;
    int cols = 1000;

    //    Part 1.1

    Mat p1(rows, cols, CV_8UC3, Scalar::all(55));
    imshow("1.1 - Grayscale 55", p1);
    imwrite("Grayscale.jpg", p1);

    //    Part 1.2

    vector<uchar> pattern = { 0,0,0,0,0,0,0,0, 255,255,255,255,255,255,255,255 };
    Mat p2;
    repeat(pattern, rows, cols / 16 + 1, p2);
    Rect crop(0, 0, rows, cols);
    imshow("1.2 - Pattern", p2(crop));
    imwrite("Line Pattern.jpg", p2(crop));

    // Part 1.3

    pattern.clear();
    for (int i = 0; i <= 1000; i++)
    {
        pattern.push_back(i / 4);
    }
    Mat p3;
    repeat(pattern, rows, 1, p3);
    transpose(p3, p3);
    imshow("Part 1.3 - Intensity Distribution", p3);
    imwrite("Intensity Distribution.jpg", p3);

    // Part 1.4

    Mat p4(rows, cols, CV_8U, Scalar::all(0));

    double rangeMin = 999999, rangeMax = 0;
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            double temp = 255.0 * exp( - (pow((i - 128), 2) + pow((j - 128), 2)) / pow(200, 2));
            rangeMin = min(rangeMin, temp);
            rangeMax = max(rangeMax, temp);
            matrix[i][j] = temp;
        }
    }

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            double temp = myMap(rangeMin, rangeMax, 0.0, 255.0, matrix[i][j]);
            p4.at<uchar>(i, j) = (uchar)temp;
        }
    }
    imshow("1.4 - Gaussian Distribution", p4);
    imwrite("Gaussian Distribution.jpg", p4);

    // Part 1.5 - Colored Squares

    Mat squares(rows, cols, CV_8UC3, Scalar(0, 255, 255, 0));
    squares(Rect(500, 0, 500, 500)) = Scalar(0, 255, 0, 0);
    squares(Rect(0, 500, 500, 500)) = Scalar(0, 0, 255, 0);
    squares(Rect(500, 500, 500, 500)) = Scalar::all(0);
    imshow("1.5 - Colored Squares", squares);
    imshow("Colored Squares.jpg", squares);

    // Part 2 - Mosaic
    string targetImage;
    Mat target;
    while (true) {
        cout << "Enter the target image name (including extension): ";
        cin >> targetImage;
        target = imread(targetImage);
        if (!target.empty())
            break;
        cout << "Image not found!\n";
    }

    cout << "Enter the source images' directory: ";
    string DBDirectory;
    getline(cin, DBDirectory);
    getline(cin, DBDirectory);
    //DBDirectory = "Starwars";
    //DBDirectory = "Spongebob";

    cout << "Enter mosaic type (rectangle/triangle/hexagon): ";
    string mosType;
    cin >> mosType;

    if (mosType == "rectangle")
    {
        int width = 20, height = 20;
        cout << "Enter patch size width: ";
        cin >> width;
        cout << "Enter patch size height: ";
        cin >> height;
        Mat rectangular = mosaicRectangular(target, width, height, DBDirectory);
        imshow("2 - Mosaic Square", rectangular);
        stringstream ss;
        ss << "mosaic_square_" << width << "x" << height << "_" << targetImage;
        targetImage = ss.str();
        imwrite(targetImage, rectangular);
        cout << "\n\nResult saved as: " << targetImage << endl;
    }

    else if (mosType == "triangle")
    {
        int width = 20, height = 20;
        cout << "Enter patch size width: ";
        cin >> width;
        cout << "Enter patch size height: ";
        cin >> height;
        Mat triangular = mosaicTriangular(target, width, height, DBDirectory);
        imshow("2 - Mosaic Triangle", triangular);
        stringstream ss;
        ss << "mosaic_triangle_" << width << "x" << height << "_" << targetImage;
        targetImage = ss.str();
        imwrite(targetImage, triangular);
        cout << "\n\nResult saved as: " << targetImage << endl;
    }

    else if (mosType == "hexagon")
    {
        int side = 10;
        cout << "Enter hexagon side size: ";
        cin >> side;
        Mat hexagonal = mosaicHexagonal(target, side, DBDirectory);
        imshow("2 - Mosaic Triangle", hexagonal);
        stringstream ss;
        ss << "mosaic_hexagonal_" << side << "_" << targetImage;
        targetImage = ss.str();
        imwrite(targetImage, hexagonal);
        cout << "\n\nResult saved as: " << targetImage << endl;
    }

    else
    {
        cout << "wrong input!";
        return -1;
    }

    // End

    int k = waitKey(0); // wait for any keypress and close all windows
    destroyAllWindows();

    return 0;
}

double myMap(double Ilow, double Ihigh, double Olow, double Ohigh, double input)
{
    double slope = (Ohigh - Olow) / (Ihigh - Ilow);
    return abs(Olow + slope * (input - Ilow));
}

Mat mosaicRectangular(Mat target, int boxWidth, int boxHeight, string DBDirectory)
{
    // define the resulting mosaic dimensions
    int height = target.rows;
    int width = target.cols;

    // create containers for the mosaic and database
    Mat mosaic(height, width, CV_8UC3, Scalar::all(0));
    vector <Mat> images;

    //import and resize the image database
    for (size_t i = 1; i <= DB_SIZE_MAX; i++) {
        stringstream ss;
        ss << DBDirectory << "/img (" << i << ").jpg";
        string filename = ss.str();
        Mat img = imread(filename);
        if (img.empty())
        {
            break;
        }
        Mat smoll;
        resize(img, smoll, Size(boxWidth, boxHeight), 0, 0, INTER_LINEAR);
        images.push_back(smoll);
    }

    if (images.empty())
        throw exception("mosaicRectangular|> Database empty!");

    int counter = 1, total = (width / boxWidth) * (height / boxHeight);

    // traverse each patch
    for (int i = 0; i < width - boxWidth + 1; i += boxWidth)
    {
        for (int j = 0; j < height - boxHeight + 1; j += boxHeight)
        {
            // crop the ROI
            Mat roi = target(Rect(i, j, boxWidth, boxHeight));
            long double best_match = INFINITY;
            size_t bmi = 0;

            // traverse image database
            for (size_t k = 0; k < images.size(); k++ )
            {
                long double sum = 0;

                // compare each pixel's colors
                for (int x = 0; x < boxHeight; x++) {
                    for (int y = 0; y < boxWidth; y++) {
                        sum += pow(roi.at<Vec3b>(x,y)[0] - images[k].at<Vec3b>(x, y)[0], 2); // Blue channel
                        sum += pow(roi.at<Vec3b>(x,y)[1] - images[k].at<Vec3b>(x, y)[1], 2); // Green channel
                        sum += pow(roi.at<Vec3b>(x,y)[2] - images[k].at<Vec3b>(x, y)[2], 2); // Red channel
                    }
                }
                
                // keep track of the best matching image
                sum = sqrt(sum);
                if (sum < best_match)
                {
                    best_match = sum;
                    bmi = k;
                }
            }

            // paste the found image into the corresponding patch place
            images[bmi].copyTo(mosaic(Rect(i, j, boxWidth, boxHeight)));

            // progress monitor
            std::cout.flush();
            std::printf("Progress: %.2f%%   \r", (double)counter / total * 100);
            counter++;
        }
    }

    return mosaic;
}

Mat mosaicTriangular(Mat target, int boxWidth, int boxHeight, string DBDirectory)
{
    // define the resulting mosaic dimensions
    int height = target.rows;
    int width = target.cols;

    // create containers for the mosaic and database
    Mat mosaic(height, width, CV_8UC3, Scalar::all(0));
    vector <Mat> images;

    // upper triangle mask
    vector<vector<Point>> upperTri = {
        vector<Point> {
            Point(0, 0),
            Point(0, boxHeight),
            Point(boxWidth, 0)
        }
    };
    Mat maskUpper(boxHeight, boxWidth, CV_8U, Scalar(0));
    fillPoly(maskUpper, upperTri, Scalar(255), 0);

    // lower triangle mask
    vector<vector<Point>> lowerTri = {
        vector<Point> {
            Point(0, boxHeight),
            Point(boxWidth, 0),
            Point(boxWidth, boxHeight)
        }
    };
    Mat maskLower(boxHeight, boxWidth, CV_8U, Scalar(0));
    fillPoly(maskLower, lowerTri, Scalar(255), 0);

    //import and resize the image database
    for (size_t i = 1; i <= DB_SIZE_MAX; i++) {
        stringstream ss;
        ss << DBDirectory << "/img (" << i << ").jpg";
        string filename = ss.str();
        Mat img = imread(filename);
        if (img.empty())
        {
            break;
        }
        Mat smoll;
        resize(img, smoll, Size(boxWidth, boxHeight), 0, 0, INTER_LINEAR);
        images.push_back(smoll);
    }

    if (images.empty())
        throw exception("mosaicTriangular|> Database empty!\n");

    int counter = 1, total = (width / boxWidth) * (height / boxHeight);

    // traverse each patch
    for (int i = 0; i < width - boxWidth + 1; i += boxWidth)
    {
        for (int j = 0; j < height - boxHeight + 1; j += boxHeight)
        {
            // crop the ROI
            Mat roi = target(Rect(i, j, boxWidth, boxHeight));
            double best_match = INFINITY;
            size_t bmiUp = 0;
            size_t bmiDown = 0;

            // iteration through image database
            for (size_t k = 0; k < images.size(); k++ )
            {
                if (images[k].empty()) continue;
                long double sum = 0;
                // compare upper triangle pixels' colors
                for (int x = 0; x < boxHeight; x++) {
                    for (int y = 0; y < boxWidth; y++) {
                        if (maskUpper.at<uchar>(x,y) == 0) continue; // compare only the pixels that we care about
                        sum += pow(roi.at<Vec3b>(x, y)[0] - images[k].at<Vec3b>(x, y)[0], 2); // Blue Channel Comparison
                        sum += pow(roi.at<Vec3b>(x, y)[1] - images[k].at<Vec3b>(x, y)[1], 2); // Green Channel Comparison
                        sum += pow(roi.at<Vec3b>(x, y)[2] - images[k].at<Vec3b>(x, y)[2], 2); // Red Channel Comparison
                    }
                }

                //keep track of the best matching image
                sum = sqrt(sum);
                if (sum < best_match)
                {
                    best_match = sum;
                    bmiUp = k;
                }



                sum = 0;
                // compare lower triangle pixels' colors
                for (int x = 0; x < boxHeight; x++) {
                    for (int y = 0; y < boxWidth; y++) {
                        if (maskLower.at<uchar>(x, y) == 0) continue; // compare only the pixels that we care about
                        sum += pow(roi.at<Vec3b>(x, y)[0] - images[k].at<Vec3b>(x, y)[0], 2); // Blue Channel Comparison
                        sum += pow(roi.at<Vec3b>(x, y)[1] - images[k].at<Vec3b>(x, y)[1], 2); // Green Channel Comparison
                        sum += pow(roi.at<Vec3b>(x, y)[2] - images[k].at<Vec3b>(x, y)[2], 2); // Red Channel Comparison
                    }
                }

                //keep track of the best matching image
                sum = sqrt(sum);
                if (sum < best_match)
                {
                    best_match = sum;
                    bmiDown = k;
                }
            }
            
            // paste the found images into the corresponding patch place
            images[bmiUp].copyTo(mosaic(Rect(i, j, boxWidth, boxHeight)), maskUpper);
            images[bmiDown].copyTo(mosaic(Rect(i, j, boxWidth, boxHeight)), maskLower);

            // progress monitor
            std::cout.flush();
            std::printf("Progress: %.2f%%   \r", (double)counter / total * 100);
            counter++;
        }
    }

    return mosaic;
}

Mat mosaicHexagonal(Mat target, int hexSize, string DBDirectory)
{
    // define the resulting mosaic dimensions
    int height = target.rows;
    int width = target.cols;

    int boxWidth = hexSize * 1.732;
    int boxHeight = hexSize * 2;
    
    int heightStep = boxHeight * 3 / 4;

    // create containers for the mosaic and database
    Mat mosaic(height, width, CV_8UC3, Scalar::all(255));
    vector <Mat> images;

    // hexagon mask
    vector<vector<Point>> hexagon = {
        vector<Point> {
            Point(0, boxHeight / 4), // upper left corner
            Point(boxWidth / 2, 0), // top corner
            Point(boxWidth, boxHeight / 4), // upper right corner
            Point(boxWidth, heightStep), // lower right corner
            Point(boxWidth / 2, boxHeight), // bottom corner
            Point(0, heightStep), // lower left corner
        }
    };
    Mat mask(boxHeight, boxWidth, CV_8U, Scalar(0));
    fillPoly(mask, hexagon, Scalar(255), 0);

    //import and resize the image database
    for (size_t i = 1; i <= DB_SIZE_MAX; i++) {
        stringstream ss;
        ss << DBDirectory << "/img (" << i << ").jpg";
        string filename = ss.str();
        Mat img = imread(filename);
        if (img.empty())
        {
            break;
        }
        Mat smoll;
        resize(img, smoll, Size(boxWidth, boxHeight), 0, 0, INTER_LINEAR);
        images.push_back(smoll);
    }

    if (images.empty())
        throw exception("mosaicHexagonal|> Database empty!\n");

    int counter = 1, total = (width / boxWidth) * (height / heightStep);

    // traverse each patch
    for (int i = 0; i < width - boxWidth + 1; i += boxWidth) // horizontally
    {
        for (int j = 0; j < height - heightStep; j += heightStep) // vertically
        {
            if ((j / heightStep) % 2 && (i + boxWidth * 1.5) > width) // check if there is place for offsetting the hexagon
                continue;

            int offset = ((j / heightStep) % 2) * (boxWidth / 2); // depends on the row

            // crop the ROI
            Mat roi = target(Rect(i + offset, j, boxWidth, boxHeight));
            double best_match = INFINITY;
            size_t bmi = 0;

            // iteration through image database
            for (size_t k = 0; k < images.size(); k++)
            {
                long double sum = 0;
                // compare each pixels' colors
                for (int x = 0; x < boxHeight; x++) {
                    for (int y = 0; y < boxWidth; y++) {
                        if (mask.at<uchar>(x, y) == 0) continue; // compare only the pixels that we care about
                        sum += pow(roi.at<Vec3b>(x, y)[0] - images[k].at<Vec3b>(x, y)[0], 2); // Blue Channel Comparison
                        sum += pow(roi.at<Vec3b>(x, y)[1] - images[k].at<Vec3b>(x, y)[1], 2); // Green Channel Comparison
                        sum += pow(roi.at<Vec3b>(x, y)[2] - images[k].at<Vec3b>(x, y)[2], 2); // Red Channel Comparison
                    }
                }

                //keep track of the best matching image
                sum = sqrt(sum);
                if (sum < best_match)
                {
                    best_match = sum;
                    bmi = k;
                }
            }

            // paste the found image into the corresponding patch place
            images[bmi].copyTo(mosaic(Rect(i + offset, j, boxWidth, boxHeight)), mask);

            // progress monitor
            std::cout.flush();
            std::printf("Progress: %.2f%%   \r", (double)counter / total * 100);
            counter++;
        }
    }

    return mosaic;
}
