#include <iostream>
#include <vector>
#include <filesystem>
#include <opencv2/opencv.hpp>

#define HIST_SIZE 256

void histogram(const unsigned char* image, int* hist, int width, int height) {
    for (int i = 0; i < width * height; i++) {
        hist[image[i]]++;
    }
}
void equalizer(unsigned char* image, const int* cdf, int cdf_val_min, int width, int height) {
    for (int i = 0; i < width * height; i++) {
        image[i] = long(cdf[image[i]] - cdf_val_min) * (HIST_SIZE - 1) / (width * height - cdf_val_min);
    }
}

int main() {
    std::vector<std::pair<std::string, std::list<std::filesystem::path>>> imagesPaths;
    for (const auto& dir : std::filesystem::directory_iterator("../../images/")) {
        std::string dirName = dir.path().filename().string();
        std::list<std::filesystem::path> imagesList;
        for (const auto& imagePath : std::filesystem::directory_iterator(dir)) {
            imagesList.push_back(std::filesystem::relative(imagePath.path()));
        }
        imagesPaths.push_back(make_pair(dirName, imagesList));
    }
    std::sort(imagesPaths.begin(), imagesPaths.end(), [](auto &left, auto &right) {
        return left.first > right.first;
    });
    for (const auto& imageSize : imagesPaths) {
        printf("Size %s\n", imageSize.first.c_str());
        double avg_time = 0;
        for (const auto &imagePath: imageSize.second) {
            cv::Mat origImage = cv::imread(imagePath);
            if (!origImage.data) {
                printf("Image not found\n");
                continue;
            }
            int width = origImage.cols;
            int height = origImage.rows;
            cv::Mat imageHSV;
            cvtColor(origImage, imageHSV, cv::COLOR_BGR2HSV);
            cv::Mat HSVchannels[3];
            split(imageHSV, HSVchannels);
            cv::Mat image = HSVchannels[2];

            auto startTime = std::chrono::high_resolution_clock::now();
            unsigned char *h_img = image.data;
            int h_hist[HIST_SIZE];
            for (int i = 0; i < HIST_SIZE; i++) {
                h_hist[i] = 0;
            }
            histogram(h_img, h_hist, width, height);

            //Calcolo la CDF
            int h_cdf[HIST_SIZE];
            int cdf_val_min = h_hist[0];

            h_cdf[0] = h_hist[0];
            for (int i = 1; i < HIST_SIZE; i++) {
                h_cdf[i] = h_cdf[i - 1] + h_hist[i];
                if (h_cdf[i] < cdf_val_min && h_cdf[i] > 0) {
                    cdf_val_min = h_cdf[i];
                    break;
                }
            }

            equalizer(h_img, h_cdf, cdf_val_min, width, height);

            auto endTime = std::chrono::high_resolution_clock::now();
            double elapsedTime = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                    endTime - startTime).count()) / 1000000000;
            avg_time += elapsedTime;
            printf("%s\t\t%dx%d\t%fs\n", imagePath.filename().c_str(), width, height, elapsedTime);
            cv::Mat valueEq = cv::Mat(height, width, CV_8U, h_img);
            HSVchannels[2] = valueEq;
            cv::merge(HSVchannels, 3, imageHSV);
            cv::Mat eqImageBGR;
            cvtColor(imageHSV, eqImageBGR, cv::COLOR_HSV2BGR);
            auto outputDir = std::filesystem::path("../../seq/results") / imageSize.first;
            std::filesystem::create_directories(outputDir);
            cv::imwrite(outputDir / imagePath.filename(), eqImageBGR);
        }
        printf("Average time: %f\n\n", avg_time / imageSize.second.size());
    }
    return 0;
}
