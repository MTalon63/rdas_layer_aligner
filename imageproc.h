#pragma once

#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

class ImageProc {
public:
  struct HueShiftRule {
    int hMin;  // inclusive, 0–179
    int hMax;  // inclusive, 0–179
    int shift; // signed shift, wrap at 0–179
  };

private:
  static constexpr double cTerminationEps = 1e-6;
  static constexpr int cFallbackIterations = 500;

  static constexpr int cWarpMode = cv::MOTION_AFFINE;

  static const std::vector<double>& pyramidScales() {
    static const std::vector<double> scales = {1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0 / 2, 1.0};
    return scales;
  }

  static const std::vector<int>& pyramidIterations() {
    static const std::vector<int> iterations = {100, 60, 40, 30, 30};
    return iterations;
  }

public:
  static cv::Mat findMatrix(cv::Mat reference, cv::Mat target) {
    if (cWarpMode == cv::MOTION_HOMOGRAPHY) {
      cv::Mat homography = cv::Mat::eye(3, 3, CV_32F);
      cv::TermCriteria criteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, cFallbackIterations, cTerminationEps);
      cv::findTransformECC(reference, target, homography, cWarpMode, criteria);
      return homography;
    }

    cv::Mat warpMatrix = cv::Mat::eye(2, 3, CV_32F);
    const auto& scales = pyramidScales();
    const auto& iterations = pyramidIterations();

    for (size_t level = 0; level < scales.size(); ++level) {
      const double scale = scales[level];

      cv::Mat levelReference;
      cv::Mat levelTarget;
      cv::resize(reference, levelReference, cv::Size(), scale, scale, cv::INTER_AREA);
      cv::resize(target, levelTarget, cv::Size(), scale, scale, cv::INTER_AREA);

      // only the translation scales with the pyramid level
      cv::Mat levelWarp = warpMatrix.clone();
      levelWarp.at<float>(0, 2) *= (float)scale;
      levelWarp.at<float>(1, 2) *= (float)scale;

      cv::TermCriteria criteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, iterations[level], cTerminationEps);
      cv::findTransformECC(levelReference, levelTarget, levelWarp, cWarpMode, criteria);

      levelWarp.at<float>(0, 2) /= (float)scale;
      levelWarp.at<float>(1, 2) /= (float)scale;
      warpMatrix = levelWarp;
    }

    return warpMatrix;
  }

  static cv::Mat shiftImageMatrix(float x, float y) {
    // Create 2x3 affine transform matrix for translation
    return (cv::Mat_<float>(2, 3) << 1, 0, x, 0, 1, y);
  }

  /// @brief Apply hue. OpenCV uses 0–179° (not 0–360°)
  /// @param image
  /// @param rules
  /// @param blend
  /// @return
  static cv::UMat hue(cv::UMat image, const std::vector<HueShiftRule>& rules, float blend);
};