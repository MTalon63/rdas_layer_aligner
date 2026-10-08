#include <filesystem>
#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <thread>
#include <tuple>

#include "configstorage.h"
#include "imageproc.h"
#include "opencv2/core/ocl.hpp"
#include "parameterparser.h"
#include "productcbor.h"

namespace cv {
    inline Mat imread(const std::filesystem::path& path, int flags = IMREAD_COLOR) {
        return cv::imread(path.string(), flags);
    }
    inline Mat imread(const char* path, int flags = IMREAD_COLOR) {
        return cv::imread(std::string(path), flags);
    }

    inline bool imwrite(const std::filesystem::path& path, InputArray img,
                        const std::vector<int>& params = std::vector<int>()) {
        return cv::imwrite(path.string(), img, params);
    }
    inline bool imwrite(const char* path, InputArray img,
                        const std::vector<int>& params = std::vector<int>()) {
        return cv::imwrite(std::string(path), img, params);
    }
}
static constexpr int CH2_OFFSET = 3606;
static constexpr int CH3_OFFSET = 1804;

void valuesFromEuclidean(const cv::Mat& warp);
std::tuple<cv::Mat, cv::Mat> calculateMatrix(cv::Mat reference, cv::Mat ch1, cv::Mat ch2);
std::tuple<cv::Mat, cv::Mat> loadMatrix(std::filesystem::path ch1Config, std::filesystem::path ch2Config);
void valuesFromAffine(const cv::Mat& warp, const std::string& prefix);

std::tuple<cv::Mat, cv::Mat> calculateMatrix(cv::Mat reference, cv::Mat ch1, cv::Mat ch2) {
  cv::Mat warpCh1;
  cv::Mat warpCh2;
  // Run the ECC algorithm. The results are stored in warp_matrix.
  std::thread ch1Thread([&] {
    warpCh1 = ImageProc::findMatrix(reference, ch1);
  });
  std::thread ch2Thread([&] {
    warpCh2 = ImageProc::findMatrix(reference, ch2);
  });
  ch1Thread.join();
  ch2Thread.join();
  return {warpCh1, warpCh2};
}

void valuesFromAffine(const cv::Mat& warp, const std::string& prefix) {
  // Extract affine components
  float a = warp.at<float>(0, 0);
  float b = warp.at<float>(0, 1);
  float c = warp.at<float>(1, 0);
  float d = warp.at<float>(1, 1);
  float dx = warp.at<float>(0, 2);
  float dy = warp.at<float>(1, 2);

  // --- Rotation (degrees) ---
  float rotation_rad = atan2(c, a);
  float rotation_deg = rotation_rad * 180.0 / CV_PI;

  // --- Scale ---
  float scale_x = std::sqrt(a * a + b * b);
  float scale_y = std::sqrt(c * c + d * d);

  // --- Shear (slant) ---
  float shear_rad = atan2(-b, d);
  float shear_deg = shear_rad * 180.0 / CV_PI;

  std::cout << prefix << "Translation: dx=" << dx << ", dy=" << dy << std::endl;
  std::cout << prefix << "Rotation(rad/deg)= " << rotation_rad << "/" << rotation_deg << "°" << std::endl;
  std::cout << prefix << "Scale: x=" << scale_x << ", y=" << scale_y << std::endl;
  std::cout << prefix << "Shear: (rad/deg)= " << shear_rad << "/" << shear_deg << "°" << std::endl;
}

void valuesFromEuclidean(const cv::Mat& warp) {
  float dx = warp.at<float>(0, 2);
  float dy = warp.at<float>(1, 2);

  float cos_theta = warp.at<float>(0, 0);
  float sin_theta = warp.at<float>(1, 0);
  float theta_rad = atan2(sin_theta, cos_theta);
  float theta_deg = theta_rad * 180.0 / CV_PI;

  std::cout << "dx = " << dx << "  dy = " << dy << "  rotation(rad/deg) = " << theta_rad << "/" << theta_deg << "°" << std::endl;
}

/**
 * @brief Finds affines between ch1 and ch2/3, saves them as affineIndex
 *
 * @param config Affine value config file
 * @param affineIndex Index to save affine values as
 * @param input 3 input channels to find the affines for
 */
void findAffines(Config& config, const std::string& affineIndex, const std::vector<cv::Mat>& input) {

  // Vertical offsets between channels on MSU-GS, consistent between all satellites
  auto ch2Shift = ImageProc::shiftImageMatrix(0, CH2_OFFSET);
  auto ch3Shift = ImageProc::shiftImageMatrix(0, CH3_OFFSET);

  std::cout << "Calibration started for " << affineIndex << std::endl;

  // - Alignment and affine parameter finding for channel 2
  cv::Mat ch2WarpMatrix = (cv::Mat_<float>(2, 3) << 1, 0, 0, 0, 1, 0);
  auto ch2thread = std::thread([&]() {
    cv::Mat ch2Aligned;
    cv::warpAffine(input[1], ch2Aligned, ch2Shift, input[1].size(), cv::INTER_LINEAR + cv::WARP_INVERSE_MAP);
    ch2WarpMatrix = ImageProc::findMatrix(input[0], ch2Aligned);
  });

  // - Alignment and affine parameter finding for channel 3
  cv::Mat ch3WarpMatrix = (cv::Mat_<float>(2, 3) << 1, 0, 0, 0, 1, 0);
  auto ch3thread = std::thread([&]() {
    cv::Mat ch3Aligned;
    cv::warpAffine(input[2], ch3Aligned, ch3Shift, input[2].size(), cv::INTER_LINEAR + cv::WARP_INVERSE_MAP);
    ch3WarpMatrix = ImageProc::findMatrix(input[0], ch3Aligned);
  });

  ch2thread.join();
  ch3thread.join();

  ch2WarpMatrix.at<float>(0, 2) += ch2Shift.at<float>(0, 2); // Horizontal
  ch2WarpMatrix.at<float>(1, 2) += ch2Shift.at<float>(1, 2); // Vertical

  ch3WarpMatrix.at<float>(0, 2) += ch3Shift.at<float>(0, 2); // Horizontal
  ch3WarpMatrix.at<float>(1, 2) += ch3Shift.at<float>(1, 2); // Vertical

  std::cout << "Calibration done for " << affineIndex << std::endl;

  valuesFromAffine(ch2WarpMatrix, "CH2 ");
  valuesFromAffine(ch3WarpMatrix, "CH3 ");

  config.setTransfromMatrix(ch2WarpMatrix, affineIndex, "CH2");
  config.setTransfromMatrix(ch3WarpMatrix, affineIndex, "CH3");
}

// Todo: this doesn't work yet
void findOverlap(const cv::UMat& left, const cv::UMat& right) {
  int stripWidth = 200;
  cv::UMat strip = left(cv::Rect(left.cols - stripWidth, 0, stripWidth, left.rows));
  cv::imwrite("test1.jpg", strip);

  // ----- 2. Extract search region from right image -----
  cv::UMat search = right(cv::Rect(0, 0, stripWidth, right.rows));
  cv::imwrite("test2.jpg", search);

  // ----- 3. Template match to find vertical shift -----
  cv::UMat result;
  cv::matchTemplate(search, strip, result, cv::TM_CCORR_NORMED);
  cv::imwrite("result.jpg", search);

  // result is (right.rows - left.rows + 1) tall, find best match
  cv::Point best;
  cv::minMaxLoc(result, nullptr, nullptr, nullptr, &best);

  std::cout << "Offset=" << best << std::endl;
}

/**
 * @brief Writes out an NC from a vector of RGB channels
 *
 * @param channels R, G, B channels for the NC
 * @param outputPath File to write the NC to
 */
void writeNaturalColor(const std::vector<cv::UMat>& channels, const std::filesystem::path& outputPath) {
  cv::UMat RGB321Image;
  cv::merge(channels, RGB321Image);

  // Hue shift red +60°, yellow -7.2°, overlap 40% - this is the NC
  std::vector<ImageProc::HueShiftRule> rules = {
    // RED
    {170, 20, 30},
    // YELLOW
    {20, 50, -3},
  };
  auto hueCorrectedImg = ImageProc::hue(RGB321Image, rules, 40.0);

  // For now just write out the raw 321
  cv::imwrite(outputPath, hueCorrectedImg);
}

/**
 * @brief Aligns the input array of channels, resulting in three uniform VIS channels
 *
 * @param config Affine value config file
 * @param affineIndex Index of the affine values
 * @param input Vector of 3 raw input channels
 * @param output Vector of 3 aligned output channels
 */
void alignChannels(Config& config, const std::string& affineIndex, const std::vector<cv::Mat>& input, std::vector<cv::UMat>& output) {

  cv::Mat warpCh2 = config.getTransformMatrix(affineIndex, "CH2");
  cv::Mat warpCh3 = config.getTransformMatrix(affineIndex, "CH3");

  valuesFromAffine(warpCh2, "CH2 ");
  valuesFromAffine(warpCh3, "CH3 ");

  cv::UMat ch2Aligned;
  cv::UMat ch3Aligned;
  cv::warpAffine(input[1], ch2Aligned, warpCh2, input[1].size(), cv::INTER_LINEAR + cv::WARP_INVERSE_MAP);
  cv::warpAffine(input[2], ch3Aligned, warpCh3, input[2].size(), cv::INTER_LINEAR + cv::WARP_INVERSE_MAP);

  // Channel 1 is left unchanged, calibration ramp will stay on it
  output.push_back(input[0].getUMat(cv::ACCESS_READ));
  output.push_back(ch2Aligned);
  output.push_back(ch3Aligned);

  // Crops away useless empty bottom, this is present because of the shifted channels
  cv::Rect cropSize(0, 0, output[0].cols, output[0].rows - 3606);
  output[0] = output[0](cropSize);
  output[1] = output[1](cropSize);
  output[2] = output[2](cropSize);
}

cv::UMat mergeLeftRight(Config& config, const std::string& satellite, const cv::UMat& left, const cv::UMat& right) {
  int width = left.cols + right.cols;
  int height = std::max(left.rows, right.rows);
  cv::UMat merged(height, width, left.type());

  cv::Rect leftCrop = config.getROI(satellite + "VIS1");
  cv::Rect rightCrop = config.getROI(satellite + "VIS2");

  auto leftROI = left(leftCrop);
  auto rightROI = right(rightCrop);
  leftROI.copyTo(merged(cv::Rect(0, 0, leftROI.cols, leftROI.rows)));
  rightROI.copyTo(merged(cv::Rect(leftROI.cols, 0, rightROI.cols, rightROI.rows)));

  return merged;
}


// Drops the raw channel transforms; the channels are already physically aligned.
static void saveAlignedProductCbor(const std::filesystem::path& sourceDirectory,
                                   const std::filesystem::path& outputDirectory) {
  const auto source = sourceDirectory / "product.cbor";
  const auto target = outputDirectory / "product.cbor";

  if (std::filesystem::exists(target)) {
    std::filesystem::remove(target);
  }

  if (!ProductCbor::writeIdentityTransforms(source, target)) {
    std::cout << "Warning: could not strip channel transforms from product.cbor, copying it unchanged!" << std::endl;
    std::filesystem::copy(source, target);
  }
}

static int run(int argc, char** argv) {
  cv::ocl::setUseOpenCL(true);
  if (cv::ocl::useOpenCL()) {
    std::cout << "OpenCL is enabled!" << std::endl;
  } else {
    std::cout << "OpenCL is disabled or not supported." << std::endl;
  }

  ParameterParser parameters;
  parameters.parseArgs(argc, argv);

  Config config("./config.json");

  // - Calibration mode -
  // Gets the affines for both channels (2 & 3) to overlay them over ch1 with the best overlap

  if (parameters.getMode() == ParameterParser::Mode::cCalibrate) {
    auto RDASDirectory = parameters.getRDASDirectory();

    if (!std::filesystem::exists(RDASDirectory / "MSUGS_VIS1") || !std::filesystem::exists(RDASDirectory / "MSUGS_VIS2")) {
      throw std::runtime_error("You didn't point to a live-decoded RDAS directory! No MSUGS-VIS1 and/or MSUGS-VIS2 folders were found");
    }

    // VIS1
    std::vector<cv::Mat> VIS1_channels;

    cv::Mat VIS1_ch1 = cv::imread(RDASDirectory / "MSUGS_VIS1" / "MSUGS-VIS-1.png", cv::IMREAD_GRAYSCALE); // R
    cv::Mat VIS1_ch2 = cv::imread(RDASDirectory / "MSUGS_VIS1" / "MSUGS-VIS-2.png", cv::IMREAD_GRAYSCALE); // G
    cv::Mat VIS1_ch3 = cv::imread(RDASDirectory / "MSUGS_VIS1" / "MSUGS-VIS-3.png", cv::IMREAD_GRAYSCALE); // B

    VIS1_channels.push_back(VIS1_ch1);
    VIS1_channels.push_back(VIS1_ch2);
    VIS1_channels.push_back(VIS1_ch3);

    findAffines(config, parameters.getSatellite() + "VIS1", VIS1_channels);

    // VIS2
    std::vector<cv::Mat> VIS2_channels;

    cv::Mat VIS2_ch1 = cv::imread(RDASDirectory / "MSUGS_VIS2" / "MSUGS-VIS-1.png", cv::IMREAD_GRAYSCALE); // R
    cv::Mat VIS2_ch2 = cv::imread(RDASDirectory / "MSUGS_VIS2" / "MSUGS-VIS-2.png", cv::IMREAD_GRAYSCALE); // G
    cv::Mat VIS2_ch3 = cv::imread(RDASDirectory / "MSUGS_VIS2" / "MSUGS-VIS-3.png", cv::IMREAD_GRAYSCALE); // B

    VIS2_channels.push_back(VIS2_ch1);
    VIS2_channels.push_back(VIS2_ch2);
    VIS2_channels.push_back(VIS2_ch3);

    findAffines(config, parameters.getSatellite() + "VIS2", VIS2_channels);

    std::cout << "Affine values for both channels have been saved!" << std::endl;

    return 0;
  }

  // - Generation mode -
  // Aligns both VIS channels, outputs products to dir

  auto RDASDirectory = parameters.getRDASDirectory();
  auto NCPath = parameters.outputPath();
  auto VIS1_directory = RDASDirectory / "MSUGS_VIS1";
  auto VIS2_directory = RDASDirectory / "MSUGS_VIS2";

  if (!std::filesystem::exists(VIS1_directory) || !std::filesystem::exists(VIS2_directory)) {
    throw std::runtime_error("You didn't point to a live-decoded RDAS directory! No MSUGS-VIS1 and/or MSUGS-VIS2 folders were found");
  }

  std::cout << "Aligning VIS1..." << std::endl;
  std::vector<cv::Mat> VIS1_channels;
  std::vector<cv::UMat> VIS1_alignedChannels;

  cv::Mat VIS1_ch1 = cv::imread(VIS1_directory / "MSUGS-VIS-1.png", cv::IMREAD_UNCHANGED); // R
  cv::Mat VIS1_ch2 = cv::imread(VIS1_directory / "MSUGS-VIS-2.png", cv::IMREAD_UNCHANGED); // G
  cv::Mat VIS3_ch3 = cv::imread(VIS1_directory / "MSUGS-VIS-3.png", cv::IMREAD_UNCHANGED); // B

  VIS1_channels.push_back(VIS1_ch1);
  VIS1_channels.push_back(VIS1_ch2);
  VIS1_channels.push_back(VIS3_ch3);

  alignChannels(config, parameters.getSatellite() + "VIS1", VIS1_channels, VIS1_alignedChannels);


  std::cout << "Aligning VIS2..." << std::endl;
  std::vector<cv::Mat> VIS2_channels;
  std::vector<cv::UMat> VIS2_alignedChannels;

  cv::Mat VIS2_ch1 = cv::imread(VIS2_directory / "MSUGS-VIS-1.png", cv::IMREAD_UNCHANGED); // R
  cv::Mat VIS2_ch2 = cv::imread(VIS2_directory / "MSUGS-VIS-2.png", cv::IMREAD_UNCHANGED); // G
  cv::Mat VIS2_ch3 = cv::imread(VIS2_directory / "MSUGS-VIS-3.png", cv::IMREAD_UNCHANGED); // B

  VIS2_channels.push_back(VIS2_ch1);
  VIS2_channels.push_back(VIS2_ch2);
  VIS2_channels.push_back(VIS2_ch3);

  alignChannels(config, parameters.getSatellite() + "VIS2", VIS2_channels, VIS2_alignedChannels);

  // Can we merge?
  if (!config.ROIExists(parameters.getSatellite() + "VIS1") || !config.ROIExists(parameters.getSatellite() + "VIS2")) {
    std::cout << "Merge parameters aren't available for the chosen satellite! Saving aligned VIS channels separately instead..." << std::endl;

    // VIS1
    {
      std::filesystem::path output_directory = RDASDirectory / "MSUGS_VIS-1_Aligned";
      if (!std::filesystem::exists(output_directory))
        std::filesystem::create_directory(output_directory);

      std::cout << "Saving VIS1..." << std::endl;
      cv::imwrite(output_directory / "MSUGS-VIS-1.png", VIS1_channels[0]);
      cv::imwrite(output_directory / "MSUGS-VIS-2.png", VIS1_channels[1]);
      cv::imwrite(output_directory / "MSUGS-VIS-3.png", VIS1_channels[2]);

      saveAlignedProductCbor(VIS1_directory, output_directory);

      std::cout << "Finished saving VIS1 products! CBOR saved at: " << output_directory / "product.cbor" << std::endl;
    }
    // VIS2
    {
      std::filesystem::path output_directory = RDASDirectory / "MSUGS_VIS-2_Aligned";
      if (!std::filesystem::exists(output_directory))
        std::filesystem::create_directory(output_directory);

      std::cout << "Saving VIS2..." << std::endl;
      cv::imwrite(output_directory / "MSUGS-VIS-1.png", VIS2_channels[0]);
      cv::imwrite(output_directory / "MSUGS-VIS-2.png", VIS2_channels[1]);
      cv::imwrite(output_directory / "MSUGS-VIS-3.png", VIS2_channels[2]);

      saveAlignedProductCbor(VIS1_directory, output_directory);

      std::cout << "Finished saving VIS2 products! CBOR saved at: " << output_directory / "product.cbor" << std::endl;
    }

    return 0;
  }

  std::cout << "Merge parameters were found, creating merged products..." << std::endl;

  // Merge channels
  auto mergedCH1 = mergeLeftRight(config, parameters.getSatellite(), VIS1_alignedChannels[0], VIS2_alignedChannels[0]);
  auto mergedCH2 = mergeLeftRight(config, parameters.getSatellite(), VIS1_alignedChannels[1], VIS2_alignedChannels[1]);
  auto mergedCH3 = mergeLeftRight(config, parameters.getSatellite(), VIS1_alignedChannels[2], VIS2_alignedChannels[2]);


  // Save results
  std::filesystem::path output_directory = RDASDirectory / "MSUGS_VIS_Aligned";

  if (!std::filesystem::exists(output_directory))
    std::filesystem::create_directory(output_directory);

  std::cout << "Saving..." << std::endl;
  cv::imwrite(output_directory / "MSUGS-VIS-1.png", mergedCH1);
  cv::imwrite(output_directory / "MSUGS-VIS-2.png", mergedCH2);
  cv::imwrite(output_directory / "MSUGS-VIS-3.png", mergedCH3);

  saveAlignedProductCbor(VIS1_directory, output_directory);

  std::cout << "Finished saving merged products! CBOR saved at: " << output_directory / "product.cbor" << std::endl;

  // Writes an NC if a path was supplied
  if (NCPath != "") {
    // TODOREWORK: NC won't work with 16-bit images. Decimate to 8 bit?
    std::cout << "Sorry, NC generation is not supported for now!" << std::endl;
    return 0;

    std::cout << "Saving NC..." << std::endl;

    std::vector<cv::UMat> merged{mergedCH1, mergedCH2, mergedCH3};
    writeNaturalColor(merged, NCPath);

    std::cout << "Natural color has been written to " << NCPath << std::endl;
  }

  return 0;
}

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}
