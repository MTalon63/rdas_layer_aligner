#pragma once

// jsoncpp header layout differs per platform/distro: Homebrew and MSYS2
// install <json/json.h>, while most Linux packages install <jsoncpp/json/json.h>.
#if defined(__has_include)
#if __has_include(<jsoncpp/json/json.h>)
#include <jsoncpp/json/json.h>
#else
#include <json/json.h>
#endif
#else
#include <json/json.h>
#endif


#include <filesystem>
#include <opencv2/opencv.hpp>
#include <string>

class Config {
public:
  Config(const std::filesystem::path& path);
  Config(const Config& other) = delete;
  Config& operator=(const Config& other) = delete;

  cv::Mat getTransformMatrix(const std::string& satellite, const std::string& channel);
  void setTransfromMatrix(const cv::Mat& matrix, const std::string& satellite, const std::string& channel);

  bool ROIExists(const std::string& key);
  cv::Rect getROI(const std::string& satellite);
  void setROI(const std::string& satellite, const cv::Rect& rect);

private:
  void load();
  void save();

private:
  const std::filesystem::path mConfigPath;
  Json::Value mJsonConfig;
};