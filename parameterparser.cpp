#include "parameterparser.h"

#include <sstream>

ParameterParser::ParameterParser() {
  mSettingsList.push_back(ParameterData("--help", "-h", "Print help")); // TODOREWORK: this is not iplemented
  mSettingsList.push_back(ParameterData("--sat", "-s", "Satellite name (L2|L3|L4|M1|M2)"));
  mSettingsList.push_back(ParameterData("--mode", "-m", "calibrate|generate (default:generate)"));
  mSettingsList.push_back(ParameterData("--directory", "-d", "Live input directory"));
  mSettingsList.push_back(ParameterData("--nc", "-n", "Output NC image (optional)"));
}

void ParameterParser::parseArgs(int argc, char** argv) {
  for (int i = 1; i < (argc - 1); i += 2) {
    mArgs.insert(std::make_pair(argv[i], argv[i + 1]));
  }
}

std::string ParameterParser::getHelp() const {
  std::list<ParameterData>::const_iterator it;
  std::stringstream ss;

  // ss << "RDAS Layer allign Version " << VERSION_MAJOR << "." << VERSION_MINOR << "." << VERSION_FIX << std::endl;
  for (it = mSettingsList.begin(); it != mSettingsList.end(); ++it) {
    ss << (*it).argNameShort << "\t" << (*it).argName << "\t" << (*it).helpText << std::endl;
  }

  return ss.str();
}

ParameterParser::Mode ParameterParser::getMode() const {
  std::string modeStr = std::string("");

  if (mArgs.count("-m")) {
    modeStr = mArgs.at("-m");
  }
  if (mArgs.count("--mode")) {
    modeStr = mArgs.at("--mode");
  }

  if (modeStr.empty() || modeStr == "generate") {
    return Mode::cGenerate;
  }
  if (modeStr == "calibrate") {
    return Mode::cCalibrate;
  }

  throw std::runtime_error("Invalid mode parameter '" + modeStr + "' (expected: calibrate|generate, default: generate)");
}

std::string ParameterParser::getSatellite() const {
  std::string sat = std::string("");

  if (mArgs.count("-s")) {
    sat = mArgs.at("-s");
  }
  if (mArgs.count("--sat")) {
    sat = mArgs.at("--sat");
  }

  if (sat == "") {
    throw std::runtime_error("No satellite is selected");
  }

  return sat;
}

std::filesystem::path ParameterParser::getRDASDirectory() const {
  std::string path = std::string("");

  if (mArgs.count("-d")) {
    path = mArgs.at("-d");
  }
  if (mArgs.count("--directory")) {
    path = mArgs.at("--directory");
  }

  if (path == "") {
    throw std::runtime_error("You must provide a live-decoded RDAS directory!");
  }

  return path;
}

std::filesystem::path ParameterParser::outputPath() const {
  std::string path = std::string("");

  if (mArgs.count("-n")) {
    path = mArgs.at("-n");
  }
  if (mArgs.count("--nc")) {
    path = mArgs.at("--nc");
  }

  return path;
}