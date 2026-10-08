#pragma once

#include <filesystem>

namespace ProductCbor {
// Channels are already aligned; keeping the raw transforms makes SatDump apply them twice.
bool writeIdentityTransforms(const std::filesystem::path& source, const std::filesystem::path& target);
}
