#include "productcbor.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {

struct Value {
  enum class Type { Null, Bool, Int, Double, String, Bytes, Array, Map } type = Type::Null;
  bool boolean = false;
  long long integer = 0;
  double real = 0.0;
  std::string str;
  std::vector<unsigned char> bytes;
  std::vector<Value> array;
  std::vector<std::pair<std::string, Value>> map;

  Value* find(const std::string& key) {
    if (type != Type::Map) {
      return nullptr;
    }
    for (auto& entry : map) {
      if (entry.first == key) {
        return &entry.second;
      }
    }
    return nullptr;
  }
};

double halfToDouble(unsigned short half) {
  const int sign = (half >> 15) & 0x1;
  const int exponent = (half >> 10) & 0x1f;
  const int fraction = half & 0x3ff;
  double value = 0.0;
  if (exponent == 0) {
    value = fraction * std::pow(2.0, -24);
  } else if (exponent == 31) {
    value = fraction ? std::numeric_limits<double>::quiet_NaN() : std::numeric_limits<double>::infinity();
  } else {
    value = std::pow(2.0, exponent - 15) * (1.0 + fraction / 1024.0);
  }
  return sign ? -value : value;
}

class Decoder {
public:
  explicit Decoder(const std::vector<unsigned char>& data)
    : mData(data) {}

  bool decode(Value& out) { return readValue(out); }

private:
  const std::vector<unsigned char>& mData;
  size_t mPos = 0;

  bool readByte(unsigned char& out) {
    if (mPos >= mData.size()) {
      return false;
    }
    out = mData[mPos++];
    return true;
  }

  bool readUint(unsigned char additional, unsigned long long& out) {
    if (additional < 24) {
      out = additional;
      return true;
    }
    if (additional == 24) {
      if (mPos + 1 > mData.size()) return false;
      out = mData[mPos++];
      return true;
    }
    if (additional == 25) {
      if (mPos + 2 > mData.size()) return false;
      out = (static_cast<unsigned long long>(mData[mPos]) << 8) | mData[mPos + 1];
      mPos += 2;
      return true;
    }
    if (additional == 26) {
      if (mPos + 4 > mData.size()) return false;
      out = 0;
      for (int i = 0; i < 4; ++i) out = (out << 8) | mData[mPos++];
      return true;
    }
    if (additional == 27) {
      if (mPos + 8 > mData.size()) return false;
      out = 0;
      for (int i = 0; i < 8; ++i) out = (out << 8) | mData[mPos++];
      return true;
    }
    return false;
  }

  bool readValue(Value& out) {
    unsigned char initial = 0;
    if (!readByte(initial)) {
      return false;
    }
    const unsigned char major = initial >> 5;
    const unsigned char additional = initial & 0x1f;
    unsigned long long length = 0;

    switch (major) {
      case 0:
        if (!readUint(additional, length)) return false;
        out.type = Value::Type::Int;
        out.integer = static_cast<long long>(length);
        return true;
      case 1:
        if (!readUint(additional, length)) return false;
        out.type = Value::Type::Int;
        out.integer = -1 - static_cast<long long>(length);
        return true;
      case 2:
        if (!readUint(additional, length)) return false;
        if (mPos + length > mData.size()) return false;
        out.type = Value::Type::Bytes;
        out.bytes.assign(mData.begin() + mPos, mData.begin() + mPos + length);
        mPos += length;
        return true;
      case 3:
        if (!readUint(additional, length)) return false;
        if (mPos + length > mData.size()) return false;
        out.type = Value::Type::String;
        out.str.assign(reinterpret_cast<const char*>(&mData[mPos]), length);
        mPos += length;
        return true;
      case 4:
        if (!readUint(additional, length)) return false;
        out.type = Value::Type::Array;
        out.array.resize(length);
        for (auto& element : out.array) {
          if (!readValue(element)) return false;
        }
        return true;
      case 5:
        if (!readUint(additional, length)) return false;
        out.type = Value::Type::Map;
        out.map.reserve(length);
        for (unsigned long long i = 0; i < length; ++i) {
          Value key;
          Value mapped;
          if (!readValue(key) || !readValue(mapped)) return false;
          out.map.emplace_back(key.type == Value::Type::String ? key.str : std::string(), std::move(mapped));
        }
        return true;
      case 6:
        if (!readUint(additional, length)) return false;
        return readValue(out);
      case 7:
        if (additional == 20) { out.type = Value::Type::Bool; out.boolean = false; return true; }
        if (additional == 21) { out.type = Value::Type::Bool; out.boolean = true; return true; }
        if (additional == 22) { out.type = Value::Type::Null; return true; }
        if (additional == 25) {
          if (mPos + 2 > mData.size()) return false;
          const unsigned short half = static_cast<unsigned short>((mData[mPos] << 8) | mData[mPos + 1]);
          mPos += 2;
          out.type = Value::Type::Double;
          out.real = halfToDouble(half);
          return true;
        }
        if (additional == 26) {
          if (mPos + 4 > mData.size()) return false;
          unsigned int bits = 0;
          for (int i = 0; i < 4; ++i) bits = (bits << 8) | mData[mPos++];
          float value = 0.0f;
          std::memcpy(&value, &bits, sizeof(value));
          out.type = Value::Type::Double;
          out.real = value;
          return true;
        }
        if (additional == 27) {
          if (mPos + 8 > mData.size()) return false;
          unsigned long long bits = 0;
          for (int i = 0; i < 8; ++i) bits = (bits << 8) | mData[mPos++];
          double value = 0.0;
          std::memcpy(&value, &bits, sizeof(value));
          out.type = Value::Type::Double;
          out.real = value;
          return true;
        }
        return false;
      default:
        return false;
    }
  }
};

void encodeValue(const Value& value, std::vector<unsigned char>& out) {
  const auto head = [&out](unsigned char major, unsigned long long length) {
    if (length < 24) {
      out.push_back(static_cast<unsigned char>((major << 5) | length));
    } else if (length < 256) {
      out.push_back(static_cast<unsigned char>((major << 5) | 24));
      out.push_back(static_cast<unsigned char>(length));
    } else if (length < 65536) {
      out.push_back(static_cast<unsigned char>((major << 5) | 25));
      out.push_back(static_cast<unsigned char>(length >> 8));
      out.push_back(static_cast<unsigned char>(length & 0xff));
    } else if (length < 0x100000000ULL) {
      out.push_back(static_cast<unsigned char>((major << 5) | 26));
      for (int shift = 24; shift >= 0; shift -= 8) out.push_back(static_cast<unsigned char>((length >> shift) & 0xff));
    } else {
      out.push_back(static_cast<unsigned char>((major << 5) | 27));
      for (int shift = 56; shift >= 0; shift -= 8) out.push_back(static_cast<unsigned char>((length >> shift) & 0xff));
    }
  };

  switch (value.type) {
    case Value::Type::Null:
      out.push_back(0xf6);
      break;
    case Value::Type::Bool:
      out.push_back(value.boolean ? 0xf5 : 0xf4);
      break;
    case Value::Type::Int:
      if (value.integer >= 0) {
        head(0, static_cast<unsigned long long>(value.integer));
      } else {
        head(1, static_cast<unsigned long long>(-1 - value.integer));
      }
      break;
    case Value::Type::Double: {
      out.push_back(0xfb);
      unsigned long long bits = 0;
      std::memcpy(&bits, &value.real, sizeof(bits));
      for (int shift = 56; shift >= 0; shift -= 8) out.push_back(static_cast<unsigned char>((bits >> shift) & 0xff));
      break;
    }
    case Value::Type::String:
      head(3, value.str.size());
      out.insert(out.end(), value.str.begin(), value.str.end());
      break;
    case Value::Type::Bytes:
      head(2, value.bytes.size());
      out.insert(out.end(), value.bytes.begin(), value.bytes.end());
      break;
    case Value::Type::Array:
      head(4, value.array.size());
      for (const auto& element : value.array) encodeValue(element, out);
      break;
    case Value::Type::Map:
      head(5, value.map.size());
      for (const auto& entry : value.map) {
        Value key;
        key.type = Value::Type::String;
        key.str = entry.first;
        encodeValue(key, out);
        encodeValue(entry.second, out);
      }
      break;
  }
}

} // namespace

bool ProductCbor::writeIdentityTransforms(const std::filesystem::path& source, const std::filesystem::path& target) {
  std::ifstream input(source, std::ios::binary);
  if (!input.is_open()) {
    return false;
  }
  const std::vector<unsigned char> data((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
  input.close();

  Decoder decoder(data);
  Value root;
  if (!decoder.decode(root)) {
    return false;
  }

  Value* images = root.find("images");
  if (images == nullptr || images->type != Value::Type::Array) {
    return false;
  }

  for (auto& image : images->array) {
    Value identity;
    identity.type = Value::Type::Map;
    Value typeField;
    typeField.type = Value::Type::Int;
    typeField.integer = 1;
    identity.map.emplace_back("type", typeField);

    if (Value* transform = image.find("transform")) {
      *transform = identity;
    } else {
      image.map.emplace_back("transform", identity);
    }
  }

  std::vector<unsigned char> encoded;
  encodeValue(root, encoded);

  std::ofstream output(target, std::ios::binary | std::ios::trunc);
  if (!output.is_open()) {
    return false;
  }
  output.write(reinterpret_cast<const char*>(encoded.data()), static_cast<std::streamsize>(encoded.size()));
  return output.good();
}
