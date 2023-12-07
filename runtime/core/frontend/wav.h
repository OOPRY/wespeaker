// Copyright (c) 2016 Personal (Binbin Zhang)
// Created on 2016-08-15
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef FRONTEND_WAV_H_
#define FRONTEND_WAV_H_

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <memory>
#include <string>
#include <vector>

#include "gflags/gflags.h"
#include "glog/logging.h"

DEFINE_int32(pcm_sample_rate, 16000, "pcm data sample rate");

namespace wenet {

class AudioReader {
 public:
  AudioReader() {}
  explicit AudioReader(const std::string& filename) {}
  virtual ~AudioReader() {}

  virtual int num_channel() const = 0;
  virtual int sample_rate() const = 0;
  virtual int bits_per_sample() const = 0;
  virtual int num_sample() const = 0;
  virtual const int16_t* data() const = 0;
};

struct WavHeader {
  char riff[4];  // "riff"
  unsigned int size;
  char wav[4];  // "WAVE"
  char fmt[4];  // "fmt "
  unsigned int fmt_size;
  uint16_t format;
  uint16_t channels;
  unsigned int sample_rate;
  unsigned int bytes_per_second;
  uint16_t block_size;
  uint16_t bit;
  char data[4];  // "data"
  unsigned int data_size;
};

class WavReader : public AudioReader {
 public:
  WavReader() {}
  explicit WavReader(const std::string& filename) { Open(filename); }

  bool Open(const std::string& filename) {
    FILE* fp = fopen(filename.c_str(), "rb");
    if (NULL == fp) {
      LOG(WARNING) << "Error in read " << filename;
      return false;
    }

    WavHeader header;
    fread(&header, 1, sizeof(header), fp);
    if (header.fmt_size < 16) {
      fprintf(stderr,
              "WaveData: expect PCM format data "
              "to have fmt chunk of at least size 16.\n");
      return false;
    } else if (header.fmt_size > 16) {
      int offset = 44 - 8 + header.fmt_size - 16;
      fseek(fp, offset, SEEK_SET);
      fread(header.data, 8, sizeof(char), fp);
    }
    // check "riff" "WAVE" "fmt " "data"

    // Skip any subchunks between "fmt" and "data".  Usually there will
    // be a single "fact" subchunk, but on Windows there can also be a
    // "list" subchunk.
    while (0 != strncmp(header.data, "data", 4)) {
      // We will just ignore the data in these chunks.
      fseek(fp, header.data_size, SEEK_CUR);
      // read next subchunk
      fread(header.data, 8, sizeof(char), fp);
    }

    num_channel_ = header.channels;
    sample_rate_ = header.sample_rate;
    bits_per_sample_ = header.bit;
    int num_data = header.data_size / (bits_per_sample_ / 8);
    data_.resize(num_data);
    int num_read = fread(&data_[0], 1, header.data_size, fp);
    if (num_read < header.data_size) {
      // If the header size is wrong, adjust
      header.data_size = num_read;
      num_data = header.data_size / (bits_per_sample_ / 8);
      data_.resize(num_data);
    }
    num_sample_ = num_data / num_channel_;
    fclose(fp);
    return true;
  }

  int num_channel() const { return num_channel_; }
  int sample_rate() const { return sample_rate_; }
  int bits_per_sample() const { return bits_per_sample_; }
  int num_sample() const { return num_sample_; }
  const int16_t* data() const { return data_.data(); }

 private:
  int num_channel_;
  int sample_rate_;
  int bits_per_sample_;
  int num_sample_;  // sample points per channel
  std::vector<int16_t> data_;
};

class WavWriter {
 public:
  WavWriter(const float* data, int num_sample, int num_channel, int sample_rate,
            int bits_per_sample)
      : data_(data),
        num_sample_(num_sample),
        num_channel_(num_channel),
        sample_rate_(sample_rate),
        bits_per_sample_(bits_per_sample) {}

  void Write(const std::string& filename) {
    FILE* fp = fopen(filename.c_str(), "w");
    // init char 'riff' 'WAVE' 'fmt ' 'data'
    WavHeader header;
    char wav_header[44] = {0x52, 0x49, 0x46, 0x46, 0x00, 0x00, 0x00, 0x00, 0x57,
                           0x41, 0x56, 0x45, 0x66, 0x6d, 0x74, 0x20, 0x10, 0x00,
                           0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                           0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                           0x64, 0x61, 0x74, 0x61, 0x00, 0x00, 0x00, 0x00};
    memcpy(&header, wav_header, sizeof(header));
    header.channels = num_channel_;
    header.bit = bits_per_sample_;
    header.sample_rate = sample_rate_;
    header.data_size = num_sample_ * num_channel_ * (bits_per_sample_ / 8);
    header.size = sizeof(header) - 8 + header.data_size;
    header.bytes_per_second =
        sample_rate_ * num_channel_ * (bits_per_sample_ / 8);
    header.block_size = num_channel_ * (bits_per_sample_ / 8);

    fwrite(&header, 1, sizeof(header), fp);

    for (int i = 0; i < num_sample_; ++i) {
      for (int j = 0; j < num_channel_; ++j) {
        switch (bits_per_sample_) {
          case 8: {
            char sample = static_cast<char>(data_[i * num_channel_ + j]);
            fwrite(&sample, 1, sizeof(sample), fp);
            break;
          }
          case 16: {
            int16_t sample = static_cast<int16_t>(data_[i * num_channel_ + j]);
            fwrite(&sample, 1, sizeof(sample), fp);
            break;
          }
          case 32: {
            int sample = static_cast<int>(data_[i * num_channel_ + j]);
            fwrite(&sample, 1, sizeof(sample), fp);
            break;
          }
        }
      }
    }
    fclose(fp);
  }

 private:
  const float* data_;
  int num_sample_;  // total float points in data_
  int num_channel_;
  int sample_rate_;
  int bits_per_sample_;
};

class PcmReader : public AudioReader {
 public:
  PcmReader() {}
  explicit PcmReader(const std::string& filename) { Open(filename); }

  bool Open(const std::string& filename) {
    FILE* fp = fopen(filename.c_str(), "rb");
    if (NULL == fp) {
      LOG(WARNING) << "Error in read " << filename;
      return false;
    }

    num_channel_ = 1;
    sample_rate_ = FLAGS_pcm_sample_rate;
    bits_per_sample_ = 16;
    fseek(fp, 0, SEEK_END);
    int data_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    num_sample_ = data_size / sizeof(int16_t);
    data_.resize(num_sample_);
    fread(&data_[0], data_size, 1, fp);
    fclose(fp);
    return true;
  }

  int num_channel() const { return num_channel_; }
  int sample_rate() const { return sample_rate_; }
  int bits_per_sample() const { return bits_per_sample_; }
  int num_sample() const { return num_sample_; }

  const int16_t* data() const { return data_.data(); }

 private:
  int num_channel_;
  int sample_rate_;
  int bits_per_sample_;
  int num_sample_;  // sample points per channel
  std::vector<int16_t> data_;
};

std::shared_ptr<AudioReader> ReadAudioFile(const std::string& filename) {
  size_t pos = filename.rfind('.');
  std::string suffix = filename.substr(pos);
  if (suffix == ".wav" || suffix == ".WAV") {
    return std::make_shared<WavReader>(filename);
  } else {
    return std::make_shared<PcmReader>(filename);
  }
}

}  // namespace wenet

#endif  // FRONTEND_WAV_H_
