// Copyright (c) 2023 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
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

#ifndef SPEAKER_SPEAKER_ENGINE_H_
#define SPEAKER_SPEAKER_ENGINE_H_

#include <memory>
#include <string>
#include <vector>

#include "frontend/feature_pipeline.h"
#include "speaker/speaker_model.h"

#ifdef USE_ONNX
#include "speaker/onnx_speaker_model.h"
#endif
#ifdef USE_MNN
#include "speaker/mnn_speaker_model.h"
#endif

#include <Eigen/Dense>

namespace wespeaker {

template <std::size_t EMBEDDING_SIZE>
class SpeakerEngine {
 public:
  using EmbeddingVector = Eigen::Matrix<float, EMBEDDING_SIZE, 1>;
  explicit SpeakerEngine(const std::string& model_path, const int feat_dim,
                             const int sample_rate,
                             const int SamplesPerChunk, const int num_threads=1) {
  // NOTE(cdliang): default num_threads = 1
  //const int kNumGemmThreads = num_threads;
  LOG(INFO) << "Reading model " << model_path;
  //embedding_size_ = embedding_size;
  LOG(INFO) << "Embedding size: " << EMBEDDING_SIZE;
  per_chunk_samples_ = SamplesPerChunk;
  LOG(INFO) << "per_chunk_samples: " << per_chunk_samples_;
  sample_rate_ = sample_rate;
  LOG(INFO) << "Sample rate: " << sample_rate_;
  feature_config_ =
      std::make_shared<wenet::FeaturePipelineConfig>(feat_dim, sample_rate);
  feature_pipeline_ =
      std::make_shared<wenet::FeaturePipeline>(*feature_config_);
  feature_pipeline_->Reset();
#ifdef USE_ONNX
  OnnxSpeakerModel::InitEngineThreads(num_threads);
#ifdef USE_GPU
  // NOTE(cdliang): default gpu_id = 0
  OnnxSpeakerModel::SetGpuDeviceId(0);
#endif
  model_ = std::make_shared<OnnxSpeakerModel>(model_path);
#elif USE_MNN
  model_ = std::make_shared<MnnSpeakerModel>(model_path, num_threads);
#elif USE_BPU
  model_ = std::make_shared<BpuSpeakerModel>(model_path);
#endif
}

void ApplyMean(std::vector<std::vector<float>>* feat,
                              unsigned int feat_dim) {
  std::vector<float> mean(feat_dim, 0);
  for (auto& i : *feat) {
    std::transform(i.begin(), i.end(), mean.begin(), mean.begin(),
                   std::plus<>{});
  }
  std::transform(mean.begin(), mean.end(), mean.begin(),
                 [&](const float d) { return d / feat->size(); });
  for (auto& i : *feat) {
    std::transform(i.begin(), i.end(), mean.begin(), i.begin(), std::minus<>{});
  }
}

  // return embedding_size
  constexpr std::size_t EmbeddingSize() const{
    return  EMBEDDING_SIZE;
  };
  // extract fbank
// 1. full mode
// When per_chunk_samples_ <= 0, extract the features of the full audio.
// 2. chunk by chunk
// Extract audio features chunk by chunk, with 198 frames for each chunk.
// If the last chunk is less than 198 frames,
// concatenate the head frame to the tail.
void ExtractFeature(
    const int16_t* data, int data_size,
    std::vector<std::vector<std::vector<float>>>* chunks_feat) {
  if (data != nullptr) {
    std::vector<std::vector<float>> chunk_feat;
    feature_pipeline_->AcceptWaveform(
        std::vector<int16_t>(data, data + data_size));
    if (per_chunk_samples_ <= 0) {
      // full mode
      feature_pipeline_->Read(feature_pipeline_->num_frames(), &chunk_feat);
      feature_pipeline_->Reset();
      chunks_feat->emplace_back(chunk_feat);
      chunk_feat.clear();
    } else {
      // NOTE(cdliang): extract feature with chunk by chunk
      int num_chunk_frames_ =
          1 + ((per_chunk_samples_ - sample_rate_ / 1000 * 25) /
               (sample_rate_ / 1000 * 10));
      int chunk_num =
          std::ceil(feature_pipeline_->num_frames() / num_chunk_frames_);
      chunks_feat->reserve(chunk_num);
      chunk_feat.reserve(num_chunk_frames_);
      while (feature_pipeline_->NumQueuedFrames() >= num_chunk_frames_) {
        feature_pipeline_->Read(num_chunk_frames_, &chunk_feat);
        chunks_feat->emplace_back(chunk_feat);
        chunk_feat.clear();
      }
      // last_chunk
      int last_frames = feature_pipeline_->NumQueuedFrames();
      if (last_frames > 0) {
        feature_pipeline_->Read(last_frames, &chunk_feat);
        if (chunks_feat->empty()) {
          // wav_len < chunk_len
          int num_pad = static_cast<int>(num_chunk_frames_ / last_frames);
          for (int i = 1; i < num_pad; i++) {
            chunk_feat.insert(chunk_feat.end(), chunk_feat.begin(),
                              chunk_feat.begin() + last_frames);
          }
          chunk_feat.insert(
              chunk_feat.end(), chunk_feat.begin(),
              chunk_feat.begin() + (num_chunk_frames_ - chunk_feat.size()));
        } else {
          chunk_feat.insert(chunk_feat.end(), (*chunks_feat)[0].begin(),
                            (*chunks_feat)[0].begin() + (num_chunk_frames_ -
                                chunk_feat.size()));
        }
        CHECK_EQ(static_cast<int32_t>(chunk_feat.size()), num_chunk_frames_);
        chunks_feat->emplace_back(chunk_feat);
        chunk_feat.clear();
      }
      feature_pipeline_->Reset();
    }
  } else {
    LOG(ERROR) << "Input is nullptr!";
  }
}
  // extract embedding
  void ExtractEmbedding(const int16_t* data, int data_size,
                                     std::vector<float>* avg_emb) {
  // chunks_feat: [nchunk, T, D]
  std::vector<std::vector<std::vector<float>>> chunks_feat;
  this->ExtractFeature(data, data_size, &chunks_feat);
  int chunk_num = chunks_feat.size();
  avg_emb->resize(EMBEDDING_SIZE, 0);
  for (int i = 0; i < chunk_num; i++) {
    std::vector<float> tmp_emb;
    this->ApplyMean(&chunks_feat[i], chunks_feat[i][0].size());
    model_->ExtractEmbedding(chunks_feat[i], &tmp_emb);
    for (std::size_t j = 0; j < tmp_emb.size(); j++) {
      (*avg_emb)[j] += tmp_emb[j];
    }
  }
  // avg_emb: [embedding_size_]
  for (size_t i = 0; i < avg_emb->size(); i++) {
    (*avg_emb)[i] /= chunk_num;
  }
}

float CosineSimilarity(const std::vector<float>& emb1,
                                      const std::vector<float>& emb2) const{
  CHECK_EQ(emb1.size(), emb2.size());
  float dot = std::inner_product(emb1.begin(), emb1.end(), emb2.begin(), 0.0f);
  float emb1_sum =
      std::inner_product(emb1.begin(), emb1.end(), emb1.begin(), 0.0f);
  float emb2_sum =
      std::inner_product(emb2.begin(), emb2.end(), emb2.begin(), 0.0f);
  dot /= std::max(std::sqrt(emb1_sum) * std::sqrt(emb2_sum),
                  std::numeric_limits<float>::epsilon());
  return dot;
}

  void ExtractEmbedding(const int16_t* data, int data_size, EmbeddingVector& avg_emb)
  {
    // chunks_feat: [nchunk, T, D]
    std::vector<std::vector<std::vector<float>>> chunks_feat;
    this->ExtractFeature(data, data_size, &chunks_feat);
    int chunk_num = chunks_feat.size();
    std::vector<float> tmp_emb;
    float inv_chunk_num = 1.0 / chunk_num;

    tmp_emb.reserve(EMBEDDING_SIZE);
    avg_emb = Eigen::Matrix<float, EMBEDDING_SIZE, 1>::Zero();
    for (int i = 0; i < chunk_num; i++) {
      tmp_emb.clear();
      this->ApplyMean(&chunks_feat[i], chunks_feat[i][0].size());
      model_->ExtractEmbedding(chunks_feat[i], &tmp_emb);
      Eigen::Map<Eigen::Matrix<float, EMBEDDING_SIZE, 1>> tmp_emb_map(tmp_emb.data());

      avg_emb += tmp_emb_map;
      /*
      for (std::size_t j = 0; j < EMBEDDING_SIZE; j++) {
        (avg_emb)[j] += tmp_emb[j];
      }*/
    }
    // avg_emb: [embedding_size_]
    avg_emb *= inv_chunk_num;
    /*
    for (size_t i = 0; i < EMBEDDING_SIZE; i++) {
      (avg_emb)[i] /= chunk_num;
    }*/
  }


  float CosineSimilarity( const EmbeddingVector& emb1, const EmbeddingVector& emb2) const {
    CHECK_EQ(emb1.rows(), emb2.rows());
    float dot = emb1.dot(emb2);
        //std::inner_product(emb1.begin(), emb1.end(), emb2.begin(), 0.0f);
    float emb1_sum = emb1.squaredNorm();
        //std::inner_product(emb1.begin(), emb1.end(), emb1.begin(), 0.0f);
    float emb2_sum = emb2.squaredNorm();
        //std::inner_product(emb2.begin(), emb2.end(), emb2.begin(), 0.0f);
    dot /= std::max(std::sqrt(emb1_sum) * std::sqrt(emb2_sum),
                    std::numeric_limits<float>::epsilon());
    return dot;
  }

 private:
  std::shared_ptr<wespeaker::SpeakerModel> model_ = nullptr;
  std::shared_ptr<wenet::FeaturePipelineConfig> feature_config_ = nullptr;
  std::shared_ptr<wenet::FeaturePipeline> feature_pipeline_ = nullptr;
  int per_chunk_samples_ = 32000;
  int sample_rate_ = 16000;
};

}  // namespace wespeaker

#endif  // SPEAKER_SPEAKER_ENGINE_H_
