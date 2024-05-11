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

#include "speaker/speaker_engine.h"
#include <algorithm>
#include <functional>
#include <limits>
#include <numeric>

#ifdef USE_ONNX
#include "speaker/onnx_speaker_model.h"
#endif
#ifdef USE_MNN
#include "speaker/mnn_speaker_model.h"
#endif

namespace wespeaker {



}  // namespace wespeaker
