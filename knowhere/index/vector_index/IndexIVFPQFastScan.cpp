// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License

#include <string>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQFastScan.h>
#include <faiss/clone_index.h>

#include "common/Exception.h"
#include "common/Log.h"
#include "common/Utils.h"
#include "index/vector_index/IndexIVFPQFastScan.h"
#include "index/vector_index/adapter/VectorAdapter.h"
#include "index/vector_index/helpers/IndexParameter.h"

namespace knowhere {

void
IVFPQFASTSCAN::Train(const DatasetPtr& dataset_ptr, const Config& config) {
    GET_TENSOR_DATA_DIM(dataset_ptr)

    utils::SetBuildOmpThread(config);
    faiss::MetricType metric_type = GetFaissMetricType(config);
    faiss::Index* coarse_quantizer = new faiss::IndexFlat(dim, metric_type);
    auto index = std::make_shared<faiss::IndexIVFPQFastScan>(coarse_quantizer, dim, GetIndexParamNlist(config),
                                                     GetIndexParamM(config), GetIndexParamNbits(config), metric_type);
    index->own_fields = true;
    index->train(rows, reinterpret_cast<const float*>(p_data));
    index_ = index;
}

VecIndexPtr
IVFPQFASTSCAN::CopyCpuToGpu(const int64_t device_id, const Config& config) {
    KNOWHERE_THROW_MSG("Calling IVFPQ::CopyCpuToGpu when we are using CPU version");
}

std::shared_ptr<faiss::IVFSearchParameters>
IVFPQFASTSCAN::GenParams(const Config& config) {
    auto params = std::make_shared<faiss::IVFSearchParameters>();
    params->nprobe = GetIndexParamNprobe(config);
    // params->scan_table_threshold = config["scan_table_threhold"]
    // params->polysemous_ht = config["polysemous_ht"]
    // params->max_codes = config["max_codes"]

    return params;
}

int64_t
IVFPQFASTSCAN::Size() {
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialize");
    }
    auto ivfpqfs_index = dynamic_cast<faiss::IndexIVFPQFastScan*>(index_.get());
    auto nb = ivfpqfs_index->invlists->compute_ntotal();
    auto code_size = ivfpqfs_index->code_size;
    auto pq = ivfpqfs_index->pq;
    auto nlist = ivfpqfs_index->nlist;
    auto d = ivfpqfs_index->d;

    // ivf codes, ivf ids and quantizer
    auto capacity = nb * code_size + nb * sizeof(int64_t) + nlist * d * sizeof(float);
    auto centroid_table = pq.M * pq.ksub * pq.dsub * sizeof(float);
    auto precomputed_table = nlist * pq.M * pq.ksub * sizeof(float);
#if 0
    if (precomputed_table > ivfpqfs_index->precomputed_table_max_bytes) {
        // will not precompute table
        precomputed_table = 0;
    }
#endif
    return (capacity + centroid_table + precomputed_table);
}

}  // namespace knowhere