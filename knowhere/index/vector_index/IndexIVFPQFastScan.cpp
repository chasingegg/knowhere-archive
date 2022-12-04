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

#include <algorithm>
#include <future>
#include <string>
#include <utility>
#include <vector>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQFastScan.h>
#include <faiss/IndexRefine.h>
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
    faiss::IndexIVFPQFastScan* index_base = new faiss::IndexIVFPQFastScan(coarse_quantizer, dim, GetIndexParamNlist(config),
                                                     GetIndexParamM(config), GetIndexParamNbits(config), metric_type);
    auto index = std::make_shared<faiss::IndexRefineFlat>(index_base, reinterpret_cast<const float*>(p_data));
    index->own_fields = true;
    index->train(rows/4, reinterpret_cast<const float*>(p_data));
    index_ = index;
}

VecIndexPtr
IVFPQFASTSCAN::CopyCpuToGpu(const int64_t device_id, const Config& config) {
    KNOWHERE_THROW_MSG("Calling IVFPQFASTSCAN::CopyCpuToGpu when we are using CPU version");
}

std::shared_ptr<faiss::IVFSearchParameters>
IVFPQFASTSCAN::GenParams(const Config& config) {
    auto params = std::make_shared<faiss::IVFSearchParameters>();
    params->nprobe = GetIndexParamNprobe(config);
    params->reorder_k = GetIndexParamReorderK(config);
    // params->scan_table_threshold = config["scan_table_threhold"]
    // params->polysemous_ht = config["polysemous_ht"]
    // params->max_codes = config["max_codes"]

    return params;
}

void
IVFPQFASTSCAN::QueryImpl(int64_t n,
               const float* xq,
               int64_t k,
               float* distances,
               int64_t* labels,
               const Config& config,
               const faiss::BitsetView bitset) {
    auto params = GenParams(config);
    utils::SetQueryOmpThread(config);
    // LOG_KNOWHERE_ERROR_ << GetIndexParamNprobe(config) << " " << GetIndexParamReorderK(config) << " "
    //     << n;

    auto ivf_index = dynamic_cast<faiss::IndexRefineFlat*>(index_.get());

    int parallel_mode = -1;
    if (params->nprobe > 1 && n <= 4) {
        parallel_mode = 1;
    } else {
        parallel_mode = 0;
    }
    size_t max_codes = 0;
    auto ivf_stats = std::dynamic_pointer_cast<IVFStatistics>(stats);
    ivf_index->search_thread_safe(n, xq, k, distances, labels, params->nprobe, parallel_mode, max_codes, params->reorder_k, bitset);
    // for (unsigned int i = 0; i < n; ++i)
    //     ivf_index->search_thread_safe(1, xq + i * Dim(), k, distances + i * k,
    //         labels + i * k, params->nprobe, parallel_mode, max_codes, params->reorder_k, bitset);
    // std::vector<std::future<void>> futures;
    // futures.reserve(n);
    // for (unsigned int i = 0; i < n; ++i) {
    //     futures.push_back(pool_->push([&, index = i]() {
    //         ivf_index->search_thread_safe(1, xq + index * Dim(), k, distances + index * k,
    //             labels + index * k, params->nprobe, parallel_mode, max_codes, params->reorder_k, bitset);
    //     }));
    // }

    // for (auto& future : futures) {
    //     future.get();
    // }
}

int64_t
IVFPQFASTSCAN::Size() {
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialize");
    }
    auto index = dynamic_cast<faiss::IndexRefineFlat*>(index_.get());
    auto ivfpqfs_index = dynamic_cast<faiss::IndexIVFPQFastScan*>(index->base_index);
    auto nb = ivfpqfs_index->invlists->compute_ntotal();
    auto code_size = ivfpqfs_index->code_size;
    auto pq = ivfpqfs_index->pq;
    auto nlist = ivfpqfs_index->nlist;
    auto d = ivfpqfs_index->d;

    // ivf codes, ivf ids and quantizer
    auto capacity = nb * code_size + nb * sizeof(int64_t) + nlist * d * sizeof(float);
    auto centroid_table = pq.M * pq.ksub * pq.dsub * sizeof(float);
    auto precomputed_table = nlist * pq.M * pq.ksub * sizeof(int8_t);
#if 0
    if (precomputed_table > ivfpqfs_index->precomputed_table_max_bytes) {
        // will not precompute table
        precomputed_table = 0;
    }
#endif
    return (capacity + centroid_table + precomputed_table);
}

}  // namespace knowhere