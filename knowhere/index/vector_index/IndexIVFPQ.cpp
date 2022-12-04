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
#include <faiss/index_io.h>
#include <faiss/IndexIVFPQFastScan.h>
#include <faiss/impl/FaissException.h>
#include <faiss/clone_index.h>
#ifdef KNOWHERE_GPU_VERSION
#include <faiss/gpu/GpuCloner.h>
#endif

#include "common/Exception.h"
#include "common/Log.h"
#include "common/Utils.h"
#include "index/vector_index/IndexIVFPQ.h"
#include "index/vector_index/adapter/VectorAdapter.h"
#include "index/vector_index/helpers/IndexParameter.h"
#include "index/vector_index/helpers/FaissIO.h"

#ifdef KNOWHERE_GPU_VERSION
#include "index/vector_index/ConfAdapter.h"
#include "index/vector_index/gpu/IndexGPUIVF.h"
#include "index/vector_index/gpu/IndexGPUIVFPQ.h"
#endif

namespace knowhere {

BinarySet
IVFPQ::Serialize(const Config& config) {
    if (!index_ || !index_->is_trained) {
        KNOWHERE_THROW_MSG("index not initialize or trained");
    }
    try {
        faiss::IndexRefineFlat* index = index_.get();

        MemoryIOWriter writer;
        faiss::write_index(index, &writer);
        std::shared_ptr<uint8_t[]> data(writer.data_);

        BinarySet res_set;
        // TODO(linxj): use virtual func Name() instead of raw string.
        res_set.Append("IVF", data, writer.rp);
        return res_set;
    } catch (std::exception& e) {
        KNOWHERE_THROW_MSG(e.what());
    }
}

void
IVFPQ::Load(const BinarySet& index_binary) {
    try {
    auto binary = index_binary.GetByName("IVF");

    MemoryIOReader reader;
    reader.total = binary->size;
    reader.data_ = binary->data.get();

    faiss::IndexRefineFlat* index = dynamic_cast<faiss::IndexRefineFlat*>(faiss::read_index(&reader));
    index_.reset(index);
    } catch (std::exception& e) {
        KNOWHERE_THROW_MSG(e.what());
    }
}

void
IVFPQ::Train(const DatasetPtr& dataset_ptr, const Config& config) {
    GET_TENSOR_DATA_DIM(dataset_ptr)

    utils::SetBuildOmpThread(config);
    faiss::MetricType metric_type = GetFaissMetricType(config);
    faiss::Index* coarse_quantizer = new faiss::IndexFlat(dim, metric_type);
    faiss::IndexIVFPQFastScan* index_base = new faiss::IndexIVFPQFastScan(coarse_quantizer, dim, GetIndexParamNlist(config),
                                                     GetIndexParamM(config), GetIndexParamNbits(config), metric_type);
    index_ = std::make_unique<faiss::IndexRefineFlat>(index_base, reinterpret_cast<const float*>(p_data));
    index_->own_fields = true;
    index_->train(rows/4, reinterpret_cast<const float*>(p_data));
}

void
IVFPQ::AddWithoutIds(const DatasetPtr& dataset_ptr, const Config& config) {
    if (!index_ || !index_->is_trained) {
        KNOWHERE_THROW_MSG("index not initialize or trained");
    }

    GET_TENSOR_DATA(dataset_ptr)
    index_->add(rows, reinterpret_cast<const float*>(p_data));
}

DatasetPtr
IVFPQ::GetVectorById(const DatasetPtr& dataset_ptr, const Config& config) {
    if (!index_ || !index_->is_trained) {
        KNOWHERE_THROW_MSG("index not initialize or trained");
    }

    GET_DATA_WITH_IDS(dataset_ptr)

    float* p_x = nullptr;
    auto release_when_exception = [&]() {
        if (p_x != nullptr) {
            delete[] p_x;
        }
    };

    try {
        p_x = new float[dim * rows];
        auto ivf_index = dynamic_cast<faiss::IndexIVF*>(index_.get());
        ivf_index->make_direct_map(true);
        for (int64_t i = 0; i < rows; i++) {
            int64_t id = p_ids[i];
            KNOWHERE_THROW_IF_NOT_FMT(id >= 0 && id < ivf_index->ntotal, "invalid id %ld", id);
            ivf_index->reconstruct(id, p_x + i * dim);
        }
        return GenResultDataset(p_x);
    } catch (std::exception& e) {
        release_when_exception();
        KNOWHERE_THROW_MSG(e.what());
    }
}

DatasetPtr
IVFPQ::Query(const DatasetPtr& dataset_ptr, const Config& config, const faiss::BitsetView bitset) {
    if (!index_ || !index_->is_trained) {
        KNOWHERE_THROW_MSG("index not initialize or trained");
    }

    GET_TENSOR_DATA(dataset_ptr)

    utils::SetQueryOmpThread(config);
    int64_t* p_id = nullptr;
    float* p_dist = nullptr;
    auto release_when_exception = [&]() {
        if (p_id != nullptr) {
            delete[] p_id;
        }
        if (p_dist != nullptr) {
            delete[] p_dist;
        }
    };

    try {
        auto k = GetMetaTopk(config);
        p_id = new int64_t[k * rows];
        p_dist = new float[k * rows];

        QueryImpl(rows, reinterpret_cast<const float*>(p_data), k, p_dist, p_id, config, bitset);

        return GenResultDataset(p_id, p_dist);
    } catch (faiss::FaissException& e) {
        release_when_exception();
        KNOWHERE_THROW_MSG(e.what());
    } catch (std::exception& e) {
        release_when_exception();
        KNOWHERE_THROW_MSG(e.what());
    }
}

void
IVFPQ::QueryImpl(int64_t n,
               const float* xq,
               int64_t k,
               float* distances,
               int64_t* labels,
               const Config& config,
               const faiss::BitsetView bitset) {
    auto params = GenParams(config);

    int parallel_mode = -1;
    if (params->nprobe > 1 && n <= 4) {
        parallel_mode = 1;
    } else {
        parallel_mode = 0;
    }
    size_t max_codes = 0;

    std::vector<std::future<void>> futures;
    futures.reserve(n);
    for (unsigned int i = 0; i < n; ++i) {
        futures.push_back(pool_->push([&, index = i]() {
            auto single_query = xq + index * Dim();
            index_->search_thread_safe(1, single_query, k, distances + index * k, labels + index * k, params->nprobe, parallel_mode, max_codes, params->reorder_k, bitset);
        }));
    }

    for (auto& future : futures) {
        future.get();
    }
}

VecIndexPtr
IVFPQ::CopyCpuToGpu(const int64_t device_id, const Config& config) {
#ifdef KNOWHERE_GPU_VERSION
    auto ivfpq_index = dynamic_cast<faiss::IndexIVFPQ*>(index_.get());
    int64_t dim = ivfpq_index->d;
    int64_t m = ivfpq_index->pq.M;
    int64_t nbits = ivfpq_index->pq.nbits;
    if (!IVFPQConfAdapter::CheckGPUPQParams(dim, m, nbits)) {
        return nullptr;
    }
    if (auto res = FaissGpuResourceMgr::GetInstance().GetRes(device_id)) {
        ResScope rs(res, device_id, false);
        auto gpu_index = faiss::gpu::index_cpu_to_gpu(res->faiss_res.get(), device_id, index_.get());

        std::shared_ptr<faiss::Index> device_index;
        device_index.reset(gpu_index);
        return std::make_shared<GPUIVFPQ>(device_index, device_id, res);
    } else {
        KNOWHERE_THROW_MSG("CopyCpuToGpu Error, can't get gpu_resource");
    }
#else
    KNOWHERE_THROW_MSG("Calling IVFPQ::CopyCpuToGpu when we are using CPU version");
#endif
}

std::shared_ptr<faiss::IVFSearchParameters>
IVFPQ::GenParams(const Config& config) {
    auto params = std::make_shared<faiss::IVFSearchParameters>();
    params->nprobe = GetIndexParamNprobe(config);
    params->reorder_k = GetIndexParamReorderK(config);
    // params->scan_table_threshold = config["scan_table_threhold"]
    // params->polysemous_ht = config["polysemous_ht"]
    // params->max_codes = config["max_codes"]

    return params;
}

int64_t
IVFPQ::Count() {
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialize");
    }
    return index_->ntotal;
}

int64_t
IVFPQ::Dim() {
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialize");
    }
    return index_->d;
}

int64_t
IVFPQ::Size() {
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialize");
    }
    return 100;
//     auto ivfpq_index = dynamic_cast<faiss::IndexIVFPQ*>(index_.get());
//     auto nb = ivfpq_index->invlists->compute_ntotal();
//     auto code_size = ivfpq_index->code_size;
//     auto pq = ivfpq_index->pq;
//     auto nlist = ivfpq_index->nlist;
//     auto d = ivfpq_index->d;

//     // ivf codes, ivf ids and quantizer
//     auto capacity = nb * code_size + nb * sizeof(int64_t) + nlist * d * sizeof(float);
//     auto centroid_table = pq.M * pq.ksub * pq.dsub * sizeof(float);
//     auto precomputed_table = nlist * pq.M * pq.ksub * sizeof(float);
// #if 0
//     if (precomputed_table > ivfpq_index->precomputed_table_max_bytes) {
//         // will not precompute table
//         precomputed_table = 0;
//     }
// #endif
//     return (capacity + centroid_table + precomputed_table);
}

}  // namespace knowhere
