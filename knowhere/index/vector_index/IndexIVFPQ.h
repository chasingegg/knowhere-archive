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

#pragma once

#include <memory>
#include <utility>

#include <faiss/IndexIVFPQFastScan.h>
#include <faiss/IndexRefine.h>
#include "knowhere/index/VecIndex.h"
#include "knowhere/common/ThreadPool.h"
namespace knowhere {

class IVFPQ : public VecIndex {
 public:
    IVFPQ() {
        index_type_ = IndexEnum::INDEX_FAISS_IVFPQ;
        pool_ = ThreadPool::GetGlobalThreadPool();
    }

    IVFPQ(const IVFPQ& index_ivfpq) = delete;

    IVFPQ&
    operator=(const IVFPQ& index_ivfpq) = delete;

    IVFPQ(IVFPQ&& index_ivfpq) noexcept = default;

    IVFPQ&
    operator=(IVFPQ&& index_ivfpq) noexcept = default;

    BinarySet
    Serialize(const Config&) override;

    void
    Load(const BinarySet&) override;

    void
    Train(const DatasetPtr&, const Config&) override;

    void
    AddWithoutIds(const DatasetPtr&, const Config&) override;

    DatasetPtr
    GetVectorById(const DatasetPtr&, const Config&) override;

    DatasetPtr
    Query(const DatasetPtr&, const Config&, const faiss::BitsetView) override;

   //  DatasetPtr
   //  QueryByRange(const DatasetPtr&, const Config&, const faiss::BitsetView) override;

   //  DatasetPtr
   //  GetIndexMeta(const Config&) override;

    int64_t
    Count() override;

    int64_t
    Dim() override;

    int64_t
    Size() override;

    VecIndexPtr
    CopyCpuToGpu(const int64_t, const Config&);

 private:
    std::shared_ptr<faiss::IVFSearchParameters>
    GenParams(const Config& config);

    void
    QueryImpl(int64_t, const float*, int64_t, float*, int64_t*, const Config&, const faiss::BitsetView);

   //  void
   //  QueryByRangeImpl(int64_t, const float*, float*&, int64_t*&, size_t*&, const Config&, const faiss::BitsetView);


 private:
    std::shared_ptr<ThreadPool> pool_;
    std::unique_ptr<faiss::IndexRefineFlat> index_;
};

// using IVFPQPtr = std::shared_ptr<IVFPQ>;

}  // namespace knowhere
