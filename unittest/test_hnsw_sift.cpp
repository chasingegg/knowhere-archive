// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#include <gtest/gtest.h>
#include <random>

#include "knowhere/common/Config.h"
#include "knowhere/feder/HNSW.h"
#include "knowhere/index/vector_index/ConfAdapterMgr.h"
#include "knowhere/index/vector_index/IndexHNSW.h"
#include "knowhere/index/vector_index/adapter/VectorAdapter.h"
#include "knowhere/index/vector_index/helpers/IndexParameter.h"
#include "unittest/Helper.h"
#include "unittest/range_utils.h"
#include "unittest/utils.h"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;

class HNSWSiftTest : public DataGen, public TestWithParam<std::string> {
 protected:
    void
    SetUp() override {
        Init_with_input("/home/liang/data", "siftsmall");
        conf_ = ParamGenerator::GetInstance().Gen(index_type_);
        index_ = std::make_shared<knowhere::IndexHNSW>();
    }

 protected:
    knowhere::Config conf_;
    knowhere::IndexMode index_mode_ = knowhere::IndexMode::MODE_CPU;
    knowhere::IndexType index_type_ = knowhere::IndexEnum::INDEX_HNSW;
    std::shared_ptr<knowhere::IndexHNSW> index_ = nullptr;
};

INSTANTIATE_TEST_CASE_P(HNSWParameters, HNSWSiftTest, Values("HNSWSift"));

TEST_P(HNSWSiftTest, HNSWSift_basic) {
    assert(!xb.empty());

    std::chrono::steady_clock::time_point build_begin = std::chrono::steady_clock::now();
    index_->BuildAll(base_dataset, conf_);
    std::chrono::steady_clock::time_point build_end = std::chrono::steady_clock::now();
    std::cout << "build cost = "
              << std::chrono::duration_cast<std::chrono::microseconds>(build_end - build_begin).count() / 1000 << "[ms]"
              << std::endl;

    EXPECT_EQ(index_->Count(), nb);
    EXPECT_EQ(index_->Dim(), dim);
    ASSERT_GT(index_->Size(), 0);

    GET_TENSOR_DATA(base_dataset)

    // Serialize and Load before Query
    knowhere::BinarySet bs = index_->Serialize(conf_);
    knowhere::BinaryPtr bptr = std::make_shared<knowhere::Binary>();
    bptr->data = std::shared_ptr<uint8_t[]>((uint8_t*)p_data, [&](uint8_t*) {});
    bptr->size = dim * rows * sizeof(float);
    bs.Append(RAW_DATA, bptr);

    index_->Load(bs);

    std::vector<int64_t> ids_invalid(nq, nb);
    auto id_dataset_invalid = knowhere::GenDatasetWithIds(nq, dim, ids_invalid.data());
    ASSERT_ANY_THROW(index_->GetVectorById(id_dataset_invalid, conf_));

    std::chrono::steady_clock::time_point query_begin = std::chrono::steady_clock::now();
    auto result = index_->Query(query_dataset, conf_, nullptr);
    std::chrono::steady_clock::time_point query_end = std::chrono::steady_clock::now();
    std::cout << "build cost = "
              << std::chrono::duration_cast<std::chrono::microseconds>(build_end - build_begin).count() / 1000 / nq << "[ms]"
              << std::endl;

    auto res_ids = knowhere::GetDatasetIDs(result);
    int cnt = 0;
    for (int64_t i = 0; i < nq; i++) {
        std::set<int32_t> ids_set;
        for (int64_t j = 0; j < k; j++) {
            ids_set.insert(xr[i * k + j]);
        }
        for (int64_t j = 0; j < k; j++) {
            if (ids_set.find(res_ids[i * k +j]) != ids_set.end()) {
                cnt++;
            }
        }
    }
    std::cout<<"recall "<<(double)cnt/(k * nq)<<std::endl;

}
