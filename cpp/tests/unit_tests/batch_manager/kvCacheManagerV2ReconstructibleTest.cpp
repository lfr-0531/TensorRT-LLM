/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "kvCacheManagerV2TestUtils.h"
#include "kv_cache_manager_v2/kvCache.h"
#include "kv_cache_manager_v2/kvCacheManager.h"
#include "kv_cache_manager_v2/storageManager.h"
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

namespace
{
using namespace tensorrt_llm::batch_manager::kv_cache_manager_v2;

KVCacheManagerConfig makeReconstructibleConfig(bool persistent = true)
{
    auto config = test::makeConfig();
    if (!persistent)
    {
        config.layers.clear();
    }
    AttentionLayerConfig transient;
    transient.layerId = persistent ? 1 : 0;
    transient.buffers.push_back(BufferConfig{"state", 4096, std::nullopt});
    transient.slidingWindowSize = 5;
    transient.reconstructible = true;
    config.layers.emplace_back(transient);
    return config;
}

TEST(KvCacheManagerV2ReconstructibleTest, PolicyIdentityAndValidation)
{
    AttentionLayerConfig ordinary;
    ordinary.layerId = 0;
    ordinary.slidingWindowSize = 5;
    EXPECT_FALSE(ordinary.reconstructible);
    auto transient = ordinary;
    transient.layerId = 1;
    transient.reconstructible = true;
    EXPECT_FALSE(makeLifeCycle(ordinary, 4) == makeLifeCycle(transient, 4));
    transient.slidingWindowSize.reset();
    EXPECT_THROW(transient.validate(), std::invalid_argument);
    transient.slidingWindowSize = 5;
    transient.numSinkTokens = 1;
    EXPECT_THROW(transient.validate(), std::invalid_argument);
}

TEST(KvCacheManagerV2ReconstructibleTest, GlobalOnlyReuseAllocatesPrivateHistory)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    cudaStream_t stream{};
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    auto streamGuard = FuncGuard([stream]() { cudaStreamDestroy(stream); });
    for (int const prefix : {8, 9})
    {
        auto manager = std::make_shared<KvCacheManager>(makeReconstructibleConfig());
        LifeCycleId const global{0};
        LifeCycleId const transient{1};
        std::vector<TokenIdExt> tokens;
        for (int i = 0; i < prefix; ++i)
        {
            tokens.emplace_back(TokenId{i});
        }
        auto source = manager->createKvCache();
        ASSERT_TRUE(source->resume(stream));
        ASSERT_TRUE(source->resize(prefix));
        source->commit(toSpan(tokens), true);
        auto tree = source->blocks()[BlockOrdinal{0}].treeBlock;
        ASSERT_NE(tree, nullptr);
        EXPECT_NE(tree->getPage(global), nullptr);
        for (auto const& block : source->blocks())
        {
            if (block.treeBlock)
            {
                EXPECT_EQ(block.treeBlock->getPage(transient), nullptr);
            }
        }
        source->close();

        auto left = manager->createKvCache({}, toSpan(tokens));
        auto right = manager->createKvCache({}, toSpan(tokens));
        auto cleanup = FuncGuard(
            [&]()
            {
                left->close();
                right->close();
            });
        ASSERT_EQ(left->numCommittedTokens(), prefix);
        ASSERT_EQ(right->numCommittedTokens(), prefix);
        EXPECT_TRUE(left->requiresReconstruction());
        EXPECT_EQ(left->getReconstructionRanges().at(transient), std::make_pair(prefix - 4, prefix));
        EXPECT_ANY_THROW(left->markReconstructed(transient));
        // Production prefetch occurs before the first resume. It must visit
        // persistent pages only and leave reconstruction pending.
        EXPECT_TRUE(left->prefetch(kHotLevel));
        EXPECT_TRUE(left->requiresReconstruction());
        EXPECT_TRUE(
            blockPageIsNull(left->blocks()[BlockOrdinal{(prefix - 1) / 4}].pages[kDefaultBeamIndex][transient]));
        ASSERT_TRUE(left->resume(stream));
        ASSERT_TRUE(right->resume(stream));
        auto const ordinal = BlockOrdinal{(prefix - 1) / 4};
        auto leftPage = blockPageGetPage(left->blocks()[ordinal].pages[kDefaultBeamIndex][transient]);
        auto rightPage = blockPageGetPage(right->blocks()[ordinal].pages[kDefaultBeamIndex][transient]);
        ASSERT_NE(leftPage, nullptr);
        ASSERT_NE(rightPage, nullptr);
        EXPECT_FALSE(leftPage->isCommitted());
        EXPECT_NE(leftPage.get(), rightPage.get());
        EXPECT_ANY_THROW(left->commit({}));
        EXPECT_ANY_THROW(left->markReconstructed(transient, prefix - 5, prefix));
        left->markReconstructed(transient, prefix - 3, prefix);
        right->markReconstructed(transient);
        EXPECT_FALSE(left->requiresReconstruction());
        EXPECT_EQ(left->getReconstructedRanges().at(transient), std::make_pair(prefix - 3, prefix));
        left->suspend();
        ASSERT_TRUE(left->resume(stream));
        EXPECT_EQ(blockPageGetPage(left->blocks()[ordinal].pages[kDefaultBeamIndex][transient]).get(), leftPage.get());
        ASSERT_TRUE(left->resize(prefix + 4));
        std::vector<TokenIdExt> suffix(4, TokenIdExt{TokenId{100}});
        left->commit(toSpan(suffix));
        for (auto const& block : left->blocks())
        {
            if (block.treeBlock)
            {
                EXPECT_EQ(block.treeBlock->getPage(transient), nullptr);
            }
        }
    }
}

TEST(KvCacheManagerV2ReconstructibleTest, FailedPrivateAllocationCanRetry)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    auto manager = std::make_shared<KvCacheManager>(makeReconstructibleConfig());
    auto source = manager->createKvCache();
    ASSERT_TRUE(source->resume(CUstream{nullptr}));
    ASSERT_TRUE(source->resize(8));
    std::vector<TokenIdExt> tokens(8, TokenIdExt{TokenId{9}});
    source->commit(toSpan(tokens), true);
    source->close();
    auto request = manager->createKvCache({}, toSpan(tokens));
    auto cleanup = FuncGuard([&]() { request->close(); });
    auto& storage = manager->storage();
    LifeCycleId const transient{1};
    auto const pool = storage.getPoolGroupIndex(kHotLevel, transient);
    TypedVec<LifeCycleId, SlotCount> needs(storage.numLifeCycles(), 0);
    needs[transient] = storage.getStatistics(kHotLevel, pool).free;
    auto held = storage.newGpuSlots(needs);
    EXPECT_FALSE(request->resume(CUstream{nullptr}));
    EXPECT_TRUE(request->requiresReconstruction());
    EXPECT_FALSE(request->isActive());
    for (auto& slot : held[transient])
    {
        storage.releaseSlot(transient, kHotLevel, std::move(slot));
    }
    held[transient].clear();
    ASSERT_TRUE(request->resume(CUstream{nullptr}));
    request->markReconstructed(transient);
    EXPECT_FALSE(request->requiresReconstruction());
}

TEST(KvCacheManagerV2ReconstructibleTest, ScratchUnlockReturnsSlotExactlyOnce)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    auto manager = std::make_shared<KvCacheManager>(makeReconstructibleConfig());
    auto cache = manager->createKvCache();
    auto cleanup = FuncGuard([&]() { cache->close(); });
    ASSERT_TRUE(cache->resume(CUstream{nullptr}));
    auto& storage = manager->storage();
    LifeCycleId const transient{1};
    auto const pool = storage.getPoolGroupIndex(kHotLevel, transient);
    TypedVec<LifeCycleId, SlotCount> needs(storage.numLifeCycles(), 0);
    needs[transient] = 1;
    auto slots = storage.newGpuSlots(needs);
    auto const freeWithSlot = storage.getStatistics(kHotLevel, pool).free;
    {
        auto eventScope = cache->recordEventScope();
        ScratchSlotLock scratch(std::move(slots[transient].back()), *cache, transient);
        slots[transient].clear();
        ASSERT_TRUE(scratch.slot().hasValidSlot());
        scratch.unlock();
        EXPECT_FALSE(scratch.slot().hasValidSlot());
        EXPECT_EQ(storage.getStatistics(kHotLevel, pool).free, freeWithSlot + 1);
    }
    // Destruction after explicit unlock must not return the same slot twice.
    EXPECT_EQ(storage.getStatistics(kHotLevel, pool).free, freeWithSlot + 1);
}

TEST(KvCacheManagerV2ReconstructibleTest, ShrinkAcrossBlockBoundaryReleasesPrivatePages)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    auto config = makeReconstructibleConfig();
    config.tokensPerBlock = 128;
    config.swaScratchReuse = SwaScratchReuseConfig{4};
    auto manager = std::make_shared<KvCacheManager>(config);
    auto cache = manager->createKvCache();
    auto cleanup = FuncGuard([&]() { cache->close(); });
    ASSERT_TRUE(cache->resume(CUstream{nullptr}));
    ASSERT_TRUE(cache->resize(131));
    ASSERT_EQ(cache->blocks().size(), BlockOrdinal{2});
    ASSERT_FALSE(cache->hasScratchSlots());
    LifeCycleId const transient{1};
    auto const firstPage = blockPageGetPage(cache->blocks()[BlockOrdinal{0}].pages[kDefaultBeamIndex][transient]);
    ASSERT_TRUE(cache->resize(127, 127));
    EXPECT_EQ(cache->blocks().size(), BlockOrdinal{1});
    EXPECT_EQ(
        blockPageGetPage(cache->blocks()[BlockOrdinal{0}].pages[kDefaultBeamIndex][transient]).get(), firstPage.get());
    ASSERT_TRUE(cache->resize(132));
    ASSERT_EQ(cache->blocks().size(), BlockOrdinal{2});
    ASSERT_TRUE(cache->resize(128, 128));
    EXPECT_EQ(cache->blocks().size(), BlockOrdinal{1});
    EXPECT_EQ(
        blockPageGetPage(cache->blocks()[BlockOrdinal{0}].pages[kDefaultBeamIndex][transient]).get(), firstPage.get());
}

TEST(KvCacheManagerV2ReconstructibleTest, SsmSnapshotStillConstrainsMatch)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    auto config = makeReconstructibleConfig(false);
    SsmLayerConfig ssm;
    ssm.layerId = 1;
    ssm.buffers.push_back(BufferConfig{"ssm", 4096, std::nullopt});
    config.layers.emplace_back(ssm);
    config.commitMinSnapshot = true;
    auto manager = std::make_shared<KvCacheManager>(config);
    std::vector<TokenIdExt> tokens;
    for (int i = 0; i < 8; ++i)
    {
        tokens.emplace_back(TokenId{i});
    }
    auto source = manager->createKvCache();
    ASSERT_TRUE(source->resume(CUstream{nullptr}));
    ASSERT_TRUE(source->resize(8));
    source->commit(toSpan(tokens), true);
    source->close();
    auto full = manager->createKvCache({}, toSpan(tokens));
    auto cleanup = FuncGuard([&]() { full->close(); });
    EXPECT_EQ(full->numCommittedTokens(), 8);
    EXPECT_TRUE(full->requiresReconstruction());
    ASSERT_TRUE(full->resume(CUstream{nullptr}));
    full->markReconstructed(LifeCycleId{0});
    std::vector<TokenIdExt> partial(tokens.begin(), tokens.begin() + 6);
    auto shorter = manager->createKvCache({}, toSpan(partial));
    // Six tokens cannot claim the recurrent state saved after eight tokens.
    EXPECT_LT(shorter->numCommittedTokens(), 6);
    shorter->close();
}

TEST(KvCacheManagerV2ReconstructibleTest, RejectTransientOnlyConfiguration)
{
    // Token-only tree nodes have no page eviction owner; reject this layout
    // rather than letting non-reusable metadata accumulate indefinitely.
    auto config = makeReconstructibleConfig(false);
    EXPECT_THROW(config.validate(), std::invalid_argument);
}
} // namespace
