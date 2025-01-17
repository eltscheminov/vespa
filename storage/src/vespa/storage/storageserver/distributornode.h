// Copyright 2017 Yahoo Holdings. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
/**
 * \class storage::DistributorNode
 * \ingroup storageserver
 *
 * \brief Class for setting up a distributor node.
 */

#pragma once

#include "distributornodecontext.h"
#include "storagenode.h"
#include <vespa/storage/common/distributorcomponent.h>
#include <vespa/storageframework/generic/thread/tickingthread.h>

namespace storage {

namespace distributor { class DistributorStripePool; }

class IStorageChainBuilder;

class DistributorNode
      : public StorageNode,
        private UniqueTimeCalculator
{
    framework::TickingThreadPool::UP _threadPool;
    std::unique_ptr<distributor::DistributorStripePool> _stripe_pool;
    DistributorNodeContext& _context;
    uint64_t _lastUniqueTimestampRequested;
    uint32_t _uniqueTimestampCounter;
    uint32_t _num_distributor_stripes;
    std::unique_ptr<StorageLink> _retrievedCommunicationManager;

public:
    typedef std::unique_ptr<DistributorNode> UP;

    DistributorNode(const config::ConfigUri & configUri,
                    DistributorNodeContext&,
                    ApplicationGenerationFetcher& generationFetcher,
                    uint32_t num_distributor_stripes,
                    std::unique_ptr<StorageLink> communicationManager,
                    std::unique_ptr<IStorageChainBuilder> storage_chain_builder);
    ~DistributorNode() override;

    const lib::NodeType& getNodeType() const override { return lib::NodeType::DISTRIBUTOR; }
    ResumeGuard pause() override;

    void handleConfigChange(vespa::config::content::core::StorDistributormanagerConfig&);
    void handleConfigChange(vespa::config::content::core::StorVisitordispatcherConfig&);

private:
    void initializeNodeSpecific() override;
    void perform_post_chain_creation_init_steps() override { /* no-op */ }
    void createChain(IStorageChainBuilder &builder) override;
    api::Timestamp getUniqueTimestamp() override;

    /**
     * Shut down necessary distributor-specific components before shutting
     * down general content node components.
     */
    void shutdownDistributor();
};

} // storage
