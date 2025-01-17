// Copyright 2017 Yahoo Holdings. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include "distributorprocess.h"
#include <vespa/storage/common/storagelink.h>
#include <vespa/storage/common/i_storage_chain_builder.h>
#include <vespa/config/helper/configgetter.hpp>

#include <vespa/log/log.h>
LOG_SETUP(".process.distributor");

namespace storage {

DistributorProcess::DistributorProcess(const config::ConfigUri & configUri)
    : Process(configUri),
      _context(),
      _num_distributor_stripes(0), // TODO STRIPE: change default when legacy single stripe mode is removed
      _node(),
      _distributorConfigHandler(),
      _visitDispatcherConfigHandler(),
      _storage_chain_builder()
{
}

DistributorProcess::~DistributorProcess() {
    shutdown();
}

void
DistributorProcess::shutdown()
{
    Process::shutdown();
    _node.reset();
}

void
DistributorProcess::setupConfig(milliseconds subscribeTimeout)
{
    using vespa::config::content::core::StorDistributormanagerConfig;
    using vespa::config::content::core::StorVisitordispatcherConfig;

    auto distr_cfg = config::ConfigGetter<StorDistributormanagerConfig>::getConfig(
            _configUri.getConfigId(), _configUri.getContext(), subscribeTimeout);
    _num_distributor_stripes = distr_cfg->numDistributorStripes;
    _distributorConfigHandler = _configSubscriber.subscribe<StorDistributormanagerConfig>(_configUri.getConfigId(), subscribeTimeout);
    _visitDispatcherConfigHandler = _configSubscriber.subscribe<StorVisitordispatcherConfig>(_configUri.getConfigId(), subscribeTimeout);
    Process::setupConfig(subscribeTimeout);
}

void
DistributorProcess::updateConfig()
{
    Process::updateConfig();
    if (_distributorConfigHandler->isChanged()) {
        _node->handleConfigChange(*_distributorConfigHandler->getConfig());
    }
    if (_visitDispatcherConfigHandler->isChanged()) {
        _node->handleConfigChange(*_visitDispatcherConfigHandler->getConfig());
    }
}

bool
DistributorProcess::configUpdated()
{
    bool changed = Process::configUpdated();
    if (_distributorConfigHandler->isChanged()) {
        LOG(info, "Distributor manager config detected changed");
        changed = true;
    }
    if (_visitDispatcherConfigHandler->isChanged()) {
        LOG(info, "Visitor dispatcher config detected changed");
        changed = true;
    }
    return changed;
}

void
DistributorProcess::createNode()
{
    _node = std::make_unique<DistributorNode>(_configUri, _context, *this, _num_distributor_stripes, StorageLink::UP(), std::move(_storage_chain_builder));
    _node->handleConfigChange(*_distributorConfigHandler->getConfig());
    _node->handleConfigChange(*_visitDispatcherConfigHandler->getConfig());
}

void
DistributorProcess::set_storage_chain_builder(std::unique_ptr<IStorageChainBuilder> builder)
{
    _storage_chain_builder = std::move(builder);
}

} // storage
