// Copyright 2017 Yahoo Holdings. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
/**
 * \class storage::DistributorStripeComponent
 * \ingroup common
 *
 * \brief Component class including some service layer specific information.
 */

/**
 * \class storage::DistributorComponentRegister
 * \ingroup common
 *
 * \brief Specialization of ComponentRegister handling service layer components.
 */

/**
 * \class storage::DistributorManagedComponent
 * \ingroup common
 *
 * \brief Specialization of StorageManagedComponent.
 *
 * A service layer component register will use this interface in order to set
 * the service layer functionality parts.
 */

#pragma once

#include "storagecomponent.h"
#include <vespa/storage/bucketdb/bucketdatabase.h>
#include <vespa/storage/config/distributorconfiguration.h>
#include <vespa/storage/config/config-stor-distributormanager.h>
#include <vespa/storage/config/config-stor-visitordispatcher.h>
#include <vespa/storageapi/defs.h>

namespace storage {

namespace bucketdb {
    class DistrBucketDatabase;
}
namespace lib {
    class IdealNodeCalculator;
}

typedef vespa::config::content::core::internal::InternalStorDistributormanagerType DistributorConfig;
typedef vespa::config::content::core::internal::InternalStorVisitordispatcherType VisitorConfig;

struct UniqueTimeCalculator {
    virtual ~UniqueTimeCalculator() {}
    virtual api::Timestamp getUniqueTimestamp() = 0;
};

struct DistributorManagedComponent
{
    virtual ~DistributorManagedComponent() {}

    virtual void setTimeCalculator(UniqueTimeCalculator&) = 0;
    virtual void setDistributorConfig(const DistributorConfig&)= 0;
    virtual void setVisitorConfig(const VisitorConfig&) = 0;
};

struct DistributorComponentRegister : public virtual StorageComponentRegister
{
    virtual void registerDistributorComponent(DistributorManagedComponent&) = 0;
};

class DistributorComponent : public StorageComponent,
                             private DistributorManagedComponent
{
    mutable UniqueTimeCalculator* _timeCalculator;
    DistributorConfig             _distributorConfig;
    VisitorConfig                 _visitorConfig;
    uint64_t                      _internal_config_generation; // Note: NOT related to config system generations
    std::shared_ptr<const DistributorConfiguration> _config_snapshot;

    void setTimeCalculator(UniqueTimeCalculator& utc) override { _timeCalculator = &utc; }
    void setDistributorConfig(const DistributorConfig& c) override {
        _distributorConfig = c;
        update_config_snapshot();
    }
    void setVisitorConfig(const VisitorConfig& c) override {
        _visitorConfig = c;
        update_config_snapshot();
    }

    void update_config_snapshot();

public:
    using UP = std::unique_ptr<DistributorComponent>;

    DistributorComponent(DistributorComponentRegister& compReg, vespalib::stringref name);
    ~DistributorComponent() override;

    api::Timestamp getUniqueTimestamp() const {
        return _timeCalculator->getUniqueTimestamp();
    }
    const DistributorConfig& getDistributorConfig() const {
        return _distributorConfig;
    }
    const VisitorConfig& getVisitorConfig() const {
        return _visitorConfig;
    }
    uint64_t internal_config_generation() const noexcept {
        return _internal_config_generation;
    }
    std::shared_ptr<const DistributorConfiguration> total_distributor_config_sp() const noexcept {
        return _config_snapshot;
    }
};

} // storage
