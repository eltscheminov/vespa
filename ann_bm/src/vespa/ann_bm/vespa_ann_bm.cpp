// Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vespa/searchcommon/attribute/hnsw_index_params.h>
#include <vespa/searchlib/attribute/attributevector.h>
#include <vespa/searchlib/attribute/attributefactory.h>
#include <vespa/searchlib/tensor/dense_tensor_attribute.h>
#include <vespa/searchlib/tensor/nearest_neighbor_index.h>
#include <vespa/eval/eval/value.h>
#include <vespa/vespalib/test/insertion_operators.h>
#include <ostream>
#include <sstream>
#include <limits>

namespace py = pybind11;

using search::AttributeFactory;
using search::AttributeVector;
using search::attribute::BasicType;
using search::attribute::Config;
using search::attribute::CollectionType;
using search::attribute::DistanceMetric;
using search::attribute::HnswIndexParams;
using search::tensor::NearestNeighborIndex;
using search::tensor::TensorAttribute;
using vespalib::eval::CellType;
using vespalib::eval::DenseValueView;
using vespalib::eval::TypedCells;
using vespalib::eval::ValueType;
using vespalib::eval::Value;

namespace vespa_ann_bm {

using TopKResult = std::vector<std::pair<uint32_t, double>>;

namespace {

std::string
make_tensor_spec(uint32_t dim_size)
{
    std::ostringstream os;
    os << "tensor<float>(x[" << dim_size << "])";
    return os.str();
}

constexpr uint32_t lid_bias = 1; // lid 0 is reserved

struct CompareTopKResult
{
    using Value = std::pair<uint32_t, double>;
    bool operator()(const Value& lhs, const Value& rhs) const {
        if (lhs.second < rhs.second) {
            return true;
        }
        if (lhs.second > rhs.second) {
            return false;
        }
        return lhs.first < rhs.first;
    }
};

}

class AnnBm
{
    ValueType                        _tensor_type;
    HnswIndexParams                  _hnsw_index_params;
    std::shared_ptr<AttributeVector> _attribute;
    TensorAttribute*                 _tensor_attribute;
    const NearestNeighborIndex*      _nearest_neighbor_index;
    size_t                           _dim_size;
    bool                             _enable_sort;

    bool check_lid(uint32_t lid);
    bool check_value(const char *op, const std::vector<float>& value);
public:
    AnnBm(uint32_t dim_size, const HnswIndexParams &hnsw_index_params);
    virtual ~AnnBm();
    uint32_t num_docs();
    void set_value(uint32_t lid, const std::vector<float>& value);
    std::vector<float> get_value(uint32_t lid);
    void clear_value(uint32_t lid);
    TopKResult find_top_k(uint32_t k, const std::vector<float>& value, uint32_t explore_k, double distance_threshold);
    void set_enable_sort(bool enable_sort);
};

AnnBm::AnnBm(uint32_t dim_size, const HnswIndexParams &hnsw_index_params)
    : _tensor_type(ValueType::error_type()),
      _hnsw_index_params(hnsw_index_params),
      _attribute(),
      _tensor_attribute(nullptr),
      _nearest_neighbor_index(nullptr),
      _dim_size(0u),
      _enable_sort(true)
{
    Config cfg(BasicType::TENSOR, CollectionType::SINGLE);
    _tensor_type = ValueType::from_spec(make_tensor_spec(dim_size));
    assert(_tensor_type.is_dense());
    assert(_tensor_type.count_indexed_dimensions() == 1u);
    _dim_size = _tensor_type.dimensions()[0].size;
    std::cout << "AnnBm::AnnBM Dimension size is " << _dim_size << std::endl;
    cfg.setTensorType(_tensor_type);
    cfg.set_distance_metric(hnsw_index_params.distance_metric());
    cfg.set_hnsw_index_params(hnsw_index_params);
    _attribute = AttributeFactory::createAttribute("tensor", cfg);
    _tensor_attribute = dynamic_cast<TensorAttribute *>(_attribute.get());
    assert(_tensor_attribute != nullptr);
    _nearest_neighbor_index = _tensor_attribute->nearest_neighbor_index();
    assert(_nearest_neighbor_index != nullptr);
}

AnnBm::~AnnBm() = default;

bool
AnnBm::check_lid(uint32_t lid)
{
    if (lid >= std::numeric_limits<uint32_t>::max() - lid_bias) {
        std::cerr << "lid is too high" << std::endl;
        return false;
    }
    return true;
}

bool
AnnBm::check_value(const char *op, const std::vector<float>& value)
{
    if (value.size() != _dim_size) {
        std::cerr << op << " failed, expected vector with size " << _dim_size << ", got vector with size " << value.size() << std::endl;
        return false;
    }
    return true;
}

uint32_t
AnnBm::num_docs()
{
    return _attribute->getNumDocs();
}

void
AnnBm::set_value(uint32_t lid, const std::vector<float>& value)
{
    if (!check_lid(lid)) {
        return;
    }
    if (!check_value("set_value", value)) {
        return;
    }
    TypedCells typed_cells(&value[0], CellType::FLOAT, value.size());
    DenseValueView tensor_view(_tensor_type, typed_cells);
    while (size_t(lid + lid_bias) >= _attribute->getNumDocs()) {
        uint32_t new_lid = 0;
        _attribute->addDoc(new_lid);
    }
    _tensor_attribute->setTensor(lid + lid_bias, tensor_view); // lid 0 is special in vespa
    _attribute->commit();
}

std::vector<float>
AnnBm::get_value(uint32_t lid)
{
    if (!check_lid(lid)) {
        return {};
    }
    TypedCells typed_cells = _tensor_attribute->extract_cells_ref(lid + lid_bias);
    assert(typed_cells.size == _dim_size);
    const float* data = static_cast<const float* >(typed_cells.data);
    return {data, data + _dim_size};
    return {};
}

void
AnnBm::clear_value(uint32_t lid)
{
    if (!check_lid(lid)) {
        return;
    }
    if (size_t(lid + lid_bias) < _attribute->getNumDocs()) {
        _attribute->clearDoc(lid + lid_bias);
        _attribute->commit();
    }
}

void
AnnBm::set_enable_sort(bool enable_sort)
{
    _enable_sort = enable_sort;
}

TopKResult
AnnBm::find_top_k(uint32_t k, const std::vector<float>& value, uint32_t explore_k, double distance_threshold)
{
    if (!check_value("find_top_k", value)) {
        return {};
    }
    TopKResult result;
    TypedCells typed_cells(&value[0], CellType::FLOAT, value.size());
    auto raw_result = _nearest_neighbor_index->find_top_k(k, typed_cells, explore_k, distance_threshold * distance_threshold);
    result.reserve(raw_result.size());
    switch (_hnsw_index_params.distance_metric()) {
    case DistanceMetric::Euclidean:
        for (auto &raw : raw_result) {
            result.emplace_back(raw.docid - lid_bias, sqrt(raw.distance));
        }
        break;
    default:
        for (auto &raw : raw_result) {
            result.emplace_back(raw.docid - lid_bias, raw.distance);
        }
    }
    if (_enable_sort) {
        std::sort(result.begin(), result.end(), CompareTopKResult());
    }
    return result;
}

}

using vespa_ann_bm::AnnBm;

PYBIND11_MODULE(vespa_ann_bm, m) {
    m.doc() = "vespa_ann_bm plugin";

    py::enum_<DistanceMetric>(m, "DistanceMetric")
        .value("Euclidean", DistanceMetric::Euclidean)
        .value("Angular", DistanceMetric::Angular);

    py::class_<HnswIndexParams>(m, "HnswIndexParams")
        .def(py::init<uint32_t, uint32_t, DistanceMetric, bool>());

    py::class_<AnnBm>(m, "AnnBm")
        .def(py::init<uint32_t, const HnswIndexParams&>())
        .def("num_docs", &AnnBm::num_docs)
        .def("set_value", &AnnBm::set_value)
        .def("get_value", &AnnBm::get_value)
        .def("clear_value", &AnnBm::clear_value)
        .def("find_top_k", &AnnBm::find_top_k)
        .def("set_enable_sort", &AnnBm::set_enable_sort);
}
