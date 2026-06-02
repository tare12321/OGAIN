#pragma once

#include "graph_types.h"

struct CompiledGraph;
struct OptOptions;

class Graph
{
private:
    std::vector<Node> _nodes;
    std::vector<TensorMeta> _metas;
    std::vector<TensorHandle> _declaredOutputs;
    Status _buildStatus = Status::ok();

public:
    struct AssignResult
    {
        TensorHandle _tensor;
        OpHandle _op;
    };

    TensorHandle placeholder(const Shape& shape, DType dt, const std::string& name);
    TensorHandle constant_f32(const Shape& shape, const std::vector<float>& data, const std::string& name);
    TensorHandle constant_scalar_f32(float v, const std::string& name);
    TensorHandle variable_f32(const Shape& shape, const std::vector<float>& init, const std::string& name);
    TensorHandle stop_gradient(TensorHandle x, const std::string& name);
    TensorHandle add(TensorHandle a, TensorHandle b, const std::string& name);
    TensorHandle add(TensorHandle a, TensorHandle b, const std::string& name, BackendKind backend);
    TensorHandle sub(TensorHandle a, TensorHandle b, const std::string& name);
    TensorHandle mul(TensorHandle a, TensorHandle b, const std::string& name);
    TensorHandle div(TensorHandle a, TensorHandle b, const std::string& name);
    TensorHandle pow(TensorHandle a, TensorHandle b, const std::string& name);
    TensorHandle neg(TensorHandle x, const std::string& name);
    TensorHandle relu(TensorHandle x, const std::string& name);
    TensorHandle relu(TensorHandle x, const std::string& name, BackendKind backend);
    TensorHandle sigmoid(TensorHandle x, const std::string& name);
    TensorHandle tanh(TensorHandle x, const std::string& name);
    TensorHandle exp(TensorHandle x, const std::string& name);
    TensorHandle log(TensorHandle x, const std::string& name);
    TensorHandle sqrt(TensorHandle x, const std::string& name);
    TensorHandle abs(TensorHandle x, const std::string& name);
    TensorHandle softmax(TensorHandle x, int64_t axis, const std::string& name);
    TensorHandle broadcast_to(TensorHandle x, const Shape& outShape, const std::string& name);
    TensorHandle transpose(TensorHandle x, const Shape& perm, const std::string& name);
    TensorHandle reshape(TensorHandle x, const Shape& newShape, const std::string& name);
    TensorHandle slice(TensorHandle x, const Shape& begin, const Shape& end, const Shape& step, const std::string& name);
    TensorHandle reduce_sum(TensorHandle x, const Shape& axes, const std::string& name);
    TensorHandle reduce_mean(TensorHandle x, const Shape& axes, const std::string& name);
    TensorHandle reduce_max(TensorHandle x, const Shape& axes, const std::string& name);
    TensorHandle matmul2d(TensorHandle a, TensorHandle b, const std::string& name);
    TensorHandle matmul2d(TensorHandle a, TensorHandle b, const std::string& name, BackendKind backend);
    TensorHandle mse_loss(TensorHandle pred, TensorHandle target, const std::string& name);
    TensorHandle cross_entropy(TensorHandle pred, TensorHandle target, const std::string& name);
    TensorHandle cross_entropy(TensorHandle pred, TensorHandle target, const std::string& name, BackendKind backend);
    TensorHandle concat(const std::vector<TensorHandle>& xs, int64_t axis, const std::string& name);
    TensorHandle gather(TensorHandle data, TensorHandle indices, int64_t axis, const std::string& name);
    TensorHandle cast(TensorHandle x, DType dtype, const std::string& name);
    AssignResult assign(TensorHandle var, TensorHandle value, const std::string& name);
    void output(TensorHandle value);
    void outputs(const std::vector<TensorHandle>& values);
    const std::vector<TensorHandle>& declared_outputs() const;

    CompiledGraph compile() const;
    CompiledGraph compile(const OptOptions& opt) const;

private:
    TensorId new_tensor_(const Shape& shape, DType dt, bool ph, bool cst, bool var, const std::string& name);
    TensorHandle binop_(OpType op, TensorHandle a, TensorHandle b, const std::string& name, BackendKind backend, bool backendExplicit);
    TensorHandle unop_(OpType op, TensorHandle x, const std::string& name, BackendKind backend, bool backendExplicit);
    TensorHandle matmul2d_(TensorHandle a, TensorHandle b, const std::string& name, BackendKind backend, bool backendExplicit);
    TensorHandle cross_entropy_(TensorHandle pred, TensorHandle target, const std::string& name, BackendKind backend, bool backendExplicit);
    bool has_build_error_() const;
    bool is_valid_tensor_(TensorHandle h) const;
    void set_build_error_(Status s);
    TensorHandle invalid_tensor_() const;
    OpHandle invalid_op_() const;
    AssignResult invalid_assign_() const;
};
