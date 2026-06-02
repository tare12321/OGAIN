#pragma once
#include "graph_types.h"

enum class BackendDeviceType
{
    Cpu,
    IntegratedGpu,
    DiscreteGpu,
    PortableGpu,
    Unknown
};

enum class BackendLayoutRequirement
{
    AnyStrides,
    ContiguousOnly
};

enum class BackendExecutionMode
{
    Sync,
    AsyncPlaceholder
};

struct BackendCooperativeMatrixConfiguration
{
    uint32_t m = 0;
    uint32_t n = 0;
    uint32_t k = 0;
    std::string scope;
    std::string a_type;
    std::string b_type;
    std::string accumulator_type;
    std::string result_type;
    bool saturating_accumulation = false;
};

struct BackendCooperativeMatrixCapability
{
    bool supported = false;
    bool khr_extension_present = false;
    bool nv_extension_present = false;
    bool nv2_extension_present = false;
    bool feature_cooperative_matrix = false;
    bool feature_robust_buffer_access = false;
    bool feature_shader_float16 = false;
    std::vector<std::string> supported_stages;
    std::vector<std::string> subgroup_properties;
    std::vector<std::string> supported_scopes;
    std::vector<std::string> supported_input_types;
    std::vector<std::string> supported_output_types;
    std::vector<std::string> supported_accumulator_types;
    std::vector<BackendCooperativeMatrixConfiguration> configurations;
    std::vector<std::string> unavailable_reasons;
};

struct BackendCapabilityDescriptor
{
    struct OperationCapability
    {
        OpType op = OpType::NoOp;
        std::vector<DType> supported_dtypes;
        BackendLayoutRequirement layout_requirement = BackendLayoutRequirement::AnyStrides;
    };

    BackendKind kind = BackendKind::CpuRef;
    std::string name;
    BackendDeviceType device_type = BackendDeviceType::Unknown;
    std::vector<DType> supported_dtypes;
    std::vector<OpType> supported_ops;
    std::vector<OperationCapability> operation_capabilities;
    BackendLayoutRequirement layout_requirement = BackendLayoutRequirement::AnyStrides;
    BackendExecutionMode execution_mode = BackendExecutionMode::Sync;
    bool supports_sync = true;
    bool supports_async = false;
    bool available = false;
    std::string unavailable_reason;
    bool supports_cooperative_matrix = false;
    std::string device_name;
    std::string api_version;
    std::string driver_version;
    BackendCooperativeMatrixCapability cooperative_matrix;
    int priority = 1000;
};

struct BackendSelectionRequest
{
    OpType op = OpType::NoOp;
    DType dtype = DType::F32;
    bool inputs_are_contiguous = true;
    bool has_explicit_backend = false;
    BackendKind explicit_backend = BackendKind::CpuRef;
    bool allow_fallback = true;
};

struct BackendSelectionResult
{
    bool ok = false;
    BackendKind selected_kind = BackendKind::CpuRef;
    std::string selected_name;
    bool used_fallback = false;
    Status status = Status::internal("backend selection did not run");
    std::vector<std::string> diagnostics;
};

const char* backend_kind_name(BackendKind kind);
const char* backend_device_type_name(BackendDeviceType type);
const char* backend_layout_requirement_name(BackendLayoutRequirement requirement);
const char* backend_execution_mode_name(BackendExecutionMode mode);
const char* op_type_name(OpType op);
const char* dtype_name(DType dtype);

class BackendRegistry
{
public:
    void register_backend(BackendCapabilityDescriptor descriptor);
    const std::vector<BackendCapabilityDescriptor>& descriptors() const;
    std::vector<BackendCapabilityDescriptor> descriptors_for_kind(BackendKind kind) const;
    std::vector<std::string> unavailable_reasons() const;
    BackendSelectionResult select_backend(const BackendSelectionRequest& request) const;

private:
    std::vector<BackendCapabilityDescriptor> _descriptors;
};

BackendRegistry make_default_backend_registry();

class IBackend
{
public:
    virtual ~IBackend() = default;

    virtual Tensor exec_node(const Node& n, const std::vector<TensorMeta>& metas, const std::vector<Tensor>& tensors, ExecContext& ctx) = 0;
};

using BackendRuntimeMap = std::unordered_map<BackendKind, std::unique_ptr<IBackend>>;
namespace act_fn
{
    float relu(float x);
    float sigmoid(float x);
    float tanh(float x);
}

namespace cpu_ref
{

    Tensor elementwise_binary(BufferPool& pool, const Tensor& A, const Tensor& B, OpType op);
    Tensor elementwise_unary(BufferPool& pool, const Tensor& X, OpType op);
    Tensor matmul2d(BufferPool& pool, const Tensor& A, const Tensor& B, size_t* materialization_count = nullptr);
    Tensor reduce_sum(BufferPool& pool, const Tensor& X, const Axes& axes);
    Tensor reduce_mean(BufferPool& pool, const Tensor& X, const Axes& axes);
    Tensor reduce_max(BufferPool& pool, const Tensor& X, const Axes& axes);
    Tensor softmax(BufferPool& pool, const Tensor& X, int64_t axis, size_t* materialization_count = nullptr);
    Tensor cross_entropy(BufferPool& pool, const Tensor& pred, const Tensor& target, size_t* materialization_count = nullptr);
    Tensor concat(BufferPool& pool, const std::vector<const Tensor*>& xs, int64_t axis, size_t* materialization_count = nullptr);
    Tensor gather(BufferPool& pool, const Tensor& data, const Tensor& indices, int64_t axis, size_t* materialization_count = nullptr);
    Tensor cast(BufferPool& pool, const Tensor& x, DType dtype, size_t* materialization_count = nullptr);
    Tensor mse_loss(BufferPool& pool, const Tensor& pred, const Tensor& target, size_t* materialization_count = nullptr);
}

namespace cpu_fast
{
    Tensor matmul2d(BufferPool& pool, ThreadPool& tp, const Tensor& A, const Tensor& B, size_t* materialization_count = nullptr);
}

class CpuRefBackend : public IBackend
{
public:
    Tensor exec_node(const Node& n, const std::vector<TensorMeta>& metas, const std::vector<Tensor>& tensors, ExecContext& ctx) override
    {
        if (ctx._pool == nullptr)
        {
            terminate_with_status(Status::internal("backend: missing workspace pool"));
        }
        BufferPool& pool = *ctx._pool;

        auto get = [&](TensorId id) -> const Tensor*
            {
                if ((size_t)id >= tensors.size())
                {
                    terminate_with_status(Status::invalid("backend: input id out of range"));
                }
                if (!tensors[id].valid())
                {
                    terminate_with_status(Status::invalid("backend: missing input tensor"));
                }
                return &tensors[id];
            };

        if (n.op == OpType::NoOp || n.op == OpType::StopGrad)
        {
            if (n.inputs.empty())
            {
                terminate_with_status(Status::invalid("backend: pass-through op missing input"));
            }
            const Tensor* X = get(n.inputs[0]);
            return *X;
        }

        if (n.op == OpType::Add || n.op == OpType::Sub || n.op == OpType::Mul || n.op == OpType::Div || n.op == OpType::Pow)
        {
            const Tensor* A = get(n.inputs[0]);
            const Tensor* B = get(n.inputs[1]);
            return cpu_ref::elementwise_binary(pool, *A, *B, n.op);
        }

        if (n.op == OpType::Neg ||
            n.op == OpType::ReLU ||
            n.op == OpType::Sigmoid ||
            n.op == OpType::Tanh ||
            n.op == OpType::Exp ||
            n.op == OpType::Log ||
            n.op == OpType::Sqrt ||
            n.op == OpType::Abs)
        {
            const Tensor* X = get(n.inputs[0]);
            return cpu_ref::elementwise_unary(pool, *X, n.op);
        }

        if (n.op == OpType::MatMul2D)
        {
            const Tensor* A = get(n.inputs[0]);
            const Tensor* B = get(n.inputs[1]);
            Tensor out = cpu_ref::matmul2d(pool, *A, *B, ctx._materializationCount);

            if (n.fused_bias != (TensorId)-1)
            {
                const Tensor* Bias = get(n.fused_bias);
                out = cpu_ref::elementwise_binary(pool, out, *Bias, OpType::Add);
            }

            if (n.fused_activation == OpType::ReLU ||
                n.fused_activation == OpType::Sigmoid ||
                n.fused_activation == OpType::Tanh)
            {
                out = cpu_ref::elementwise_unary(pool, out, n.fused_activation);
            }

            return out;
        }

        if (n.op == OpType::BroadcastTo)
        {
            const Tensor* X = get(n.inputs[0]);
            return broadcast_to(*X, n.bcast_shape());
        }

        if (n.op == OpType::Transpose)
        {
            const Tensor* X = get(n.inputs[0]);
            return transpose_view(*X, n.perm());
        }

        if (n.op == OpType::Reshape)
        {
            const Tensor* X = get(n.inputs[0]);
            return reshape_view(*X, n.reshape_shape());
        }

        if (n.op == OpType::Slice)
        {
            const Tensor* X = get(n.inputs[0]);
            return slice_view(*X, n.slice_begin(), n.slice_end(), n.slice_step());
        }

        if (n.op == OpType::ReduceSum)
        {
            const Tensor* X = get(n.inputs[0]);
            return cpu_ref::reduce_sum(pool, *X, n.reduce_axes());
        }

        if (n.op == OpType::ReduceMean)
        {
            const Tensor* X = get(n.inputs[0]);
            return cpu_ref::reduce_mean(pool, *X, n.reduce_axes());
        }

        if (n.op == OpType::ReduceMax)
        {
            const Tensor* X = get(n.inputs[0]);
            return cpu_ref::reduce_max(pool, *X, n.reduce_axes());
        }

        if (n.op == OpType::Softmax)
        {
            const Tensor* X = get(n.inputs[0]);
            return cpu_ref::softmax(pool, *X, n.axis(), ctx._materializationCount);
        }

        if (n.op == OpType::MSELoss)
        {
            const Tensor* P = get(n.inputs[0]);
            const Tensor* T = get(n.inputs[1]);
            return cpu_ref::mse_loss(pool, *P, *T, ctx._materializationCount);
        }

        if (n.op == OpType::CrossEntropy)
        {
            const Tensor* P = get(n.inputs[0]);
            const Tensor* T = get(n.inputs[1]);
            return cpu_ref::cross_entropy(pool, *P, *T, ctx._materializationCount);
        }

        if (n.op == OpType::Concat)
        {
            std::vector<const Tensor*> xs;
            xs.reserve(n.inputs.size());
            for (TensorId tid : n.inputs)
            {
                xs.push_back(get(tid));
            }
            return cpu_ref::concat(pool, xs, n.axis(), ctx._materializationCount);
        }

        if (n.op == OpType::Gather)
        {
            if (n.inputs.size() != 2)
            {
                terminate_with_status(Status::invalid("Gather: arity mismatch"));
            }
            const Tensor* data = get(n.inputs[0]);
            const Tensor* indices = get(n.inputs[1]);
            return cpu_ref::gather(pool, *data, *indices, n.axis(), ctx._materializationCount);
        }

        if (n.op == OpType::Cast)
        {
            const Tensor* X = get(n.inputs[0]);
            return cpu_ref::cast(pool, *X, n.cast_dtype(), ctx._materializationCount);
        }

        if (n.op == OpType::Assign)
        {
            if (ctx._varStore == nullptr)
            {
                terminate_with_status(Status::internal("Assign: missing var store"));
            }
            if (ctx._persistPool == nullptr)
            {
                terminate_with_status(Status::internal("Assign: missing persistent pool"));
            }

            const TensorId varId = n.assign_var();
            if ((size_t)varId >= metas.size())
            {
                terminate_with_status(Status::invalid("Assign: varId out of range"));
            }
            if ((size_t)varId >= ctx._varStore->size())
            {
                terminate_with_status(Status::invalid("Assign: var store index out of range"));
            }
            if (!metas[varId].is_variable)
            {
                terminate_with_status(Status::invalid("Assign: assign_var is not variable"));
            }

            const Tensor* value = get(n.inputs[1]);

            Tensor vb = broadcast_to(*value, metas[varId].dims);
            Tensor vc = ensure_contiguous(vb, pool, ctx._materializationCount);

            Tensor dst = make_contiguous( *ctx._persistPool, metas[varId].dims, metas[varId].dtype);
            const size_t bytes = checked_numel_bytes(metas[varId].dims, metas[varId].dtype, "Assign");
            std::memcpy(dst.raw_ptr(), vc.raw_ptr(), bytes);

            (*ctx._varStore)[varId] = dst;
            return dst;
        }

        terminate_with_status(Status::internal("CpuRefBackend: unsupported op"));
    }
};

class CpuFastBackend : public IBackend
{
public:
    Tensor exec_node(const Node& n, const std::vector<TensorMeta>& metas, const std::vector<Tensor>& tensors, ExecContext& ctx) override
    {
        if (ctx._pool == nullptr)
        {
            terminate_with_status(Status::internal("backend: missing workspace pool"));
        }
        BufferPool& pool = *ctx._pool;

        auto get = [&](TensorId id) -> const Tensor*
            {
                if ((size_t)id >= tensors.size())
                {
                    terminate_with_status(Status::invalid("backend: input id out of range"));
                }
                if (!tensors[id].valid())
                {
                    terminate_with_status(Status::invalid("backend: missing input tensor"));
                }
                return &tensors[id];
            };

        if (n.op == OpType::MatMul2D)
        {
            if (ctx._tp == nullptr)
            {
                terminate_with_status(Status::internal("CpuFastBackend: missing thread pool"));
            }
            const Tensor* A = get(n.inputs[0]);
            const Tensor* B = get(n.inputs[1]);
            Tensor out = cpu_fast::matmul2d(pool, *ctx._tp, *A, *B, ctx._materializationCount);

            if (n.fused_bias != (TensorId)-1)
            {
                const Tensor* Bias = get(n.fused_bias);
                out = cpu_ref::elementwise_binary(pool, out, *Bias, OpType::Add);
            }

            if (n.fused_activation == OpType::ReLU ||
                n.fused_activation == OpType::Sigmoid ||
                n.fused_activation == OpType::Tanh)
            {
                out = cpu_ref::elementwise_unary(pool, out, n.fused_activation);
            }

            return out;
        }

        CpuRefBackend ref;
        return ref.exec_node(n, metas, tensors, ctx);
    }
};
