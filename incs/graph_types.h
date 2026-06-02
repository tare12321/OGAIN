#pragma once
#include <variant>
#include "core.h"
#include "tensor.h"
#include "thread_pool.h"

namespace cuda_resident
{
    class Runtime;
}

struct TensorHandle
{
    TensorId id = 0;

    bool operator==(const TensorHandle& o) const
    {
        return id == o.id;
    }
};

struct TensorHandleHash
{
    size_t operator()(const TensorHandle& h) const noexcept
    {
        return std::hash<uint32_t>{}(h.id);
    }
};

struct OpHandle
{
    NodeId id = 0;

    bool operator==(const OpHandle& o) const
    {
        return id == o.id;
    }
};

struct TensorMeta
{
    Dims dims;
    DType dtype = DType::F32;

    bool is_placeholder = false;
    bool is_constant = false;
    bool is_variable = false;

    std::string name;
};

enum class BackendKind
{
    CpuRef,
    CpuFast,
    Cpu,
    MpsGraph,
    Cuda,
    CudaResident,
    Rocm,
    Vulkan,
    VulkanResidentTraining,
    Mock
};

enum class OpType
{
    Placeholder,
    Constant,
    Variable,

    Assign,
    StopGrad,

    Add,
    Sub,
    Mul,
    Div,
    Pow,

    Neg,
    ReLU,
    Sigmoid,
    Tanh,
    Exp,
    Log,
    Sqrt,
    Abs,
    Softmax,

    MatMul2D,
    BroadcastTo,
    Transpose,
    Reshape,
    Slice,
    Gather,
    Concat,
    Cast,

    ReduceSum,
    ReduceMean,
    ReduceMax,

    MSELoss,
    CrossEntropy,

    NoOp
};

struct Node
{
    OpType op = OpType::Add;
    SmallVec<TensorId, 4> inputs;
    TensorId output = 0;
    std::string name;

    BackendKind backend = BackendKind::CpuRef;
    bool backend_explicit = false;

    struct AttrNone
    {
    };
    struct AttrConstant
    {
        std::vector<uint8_t> init_bytes;
    };
    struct AttrBroadcastTo
    {
        Shape bcast_shape;
    };
    struct AttrTranspose
    {
        Rank perm;
    };
    struct AttrReshape
    {
        Shape reshape_shape;
    };
    struct AttrSlice
    {
        Dims slice_begin;
        Dims slice_end;
        Dims slice_step;
    };
    struct AttrReduceSum
    {
        Axes reduce_axes;
    };
    struct AttrAxis
    {
        int64_t axis = 0;
    };
    struct AttrCast
    {
        DType dtype = DType::F32;
    };
    struct AttrAssign
    {
        TensorId assign_var = (TensorId)-1;
    };

    using AttrData = std::variant<
        AttrNone,
        AttrConstant,
        AttrBroadcastTo,
        AttrTranspose,
        AttrReshape,
        AttrSlice,
        AttrReduceSum,
        AttrAxis,
        AttrCast,
        AttrAssign>;
    AttrData attrs = AttrNone{};

    TensorId fused_bias = (TensorId)-1;
    OpType fused_activation = OpType::NoOp;

    template <typename T>
    T* attrs_if()
    {
        return std::get_if<T>(&attrs);
    }

    template <typename T>
    const T* attrs_if() const
    {
        return std::get_if<T>(&attrs);
    }

    template <typename T>
    T& ensure_attrs()
    {
        if (auto* cur = std::get_if<T>(&attrs))
        {
            return *cur;
        }
        attrs = T{};
        return std::get<T>(attrs);
    }

    void clear_attrs()
    {
        attrs = AttrNone{};
    }

    std::vector<uint8_t>& init_bytes_mut()
    {
        return ensure_attrs<AttrConstant>().init_bytes;
    }

    const std::vector<uint8_t>& init_bytes() const
    {
        static const std::vector<uint8_t> kEmpty;
        if (auto* a = attrs_if<AttrConstant>())
        {
            return a->init_bytes;
        }
        return kEmpty;
    }

    Shape& bcast_shape_mut()
    {
        return ensure_attrs<AttrBroadcastTo>().bcast_shape;
    }

    const Shape& bcast_shape() const
    {
        static const Shape kEmpty;
        if (auto* a = attrs_if<AttrBroadcastTo>())
        {
            return a->bcast_shape;
        }
        return kEmpty;
    }

    Rank& perm_mut()
    {
        return ensure_attrs<AttrTranspose>().perm;
    }

    const Rank& perm() const
    {
        static const Rank kEmpty;
        if (auto* a = attrs_if<AttrTranspose>())
        {
            return a->perm;
        }
        return kEmpty;
    }

    Shape& reshape_shape_mut()
    {
        return ensure_attrs<AttrReshape>().reshape_shape;
    }

    const Shape& reshape_shape() const
    {
        static const Shape kEmpty;
        if (auto* a = attrs_if<AttrReshape>())
        {
            return a->reshape_shape;
        }
        return kEmpty;
    }

    Dims& slice_begin_mut()
    {
        return ensure_attrs<AttrSlice>().slice_begin;
    }

    const Dims& slice_begin() const
    {
        static const Dims kEmpty;
        if (auto* a = attrs_if<AttrSlice>())
        {
            return a->slice_begin;
        }
        return kEmpty;
    }

    Dims& slice_end_mut()
    {
        return ensure_attrs<AttrSlice>().slice_end;
    }

    const Dims& slice_end() const
    {
        static const Dims kEmpty;
        if (auto* a = attrs_if<AttrSlice>())
        {
            return a->slice_end;
        }
        return kEmpty;
    }

    Dims& slice_step_mut()
    {
        return ensure_attrs<AttrSlice>().slice_step;
    }

    const Dims& slice_step() const
    {
        static const Dims kEmpty;
        if (auto* a = attrs_if<AttrSlice>())
        {
            return a->slice_step;
        }
        return kEmpty;
    }

    Axes& reduce_axes_mut()
    {
        return ensure_attrs<AttrReduceSum>().reduce_axes;
    }

    const Axes& reduce_axes() const
    {
        static const Axes kEmpty;
        if (auto* a = attrs_if<AttrReduceSum>())
        {
            return a->reduce_axes;
        }
        return kEmpty;
    }

    int64_t& axis_mut()
    {
        return ensure_attrs<AttrAxis>().axis;
    }

    int64_t axis() const
    {
        if (auto* a = attrs_if<AttrAxis>())
        {
            return a->axis;
        }
        return 0;
    }

    DType& cast_dtype_mut()
    {
        return ensure_attrs<AttrCast>().dtype;
    }

    DType cast_dtype() const
    {
        if (auto* a = attrs_if<AttrCast>())
        {
            return a->dtype;
        }
        return DType::F32;
    }

    TensorId assign_var() const
    {
        if (auto* a = attrs_if<AttrAssign>())
        {
            return a->assign_var;
        }
        return (TensorId)-1;
    }

    void set_assign_var(TensorId id)
    {
        ensure_attrs<AttrAssign>().assign_var = id;
    }
};


struct ExecContext
{
    BufferPool* _pool = nullptr;
    BufferPool* _persistPool = nullptr;
    ThreadPool* _tp = nullptr;
    std::vector<Tensor>* _varStore = nullptr;
    cuda_resident::Runtime* _cudaRuntime = nullptr;
    bool _cudaResidentMode = false;
    bool _vulkanResidentTrainingMode = false;
    size_t* _materializationCount = nullptr;
};
