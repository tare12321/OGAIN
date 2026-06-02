#pragma once
#include "core.h"
#include "buffer_pool.h"


struct Tensor;
struct DeviceStorage;

enum class TensorDeviceKind
{
    None,
    Cuda,
    Vulkan
};

enum class TensorResidencyCategory
{
    Unspecified,
    InputBatch,
    Label,
    TrainableParameter,
    ForwardActivation,
    LogitsOrLoss,
    Gradient,
    OptimizerState,
    PredictionOrFinalOutput,
    ScratchTemporary
};

struct TensorResidencyState
{
    bool host_valid = true;
    bool device_valid = false;
    TensorDeviceKind device_kind = TensorDeviceKind::None;
    TensorResidencyCategory category = TensorResidencyCategory::Unspecified;
    size_t device_allocation_bytes = 0;
    std::shared_ptr<DeviceStorage> device_storage;
};

struct DeviceStorage
{
    virtual ~DeviceStorage() = default;
    virtual TensorDeviceKind device_kind() const = 0;
    virtual size_t size_bytes() const = 0;
    virtual void* device_ptr() = 0;
    virtual const void* device_ptr() const = 0;
};

namespace tensor_ops
{
    inline size_t dtype_size(DType dt)
    {
        switch (dt)
        {
        case DType::FP16:
            return 2;
        case DType::F32:
            return 4;
        case DType::FP64:
            return 8;
        case DType::I8:
            return 1;
        case DType::I16:
            return 2;
        case DType::I32:
            return 4;
        case DType::I64:
            return 8;
        case DType::U8:
            return 1;
        case DType::U16:
            return 2;
        case DType::U32:
            return 4;
        case DType::U64:
            return 8;
        }
        terminate_with_status(Status::invalid("dtype_size: unsupported dtype"));
    }

    template <typename Dims>
    inline size_t checked_numel_bytes(const Dims& dims, DType dt, const char* context)
    {
        const size_t elemCount = checked_size_t_from_i64(checked_numel_of(dims, context), context);
        const size_t elemBytes = dtype_size(dt);
        if (elemBytes != 0 && elemCount > (std::numeric_limits<size_t>::max() / elemBytes))
        {
            terminate_with_status(Status::invalid(std::string(context) + ": byte-size overflow"));
        }
        return elemCount * elemBytes;
    }

    template <typename Dims>
    inline Strides rowmajor_strides_bytes(const Dims& dims, DType dt)
    {
        const size_t elemBytes = dtype_size(dt);
        Strides strides = rowmajor_strides_elems(dims);
        const int64_t maxStride = std::numeric_limits<int64_t>::max() / static_cast<int64_t>(elemBytes);
        for (size_t i = 0; i < strides.size(); ++i)
        {
            if (strides[i] > maxStride)
            {
                terminate_with_status(Status::invalid("rowmajor_strides_bytes: stride overflow"));
            }
            strides[i] *= static_cast<int64_t>(elemBytes);
        }
        return strides;
    }

    Tensor make_contiguous(BufferPool& pool, const Dims& dims, DType dt);
    Tensor make_view(const Tensor& base, const Dims& dims, const Strides& strides_bytes, size_t offset_bytes);
    Tensor broadcast_to(const Tensor& x, const Shape& outShape);
    Tensor transpose_view(const Tensor& x, const Rank& perm);
    Tensor reshape_view(const Tensor& x, const Shape& newShape);
    Tensor slice_view(const Tensor& x, const Dims& begin, const Dims& end, const Dims& step);
    Tensor ensure_contiguous(const Tensor& x, BufferPool& pool);
    Tensor ensure_contiguous(const Tensor& x, BufferPool& pool, size_t* materialization_count);
    Tensor reduce_sum(BufferPool& pool, const Tensor& x, const Axes& axes);
    Tensor sum_to_shape(BufferPool& pool, const Tensor& g, const Shape& targetShape);

}

using tensor_ops::dtype_size;
using tensor_ops::checked_numel_bytes;
using tensor_ops::make_contiguous;
using tensor_ops::make_view;
using tensor_ops::broadcast_to;
using tensor_ops::transpose_view;
using tensor_ops::reshape_view;
using tensor_ops::slice_view;
using tensor_ops::ensure_contiguous;
using tensor_ops::reduce_sum;
using tensor_ops::sum_to_shape;
using tensor_ops::rowmajor_strides_bytes;

struct Tensor
{
    std::shared_ptr<StorageBlock> _storage;
    std::shared_ptr<TensorResidencyState> _residency;
    size_t _offset_bytes = 0;

    Shape _dims;
    Strides _strides;

    DType _dtype = DType::F32;

    bool valid() const
    {
        return (bool)_storage;
    }

    TensorResidencyState& ensure_residency()
    {
        if (!_residency)
        {
            _residency = std::make_shared<TensorResidencyState>();
        }
        return *_residency;
    }

    const TensorResidencyState* residency() const
    {
        return _residency.get();
    }

    bool host_copy_valid() const
    {
        return _residency ? _residency->host_valid : valid();
    }

    bool device_copy_valid() const
    {
        return _residency ? _residency->device_valid : false;
    }

    bool host_copy_stale() const
    {
        return _residency ? (!_residency->host_valid && _residency->device_valid) : false;
    }

    bool device_copy_stale() const
    {
        return _residency ? (!_residency->device_valid && _residency->host_valid) : false;
    }

    bool has_device_storage() const
    {
        return _residency && static_cast<bool>(_residency->device_storage);
    }

    TensorDeviceKind device_kind() const
    {
        return _residency ? _residency->device_kind : TensorDeviceKind::None;
    }

    TensorResidencyCategory residency_category() const
    {
        return _residency ? _residency->category : TensorResidencyCategory::Unspecified;
    }

    void set_residency_category(TensorResidencyCategory category)
    {
        ensure_residency().category = category;
    }

    void mark_host_modified()
    {
        TensorResidencyState& state = ensure_residency();
        state.host_valid = true;
        state.device_valid = false;
    }

    void mark_host_synchronized()
    {
        TensorResidencyState& state = ensure_residency();
        state.host_valid = true;
    }

    void mark_device_modified()
    {
        TensorResidencyState& state = ensure_residency();
        state.host_valid = false;
        state.device_valid = true;
    }

    void clear_device_copy()
    {
        TensorResidencyState& state = ensure_residency();
        state.device_valid = false;
        state.device_kind = TensorDeviceKind::None;
        state.device_allocation_bytes = 0;
        state.device_storage.reset();
    }

    int64_t numel() const
    {
        return checked_numel_of(_dims, "Tensor::numel");
    }

    bool is_contiguous() const
    {
        if (!valid())
        {
            return false;
        }

        const Strides contiguous = rowmajor_strides_bytes(_dims, _dtype);
        if (contiguous.size() != _strides.size())
        {
            return false;
        }
        for (size_t i = 0; i < _strides.size(); ++i)
        {
            if (_strides[i] != contiguous[i])
            {
                return false;
            }
        }
        return true;
    }

    int64_t stride_bytes(size_t dim) const
    {
        if (dim >= _strides.size())
        {
            terminate_with_status(Status::invalid("stride_bytes: dim out of range"));
        }
        return _strides[dim];
    }

    int64_t stride_elems(size_t dim) const
    {
        const int64_t byteStride = stride_bytes(dim);
        const int64_t elemBytes = static_cast<int64_t>(dtype_size(_dtype));
        if (elemBytes <= 0)
        {
            terminate_with_status(Status::invalid("stride_elems: invalid dtype size"));
        }
        if ((byteStride % elemBytes) != 0)
        {
            terminate_with_status(Status::invalid("stride_elems: byte stride is not element-aligned"));
        }
        return byteStride / elemBytes;
    }

    uint8_t* raw_ptr()
    {
        if (!valid())
        {
            return nullptr;
        }
        mark_host_modified();
        return _storage->data() + _offset_bytes;
    }

    const uint8_t* raw_ptr() const
    {
        if (!valid())
        {
            return nullptr;
        }
        return _storage->data() + _offset_bytes;
    }

    float* f32_ptr()
    {
        return (float*)raw_ptr();
    }

    const float* f32_ptr() const
    {
        return (const float*)raw_ptr();
    }

    int32_t* i32_ptr()
    {
        return (int32_t*)raw_ptr();
    }

    const int32_t* i32_ptr() const
    {
        return (const int32_t*)raw_ptr();
    }
};
