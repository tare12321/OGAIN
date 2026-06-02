
#pragma once



#include <algorithm>
#include <atomic>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>
#include <condition_variable>
#include <memory>

#include "small_vec.h"
#include "status.h"


using TensorId = uint32_t;
using NodeId = uint32_t;

using Dim = int64_t;
using RankIndex = size_t;
using Shape = SmallVec<Dim, 4>;
using Dims = Shape;
using Rank = Shape;
using Strides = Shape;
using Axes = Shape;

enum class DType
{
    FP16, F32, FP64,
    I8, I16, I32, I64,
    U8, U16, U32, U64
};


// ============================================================
// Utilities
// ============================================================

namespace shape_utils
{
    template <typename Dims>
    inline Status validate_dims(const Dims& dims, const char* context)
    {
        for (size_t i = 0; i < dims.size(); ++i)
        {
            if (dims[i] < 0)
            {
                return Status::invalid(std::string(context) + ": negative dimension at index " + std::to_string(i));
            }
        }
        return Status::ok();
    }

    template <typename Dims>
    inline int64_t numel_of(const Dims& dims)
    {
        if (dims.empty())
        {
            return 1;
        }
        int64_t n = 1;
        for (int64_t d : dims)
        {
            n *= d;
        }
        return n;
    }

    template <typename Dims>
    inline int64_t checked_numel_of(const Dims& dims, const char* context)
    {
        const Status status = validate_dims(dims, context);
        if (!status.ok_())
        {
            terminate_with_status(status);
        }

        if (dims.empty())
        {
            return 1;
        }

        int64_t n = 1;
        for (int64_t d : dims)
        {
            if (d == 0)
            {
                n = 0;
                continue;
            }
            if (n != 0 && d > (std::numeric_limits<int64_t>::max() / n))
            {
                terminate_with_status(Status::invalid(std::string(context) + ": numel overflow"));
            }
            n *= d;
        }
        return n;
    }

    inline size_t checked_size_t_from_i64(int64_t value, const char* context)
    {
        if (value < 0)
        {
            terminate_with_status(Status::invalid(std::string(context) + ": negative size"));
        }
        return static_cast<size_t>(value);
    }

    template <typename Dims>
    inline Strides rowmajor_strides_elems(const Dims& dims)
    {
        const Status status = validate_dims(dims, "rowmajor_strides_elems");
        if (!status.ok_())
        {
            terminate_with_status(status);
        }

        Strides s(dims.size(), 0);
        int64_t stride = 1;
        for (int i = (int)dims.size() - 1; i >= 0; --i)
        {
            s[(size_t)i] = stride;
            if (dims[(size_t)i] != 0 && stride != 0 &&
                dims[(size_t)i] > (std::numeric_limits<int64_t>::max() / stride))
            {
                terminate_with_status(Status::invalid("rowmajor_strides_elems: stride overflow"));
            }
            stride *= dims[(size_t)i];
        }
        return s;
    }

    template <typename Dims>
    inline std::string shape_to_string(const Dims& d)
    {
        std::string r = "{";
        for (size_t i = 0; i < d.size(); ++i)
        {
            r += std::to_string(d[i]);
            if (i + 1 < d.size())
            {
                r += ",";
            }
        }
        r += "}";
        return r;
    }

    template <typename ShapeA, typename ShapeB, typename ShapeOut = Shape>
    inline ShapeOut broadcast_shape(const ShapeA& a, const ShapeB& b)
    {
        const Status aStatus = validate_dims(a, "broadcast_shape(lhs)");
        if (!aStatus.ok_())
        {
            terminate_with_status(aStatus);
        }
        const Status bStatus = validate_dims(b, "broadcast_shape(rhs)");
        if (!bStatus.ok_())
        {
            terminate_with_status(bStatus);
        }

        const size_t na = a.size();
        const size_t nb = b.size();
        const size_t nd = std::max(na, nb);

        ShapeOut out;
        out.assign(nd, 1);

        for (size_t i = 0; i < nd; ++i)
        {
            const int ia = (int)(na - 1 - i);
            const int ib = (int)(nb - 1 - i);

            const int64_t da = (ia >= 0) ? a[(size_t)ia] : 1;
            const int64_t db = (ib >= 0) ? b[(size_t)ib] : 1;

            if (da == db)
            {
                out[nd - 1 - i] = da;
            }
            else if (da == 1)
            {
                out[nd - 1 - i] = db;
            }
            else if (db == 1)
            {
                out[nd - 1 - i] = da;
            }
            else
            {
                terminate_with_status(Status::invalid("broadcast incompatible: " + shape_to_string(a) + " vs " + shape_to_string(b)));
            }
        }

        return out;
    }
}

namespace hash_utils
{
    inline uint64_t fnv1a64(const void* data, size_t n)
    {
        const uint8_t* p = (const uint8_t*)data;
        uint64_t h = 1469598103934665603ull;
        for (size_t i = 0; i < n; ++i)
        {
            h ^= (uint64_t)p[i];
            h *= 1099511628211ull;
        }
        return h;
    }

    template <typename VecI64>
    inline uint64_t hash_i64_vec(const VecI64& v)
    {
        if (v.empty())
        {
            return 0x9e3779b97f4a7c15ull;
        }
        return fnv1a64(v.data(), v.size() * sizeof(int64_t));
    }
}

using shape_utils::numel_of;
using shape_utils::checked_numel_of;
using shape_utils::checked_size_t_from_i64;
using shape_utils::rowmajor_strides_elems;
using shape_utils::shape_to_string;
using shape_utils::broadcast_shape;
using shape_utils::validate_dims;
using hash_utils::fnv1a64;
using hash_utils::hash_i64_vec;

