#pragma once
#include "graph.h"
#include "optimizer.h"
#include "backends.h"

namespace cuda_resident
{
    class Runtime;
}

namespace vulkan_backend
{
    class ResidentRuntime;
}

enum class RuntimeExecutionPolicy
{
    Default,
    CudaResident,
    VulkanResidentTraining
};

class RuntimeSession
{
public:
    using FeedDictF32 = std::unordered_map<TensorHandle, std::vector<float>, TensorHandleHash>;
    using FetchDict = std::unordered_map<TensorHandle, Tensor, TensorHandleHash>;


private:
    struct PlanCacheKey
    {
        SmallVec<TensorId, 4> _fetchIds;
        SmallVec<NodeId, 4> _targetIds;

        bool operator==(const PlanCacheKey& o) const
        {
            return _fetchIds == o._fetchIds && _targetIds == o._targetIds;
        }
    };

    struct PlanCacheKeyHash
    {
        size_t operator()(const PlanCacheKey& k) const;
    };

    struct PlanCacheEntry
    {
        ReachabilityPlan _plan;
        std::vector<int> _lastUse;
        bool _hasLastUse = false;
    };

private:
    CompiledGraph _cg;
    mutable std::mutex _sessionMtx;

    BufferPool _pool;
    BufferPool _workspace;
    std::unique_ptr<ThreadPool> _tp;

    std::vector<Tensor> _tensors;
    std::vector<Tensor> _constStore;
    std::vector<Tensor> _varStore;
    std::vector<TensorId> _activeTensorIds;
    std::vector<uint8_t> _activeTensorMask;
    std::unordered_map<PlanCacheKey, PlanCacheEntry, PlanCacheKeyHash> _planCache;

    std::unique_ptr<IBackend> _ref;
    std::unique_ptr<IBackend> _fast;
    BackendRuntimeMap _runtimeBackends;
    BackendRegistry _backendRegistry;
    std::unique_ptr<cuda_resident::Runtime> _cudaRuntime;
    size_t _materializationCount = 0;
    bool _backendDiagnosticsEnabled = false;
    RuntimeExecutionPolicy _executionPolicy = RuntimeExecutionPolicy::Default;

public:
    explicit RuntimeSession(const CompiledGraph& cg);
    ~RuntimeSession();

    void set_variable_f32(TensorHandle var, const std::vector<float>& data);
    void get_variable_f32(TensorHandle var, std::vector<float>& out);

    const CompiledGraph& compiled_graph() const;
    const BackendRegistry& backend_registry() const;
    void register_backend(BackendCapabilityDescriptor descriptor, std::unique_ptr<IBackend> backend);
    void set_backend_diagnostics_enabled(bool enabled);
    bool backend_diagnostics_enabled() const;
    void set_execution_policy(RuntimeExecutionPolicy policy);
    RuntimeExecutionPolicy execution_policy() const;

    cuda_resident::Runtime* cuda_runtime();
    const cuda_resident::Runtime* cuda_runtime() const;
    vulkan_backend::ResidentRuntime* vulkan_resident_runtime();
    const vulkan_backend::ResidentRuntime* vulkan_resident_runtime() const;
    Tensor upload_tensor(const Tensor& value);
    Tensor download_tensor(const Tensor& value);
    void synchronize_cuda();


    void run(const FeedDictF32& feeds,  const std::vector<TensorHandle>& fetchTensors, const std::vector<OpHandle>& targetOps, FetchDict& fetched);

    void run(const FeedDictF32& feeds,FetchDict& fetched);

    void run_with_trace(const FeedDictF32& feeds,
        const std::vector<TensorHandle>& fetchTensors,
        const std::vector<OpHandle>& targetOps,
        FetchDict& fetched,
        std::vector<NodeId>& executedNodes);

    void run_with_trace(const FeedDictF32& feeds, FetchDict& fetched,std::vector<NodeId>& executedNodes);

    const std::vector<Tensor>& runtime_tensors() const;
    Tensor get_variable_tensor(TensorHandle var);
    void set_variable_tensor(TensorHandle var, const Tensor& value);

    BufferPool& pool();
    const BufferPool& pool() const;
    DebugStats debug_stats() const;
    void reset_debug_stats();

private:
    void run_internal_(const FeedDictF32& feeds,
        const std::vector<TensorHandle>& fetchTensors,
        const std::vector<OpHandle>& targetOps,
        FetchDict& fetched,
        std::vector<NodeId>* executedNodes,
        bool keepIntermediates);

    Tensor make_host_readable_tensor_(const Tensor& value);
        
    void check_tensor_handle_(TensorHandle h) const;

    TensorId resolve_tensor_alias_(TensorId tid) const;

    TensorId variable_id_unlocked_(TensorHandle var) const;

    void begin_run_();

    void stage_tensor_(TensorId tid, const Tensor& t);

    PlanCacheKey make_plan_cache_key_(const std::vector<TensorHandle>& fetchTensors, const std::vector<OpHandle>& targetOps) const;

    PlanCacheEntry* get_or_build_plan_(const std::vector<TensorHandle>& fetchTensors, const std::vector<OpHandle>& targetOps);
};
