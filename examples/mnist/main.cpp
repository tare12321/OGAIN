#include "mnist_common.hpp"

#include <computegraph/config.hpp>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>

#ifndef COMPUTEGRAPH_DEFAULT_MNIST_DATA_DIR
#define COMPUTEGRAPH_DEFAULT_MNIST_DATA_DIR ""
#endif

namespace
{
    enum class Mode
    {
        Infer,
        Train,
    };

    struct Options
    {
        Mode mode = Mode::Infer;
        BackendKind backend = BackendKind::Cpu;
        bool compare_cpu_cuda = false;
        bool verbose_backend = false;
        bool verify_no_intermediate_transfer = false;
        bool samples_explicit = false;
        std::optional<std::filesystem::path> data_dir;
        std::optional<std::filesystem::path> load_weights_path;
        std::optional<std::filesystem::path> save_weights_path;
        size_t samples = 1000;
        size_t epochs = 1;
        size_t batch_size = 64;
        float learning_rate = 0.01f;
        int64_t hidden = 128;
        uint32_t seed = 123;
    };

    void print_usage(const char* argv0)
    {
        std::cout
            << "Usage: " << argv0 << " [--mode=infer|train] [--backend=cpu|cuda|cuda-resident|vulkan-resident] [--compare-cpu-cuda]\n"
            << "       [--data-dir=<MNIST_DIR>] [--samples=<N>] [--epochs=<N>] [--batch-size=<N>]\n"
            << "       [--learning-rate=<value>] [--hidden=<N>] [--seed=<N>]\n"
            << "       [--weights=<path>|--load-weights=<path>] [--save-weights=<path>]\n"
            << "       [--verbose-backend] [--verify-no-intermediate-transfer]\n\n"
            << "MNIST data-dir expects IDX files:\n"
            << "  train-images-idx3-ubyte and train-labels-idx1-ubyte, or\n"
            << "  t10k-images-idx3-ubyte and t10k-labels-idx1-ubyte\n\n"
    }

    size_t parse_size_arg(const std::string& value, const std::string& name)
    {
        size_t consumed = 0;
        const unsigned long long parsed = std::stoull(value, &consumed);
        if (consumed != value.size() || parsed == 0 || parsed > static_cast<unsigned long long>(std::numeric_limits<size_t>::max()))
        {
            throw std::runtime_error(name + " must be a positive integer");
        }
        return static_cast<size_t>(parsed);
    }

    uint32_t parse_u32_arg(const std::string& value, const std::string& name)
    {
        size_t consumed = 0;
        const unsigned long parsed = std::stoul(value, &consumed);
        if (consumed != value.size() || parsed > static_cast<unsigned long>(std::numeric_limits<uint32_t>::max()))
        {
            throw std::runtime_error(name + " must be an unsigned 32-bit integer");
        }
        return static_cast<uint32_t>(parsed);
    }

    int64_t parse_i64_arg(const std::string& value, const std::string& name)
    {
        size_t consumed = 0;
        const long long parsed = std::stoll(value, &consumed);
        if (consumed != value.size() || parsed <= 0 || parsed > static_cast<long long>(std::numeric_limits<int64_t>::max()))
        {
            throw std::runtime_error(name + " must be a positive integer");
        }
        return static_cast<int64_t>(parsed);
    }

    float parse_f32_arg(const std::string& value, const std::string& name)
    {
        size_t consumed = 0;
        const float parsed = std::stof(value, &consumed);
        if (consumed != value.size() || !(parsed > 0.0f) || !std::isfinite(parsed))
        {
            throw std::runtime_error(name + " must be a positive finite value");
        }
        return parsed;
    }

    Options parse_options(int argc, char** argv)
    {
        Options options;
        for (int i = 1; i < argc; ++i)
        {
            const std::string arg = argv[i];
            auto value_after = [&](const std::string& prefix) -> std::optional<std::string>
                {
                    if (arg.rfind(prefix, 0) == 0)
                    {
                        return arg.substr(prefix.size());
                    }
                    return std::nullopt;
                };

            if (arg == "--help" || arg == "-h")
            {
                print_usage(argv[0]);
                std::exit(0);
            }
            if (auto value = value_after("--backend="))
            {
                if (*value == "cpu")
                {
                    options.backend = BackendKind::Cpu;
                }
                else if (*value == "cuda")
                {
                    options.backend = BackendKind::Cuda;
                }
                else if (*value == "cuda-resident" || *value == "cuda_resident")
                {
                    options.backend = BackendKind::CudaResident;
                }
                else if (*value == "vulkan-resident" || *value == "vulkan_resident")
                {
                    options.backend = BackendKind::VulkanResidentTraining;
                }
                else
                {
                    throw std::runtime_error("--backend must be cpu, cuda, cuda-resident, or vulkan-resident");
                }
                continue;
            }
            if (arg == "--compare-cpu-cuda")
            {
                options.compare_cpu_cuda = true;
                continue;
            }
            if (arg == "--verbose-backend")
            {
                options.verbose_backend = true;
                continue;
            }
            if (arg == "--verify-no-intermediate-transfer")
            {
                options.verify_no_intermediate_transfer = true;
                continue;
            }
            if (auto value = value_after("--data-dir="))
            {
                options.data_dir = *value;
                continue;
            }
            if (auto value = value_after("--samples="))
            {
                options.samples = parse_size_arg(*value, "--samples");
                options.samples_explicit = true;
                continue;
            }
            if (auto value = value_after("--epochs="))
            {
                options.epochs = parse_size_arg(*value, "--epochs");
                continue;
            }
            if (auto value = value_after("--batch-size="))
            {
                options.batch_size = parse_size_arg(*value, "--batch-size");
                continue;
            }
            if (auto value = value_after("--learning-rate="))
            {
                options.learning_rate = parse_f32_arg(*value, "--learning-rate");
                continue;
            }
            if (auto value = value_after("--hidden="))
            {
                options.hidden = parse_i64_arg(*value, "--hidden");
                continue;
            }
            if (auto value = value_after("--seed="))
            {
                options.seed = parse_u32_arg(*value, "--seed");
                continue;
            }
            if (auto value = value_after("--weights="))
            {
                options.load_weights_path = *value;
                continue;
            }
            if (auto value = value_after("--load-weights="))
            {
                options.load_weights_path = *value;
                continue;
            }
            if (auto value = value_after("--save-weights="))
            {
                options.save_weights_path = *value;
                continue;
            }
            if (auto value = value_after("--mode="))
            {
                if (*value == "infer")
                {
                    options.mode = Mode::Infer;
                }
                else if (*value == "train")
                {
                    options.mode = Mode::Train;
                }
                else
                {
                    throw std::runtime_error("--mode must be infer or train");
                }
                continue;
            }
            throw std::runtime_error("unknown argument: " + arg);
        }
        return options;
    }

    void print_run_result(
        const std::string& name,
        const computegraph_mnist::RunResult& result,
        const computegraph_mnist::Weights& weights)
    {
        std::cout << "backend: " << name << "\n";
        std::cout << "matmul_backend: " << (name == "cpu" ? "cpu" : (name == "cuda" ? "cuda/cublas" : name)) << "\n";
        std::cout << "samples: " << result.samples << "\n";
        std::cout << "batch_size: " << result.batch_size << "\n";
        std::cout << "timing_ms: " << std::fixed << std::setprecision(3) << result.elapsed_ms << "\n";
        if (result.has_accuracy)
        {
            const double accuracy = result.samples == 0
                ? 0.0
                : static_cast<double>(result.correct) / static_cast<double>(result.samples);
            if (weights.untrained_smoke)
            {
                std::cout << "accuracy_untrained_smoke: " << std::setprecision(6) << accuracy
                    << " (" << result.correct << "/" << result.samples << ")\n";
                std::cout << "accuracy_note: untrained random-weight accuracy is not meaningful\n";
            }
            else
            {
                std::cout << "accuracy: " << std::setprecision(6) << accuracy
                    << " (" << result.correct << "/" << result.samples << ")\n";
            }
        }
    }

    void print_accuracy_result(const std::string& prefix, const computegraph_mnist::RunResult& result)
    {
        if (!result.has_accuracy)
        {
            std::cout << prefix << "_accuracy: unavailable\n";
            return;
        }
        const double accuracy = result.samples == 0
            ? 0.0
            : static_cast<double>(result.correct) / static_cast<double>(result.samples);
        std::cout << prefix << "_samples: " << result.samples << "\n";
        std::cout << prefix << "_accuracy: " << std::fixed << std::setprecision(6) << accuracy
            << " (" << result.correct << "/" << result.samples << ")\n";
        std::cout << prefix << "_timing_ms: " << std::fixed << std::setprecision(3) << result.elapsed_ms << "\n";
    }

    std::string backend_name(BackendKind backend)
    {
        if (backend == BackendKind::Cuda)
        {
            return "cuda";
        }
        if (backend == BackendKind::CudaResident)
        {
            return "cuda-resident";
        }
        if (backend == BackendKind::VulkanResidentTraining)
        {
            return "vulkan-resident";
        }
        return "cpu";
    }

    std::string mode_name(Mode mode)
    {
        return mode == Mode::Train ? "train" : "infer";
    }

    void print_cuda_runtime_info()
    {
        const cuda_resident::RuntimeInfo info = cuda_resident::query_runtime_info();
        std::cout << "cuda_runtime_available: " << (info.available ? "yes" : "no") << "\n";
        if (!info.available && !info.unavailable_reason.empty())
        {
            std::cout << "cuda_runtime_unavailable_reason: " << info.unavailable_reason << "\n";
        }
        std::cout << "cuda_device_count: " << info.device_count << "\n";
        std::cout << "cuda_selected_device: " << info.selected_device << "\n";
        std::cout << "cuda_device_name: " << (info.device_name.empty() ? "<unavailable>" : info.device_name) << "\n";
        std::cout << "cuda_driver_version: " << info.cuda_driver_version << "\n";
        std::cout << "cuda_runtime_version: " << info.cuda_runtime_version << "\n";
        std::cout << "cublas_version: " << info.cublas_version << "\n";
    }

    std::optional<std::filesystem::path> configured_default_data_dir()
    {
        const std::filesystem::path path = COMPUTEGRAPH_DEFAULT_MNIST_DATA_DIR;
        if (!path.empty() && std::filesystem::exists(path) && std::filesystem::is_directory(path))
        {
            return path;
        }
        return std::nullopt;
    }
}

int main(int argc, char** argv)
{
    try
    {
        const Options options = parse_options(argc, argv);
        const std::optional<std::filesystem::path> default_data_dir = configured_default_data_dir();

        std::cout << "ComputeGraph MNIST MLP validation\n";
        std::cout << "mode: " << mode_name(options.mode) << "\n";
        std::cout << "backend_diagnostics: " << (options.verbose_backend ? "verbose" : "quiet") << "\n";
        std::cout << "build_cuda_enabled: " << (computegraph::build_config::enable_cuda ? "yes" : "no") << "\n";
        std::cout << "build_cublas_available: " << (computegraph::build_config::has_cublas ? "yes" : "no") << "\n";
        std::cout << "build_cuda_compiler_available: " << (computegraph::build_config::has_cuda_compiler ? "yes" : "no") << "\n";
        std::cout << "build_vulkan_enabled: " << (computegraph::build_config::enable_vulkan ? "yes" : "no") << "\n";
        std::cout << "build_vulkan_available: " << (computegraph::build_config::has_vulkan ? "yes" : "no") << "\n";
        if (options.backend == BackendKind::Cuda || options.backend == BackendKind::CudaResident || options.compare_cpu_cuda)
        {
            print_cuda_runtime_info();
        }

        if (options.mode == Mode::Train)
        {
            if (options.compare_cpu_cuda)
            {
                throw std::runtime_error("--compare-cpu-cuda is supported only with --mode=infer");
            }
            const std::optional<std::filesystem::path> train_data_dir = options.data_dir ? options.data_dir : default_data_dir;
            if (!train_data_dir)
            {
                throw std::runtime_error("--mode=train requires --data-dir or a configured default MNIST data directory");
            }

            const computegraph_mnist::Dataset dataset = computegraph_mnist::load_dataset_split_from_dir(
                *train_data_dir,
                computegraph_mnist::DatasetSplit::Train);
            std::optional<computegraph_mnist::Dataset> test_dataset;
            try
            {
                test_dataset = computegraph_mnist::load_dataset_split_from_dir(
                    *train_data_dir,
                    computegraph_mnist::DatasetSplit::Test);
            }
            catch (const std::exception& ex)
            {
                std::cout << "test_dataset: unavailable (" << ex.what() << ")\n";
            }
            const std::optional<computegraph_mnist::Weights> loaded_weights = options.load_weights_path
                ? std::optional<computegraph_mnist::Weights>(computegraph_mnist::load_weights_text(*options.load_weights_path, options.hidden))
                : std::nullopt;

            computegraph_mnist::TrainingOptions train_options;
            train_options.epochs = options.epochs;
            train_options.batch_size = options.batch_size;
            train_options.max_samples = options.samples_explicit ? options.samples : 0;
            train_options.eval_samples = options.samples_explicit ? std::min<size_t>(options.samples, 1000) : 1000;
            train_options.learning_rate = options.learning_rate;
            train_options.hidden = loaded_weights ? loaded_weights->hidden : options.hidden;
            train_options.seed = options.seed;
            train_options.verbose_backend = options.verbose_backend;
            train_options.verify_no_intermediate_transfer = options.verify_no_intermediate_transfer;
            train_options.download_final_weights =
                (options.backend != BackendKind::CudaResident && options.backend != BackendKind::VulkanResidentTraining) ||
                static_cast<bool>(options.save_weights_path);
            train_options.initial_weights = loaded_weights ? &*loaded_weights : nullptr;

            std::cout << "dataset: " << dataset.description << "\n";
            std::cout << "dataset_path: " << train_data_dir->string() << "\n";
            std::cout << "train_samples: " << dataset.sample_count() << "\n";
            std::cout << "test_samples: " << (test_dataset ? test_dataset->sample_count() : 0) << "\n";
            std::cout << "weights: " << (loaded_weights ? loaded_weights->description : "deterministic He initialization, seed=" + std::to_string(options.seed)) << "\n";
            std::cout << "load_weights: " << (options.load_weights_path ? options.load_weights_path->string() : "<none>") << "\n";
            std::cout << "save_weights: " << (options.save_weights_path ? options.save_weights_path->string() : "<none>") << "\n";
            std::cout << "hidden: " << train_options.hidden << "\n";
            std::cout << "epochs: " << train_options.epochs << "\n";
            std::cout << "batch_size: " << train_options.batch_size << "\n";
            std::cout << "learning_rate: " << std::setprecision(6) << train_options.learning_rate << "\n";
            std::cout << "seed: " << train_options.seed << "\n";
            std::cout << "matmul_backend: " << (options.backend == BackendKind::Cpu ? "cpu" : (options.backend == BackendKind::Cuda ? "cuda/cublas" : backend_name(options.backend))) << "\n";
            std::cout << "verify_no_intermediate_transfer: " << (options.verify_no_intermediate_transfer ? "yes" : "no") << "\n";

            const computegraph_mnist::TrainingResult result = computegraph_mnist::train_mlp(dataset, options.backend, train_options);
            for (const computegraph_mnist::TrainingEpochResult& epoch : result.epochs)
            {
                std::cout << "epoch: " << epoch.epoch
                    << " loss: " << std::fixed << std::setprecision(6) << epoch.average_loss
                    << " accuracy: " << std::setprecision(6) << epoch.accuracy
                    << " (" << epoch.correct << "/" << epoch.eval_samples << ")\n";
            }
            std::cout << "samples_trained: " << result.trained_samples << "\n";
            std::cout << "steps: " << result.steps << "\n";
            std::cout << "initial_loss: " << std::fixed << std::setprecision(6) << result.initial_loss << "\n";
            std::cout << "final_loss: " << std::fixed << std::setprecision(6) << result.final_loss << "\n";
            std::cout << "loss_decreased: " << (result.final_loss < result.initial_loss ? "yes" : "no") << "\n";
            std::cout << "cuda_cublas_used: " << (result.used_cuda_cublas ? "yes" : "no") << "\n";
            std::cout << "cuda_resident_used: " << (result.used_cuda_resident ? "yes" : "no") << "\n";
            std::cout << "vulkan_resident_used: " << (result.used_vulkan_resident ? "yes" : "no") << "\n";
            std::cout << "final_weight_download: " << (result.weights_downloaded ? "yes" : "no") << "\n";
            std::cout << "training_cuda_upload_count: " << result.training_cuda_upload_count << "\n";
            std::cout << "training_cuda_download_count: " << result.training_cuda_download_count << "\n";
            std::cout << "training_cuda_uploaded_bytes: " << result.training_cuda_uploaded_bytes << "\n";
            std::cout << "training_cuda_downloaded_bytes: " << result.training_cuda_downloaded_bytes << "\n";
            if (result.used_cuda_resident)
            {
                std::cout << "training_cuda_batch_image_upload_count: " << result.training_cuda_batch_image_upload_count << "\n";
                std::cout << "training_cuda_batch_image_uploaded_bytes: " << result.training_cuda_batch_image_uploaded_bytes << "\n";
                std::cout << "training_cuda_label_upload_count: " << result.training_cuda_label_upload_count << "\n";
                std::cout << "training_cuda_label_uploaded_bytes: " << result.training_cuda_label_uploaded_bytes << "\n";
                std::cout << "training_cuda_weight_upload_count: " << result.training_cuda_weight_upload_count << "\n";
                std::cout << "training_cuda_weight_uploaded_bytes: " << result.training_cuda_weight_uploaded_bytes << "\n";
                std::cout << "training_cuda_scalar_loss_download_count: " << result.training_cuda_scalar_loss_download_count << "\n";
                std::cout << "training_cuda_scalar_loss_downloaded_bytes: " << result.training_cuda_scalar_loss_downloaded_bytes << "\n";
                std::cout << "final_weight_download_count: " << result.final_weight_download_count << "\n";
                std::cout << "final_weight_downloaded_bytes: " << result.final_weight_downloaded_bytes << "\n";
            }
            if (result.used_vulkan_resident)
            {
                std::cout << "training_vulkan_upload_count: " << result.training_vulkan_upload_count << "\n";
                std::cout << "training_vulkan_download_count: " << result.training_vulkan_download_count << "\n";
                std::cout << "training_vulkan_uploaded_bytes: " << result.training_vulkan_uploaded_bytes << "\n";
                std::cout << "training_vulkan_downloaded_bytes: " << result.training_vulkan_downloaded_bytes << "\n";
                std::cout << "training_vulkan_batch_image_upload_count: " << result.training_vulkan_batch_image_upload_count << "\n";
                std::cout << "training_vulkan_batch_image_uploaded_bytes: " << result.training_vulkan_batch_image_uploaded_bytes << "\n";
                std::cout << "training_vulkan_label_upload_count: " << result.training_vulkan_label_upload_count << "\n";
                std::cout << "training_vulkan_label_uploaded_bytes: " << result.training_vulkan_label_uploaded_bytes << "\n";
                std::cout << "training_vulkan_weight_upload_count: " << result.training_vulkan_weight_upload_count << "\n";
                std::cout << "training_vulkan_weight_uploaded_bytes: " << result.training_vulkan_weight_uploaded_bytes << "\n";
                std::cout << "training_vulkan_scalar_loss_download_count: " << result.training_vulkan_scalar_loss_download_count << "\n";
                std::cout << "training_vulkan_scalar_loss_downloaded_bytes: " << result.training_vulkan_scalar_loss_downloaded_bytes << "\n";
                std::cout << "training_vulkan_intermediate_download_count: " << result.training_vulkan_intermediate_download_count << "\n";
                std::cout << "training_vulkan_unknown_transfer_count: " << result.training_vulkan_unknown_transfer_count << "\n";
                std::cout << "training_vulkan_dispatch_count: " << result.final_debug_stats.vulkan_dispatch_count << "\n";
                std::cout << "training_vulkan_pipeline_cache_hits: " << result.final_debug_stats.vulkan_pipeline_cache_hits << "\n";
                std::cout << "training_vulkan_pipeline_cache_misses: " << result.final_debug_stats.vulkan_pipeline_cache_misses << "\n";
                std::cout << "training_vulkan_descriptor_cache_hits: " << result.final_debug_stats.vulkan_descriptor_cache_hits << "\n";
                std::cout << "training_vulkan_descriptor_cache_misses: " << result.final_debug_stats.vulkan_descriptor_cache_misses << "\n";
                std::cout << "training_vulkan_explicit_synchronization_count: " << result.final_debug_stats.vulkan_explicit_synchronization_count << "\n";
                std::cout << "final_weight_download_count: " << result.final_weight_download_count << "\n";
                std::cout << "final_weight_downloaded_bytes: " << result.final_weight_downloaded_bytes << "\n";
            }
            std::cout << "total_cuda_upload_count: " << result.final_debug_stats.cuda_upload_count << "\n";
            std::cout << "total_cuda_download_count: " << result.final_debug_stats.cuda_download_count << "\n";
            std::cout << "total_cuda_uploaded_bytes: " << result.final_debug_stats.cuda_uploaded_bytes << "\n";
            std::cout << "total_cuda_downloaded_bytes: " << result.final_debug_stats.cuda_downloaded_bytes << "\n";
            std::cout << "total_vulkan_upload_count: " << result.final_debug_stats.vulkan_upload_count << "\n";
            std::cout << "total_vulkan_download_count: " << result.final_debug_stats.vulkan_download_count << "\n";
            std::cout << "total_vulkan_uploaded_bytes: " << result.final_debug_stats.vulkan_uploaded_bytes << "\n";
            std::cout << "total_vulkan_downloaded_bytes: " << result.final_debug_stats.vulkan_downloaded_bytes << "\n";
            std::cout << "timing_ms: " << std::fixed << std::setprecision(3) << result.elapsed_ms << "\n";

            const size_t final_eval_samples = options.samples_explicit ? options.samples : 10000;
            if (result.weights_downloaded)
            {
                const computegraph_mnist::RunResult train_eval = computegraph_mnist::run_mlp(
                    dataset,
                    result.weights,
                    options.backend,
                    final_eval_samples,
                    train_options.batch_size,
                    options.verbose_backend);
                print_accuracy_result("train", train_eval);

                if (test_dataset)
                {
                    const computegraph_mnist::RunResult test_eval = computegraph_mnist::run_mlp(
                        *test_dataset,
                        result.weights,
                        options.backend,
                        final_eval_samples,
                        train_options.batch_size,
                        options.verbose_backend);
                    print_accuracy_result("test", test_eval);
                }
            }
            else
            {
                std::cout << "final_eval: skipped (trained weights remain resident; use --save-weights to request final download)\n";
            }

            if (options.save_weights_path)
            {
                computegraph_mnist::save_weights_text(*options.save_weights_path, result.weights);
                std::cout << "saved_weights: " << options.save_weights_path->string() << "\n";
            }
            return 0;
        }

        if (options.save_weights_path)
        {
            throw std::runtime_error("--save-weights is supported only with --mode=train");
        }

        const computegraph_mnist::Dataset dataset = options.data_dir
            ? computegraph_mnist::load_dataset_from_dir(*options.data_dir)
            : (default_data_dir
                ? computegraph_mnist::load_dataset_from_dir(*default_data_dir)
                : computegraph_mnist::make_smoke_dataset(options.samples));
        const computegraph_mnist::Weights weights = options.load_weights_path
            ? computegraph_mnist::load_weights_text(*options.load_weights_path, options.hidden)
            : computegraph_mnist::make_deterministic_weights(options.hidden, options.seed);

        std::cout << "dataset: " << dataset.description << "\n";
        std::cout << "dataset_path: " << (options.data_dir ? options.data_dir->string() : (default_data_dir ? default_data_dir->string() : "<synthetic-smoke>")) << "\n";
        std::cout << "weights: " << weights.description << "\n";
        std::cout << "load_weights: " << (options.load_weights_path ? options.load_weights_path->string() : "<none>") << "\n";
        std::cout << "weight_mode: " << (weights.untrained_smoke ? "untrained_random_smoke" : "loaded_weights") << "\n";
        std::cout << "hidden: " << weights.hidden << "\n";
        std::cout << "batch_size: " << options.batch_size << "\n";
        std::cout << "seed: " << options.seed << "\n";

        if (options.compare_cpu_cuda)
        {
            const computegraph_mnist::RunResult cpu = computegraph_mnist::run_mlp(
                dataset, weights, BackendKind::Cpu, options.samples, options.batch_size, options.verbose_backend);
            const computegraph_mnist::RunResult cuda = computegraph_mnist::run_mlp(
                dataset, weights, BackendKind::Cuda, options.samples, options.batch_size, options.verbose_backend);
            const computegraph_mnist::Difference diff = computegraph_mnist::compare_results(cpu, cuda);

            print_run_result("cpu", cpu, weights);
            print_run_result("cuda", cuda, weights);
            std::cout << "cpu_vs_cuda_max_abs_diff: " << std::setprecision(9) << diff.max_abs << "\n";
            std::cout << "cpu_vs_cuda_mean_abs_diff: " << std::setprecision(9) << diff.mean_abs << "\n";
            std::cout << "cpu_vs_cuda_max_rel_diff: " << std::setprecision(9) << diff.max_rel << "\n";
            std::cout << "cpu_vs_cuda_prediction_mismatches: " << diff.mismatched_predictions << "\n";
        }
        else
        {
            const computegraph_mnist::RunResult result = computegraph_mnist::run_mlp(
                dataset, weights, options.backend, options.samples, options.batch_size, options.verbose_backend, options.verify_no_intermediate_transfer);
            print_run_result(backend_name(options.backend), result, weights);
            if (options.backend == BackendKind::CudaResident)
            {
                std::cout << "verify_no_intermediate_transfer: " << (options.verify_no_intermediate_transfer ? "yes" : "no") << "\n";
                std::cout << "inference_cuda_upload_count: " << result.cuda_upload_count << "\n";
                std::cout << "inference_cuda_download_count: " << result.cuda_download_count << "\n";
                std::cout << "inference_cuda_uploaded_bytes: " << result.cuda_uploaded_bytes << "\n";
                std::cout << "inference_cuda_downloaded_bytes: " << result.cuda_downloaded_bytes << "\n";
            }
            if (options.backend == BackendKind::VulkanResidentTraining)
            {
                std::cout << "verify_no_intermediate_transfer: " << (options.verify_no_intermediate_transfer ? "yes" : "no") << "\n";
                std::cout << "inference_vulkan_upload_count: " << result.vulkan_upload_count << "\n";
                std::cout << "inference_vulkan_download_count: " << result.vulkan_download_count << "\n";
                std::cout << "inference_vulkan_uploaded_bytes: " << result.vulkan_uploaded_bytes << "\n";
                std::cout << "inference_vulkan_downloaded_bytes: " << result.vulkan_downloaded_bytes << "\n";
                std::cout << "inference_vulkan_dispatch_count: " << result.vulkan_dispatch_count << "\n";
                std::cout << "inference_vulkan_pipeline_cache_hits: " << result.vulkan_pipeline_cache_hits << "\n";
                std::cout << "inference_vulkan_pipeline_cache_misses: " << result.vulkan_pipeline_cache_misses << "\n";
                std::cout << "inference_vulkan_descriptor_cache_hits: " << result.vulkan_descriptor_cache_hits << "\n";
                std::cout << "inference_vulkan_descriptor_cache_misses: " << result.vulkan_descriptor_cache_misses << "\n";
                std::cout << "inference_vulkan_explicit_synchronization_count: " << result.vulkan_explicit_synchronization_count << "\n";
            }
        }
        return 0;
    }
    catch (const std::exception& ex)
    {
        std::cerr << "computegraph_mnist: " << ex.what() << "\n";
        return 1;
    }
}
