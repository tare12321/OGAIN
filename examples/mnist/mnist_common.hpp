#pragma once

#include <computegraph/compute_graph.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace computegraph_mnist
{
    constexpr int64_t kInputFeatures = 784;
    constexpr int64_t kClasses = 10;
    constexpr uint32_t kIdxImageMagic = 2051;
    constexpr uint32_t kIdxLabelMagic = 2049;

    enum class DatasetSplit
    {
        Train,
        Test,
    };

    struct Dataset
    {
        std::vector<float> images;
        std::vector<uint8_t> labels;
        int64_t rows = 28;
        int64_t cols = 28;
        bool has_labels = false;
        bool synthetic_smoke = false;
        std::string description;

        size_t sample_count() const
        {
            return images.size() / (size_t)kInputFeatures;
        }
    };

    struct Weights
    {
        int64_t hidden = 64;
        std::vector<float> w1;
        std::vector<float> b1;
        std::vector<float> w2;
        std::vector<float> b2;
        std::string description;
        bool loaded_from_file = false;
        bool untrained_smoke = false;
    };

    struct RunResult
    {
        std::vector<float> logits;
        std::vector<int> predictions;
        size_t samples = 0;
        size_t batch_size = 0;
        double elapsed_ms = 0.0;
        int correct = 0;
        bool has_accuracy = false;
        bool used_cuda_resident = false;
        bool used_vulkan_resident = false;
        size_t cuda_upload_count = 0;
        size_t cuda_download_count = 0;
        size_t cuda_uploaded_bytes = 0;
        size_t cuda_downloaded_bytes = 0;
        size_t vulkan_upload_count = 0;
        size_t vulkan_download_count = 0;
        size_t vulkan_uploaded_bytes = 0;
        size_t vulkan_downloaded_bytes = 0;
        size_t vulkan_dispatch_count = 0;
        size_t vulkan_pipeline_cache_hits = 0;
        size_t vulkan_pipeline_cache_misses = 0;
        size_t vulkan_descriptor_cache_hits = 0;
        size_t vulkan_descriptor_cache_misses = 0;
        size_t vulkan_explicit_synchronization_count = 0;
    };

    struct Difference
    {
        float max_abs = 0.0f;
        float mean_abs = 0.0f;
        float max_rel = 0.0f;
        size_t mismatched_predictions = 0;
    };

    struct TrainingOptions
    {
        size_t epochs = 1;
        size_t batch_size = 64;
        size_t max_samples = 0;
        size_t eval_samples = 1000;
        float learning_rate = 0.01f;
        int64_t hidden = 128;
        uint32_t seed = 123;
        bool verbose_backend = false;
        bool verify_no_intermediate_transfer = false;
        bool download_final_weights = true;
        const Weights* initial_weights = nullptr;
    };

    struct TrainingEpochResult
    {
        size_t epoch = 0;
        float average_loss = 0.0f;
        double accuracy = 0.0;
        int correct = 0;
        size_t eval_samples = 0;
    };

    struct TrainingResult
    {
        Weights weights;
        std::vector<TrainingEpochResult> epochs;
        float initial_loss = 0.0f;
        float final_loss = 0.0f;
        size_t steps = 0;
        size_t trained_samples = 0;
        size_t batch_size = 0;
        bool used_cuda_cublas = false;
        bool used_cuda_resident = false;
        bool used_vulkan_resident = false;
        bool weights_downloaded = false;
        size_t training_cuda_upload_count = 0;
        size_t training_cuda_download_count = 0;
        size_t training_cuda_uploaded_bytes = 0;
        size_t training_cuda_downloaded_bytes = 0;
        size_t training_cuda_batch_image_upload_count = 0;
        size_t training_cuda_batch_image_uploaded_bytes = 0;
        size_t training_cuda_label_upload_count = 0;
        size_t training_cuda_label_uploaded_bytes = 0;
        size_t training_cuda_weight_upload_count = 0;
        size_t training_cuda_weight_uploaded_bytes = 0;
        size_t training_cuda_scalar_loss_download_count = 0;
        size_t training_cuda_scalar_loss_downloaded_bytes = 0;
        size_t training_vulkan_upload_count = 0;
        size_t training_vulkan_download_count = 0;
        size_t training_vulkan_uploaded_bytes = 0;
        size_t training_vulkan_downloaded_bytes = 0;
        size_t training_vulkan_batch_image_upload_count = 0;
        size_t training_vulkan_batch_image_uploaded_bytes = 0;
        size_t training_vulkan_label_upload_count = 0;
        size_t training_vulkan_label_uploaded_bytes = 0;
        size_t training_vulkan_weight_upload_count = 0;
        size_t training_vulkan_weight_uploaded_bytes = 0;
        size_t training_vulkan_scalar_loss_download_count = 0;
        size_t training_vulkan_scalar_loss_downloaded_bytes = 0;
        size_t training_vulkan_intermediate_download_count = 0;
        size_t training_vulkan_unknown_transfer_count = 0;
        size_t final_weight_download_count = 0;
        size_t final_weight_downloaded_bytes = 0;
        RuntimeSession::DebugStats final_debug_stats;
        double elapsed_ms = 0.0;
    };

    struct TrainingGraphHandles
    {
        TensorHandle input;
        TensorHandle labels;
        TensorHandle w1;
        TensorHandle b1;
        TensorHandle w2;
        TensorHandle b2;
        TensorHandle logits;
        TensorHandle loss;
    };

    struct InferenceGraphHandles
    {
        TensorHandle input;
        TensorHandle w1;
        TensorHandle b1;
        TensorHandle w2;
        TensorHandle b2;
        TensorHandle logits;
    };

    inline uint32_t read_be_u32(std::istream& in, const std::string& path)
    {
        uint8_t bytes[4] = {};
        in.read(reinterpret_cast<char*>(bytes), 4);
        if (!in)
        {
            throw std::runtime_error("MNIST IDX read failed in " + path);
        }
        return (uint32_t(bytes[0]) << 24) |
            (uint32_t(bytes[1]) << 16) |
            (uint32_t(bytes[2]) << 8) |
            uint32_t(bytes[3]);
    }

    inline std::vector<uint8_t> read_file_tail(std::ifstream& in, size_t bytes, const std::string& path)
    {
        std::vector<uint8_t> out(bytes);
        if (bytes != 0)
        {
            in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(bytes));
        }
        if (!in)
        {
            throw std::runtime_error("MNIST IDX payload is truncated: " + path);
        }
        return out;
    }

    inline void require_idx_eof(std::ifstream& in, const std::string& path)
    {
        const int next = in.peek();
        if (next != std::char_traits<char>::eof())
        {
            throw std::runtime_error("MNIST IDX file has trailing bytes after expected payload: " + path);
        }
    }

    inline Dataset load_idx_dataset_pair(const std::filesystem::path& image_path, const std::filesystem::path& label_path)
    {
        std::ifstream images_in(image_path, std::ios::binary);
        if (!images_in)
        {
            throw std::runtime_error("MNIST images file not found: " + image_path.string());
        }
        std::ifstream labels_in(label_path, std::ios::binary);
        if (!labels_in)
        {
            throw std::runtime_error("MNIST labels file not found: " + label_path.string());
        }

        const uint32_t image_magic = read_be_u32(images_in, image_path.string());
        const uint32_t image_count = read_be_u32(images_in, image_path.string());
        const uint32_t rows = read_be_u32(images_in, image_path.string());
        const uint32_t cols = read_be_u32(images_in, image_path.string());
        if (image_magic != kIdxImageMagic)
        {
            throw std::runtime_error("MNIST images file has invalid IDX magic: " + image_path.string());
        }
        if (rows != 28 || cols != 28)
        {
            throw std::runtime_error("MNIST images must be 28x28: " + image_path.string());
        }

        const uint32_t label_magic = read_be_u32(labels_in, label_path.string());
        const uint32_t label_count = read_be_u32(labels_in, label_path.string());
        if (label_magic != kIdxLabelMagic)
        {
            throw std::runtime_error("MNIST labels file has invalid IDX magic: " + label_path.string());
        }
        if (image_count != label_count)
        {
            throw std::runtime_error("MNIST image/label count mismatch: " + image_path.string() + " vs " + label_path.string());
        }

        const size_t pixel_count = (size_t)image_count * (size_t)rows * (size_t)cols;
        const std::vector<uint8_t> raw_images = read_file_tail(images_in, pixel_count, image_path.string());
        std::vector<uint8_t> raw_labels = read_file_tail(labels_in, (size_t)label_count, label_path.string());
        require_idx_eof(images_in, image_path.string());
        require_idx_eof(labels_in, label_path.string());
        for (uint8_t label : raw_labels)
        {
            if (label >= kClasses)
            {
                throw std::runtime_error("MNIST label out of range in " + label_path.string());
            }
        }

        Dataset dataset;
        dataset.images.resize(pixel_count);
        std::transform(raw_images.begin(), raw_images.end(), dataset.images.begin(), [](uint8_t v)
            {
                return static_cast<float>(v) / 255.0f;
            });
        dataset.labels = std::move(raw_labels);
        dataset.rows = rows;
        dataset.cols = cols;
        dataset.has_labels = true;
        dataset.synthetic_smoke = false;
        dataset.description = image_path.string();
        return dataset;
    }

    inline std::string expected_dataset_message(const std::filesystem::path& data_dir)
    {
        std::ostringstream oss;
        oss << "Expected MNIST IDX files in " << data_dir.string() << ":\n"
            << "  train-images-idx3-ubyte and train-labels-idx1-ubyte, or\n"
            << "  t10k-images-idx3-ubyte and t10k-labels-idx1-ubyte\n"
            << "Nested files are also supported, e.g. train-images-idx3-ubyte/train-images.idx3-ubyte.";
        return oss.str();
    }

    inline std::filesystem::path first_existing_path(std::initializer_list<std::filesystem::path> candidates)
    {
        for (const std::filesystem::path& candidate : candidates)
        {
            if (std::filesystem::exists(candidate) && std::filesystem::is_regular_file(candidate))
            {
                return candidate;
            }
        }
        return {};
    }

    struct DatasetSplitPaths
    {
        std::filesystem::path images;
        std::filesystem::path labels;
    };

    inline DatasetSplitPaths resolve_split_paths(
        const std::filesystem::path& data_dir,
        const std::string& image_stem,
        const std::string& image_nested_name,
        const std::string& label_stem,
        const std::string& label_nested_name)
    {
        DatasetSplitPaths paths;
        paths.images = first_existing_path({
            data_dir / image_stem,
            data_dir / image_nested_name,
            data_dir / image_stem / image_nested_name,
        });
        paths.labels = first_existing_path({
            data_dir / label_stem,
            data_dir / label_nested_name,
            data_dir / label_stem / label_nested_name,
        });
        return paths;
    }

    inline bool split_is_present(const DatasetSplitPaths& paths)
    {
        return !paths.images.empty() || !paths.labels.empty();
    }

    inline void require_complete_split(const DatasetSplitPaths& paths, const std::string& split_name)
    {
        if (paths.images.empty())
        {
            throw std::runtime_error("MNIST " + split_name + " images file is missing while labels are present");
        }
        if (paths.labels.empty())
        {
            throw std::runtime_error("MNIST " + split_name + " labels file is missing while images are present");
        }
    }

    inline Dataset load_dataset_split_from_dir(const std::filesystem::path& data_dir, DatasetSplit split)
    {
        if (!std::filesystem::exists(data_dir) || !std::filesystem::is_directory(data_dir))
        {
            throw std::runtime_error("MNIST data directory is missing. " + expected_dataset_message(data_dir));
        }

        const DatasetSplitPaths train = resolve_split_paths(
            data_dir,
            "train-images-idx3-ubyte",
            "train-images.idx3-ubyte",
            "train-labels-idx1-ubyte",
            "train-labels.idx1-ubyte");
        const DatasetSplitPaths test = resolve_split_paths(
            data_dir,
            "t10k-images-idx3-ubyte",
            "t10k-images.idx3-ubyte",
            "t10k-labels-idx1-ubyte",
            "t10k-labels.idx1-ubyte");

        if (split == DatasetSplit::Train)
        {
            if (!split_is_present(train))
            {
                throw std::runtime_error("MNIST train IDX files are missing. " + expected_dataset_message(data_dir));
            }
            require_complete_split(train, "train");
            return load_idx_dataset_pair(train.images, train.labels);
        }

        if (!split_is_present(test))
        {
            throw std::runtime_error("MNIST test IDX files are missing. " + expected_dataset_message(data_dir));
        }
        require_complete_split(test, "test");
        return load_idx_dataset_pair(test.images, test.labels);
    }

    inline Dataset load_dataset_from_dir(const std::filesystem::path& data_dir)
    {
        if (!std::filesystem::exists(data_dir) || !std::filesystem::is_directory(data_dir))
        {
            throw std::runtime_error("MNIST data directory is missing. " + expected_dataset_message(data_dir));
        }

        const DatasetSplitPaths train = resolve_split_paths(
            data_dir,
            "train-images-idx3-ubyte",
            "train-images.idx3-ubyte",
            "train-labels-idx1-ubyte",
            "train-labels.idx1-ubyte");
        const DatasetSplitPaths test = resolve_split_paths(
            data_dir,
            "t10k-images-idx3-ubyte",
            "t10k-images.idx3-ubyte",
            "t10k-labels-idx1-ubyte",
            "t10k-labels.idx1-ubyte");

        if (split_is_present(test))
        {
            require_complete_split(test, "test");
            return load_idx_dataset_pair(test.images, test.labels);
        }
        if (split_is_present(train))
        {
            require_complete_split(train, "train");
            return load_idx_dataset_pair(train.images, train.labels);
        }

        throw std::runtime_error("MNIST IDX files are missing. " + expected_dataset_message(data_dir));
    }

    inline Dataset make_smoke_dataset(size_t samples)
    {
        Dataset dataset;
        dataset.images.resize(samples * (size_t)kInputFeatures);
        dataset.labels.resize(samples);
        for (size_t s = 0; s < samples; ++s)
        {
            dataset.labels[s] = static_cast<uint8_t>((s * 7) % kClasses);
            for (size_t j = 0; j < (size_t)kInputFeatures; ++j)
            {
                const int centered = static_cast<int>((s * 31 + j * 17) % 257) - 128;
                dataset.images[s * (size_t)kInputFeatures + j] = static_cast<float>(centered) / 128.0f;
            }
        }
        dataset.has_labels = true;
        dataset.synthetic_smoke = true;
        dataset.description = "deterministic MNIST-shaped smoke data; not real MNIST";
        return dataset;
    }

    inline Weights make_deterministic_weights(int64_t hidden, uint32_t seed = 1337)
    {
        if (hidden <= 0)
        {
            throw std::runtime_error("hidden size must be positive");
        }

        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(-0.05f, 0.05f);

        Weights weights;
        weights.hidden = hidden;
        weights.w1.resize((size_t)(kInputFeatures * hidden));
        weights.b1.resize((size_t)hidden);
        weights.w2.resize((size_t)(hidden * kClasses));
        weights.b2.resize((size_t)kClasses);
        for (float& value : weights.w1) value = dist(rng);
        for (float& value : weights.b1) value = dist(rng) * 0.2f;
        for (float& value : weights.w2) value = dist(rng);
        for (float& value : weights.b2) value = dist(rng) * 0.2f;
        weights.description = "UNTRAINED deterministic pseudo-random weights, seed=" + std::to_string(seed) +
            "; validates backend correctness only, accuracy is not meaningful";
        weights.loaded_from_file = false;
        weights.untrained_smoke = true;
        return weights;
    }

    inline Weights make_training_initial_weights(int64_t hidden, uint32_t seed = 123)
    {
        if (hidden <= 0)
        {
            throw std::runtime_error("hidden size must be positive");
        }

        std::mt19937 rng(seed);
        const float w1_stddev = std::sqrt(2.0f / static_cast<float>(kInputFeatures));
        const float w2_stddev = std::sqrt(2.0f / static_cast<float>(hidden));
        std::normal_distribution<float> w1_dist(0.0f, w1_stddev);
        std::normal_distribution<float> w2_dist(0.0f, w2_stddev);

        Weights weights;
        weights.hidden = hidden;
        weights.w1.resize((size_t)(kInputFeatures * hidden));
        weights.b1.assign((size_t)hidden, 0.0f);
        weights.w2.resize((size_t)(hidden * kClasses));
        weights.b2.assign((size_t)kClasses, 0.0f);
        for (float& value : weights.w1) value = w1_dist(rng);
        for (float& value : weights.w2) value = w2_dist(rng);
        weights.description = "deterministic He initialization for MNIST training, seed=" + std::to_string(seed);
        weights.loaded_from_file = false;
        weights.untrained_smoke = false;
        return weights;
    }

    inline void validate_weight_shapes(const Weights& weights)
    {
        if (weights.hidden <= 0)
        {
            throw std::runtime_error("weights hidden size must be positive");
        }
        const size_t hidden = (size_t)weights.hidden;
        if (weights.w1.size() != (size_t)kInputFeatures * hidden)
        {
            throw std::runtime_error("weights W1 shape is incompatible; expected [784," + std::to_string(weights.hidden) + "]");
        }
        if (weights.b1.size() != hidden)
        {
            throw std::runtime_error("weights b1 shape is incompatible; expected [" + std::to_string(weights.hidden) + "]");
        }
        if (weights.w2.size() != hidden * (size_t)kClasses)
        {
            throw std::runtime_error("weights W2 shape is incompatible; expected [" + std::to_string(weights.hidden) + ",10]");
        }
        if (weights.b2.size() != (size_t)kClasses)
        {
            throw std::runtime_error("weights b2 shape is incompatible; expected [10]");
        }
    }

    inline size_t weight_value_count(const Weights& weights)
    {
        return weights.w1.size() + weights.b1.size() + weights.w2.size() + weights.b2.size();
    }

    inline std::string weight_format_help()
    {
        return "Expected text weight format: computegraph_mnist_mlp_v1, dtype f32, hidden <N>, "
            "then W1 784 <N>, b1 <N>, W2 <N> 10, b2 10 followed by float values";
    }

    inline void read_expected_token(std::istream& in, const std::string& expected, const std::filesystem::path& path)
    {
        std::string token;
        if (!(in >> token) || token != expected)
        {
            throw std::runtime_error("weights file " + path.string() + " expected token '" + expected + "'. " + weight_format_help());
        }
    }

    inline int64_t read_positive_i64(std::istream& in, const std::string& field, const std::filesystem::path& path)
    {
        int64_t value = 0;
        if (!(in >> value) || value <= 0)
        {
            throw std::runtime_error("weights file " + path.string() + " has invalid positive integer for " + field);
        }
        return value;
    }

    inline std::vector<float> read_weight_values(std::istream& in, size_t count, const std::string& tensor_name, const std::filesystem::path& path)
    {
        std::vector<float> values(count);
        for (size_t i = 0; i < count; ++i)
        {
            if (!(in >> values[i]))
            {
                throw std::runtime_error("weights file " + path.string() + " ended while reading " + tensor_name +
                    "; expected " + std::to_string(count) + " f32 values");
            }
        }
        return values;
    }

    inline Weights load_legacy_weights_text(std::istream& in, const std::filesystem::path& path, int64_t hidden, const std::string& first_token)
    {
        if (hidden <= 0)
        {
            throw std::runtime_error("legacy weights require positive --hidden because the file has no shape header");
        }
        const size_t expected = (size_t)(kInputFeatures * hidden + hidden + hidden * kClasses + kClasses);
        std::vector<float> values;
        values.reserve(expected);
        try
        {
            size_t consumed = 0;
            const float first_value = std::stof(first_token, &consumed);
            if (consumed != first_token.size())
            {
                throw std::runtime_error("partial float token");
            }
            values.push_back(first_value);
        }
        catch (const std::exception&)
        {
            throw std::runtime_error("weights file " + path.string() + " is not the documented Sprint 12 format and does not start with a float");
        }

        float value = 0.0f;
        while (in >> value)
        {
            values.push_back(value);
        }
        if (!in.eof())
        {
            throw std::runtime_error("weights file contains a non-float token: " + path.string());
        }
        if (values.size() != expected)
        {
            throw std::runtime_error("legacy weights file has " + std::to_string(values.size()) +
                " floats; expected " + std::to_string(expected) +
                " for [W1,b1,W2,b2] with hidden=" + std::to_string(hidden));
        }

        Weights weights;
        weights.hidden = hidden;
        size_t offset = 0;
        auto take = [&](size_t count)
            {
                std::vector<float> out(values.begin() + static_cast<std::ptrdiff_t>(offset),
                    values.begin() + static_cast<std::ptrdiff_t>(offset + count));
                offset += count;
                return out;
            };
        weights.w1 = take((size_t)(kInputFeatures * hidden));
        weights.b1 = take((size_t)hidden);
        weights.w2 = take((size_t)(hidden * kClasses));
        weights.b2 = take((size_t)kClasses);
        weights.description = "loaded legacy raw-float weights from " + path.string();
        weights.loaded_from_file = true;
        weights.untrained_smoke = false;
        validate_weight_shapes(weights);
        return weights;
    }

    inline Weights load_weights_text(const std::filesystem::path& path, int64_t hidden_hint = 0)
    {
        std::ifstream in(path);
        if (!in)
        {
            throw std::runtime_error("weights file not found: " + path.string());
        }

        std::string magic;
        if (!(in >> magic))
        {
            throw std::runtime_error("weights file is empty: " + path.string());
        }
        if (magic != "computegraph_mnist_mlp_v1")
        {
            return load_legacy_weights_text(in, path, hidden_hint, magic);
        }

        read_expected_token(in, "dtype", path);
        std::string dtype;
        if (!(in >> dtype) || dtype != "f32")
        {
            throw std::runtime_error("weights file " + path.string() + " has unsupported dtype; expected dtype f32");
        }
        read_expected_token(in, "hidden", path);
        const int64_t hidden = read_positive_i64(in, "hidden", path);

        Weights weights;
        weights.hidden = hidden;
        const int64_t w1_rows = (read_expected_token(in, "W1", path), read_positive_i64(in, "W1 rows", path));
        const int64_t w1_cols = read_positive_i64(in, "W1 cols", path);
        if (w1_rows != kInputFeatures || w1_cols != hidden)
        {
            throw std::runtime_error("weights file " + path.string() + " has incompatible W1 shape [" +
                std::to_string(w1_rows) + "," + std::to_string(w1_cols) + "]; expected [784," + std::to_string(hidden) + "]");
        }
        weights.w1 = read_weight_values(in, (size_t)(kInputFeatures * hidden), "W1", path);

        read_expected_token(in, "b1", path);
        const int64_t b1_size = read_positive_i64(in, "b1 size", path);
        if (b1_size != hidden)
        {
            throw std::runtime_error("weights file " + path.string() + " has incompatible b1 shape [" +
                std::to_string(b1_size) + "]; expected [" + std::to_string(hidden) + "]");
        }
        weights.b1 = read_weight_values(in, (size_t)hidden, "b1", path);

        read_expected_token(in, "W2", path);
        const int64_t w2_rows = read_positive_i64(in, "W2 rows", path);
        const int64_t w2_cols = read_positive_i64(in, "W2 cols", path);
        if (w2_rows != hidden || w2_cols != kClasses)
        {
            throw std::runtime_error("weights file " + path.string() + " has incompatible W2 shape [" +
                std::to_string(w2_rows) + "," + std::to_string(w2_cols) + "]; expected [" + std::to_string(hidden) + ",10]");
        }
        weights.w2 = read_weight_values(in, (size_t)(hidden * kClasses), "W2", path);

        read_expected_token(in, "b2", path);
        const int64_t b2_size = read_positive_i64(in, "b2 size", path);
        if (b2_size != kClasses)
        {
            throw std::runtime_error("weights file " + path.string() + " has incompatible b2 shape [" +
                std::to_string(b2_size) + "]; expected [10]");
        }
        weights.b2 = read_weight_values(in, (size_t)kClasses, "b2", path);

        std::string trailing;
        if (in >> trailing)
        {
            throw std::runtime_error("weights file " + path.string() + " has trailing token after b2 values: " + trailing);
        }

        weights.description = "loaded Sprint 12 f32 MLP weights from " + path.string() +
            "; measured accuracy is meaningful only if these weights were trained";
        weights.loaded_from_file = true;
        weights.untrained_smoke = false;
        validate_weight_shapes(weights);
        return weights;
    }

    inline void save_weights_text(const std::filesystem::path& path, const Weights& weights)
    {
        validate_weight_shapes(weights);
        std::ofstream out(path);
        if (!out)
        {
            throw std::runtime_error("cannot open weights file for writing: " + path.string());
        }
        out << std::setprecision(9);
        out << "computegraph_mnist_mlp_v1\n";
        out << "dtype f32\n";
        out << "hidden " << weights.hidden << "\n";
        out << "W1 " << kInputFeatures << " " << weights.hidden << "\n";
        for (float value : weights.w1) out << value << "\n";
        out << "b1 " << weights.hidden << "\n";
        for (float value : weights.b1) out << value << "\n";
        out << "W2 " << weights.hidden << " " << kClasses << "\n";
        for (float value : weights.w2) out << value << "\n";
        out << "b2 " << kClasses << "\n";
        for (float value : weights.b2) out << value << "\n";
    }

    inline std::vector<float> tensor_to_vec_f32(const Tensor& tensor)
    {
        BufferPool pool;
        Tensor contiguous = ensure_contiguous(tensor, pool);
        return std::vector<float>(contiguous.f32_ptr(), contiguous.f32_ptr() + contiguous.numel());
    }

    inline std::vector<int> argmax_rows(const std::vector<float>& logits, size_t rows)
    {
        std::vector<int> predictions(rows, 0);
        for (size_t row = 0; row < rows; ++row)
        {
            int best = 0;
            float best_value = logits[row * (size_t)kClasses];
            for (int cls = 1; cls < kClasses; ++cls)
            {
                const float candidate = logits[row * (size_t)kClasses + (size_t)cls];
                if (candidate > best_value)
                {
                    best_value = candidate;
                    best = cls;
                }
            }
            predictions[row] = best;
        }
        return predictions;
    }

    inline Status cuda_matmul_available_status()
    {
        BackendRegistry registry = make_default_backend_registry();
        BackendSelectionRequest request;
        request.op = OpType::MatMul2D;
        request.dtype = DType::F32;
        request.inputs_are_contiguous = true;
        request.has_explicit_backend = true;
        request.explicit_backend = BackendKind::Cuda;
        request.allow_fallback = false;

        const BackendSelectionResult result = registry.select_backend(request);
        if (result.ok)
        {
            return Status::ok();
        }

        std::ostringstream oss;
        oss << result.status._msg;
        for (const std::string& diagnostic : result.diagnostics)
        {
            oss << "; " << diagnostic;
        }
        return Status::not_found(oss.str());
    }

    inline void throw_if_backend_unavailable(BackendKind backend)
    {
        if (backend == BackendKind::Cpu)
        {
            return;
        }
        if (backend == BackendKind::Cuda)
        {
            const Status status = cuda_matmul_available_status();
            if (!status.ok_())
            {
                throw std::runtime_error("explicit CUDA backend requested but CUDA/cuBLAS is unavailable: " + status._msg);
            }
            return;
        }
        if (backend == BackendKind::CudaResident)
        {
            std::string reason;
            if (!cuda_resident::runtime_available(reason))
            {
                throw std::runtime_error("explicit CUDA-resident backend requested but resident runtime is unavailable: " + reason);
            }
            return;
        }
        if (backend == BackendKind::VulkanResidentTraining)
        {
            std::string reason;
            if (!vulkan_backend::runtime_available(reason))
            {
                throw std::runtime_error("explicit Vulkan-resident backend requested but resident runtime is unavailable: " + reason);
            }
            return;
        }
        throw std::runtime_error("unsupported MNIST backend requested");
    }

    inline TensorHandle build_logits_graph(Graph& graph, int64_t batch, const Weights& weights, BackendKind matmul_backend)
    {
        auto input = graph.placeholder({ batch, kInputFeatures }, DType::F32, "input");
        auto w1 = graph.placeholder({ kInputFeatures, weights.hidden }, DType::F32, "w1");
        auto b1 = graph.placeholder({ weights.hidden }, DType::F32, "b1");
        auto w2 = graph.placeholder({ weights.hidden, kClasses }, DType::F32, "w2");
        auto b2 = graph.placeholder({ kClasses }, DType::F32, "b2");
        const bool force_cpu = matmul_backend == BackendKind::Cpu;
        auto add = [&](TensorHandle lhs, TensorHandle rhs, const std::string& name)
            {
                return force_cpu ? graph.add(lhs, rhs, name, BackendKind::Cpu) : graph.add(lhs, rhs, name);
            };
        auto relu = [&](TensorHandle value, const std::string& name)
            {
                return force_cpu ? graph.relu(value, name, BackendKind::Cpu) : graph.relu(value, name);
            };

        auto dense1 = graph.matmul2d(input, w1, "dense1", matmul_backend);
        auto hidden = relu(add(dense1, b1, "bias1"), "relu1");
        auto dense2 = graph.matmul2d(hidden, w2, "dense2", matmul_backend);
        auto logits = add(dense2, b2, "logits");
        graph.output(logits);
        return logits;
    }

    inline InferenceGraphHandles build_resident_inference_graph(Graph& graph, int64_t batch, const Weights& weights, BackendKind resident_backend)
    {
        InferenceGraphHandles handles;
        handles.input = graph.placeholder({ batch, kInputFeatures }, DType::F32, "input");
        handles.w1 = graph.variable_f32({ kInputFeatures, weights.hidden }, weights.w1, "w1");
        handles.b1 = graph.variable_f32({ weights.hidden }, weights.b1, "b1");
        handles.w2 = graph.variable_f32({ weights.hidden, kClasses }, weights.w2, "w2");
        handles.b2 = graph.variable_f32({ kClasses }, weights.b2, "b2");

        auto dense1 = graph.matmul2d(handles.input, handles.w1, "dense1", resident_backend);
        auto hidden = graph.relu(graph.add(dense1, handles.b1, "bias1"), "relu1");
        auto dense2 = graph.matmul2d(hidden, handles.w2, "dense2", resident_backend);
        handles.logits = graph.add(dense2, handles.b2, "logits");
        graph.output(handles.logits);
        return handles;
    }

    inline RunResult run_mlp(
        const Dataset& dataset,
        const Weights& weights,
        BackendKind backend,
        size_t requested_samples,
        size_t batch_size,
        bool verbose_backend = false,
        bool verify_no_intermediate_transfer = false);

    inline TrainingGraphHandles build_training_graph(Graph& graph, int64_t batch, const Weights& weights, BackendKind matmul_backend)
    {
        TrainingGraphHandles handles;
        handles.input = graph.placeholder({ batch, kInputFeatures }, DType::F32, "input");
        handles.labels = graph.placeholder({ batch }, DType::F32, "labels");
        handles.w1 = graph.variable_f32({ kInputFeatures, weights.hidden }, weights.w1, "w1");
        handles.b1 = graph.variable_f32({ weights.hidden }, weights.b1, "b1");
        handles.w2 = graph.variable_f32({ weights.hidden, kClasses }, weights.w2, "w2");
        handles.b2 = graph.variable_f32({ kClasses }, weights.b2, "b2");
        const bool force_cpu = matmul_backend == BackendKind::Cpu;
        auto add = [&](TensorHandle lhs, TensorHandle rhs, const std::string& name)
            {
                return force_cpu ? graph.add(lhs, rhs, name, BackendKind::Cpu) : graph.add(lhs, rhs, name);
            };
        auto relu = [&](TensorHandle value, const std::string& name)
            {
                return force_cpu ? graph.relu(value, name, BackendKind::Cpu) : graph.relu(value, name);
            };
        auto cross_entropy = [&](TensorHandle pred, TensorHandle target, const std::string& name)
            {
                return force_cpu ? graph.cross_entropy(pred, target, name, BackendKind::Cpu) : graph.cross_entropy(pred, target, name);
            };

        auto dense1 = graph.matmul2d(handles.input, handles.w1, "dense1", matmul_backend);
        auto hidden = relu(add(dense1, handles.b1, "bias1"), "relu1");
        auto dense2 = graph.matmul2d(hidden, handles.w2, "dense2", matmul_backend);
        handles.logits = add(dense2, handles.b2, "logits");
        handles.loss = cross_entropy(handles.logits, handles.labels, "softmax_cross_entropy");
        graph.output(handles.loss);
        return handles;
    }

    inline std::vector<float> copy_image_batch(const Dataset& dataset, size_t offset, size_t batch)
    {
        const auto begin = dataset.images.begin() + static_cast<std::ptrdiff_t>(offset * (size_t)kInputFeatures);
        const auto end = begin + static_cast<std::ptrdiff_t>(batch * (size_t)kInputFeatures);
        return std::vector<float>(begin, end);
    }

    inline std::vector<float> copy_image_batch_padded(const Dataset& dataset, size_t offset, size_t actual, size_t padded_batch)
    {
        std::vector<float> images(padded_batch * (size_t)kInputFeatures, 0.0f);
        const auto begin = dataset.images.begin() + static_cast<std::ptrdiff_t>(offset * (size_t)kInputFeatures);
        const auto end = begin + static_cast<std::ptrdiff_t>(actual * (size_t)kInputFeatures);
        std::copy(begin, end, images.begin());
        return images;
    }

    inline std::vector<float> copy_label_index_batch(const Dataset& dataset, size_t offset, size_t batch)
    {
        std::vector<float> labels(batch);
        for (size_t i = 0; i < batch; ++i)
        {
            labels[i] = static_cast<float>(dataset.labels[offset + i]);
        }
        return labels;
    }

    inline std::vector<float> copy_image_batch_indexed(const Dataset& dataset, const std::vector<size_t>& indices, size_t offset, size_t batch)
    {
        std::vector<float> images(batch * (size_t)kInputFeatures);
        for (size_t i = 0; i < batch; ++i)
        {
            const size_t sample = indices[offset + i];
            const auto src_begin = dataset.images.begin() + static_cast<std::ptrdiff_t>(sample * (size_t)kInputFeatures);
            const auto src_end = src_begin + static_cast<std::ptrdiff_t>(kInputFeatures);
            std::copy(src_begin, src_end, images.begin() + static_cast<std::ptrdiff_t>(i * (size_t)kInputFeatures));
        }
        return images;
    }

    inline std::vector<float> copy_label_index_batch_indexed(const Dataset& dataset, const std::vector<size_t>& indices, size_t offset, size_t batch)
    {
        std::vector<float> labels(batch);
        for (size_t i = 0; i < batch; ++i)
        {
            labels[i] = static_cast<float>(dataset.labels[indices[offset + i]]);
        }
        return labels;
    }

    inline Weights read_trained_weights(RuntimeSession& session, const TrainingGraphHandles& handles, int64_t hidden, const std::string& description)
    {
        Weights weights;
        weights.hidden = hidden;
        session.get_variable_f32(handles.w1, weights.w1);
        session.get_variable_f32(handles.b1, weights.b1);
        session.get_variable_f32(handles.w2, weights.w2);
        session.get_variable_f32(handles.b2, weights.b2);
        weights.description = description;
        weights.loaded_from_file = false;
        weights.untrained_smoke = false;
        validate_weight_shapes(weights);
        return weights;
    }

    inline RunResult run_training_session_mlp(
        RuntimeSession& session,
        const TrainingGraphHandles& handles,
        const Dataset& dataset,
        size_t requested_samples,
        size_t batch_size)
    {
        if (batch_size == 0)
        {
            throw std::runtime_error("batch size must be positive");
        }
        const size_t available = dataset.sample_count();
        const size_t requested = std::min(requested_samples == 0 ? available : requested_samples, available);
        const size_t samples = requested - (requested % batch_size);

        RunResult result;
        result.samples = samples;
        result.batch_size = batch_size;
        result.logits.reserve(samples * (size_t)kClasses);

        const auto start = std::chrono::steady_clock::now();
        for (size_t offset = 0; offset < samples; offset += batch_size)
        {
            RuntimeSession::FetchDict fetched;
            session.run(
                {
                    { handles.input, copy_image_batch(dataset, offset, batch_size) },
                },
                { handles.logits },
                {},
                fetched);
            std::vector<float> batch_logits = tensor_to_vec_f32(fetched.at(handles.logits));
            result.logits.insert(result.logits.end(), batch_logits.begin(), batch_logits.end());
        }
        const auto end = std::chrono::steady_clock::now();
        result.elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

        result.predictions = argmax_rows(result.logits, samples);
        if (dataset.has_labels)
        {
            result.has_accuracy = true;
            for (size_t i = 0; i < samples; ++i)
            {
                if (result.predictions[i] == static_cast<int>(dataset.labels[i]))
                {
                    result.correct += 1;
                }
            }
        }
        return result;
    }

    inline TrainingResult train_mlp(
        const Dataset& dataset,
        BackendKind backend,
        const TrainingOptions& options)
    {
        throw_if_backend_unavailable(backend);
        if (!dataset.has_labels)
        {
            throw std::runtime_error("MNIST training requires labels");
        }
        if (options.epochs == 0)
        {
            throw std::runtime_error("epochs must be positive");
        }
        if (options.batch_size == 0)
        {
            throw std::runtime_error("batch size must be positive");
        }
        if (!(options.learning_rate > 0.0f))
        {
            throw std::runtime_error("learning rate must be positive");
        }

        const size_t available = dataset.sample_count();
        if (available == 0)
        {
            throw std::runtime_error("dataset contains no samples");
        }

        const size_t requested = std::min(options.max_samples == 0 ? available : options.max_samples, available);
        if (requested == 0)
        {
            throw std::runtime_error("training sample count must be positive");
        }

        const size_t batch_size = std::min(options.batch_size, requested);
        const size_t usable_samples = requested - (requested % batch_size);
        if (usable_samples == 0)
        {
            throw std::runtime_error("training sample count is smaller than one mini-batch");
        }

        Weights initial = options.initial_weights != nullptr
            ? *options.initial_weights
            : make_training_initial_weights(options.hidden, options.seed);
        validate_weight_shapes(initial);

        std::vector<size_t> order(requested);
        std::iota(order.begin(), order.end(), (size_t)0);
        std::mt19937 shuffle_rng(options.seed);

        Graph graph;
        const TrainingGraphHandles handles = build_training_graph(
            graph,
            static_cast<int64_t>(batch_size),
            initial,
            backend);
        RuntimeSession session(graph.compile());
        session.set_backend_diagnostics_enabled(options.verbose_backend);
        if (backend == BackendKind::CudaResident)
        {
            session.set_execution_policy(RuntimeExecutionPolicy::CudaResident);
        }
        else if (backend == BackendKind::VulkanResidentTraining)
        {
            session.set_execution_policy(RuntimeExecutionPolicy::VulkanResidentTraining);
        }
        Optimizer optimizer = Optimizer::sgd(options.learning_rate);

        TrainingResult result;
        result.used_cuda_cublas = backend == BackendKind::Cuda;
        result.used_cuda_resident = backend == BackendKind::CudaResident;
        result.used_vulkan_resident = backend == BackendKind::VulkanResidentTraining;
        result.trained_samples = usable_samples;
        result.batch_size = batch_size;
        result.epochs.reserve(options.epochs);

        const auto start = std::chrono::steady_clock::now();
        bool saw_loss = false;
        for (size_t epoch = 0; epoch < options.epochs; ++epoch)
        {
            std::shuffle(order.begin(), order.end(), shuffle_rng);
            double epoch_loss = 0.0;
            size_t epoch_steps = 0;
            for (size_t offset = 0; offset < usable_samples; offset += batch_size)
            {
                RuntimeSession::FeedDictF32 feeds;
                feeds.emplace(handles.input, copy_image_batch_indexed(dataset, order, offset, batch_size));
                feeds.emplace(handles.labels, copy_label_index_batch_indexed(dataset, order, offset, batch_size));

                float loss = 0.0f;
                const RuntimeSession::DebugStats stats_before = session.debug_stats();
                const bool first_resident_step = (result.used_cuda_resident || result.used_vulkan_resident) && result.steps == 0;
                training_ops::train_step(
                    session,
                    optimizer,
                    feeds,
                    handles.loss,
                    { handles.w1, handles.b1, handles.w2, handles.b2 },
                    &loss);
                if (result.used_cuda_resident || result.used_vulkan_resident)
                {
                    const RuntimeSession::DebugStats stats_after = session.debug_stats();
                    const size_t upload_delta = result.used_cuda_resident
                        ? stats_after.cuda_upload_count - stats_before.cuda_upload_count
                        : stats_after.vulkan_upload_count - stats_before.vulkan_upload_count;
                    const size_t download_delta = result.used_cuda_resident
                        ? stats_after.cuda_download_count - stats_before.cuda_download_count
                        : stats_after.vulkan_download_count - stats_before.vulkan_download_count;
                    const size_t uploaded_bytes_delta = result.used_cuda_resident
                        ? stats_after.cuda_uploaded_bytes - stats_before.cuda_uploaded_bytes
                        : stats_after.vulkan_uploaded_bytes - stats_before.vulkan_uploaded_bytes;
                    const size_t downloaded_bytes_delta = result.used_cuda_resident
                        ? stats_after.cuda_downloaded_bytes - stats_before.cuda_downloaded_bytes
                        : stats_after.vulkan_downloaded_bytes - stats_before.vulkan_downloaded_bytes;
                    if (result.used_cuda_resident)
                    {
                        result.training_cuda_upload_count += upload_delta;
                        result.training_cuda_download_count += download_delta;
                        result.training_cuda_uploaded_bytes += uploaded_bytes_delta;
                        result.training_cuda_downloaded_bytes += downloaded_bytes_delta;
                    }
                    else
                    {
                        result.training_vulkan_upload_count += upload_delta;
                        result.training_vulkan_download_count += download_delta;
                        result.training_vulkan_uploaded_bytes += uploaded_bytes_delta;
                        result.training_vulkan_downloaded_bytes += downloaded_bytes_delta;
                        result.training_vulkan_intermediate_download_count +=
                            stats_after.vulkan_activation_logit_gradient_download_count -
                            stats_before.vulkan_activation_logit_gradient_download_count;
                        result.training_vulkan_unknown_transfer_count +=
                            stats_after.vulkan_unknown_transfer_count -
                            stats_before.vulkan_unknown_transfer_count;
                    }

                    if (options.verify_no_intermediate_transfer)
                    {
                        const char* backend_name = result.used_cuda_resident ? "cuda-resident" : "vulkan-resident";
                        const size_t expected_uploads = first_resident_step ? 6u : 2u;
                        const size_t expected_upload_bytes =
                            sizeof(float) * batch_size * (size_t)kInputFeatures +
                            sizeof(float) * batch_size +
                            (first_resident_step ? sizeof(float) * weight_value_count(initial) : 0u);
                        if (upload_delta != expected_uploads)
                        {
                            throw std::runtime_error(
                                std::string(backend_name) + " transfer verification failed: expected " +
                                std::to_string(expected_uploads) +
                                " uploads in training step, got " + std::to_string(upload_delta));
                        }
                        if (uploaded_bytes_delta != expected_upload_bytes)
                        {
                            throw std::runtime_error(
                                std::string(backend_name) + " transfer verification failed: expected " +
                                std::to_string(expected_upload_bytes) +
                                " uploaded bytes in training step, got " +
                                std::to_string(uploaded_bytes_delta));
                        }
                        if (download_delta != 1u)
                        {
                            throw std::runtime_error(
                                std::string(backend_name) + " transfer verification failed: expected only explicit scalar loss download, got " +
                                std::to_string(download_delta) + " downloads");
                        }
                        if (downloaded_bytes_delta != sizeof(float))
                        {
                            throw std::runtime_error(
                                std::string(backend_name) + " transfer verification failed: expected scalar loss download of " +
                                std::to_string(sizeof(float)) + " bytes, got " +
                                std::to_string(downloaded_bytes_delta));
                        }
                        if (result.used_vulkan_resident)
                        {
                            if (stats_after.vulkan_activation_logit_gradient_download_count != stats_before.vulkan_activation_logit_gradient_download_count)
                            {
                                throw std::runtime_error("vulkan-resident transfer verification failed: hidden activation/logit/gradient download occurred");
                            }
                            if (stats_after.vulkan_unknown_transfer_count != stats_before.vulkan_unknown_transfer_count)
                            {
                                throw std::runtime_error("vulkan-resident transfer verification failed: unknown hidden transfer occurred");
                            }
                        }
                    }
                }

                if (!saw_loss)
                {
                    result.initial_loss = loss;
                    saw_loss = true;
                }
                result.final_loss = loss;
                epoch_loss += (double)loss;
                epoch_steps += 1;
                result.steps += 1;
            }

            const std::string desc = "trained MNIST MLP weights after epoch " + std::to_string(epoch + 1);
            const size_t eval_count = std::min(options.eval_samples == 0 ? usable_samples : options.eval_samples, usable_samples);
            const RunResult eval = (result.used_cuda_resident || result.used_vulkan_resident)
                ? run_training_session_mlp(session, handles, dataset, eval_count, batch_size)
                : run_mlp(dataset, read_trained_weights(session, handles, initial.hidden, desc), backend, eval_count, batch_size, options.verbose_backend);

            TrainingEpochResult epoch_result;
            epoch_result.epoch = epoch + 1;
            epoch_result.average_loss = epoch_steps == 0 ? 0.0f : static_cast<float>(epoch_loss / (double)epoch_steps);
            epoch_result.correct = eval.correct;
            epoch_result.eval_samples = eval.samples;
            epoch_result.accuracy = eval.samples == 0
                ? 0.0
                : static_cast<double>(eval.correct) / static_cast<double>(eval.samples);
            result.epochs.push_back(epoch_result);
        }

        const auto end = std::chrono::steady_clock::now();
        result.elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        if (result.used_cuda_resident)
        {
            result.training_cuda_batch_image_upload_count = result.steps;
            result.training_cuda_batch_image_uploaded_bytes =
                result.steps * batch_size * (size_t)kInputFeatures * sizeof(float);
            result.training_cuda_label_upload_count = result.steps;
            result.training_cuda_label_uploaded_bytes = result.steps * batch_size * sizeof(float);
            result.training_cuda_weight_upload_count = result.steps == 0 ? 0u : 4u;
            result.training_cuda_weight_uploaded_bytes = result.steps == 0 ? 0u : weight_value_count(initial) * sizeof(float);
            result.training_cuda_scalar_loss_download_count = result.training_cuda_download_count;
            result.training_cuda_scalar_loss_downloaded_bytes = result.training_cuda_downloaded_bytes;
        }
        if (result.used_vulkan_resident)
        {
            result.training_vulkan_batch_image_upload_count = result.steps;
            result.training_vulkan_batch_image_uploaded_bytes =
                result.steps * batch_size * (size_t)kInputFeatures * sizeof(float);
            result.training_vulkan_label_upload_count = result.steps;
            result.training_vulkan_label_uploaded_bytes = result.steps * batch_size * sizeof(float);
            result.training_vulkan_weight_upload_count = result.steps == 0 ? 0u : 4u;
            result.training_vulkan_weight_uploaded_bytes = result.steps == 0 ? 0u : weight_value_count(initial) * sizeof(float);
            result.training_vulkan_scalar_loss_download_count = result.training_vulkan_download_count;
            result.training_vulkan_scalar_loss_downloaded_bytes = result.training_vulkan_downloaded_bytes;
        }
        if (options.download_final_weights)
        {
            const RuntimeSession::DebugStats stats_before_final_download = session.debug_stats();
            result.weights = read_trained_weights(
                session,
                handles,
                initial.hidden,
                "trained MNIST MLP weights, epochs=" + std::to_string(options.epochs));
            if (result.used_cuda_resident)
            {
                const RuntimeSession::DebugStats stats_after_final_download = session.debug_stats();
                result.final_weight_download_count =
                    stats_after_final_download.cuda_download_count - stats_before_final_download.cuda_download_count;
                result.final_weight_downloaded_bytes =
                    stats_after_final_download.cuda_downloaded_bytes - stats_before_final_download.cuda_downloaded_bytes;
            }
            else if (result.used_vulkan_resident)
            {
                const RuntimeSession::DebugStats stats_after_final_download = session.debug_stats();
                result.final_weight_download_count =
                    stats_after_final_download.vulkan_download_count - stats_before_final_download.vulkan_download_count;
                result.final_weight_downloaded_bytes =
                    stats_after_final_download.vulkan_downloaded_bytes - stats_before_final_download.vulkan_downloaded_bytes;
            }
            result.weights_downloaded = true;
        }
        else
        {
            result.weights.hidden = initial.hidden;
            result.weights.description = result.used_vulkan_resident
                ? "trained MNIST MLP weights remain Vulkan-resident; final download was not requested"
                : "trained MNIST MLP weights remain CUDA-resident; final download was not requested";
            result.weights.loaded_from_file = false;
            result.weights.untrained_smoke = false;
        }
        result.final_debug_stats = session.debug_stats();
        return result;
    }

    inline RunResult run_mlp(
        const Dataset& dataset,
        const Weights& weights,
        BackendKind backend,
        size_t requested_samples,
        size_t batch_size,
        bool verbose_backend,
        bool verify_no_intermediate_transfer)
    {
        throw_if_backend_unavailable(backend);
        if (batch_size == 0)
        {
            throw std::runtime_error("batch size must be positive");
        }
        const size_t available = dataset.sample_count();
        if (available == 0)
        {
            throw std::runtime_error("dataset contains no samples");
        }
        const size_t samples = std::min(requested_samples == 0 ? available : requested_samples, available);
        if (samples == 0)
        {
            throw std::runtime_error("sample count must be positive");
        }

        RunResult result;
        result.samples = samples;
        result.batch_size = batch_size;
        result.used_cuda_resident = backend == BackendKind::CudaResident;
        result.used_vulkan_resident = backend == BackendKind::VulkanResidentTraining;
        result.logits.reserve(samples * (size_t)kClasses);

        const auto start = std::chrono::steady_clock::now();
        if (backend == BackendKind::CudaResident || backend == BackendKind::VulkanResidentTraining)
        {
            auto run_resident_batch_group = [&](size_t group_batch_size, size_t first_offset, size_t group_batches) -> void
                {
                    if (group_batches == 0)
                    {
                        return;
                    }

                    Graph graph;
                    const InferenceGraphHandles handles = build_resident_inference_graph(
                        graph,
                        static_cast<int64_t>(group_batch_size),
                        weights,
                        backend);
                    RuntimeSession session(graph.compile());
                    session.set_execution_policy(backend == BackendKind::CudaResident
                        ? RuntimeExecutionPolicy::CudaResident
                        : RuntimeExecutionPolicy::VulkanResidentTraining);
                    session.set_backend_diagnostics_enabled(verbose_backend);

                    for (size_t batch_index = 0; batch_index < group_batches; ++batch_index)
                    {
                        const size_t offset = first_offset + batch_index * group_batch_size;
                        const size_t actual_batch = std::min(group_batch_size, samples - offset);
                        const RuntimeSession::DebugStats stats_before = session.debug_stats();

                        RuntimeSession::FetchDict fetched;
                        session.run(
                            {
                                { handles.input, actual_batch == group_batch_size
                                    ? copy_image_batch(dataset, offset, group_batch_size)
                                    : copy_image_batch_padded(dataset, offset, actual_batch, group_batch_size) },
                            },
                            { handles.logits },
                            {},
                            fetched);

                        const RuntimeSession::DebugStats stats_after = session.debug_stats();
                        const size_t upload_delta = backend == BackendKind::CudaResident
                            ? stats_after.cuda_upload_count - stats_before.cuda_upload_count
                            : stats_after.vulkan_upload_count - stats_before.vulkan_upload_count;
                        const size_t download_delta = backend == BackendKind::CudaResident
                            ? stats_after.cuda_download_count - stats_before.cuda_download_count
                            : stats_after.vulkan_download_count - stats_before.vulkan_download_count;
                        const size_t uploaded_bytes_delta = backend == BackendKind::CudaResident
                            ? stats_after.cuda_uploaded_bytes - stats_before.cuda_uploaded_bytes
                            : stats_after.vulkan_uploaded_bytes - stats_before.vulkan_uploaded_bytes;
                        const size_t downloaded_bytes_delta = backend == BackendKind::CudaResident
                            ? stats_after.cuda_downloaded_bytes - stats_before.cuda_downloaded_bytes
                            : stats_after.vulkan_downloaded_bytes - stats_before.vulkan_downloaded_bytes;
                        if (backend == BackendKind::CudaResident)
                        {
                            result.cuda_upload_count += upload_delta;
                            result.cuda_download_count += download_delta;
                            result.cuda_uploaded_bytes += uploaded_bytes_delta;
                            result.cuda_downloaded_bytes += downloaded_bytes_delta;
                        }
                        else
                        {
                            result.vulkan_upload_count += upload_delta;
                            result.vulkan_download_count += download_delta;
                            result.vulkan_uploaded_bytes += uploaded_bytes_delta;
                            result.vulkan_downloaded_bytes += downloaded_bytes_delta;
                            result.vulkan_dispatch_count += stats_after.vulkan_dispatch_count - stats_before.vulkan_dispatch_count;
                            result.vulkan_pipeline_cache_hits += stats_after.vulkan_pipeline_cache_hits - stats_before.vulkan_pipeline_cache_hits;
                            result.vulkan_pipeline_cache_misses += stats_after.vulkan_pipeline_cache_misses - stats_before.vulkan_pipeline_cache_misses;
                            result.vulkan_descriptor_cache_hits += stats_after.vulkan_descriptor_cache_hits - stats_before.vulkan_descriptor_cache_hits;
                            result.vulkan_descriptor_cache_misses += stats_after.vulkan_descriptor_cache_misses - stats_before.vulkan_descriptor_cache_misses;
                            result.vulkan_explicit_synchronization_count +=
                                stats_after.vulkan_explicit_synchronization_count - stats_before.vulkan_explicit_synchronization_count;
                        }

                        if (verify_no_intermediate_transfer)
                        {
                            const char* resident_name = backend == BackendKind::CudaResident ? "cuda-resident" : "vulkan-resident";
                            const size_t expected_uploads = batch_index == 0 ? 5u : 1u;
                            const size_t expected_upload_bytes = sizeof(float) * group_batch_size * (size_t)kInputFeatures +
                                (batch_index == 0
                                    ? sizeof(float) * (weights.w1.size() + weights.b1.size() + weights.w2.size() + weights.b2.size())
                                    : 0u);
                            const size_t expected_download_bytes = sizeof(float) * group_batch_size * (size_t)kClasses;
                            if (upload_delta != expected_uploads)
                            {
                                throw std::runtime_error(
                                    std::string(resident_name) + " inference transfer verification failed: expected " +
                                    std::to_string(expected_uploads) + " uploads, got " +
                                    std::to_string(upload_delta));
                            }
                            if (uploaded_bytes_delta != expected_upload_bytes)
                            {
                                throw std::runtime_error(
                                    std::string(resident_name) + " inference transfer verification failed: expected " +
                                    std::to_string(expected_upload_bytes) + " uploaded bytes, got " +
                                    std::to_string(uploaded_bytes_delta));
                            }
                            if (download_delta != 1u)
                            {
                                throw std::runtime_error(
                                    std::string(resident_name) + " inference transfer verification failed: expected only explicit logits download, got " +
                                    std::to_string(download_delta) + " downloads");
                            }
                            if (downloaded_bytes_delta != expected_download_bytes)
                            {
                                throw std::runtime_error(
                                    std::string(resident_name) + " inference transfer verification failed: expected logits download of " +
                                    std::to_string(expected_download_bytes) + " bytes, got " +
                                    std::to_string(downloaded_bytes_delta));
                            }
                            if (backend == BackendKind::VulkanResidentTraining)
                            {
                                if (stats_after.vulkan_activation_logit_gradient_download_count != stats_before.vulkan_activation_logit_gradient_download_count)
                                {
                                    throw std::runtime_error("vulkan-resident inference transfer verification failed: hidden activation/logit/gradient download occurred");
                                }
                                if (stats_after.vulkan_unknown_transfer_count != stats_before.vulkan_unknown_transfer_count)
                                {
                                    throw std::runtime_error("vulkan-resident inference transfer verification failed: unknown hidden transfer occurred");
                                }
                            }
                        }

                        std::vector<float> batch_logits = tensor_to_vec_f32(fetched.at(handles.logits));
                        result.logits.insert(
                            result.logits.end(),
                            batch_logits.begin(),
                            batch_logits.begin() + static_cast<std::ptrdiff_t>(actual_batch * (size_t)kClasses));
                    }
                };

            const size_t resident_batches = (samples + batch_size - 1) / batch_size;
            run_resident_batch_group(batch_size, 0, resident_batches);
        }
        else
        {
            for (size_t offset = 0; offset < samples; offset += batch_size)
            {
                const size_t current_batch = std::min(batch_size, samples - offset);
                Graph graph;
                TensorHandle logits = build_logits_graph(graph, static_cast<int64_t>(current_batch), weights, backend);
                RuntimeSession session(graph.compile());

                const auto input_begin = dataset.images.begin() + static_cast<std::ptrdiff_t>(offset * (size_t)kInputFeatures);
                const auto input_end = input_begin + static_cast<std::ptrdiff_t>(current_batch * (size_t)kInputFeatures);
                std::vector<float> input_values(input_begin, input_end);

                RuntimeSession::FetchDict fetched;
                session.set_backend_diagnostics_enabled(verbose_backend);
                session.run(
                    {
                        { TensorHandle{0}, input_values },
                        { TensorHandle{1}, weights.w1 },
                        { TensorHandle{2}, weights.b1 },
                        { TensorHandle{3}, weights.w2 },
                        { TensorHandle{4}, weights.b2 },
                    },
                    fetched);
                std::vector<float> batch_logits = tensor_to_vec_f32(fetched.at(logits));
                result.logits.insert(result.logits.end(), batch_logits.begin(), batch_logits.end());
            }
        }
        const auto end = std::chrono::steady_clock::now();
        result.elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

        result.predictions = argmax_rows(result.logits, samples);
        if (dataset.has_labels)
        {
            result.has_accuracy = true;
            for (size_t i = 0; i < samples; ++i)
            {
                if (result.predictions[i] == static_cast<int>(dataset.labels[i]))
                {
                    result.correct += 1;
                }
            }
        }
        return result;
    }

    inline Difference compare_results(const RunResult& lhs, const RunResult& rhs)
    {
        if (lhs.logits.size() != rhs.logits.size() || lhs.predictions.size() != rhs.predictions.size())
        {
            throw std::runtime_error("cannot compare MNIST results with different output sizes");
        }

        Difference diff;
        double sum_abs = 0.0;
        for (size_t i = 0; i < lhs.logits.size(); ++i)
        {
            const float abs_diff = std::fabs(lhs.logits[i] - rhs.logits[i]);
            diff.max_abs = std::max(diff.max_abs, abs_diff);
            sum_abs += abs_diff;
            const float denom = std::max(std::fabs(lhs.logits[i]), 1e-6f);
            diff.max_rel = std::max(diff.max_rel, abs_diff / denom);
        }
        diff.mean_abs = lhs.logits.empty() ? 0.0f : static_cast<float>(sum_abs / static_cast<double>(lhs.logits.size()));
        for (size_t i = 0; i < lhs.predictions.size(); ++i)
        {
            if (lhs.predictions[i] != rhs.predictions[i])
            {
                diff.mismatched_predictions += 1;
            }
        }
        return diff;
    }
}
