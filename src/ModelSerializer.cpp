//27-11-26


#include "ModelSerializer.h"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <vector>

#include <utility>

namespace {

constexpr char MAGIC[8] = {
    'N', 'N', 'C', 'K', 'P', 'T', '0', '1'
};

constexpr std::uint32_t FORMAT_VERSION = 1;

constexpr std::uint64_t MAX_LAYERS = 1024;
constexpr std::uint32_t MAX_TENSOR_RANK = 8;
constexpr std::uint64_t MAX_TENSOR_ELEMENTS = 10'000'000;

// Schrijf een vaste hoeveelheid bytes.
template <typename T>
void write_value(std::ostream& stream, const T& value) {
    stream.write(
        reinterpret_cast<const char*>(&value),
        static_cast<std::streamsize>(sizeof(T))
    );

    if (!stream) {
        throw std::runtime_error(
            "Failed to write checkpoint"
        );
    }
}

// Lees een vaste hoeveelheid bytes.
template <typename T>
T read_value(std::istream& stream) {
    T value{};

    stream.read(
        reinterpret_cast<char*>(&value),
        static_cast<std::streamsize>(sizeof(T))
    );

    if (!stream) {
        throw std::runtime_error(
            "Checkpoint is incomplete or corrupted"
        );
    }

    return value;
}

void write_tensor(
    std::ostream& stream,
    const Tensor& tensor
) {
    const auto& shape = tensor.shape();
    const auto& data = tensor.data();

    if (shape.empty() || shape.size() > MAX_TENSOR_RANK) {
        throw std::invalid_argument(
            "Tensor rank is not supported"
        );
    }

    if (data.size() > MAX_TENSOR_ELEMENTS) {
        throw std::invalid_argument(
            "Tensor contains too many elements"
        );
    }

    write_value(
        stream,
        static_cast<std::uint32_t>(shape.size())
    );

    for (std::size_t dimension : shape) {
        write_value(
            stream,
            static_cast<std::uint64_t>(dimension)
        );
    }

    for (float value : data) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                "Cannot save non-finite tensor values"
            );
        }

        write_value(stream, value);
    }
}

Tensor read_tensor(std::istream& stream) {
    const std::uint32_t rank =
        read_value<std::uint32_t>(stream);

    if (rank == 0 || rank > MAX_TENSOR_RANK) {
        throw std::runtime_error(
            "Invalid tensor rank in checkpoint"
        );
    }

    std::vector<std::size_t> shape;
    shape.reserve(rank);

    std::uint64_t element_count = 1;

    for (std::uint32_t i = 0; i < rank; ++i) {
        const std::uint64_t dimension =
            read_value<std::uint64_t>(stream);

        if (dimension == 0 ||
            dimension > MAX_TENSOR_ELEMENTS ||
            element_count > MAX_TENSOR_ELEMENTS / dimension) {
            throw std::runtime_error(
                "Invalid tensor dimensions in checkpoint"
            );
        }

        element_count *= dimension;

        shape.push_back(
            static_cast<std::size_t>(dimension)
        );
    }

    std::vector<float> data;
    data.reserve(static_cast<std::size_t>(element_count));

    for (std::uint64_t i = 0; i < element_count; ++i) {
        const float value = read_value<float>(stream);

        if (!std::isfinite(value)) {
            throw std::runtime_error(
                "Checkpoint contains non-finite tensor values"
            );
        }

        data.push_back(value);
    }

    return Tensor(std::move(data), std::move(shape));
}

} // namespace

void ModelSerializer::save(
    const Sequential& network,
    const std::string& filename
) const {
    if (network.size() == 0 ||
        network.size() > MAX_LAYERS) {
        throw std::invalid_argument(
            "Network has an unsupported number of layers"
        );
    }

    std::ofstream file(
        filename,
        std::ios::binary | std::ios::trunc
    );

    if (!file) {
        throw std::runtime_error(
            "Could not open checkpoint file for writing"
        );
    }

    file.write(MAGIC, sizeof(MAGIC));

    if (!file) {
        throw std::runtime_error(
            "Failed to write checkpoint header"
        );
    }

    write_value(file, FORMAT_VERSION);

    write_value(
        file,
        static_cast<std::uint64_t>(network.size())
    );

    for (std::size_t i = 0; i < network.size(); ++i) {
        const LayerType type = network.layer_type(i);

        write_value(
            file,
            static_cast<std::uint8_t>(type)
        );

        write_value(file, network.negative_slope(i));

        write_tensor(file, network.weights(i));
        write_tensor(file, network.bias(i));
    }

    file.flush();

    if (!file) {
        throw std::runtime_error(
            "Failed to finish writing checkpoint"
        );
    }
}

void ModelSerializer::load(
    Sequential& network,
    const std::string& filename
) const {
    std::ifstream file(
        filename,
        std::ios::binary
    );

    if (!file) {
        throw std::runtime_error(
            "Could not open checkpoint file"
        );
    }

    char magic[sizeof(MAGIC)]{};

    file.read(magic, sizeof(magic));

    if (!file || magic[0] != MAGIC[0] ||
        magic[1] != MAGIC[1] ||
        magic[2] != MAGIC[2] ||
        magic[3] != MAGIC[3] ||
        magic[4] != MAGIC[4] ||
        magic[5] != MAGIC[5] ||
        magic[6] != MAGIC[6] ||
        magic[7] != MAGIC[7]) {
        throw std::runtime_error(
            "Invalid checkpoint file header"
        );
    }

    const std::uint32_t version =
        read_value<std::uint32_t>(file);

    if (version != FORMAT_VERSION) {
        throw std::runtime_error(
            "Unsupported checkpoint format version"
        );
    }

    const std::uint64_t layer_count =
        read_value<std::uint64_t>(file);

    if (layer_count == 0 || layer_count > MAX_LAYERS) {
        throw std::runtime_error(
            "Invalid layer count in checkpoint"
        );
    }

    std::vector<LayerParameters> parameters;
    parameters.reserve(static_cast<std::size_t>(layer_count));

    for (std::uint64_t i = 0; i < layer_count; ++i) {
        const std::uint8_t raw_type =
            read_value<std::uint8_t>(file);

        if (raw_type >
            static_cast<std::uint8_t>(LayerType::LeakyReLU)) {
            throw std::runtime_error(
                "Invalid layer type in checkpoint"
            );
        }

        const LayerType type =
            static_cast<LayerType>(raw_type);

        const float slope = read_value<float>(file);

        if (!std::isfinite(slope)) {
            throw std::runtime_error(
                "Invalid activation slope in checkpoint"
            );
        }

        Tensor weights = read_tensor(file);
        Tensor bias = read_tensor(file);

        parameters.push_back({
            type,
            std::move(weights),
            std::move(bias),
            slope
        });
    }

    // Extra bytes wijzen op een ongeldig bestandsformaat.
    if (file.peek() != std::char_traits<char>::eof()) {
        throw std::runtime_error(
            "Checkpoint contains unexpected trailing data"
        );
    }

    // Pas parameters pas toe nadat het volledige bestand
    // succesvol is gelezen en gecontroleerd.
    network.set_parameters(parameters);
}
