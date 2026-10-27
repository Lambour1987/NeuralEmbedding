//27-10-2026


#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <utility>
#include <vector>

#include "Tensor.h"
#include "TensorOps.h"

void benchmarkMatmul(std::size_t n, int repetitions)
{
    if (repetitions <= 0) {
        std::cerr << "Aantal herhalingen moet groter zijn dan 0.\n";
        return;
    }

    // Maak de matrices buiten de tijdmeting.
    std::vector<float> dataA(n * n, 0.5f);
    std::vector<float> dataB(n * n, 0.25f);

    Tensor a(std::move(dataA), {n, n});
    Tensor b(std::move(dataB), {n, n});

    // Warm-up: eerste uitvoering niet meetellen.
    Tensor warmup = matmul(a, b);
    volatile float sink = warmup.data()[0];
    (void)sink;

    double totalMilliseconds = 0.0;
    float checksum = 0.0f;

    for (int r = 0; r < repetitions; ++r) {
        const auto start = std::chrono::steady_clock::now();

        Tensor result = matmul(a, b);

        const auto end = std::chrono::steady_clock::now();

        totalMilliseconds +=
            std::chrono::duration<double, std::milli>(
                end - start
            ).count();

        checksum += result.data()[0];
    }

    const double averageMilliseconds =
        totalMilliseconds / repetitions;

    std::cout << "Matrix: " << n << " x " << n
              << " | Gemiddeld: "
              << std::fixed << std::setprecision(3)
              << averageMilliseconds << " ms"
              << " | Checksum: " << checksum
              << '\n';
}

int main()
{
    std::cout << "=== Matmul Benchmark ===\n";

    benchmarkMatmul(64, 3);
    benchmarkMatmul(128, 3);
    benchmarkMatmul(256, 3);

    return 0;
}
