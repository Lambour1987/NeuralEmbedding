//27-10-2026: Nog geen gebruik gemaakt van externe bibliotheken zoals 'Eigen' en 'oneDNN'
// Later ook'cuBLAS' gaan gebruiken: geoptimaliseerde lineaire algebra voor NVIDIA-GPU's

#include <cstddef>
#include <exception>
#include <iostream>
#include <utility>
#include <vector>
#include <cassert>
#include <iomanip>

//27-10-26: Include Chrono
#include <chrono>

#include "Tensor.h"
#include "TensorOps.h"

void print_tensor_info(const Tensor& tensor)
{
    std::cout << "Shape:   ";

    for (const auto dim : tensor.shape()) {
        std::cout << dim << ' ';
    }

    std::cout << "\nStrides: ";

    for (const auto stride : tensor.strides()) {
        std::cout << stride << ' ';
    }

    std::cout << "\n";
}

//27-10-2026: Benchmark Functie toegevoegd

void benchmarkMatmul(std::size_t n, int repetitions)
{
    // Maak twee n x n matrices buiten de meting.
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

        // Gebruik de output ook buiten de tijdmeting.
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
    try {
        // 1D: een vector met 4 getallen
        Tensor a({10, 20, 30, 40}, {4});

        std::cout << "=== 1D Tensor ===\n";
        print_tensor_info(a);
        std::cout << "a.at({2}) = " << a.at({2}) << "\n\n";

        // 2D: 3 rijen, 4 kolommen
        Tensor b(
            {1, 2, 3, 4,
             5, 6, 7, 8,
             9, 10, 11, 12},
            {3, 4}
        );

        std::cout << "=== 2D Tensor ===\n";
        print_tensor_info(b);
        std::cout << "b.at({1, 2}) = "
                  << b.at({1, 2}) << "\n\n";

        // 3D: 2 blokken, 3 rijen, 4 kolommen
        std::vector<float> data(24);

        for (std::size_t i = 0; i < data.size(); ++i) {
            data[i] = static_cast<float>(i + 1);
        }

        Tensor c(std::move(data), {2, 3, 4});

        std::cout << "=== 3D Tensor ===\n";
        print_tensor_info(c);
        std::cout << "c.at({1, 2, 3}) = "
                  << c.at({1, 2, 3}) << "\n\n";

        // Ongeldige index: moet een exception geven
        std::cout << "=== Grenscontrole ===\n";

        
        std::cout << "\n=== Tensor Addition ===\n";

        Tensor x({1, 2, 3, 4}, {2, 2});
        Tensor y({10, 20, 30, 40}, {2, 2});

        Tensor z = add(x, y);

        std::cout << "z.at({0, 0}) = " << z.at({0, 0}) << '\n';
        std::cout << "z.at({0, 1}) = " << z.at({0, 1}) << '\n';
        std::cout << "z.at({1, 0}) = " << z.at({1, 0}) << '\n';
        std::cout << "z.at({1, 1}) = " << z.at({1, 1}) << '\n';

        try {
            std::cout << c.at({2, 0, 0}) << '\n';
        }
        catch (const std::out_of_range& e) {
            std::cout << "Correct afgevangen: "
                      << e.what() << '\n';
        }
        std::cout << "\n=== Shape validatie ===\n";

        
        std::cout << "\n=== Tensor Subtraction ===\n";

        Tensor difference = subtract(y, x);

        std::cout << "difference.at({0, 0}) = "
                << difference.at({0, 0}) << '\n';

        std::cout << "\n=== Element-wise Multiply ===\n";

        Tensor product = multiply(x, y);

        std::cout << "product.at({0, 0}) = "
                << product.at({0, 0}) << '\n';
        std::cout << "product.at({1, 1}) = "
                << product.at({1, 1}) << '\n';


    try {
        Tensor p({1, 2, 3, 4}, {2, 2});
        Tensor q({10, 20, 30}, {3});

        Tensor r = add(p, q);

        std::cout << "Onverwacht resultaat!\n";
    }
    catch (const std::invalid_argument& e) {
        std::cout << "Correct afgevangen: "
                << e.what() << '\n';
}

    }
    catch (const std::exception& e) {
        std::cerr << "Test mislukt: " << e.what() << '\n';
        return 1;
    }

    //27-10-2026: testen toegevoegd
    
    Tensor a(
        {1, 2, 3,
        4, 5, 6},
        {2, 3}
    );

    Tensor b(
        {7, 8,
        9, 10,
        11, 12},
        {3, 2}
    );

    Tensor c = matmul(a, b);

    std::cout << "Result shape: {"
            << c.shape()[0] << ", "
            << c.shape()[1] << "}\n";

    for (std::size_t i = 0; i < c.shape()[0]; ++i) {
        for (std::size_t j = 0; j < c.shape()[1]; ++j) {
            std::cout << c.at({i, j}) << " ";
        }
        std::cout << '\n';
    }

    //27-10-2027: nieuwe tests
    
    std::cout << "\n=== Matmul Tests ===\n";

    // Test 1: Ongeldige dimensies
    Tensor testA(
        {1, 2, 3,
        4, 5, 6},
        {2, 3}
    );

    Tensor testB(
        {1, 2,
        3, 4},
        {2, 2}
    );

    bool invalidDimensionsCaught = false;

    try {
        Tensor invalidResult = matmul(testA, testB);
    }
    catch (const std::invalid_argument& e) {
        invalidDimensionsCaught = true;
        std::cout << "Correct afgevangen: "
                << e.what() << '\n';
    }

    assert(invalidDimensionsCaught);

    // Test 2: Geldige matrixvermenigvuldiging
    Tensor testC = matmul(
        Tensor({2, 3}, {1, 2}),
        Tensor({4, 5, 6, 7}, {2, 2})
    );

    assert(testC.shape() == std::vector<std::size_t>({1, 2}));
    assert(testC.at({0, 0}) == 26.0f);
    assert(testC.at({0, 1}) == 31.0f);

    std::cout << "Geldige matrixvermenigvuldiging: OK\n";
    std::cout << "Alle matmul-tests geslaagd!\n";

    
    std::cout << "\n=== Matmul Edge Cases ===\n";

    // Test 3: Een rij maal een kolom
    Tensor row({2, 3, 4}, {1, 3});
    Tensor col({5, 6, 7}, {3, 1});

    Tensor scalarResult = matmul(row, col);

    assert(scalarResult.shape() ==
        std::vector<std::size_t>({1, 1}));
    assert(scalarResult.at({0, 0}) == 56.0f);

    std::cout << "1x3 maal 3x1: OK\n";

    // Test 4: Een 3D-tensor moet worden afgewezen
    Tensor threeD({1, 2, 3, 4}, {1, 2, 2});

    bool non2DCaught = false;

    try {
        Tensor invalidResult = matmul(threeD, row);
    }
    catch (const std::invalid_argument& e) {
        non2DCaught = true;
        std::cout << "3D-input correct afgewezen: "
                << e.what() << '\n';
    }

    assert(non2DCaught);

    std::cout << "Alle edge-case-tests geslaagd!\n";

    std::cout << "\n=== Transpose test ===\n";

    Tensor matrix(
        {1, 2, 3, 4, 5, 6},
        {2, 3}
    );

    Tensor transposed = transpose(matrix);

    print_tensor_info(transposed);

    for (float value : transposed.data()) {
        std::cout << value << " ";
    }
    std::cout << "\n";


    return 0;
}

