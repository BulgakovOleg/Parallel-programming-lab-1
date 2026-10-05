#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <random>
#include <omp.h>

bool generator(const std::string& filename, int N) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error while opening file: " << filename << "\n";
        return false;
    }
    file.precision(2);
    file << std::fixed;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(1.0, 100.0);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            file << dis(gen) << (j == N - 1 ? "" : " ");
        }
        file << "\n";
    }
    return true;
}

bool readMatrix(const std::string& filename, std::vector<double>& matrix, int N) {
    std::ifstream file(filename);
    if (!file.is_open()) return false;
    matrix.resize(static_cast<size_t>(N) * N);
    for (int i = 0; i < N * N; ++i) {
        if (!(file >> matrix[i])) return false;
    }
    return true;
}

bool writeMatrix(const std::string& filename, const std::vector<double>& matrix, int N) {
    std::ofstream file(filename);
    if (!file.is_open()) return false;
    file.precision(2);
    file << std::fixed;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            file << matrix[static_cast<size_t>(i) * N + j] << (j == N - 1 ? "" : " ");
        }
        file << "\n";
    }
    return true;
}

int number_input() {
	int N = 0;
	while (true) {
        std::cout << "Enter the size of the square matrix N (e.g., 500): ";
        if (std::cin >> N && N > 0) {
            break;
        }
		else {
            std::cout << "Error! The size must be a positive integer.\n";
            std::cin.clear(); 
            std::cin.ignore(10000, '\n'); 
        }
    }
	return N;
}

double multiplication_process(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, int N) {
	double start = omp_get_wtime();

    #pragma omp parallel for shared(A, B, C, N) schedule(static)
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            double r = A[static_cast<size_t>(i) * N + k];
            for (int j = 0; j < N; ++j) {
                C[static_cast<size_t>(i) * N + j] += r * B[static_cast<size_t>(k) * N + j];
            }
        }
    }

    double duration_ms = (omp_get_wtime() - start) * 1000.0;
	return duration_ms;
}

bool Python_check(int N, const std::string& fileA, const std::string& fileB, const std::string& fileC){
	std::string python_path = "C:\\Users\\Pro\\AppData\\Local\\Programs\\Python\\Python313\\python.exe";
    std::string verify_command = "\"" + python_path + "\" verify.py " + std::to_string(N) + " " + fileA + " " + fileB + " " + fileC;
    int verify_result = std::system(verify_command.c_str());
    return (verify_result == 0);
}

void conclusion(int N, int num_threads, double duration_ms, double total_mem_mb) {
	std::cout << "\n==================== EXECUTION REPORT ====================\n"
              << "TASK_SIZE:         " << N << "x" << N << "\n"
              << "THREADS_USED:      " << num_threads << "\n"
              << "EXECUTION_TIME_MS: " << duration_ms << " ms\n"
              << "MEMORY_USAGE_MB:   " << total_mem_mb << " MB\n"
              << "==========================================================\n";
}

int main(int argc, char* argv[]) {
    int N = number_input();
    
    std::string fileA = "A_" + std::to_string(N) + ".txt";
    std::string fileB = "B_" + std::to_string(N) + ".txt";
    std::string fileC = "C_" + std::to_string(N) + ".txt";

    std::cout << "\n[Step 1] Automatically generating filenames:\n"
              << " -> Matrix A: " << fileA << "\n"
              << " -> Matrix B: " << fileB << "\n"
              << " -> Result C: " << fileC << "\n\n"
              << "Generating random matrices data...";

    if (!generator(fileA, N) || !generator(fileB, N)) {
        std::cerr << "An error occurred during matrix generation. Exiting.\n";
        return 1;
    }
    std::cout << " OK!\n";

    std::cout << "[Step 2] Reading generated files into memory variables...";
    std::vector<double> A, B;
    std::vector<double> C(static_cast<size_t>(N) * N, 0.0);

    if (!readMatrix(fileA, A, N) || !readMatrix(fileB, B, N)) {
        std::cerr << "\nError reading input files.\n";
        return 1;
    }
    std::cout << " OK!\n";

    std::cout << "[Step 3] Performing parallel matrix multiplication (OpenMP)...";
    int num_threads = omp_get_max_threads();
	double duration_ms = multiplication_process(A,B,C,N);
    std::cout << " OK!\n";

    if (!writeMatrix(fileC, C, N)) {
        std::cerr << "Error writing the output file.\n";
        return 1;
    }

    double total_mem_mb = (static_cast<double>(sizeof(double)) * N * N * 3) / (1024.0 * 1024.0);

    conclusion(N, num_threads, duration_ms, total_mem_mb);

    std::cout << "\n[Step 4] Launching automatic verification of results via Python...\n";
	if (Python_check(N,fileA,fileB,fileC)) {
        std::cout << "Verification successfully completed! Results match NumPy.\n";
		return 0;
    }
	else {
        std::cerr << "Result verification error! Python script rejected the result.\n";
        return 1; 
    }
    
    return 0;
}


