#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include <atomic>
#include <string>
#include <fstream>
#include <Windows.h>
#include <pdh.h>
#include <ctime>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "json.hpp"

using json = nlohmann::json;

std::atomic<int> cells_processed(0);
std::atomic<int> total_cycles(0);
int data_size = 16;
int num_threads = 16;
int benchmark_duration = 10;
bool use_gpu = false;
std::string log_file_name = "benchmark_log.txt";

// logging
void logBenchmark(const std::string& message) {
    std::ofstream log_file(log_file_name, std::ios_base::app);
    if (log_file.is_open()) {
        log_file << message << std::endl;
        log_file.close();
    }
    else {
        std::cerr << "Error opening log file!" << std::endl;
    }
}

// load config
void loadConfiguration() {
    std::ifstream config_file("config.json");
    if (config_file.is_open()) {
        json config;
        config_file >> config;
        data_size = config["data_size"].get<int>();
        num_threads = config["num_threads"].get<int>();
        benchmark_duration = config["benchmark_duration"].get<int>();
        use_gpu = config["use_gpu"].get<bool>();
        config_file.close();
    }
    else {
        std::cerr << "Error opening configuration file!" << std::endl;
    }
}

// save config
void saveConfiguration() {
    json config;
    config["data_size"] = data_size;
    config["num_threads"] = num_threads;
    config["benchmark_duration"] = benchmark_duration;
    config["use_gpu"] = use_gpu;

    std::ofstream config_file("config.json");
    if (config_file.is_open()) {
        config_file << config.dump(4); // Pretty print with 4 spaces
        config_file.close();
    }
    else {
        std::cerr << "Error opening configuration file!" << std::endl;
    }
}

// executed by each thread
void processMatrixMultiplication(int thread_id, const std::vector<std::vector<int>>& A, const std::vector<std::vector<int>>& B, std::vector<std::vector<int>>& C) {
    int n = A.size();
    for (int i = thread_id; i < n; i += num_threads) {
        for (int j = 0; j < n; ++j) {
            C[i][j] = 0;
            for (int k = 0; k < n; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// CUDA kernel
__global__ void matrixMultiplicationKernel(int* A, int* B, int* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        int sum = 0;
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

// process one cell (matrix multiplication) using GPU
void processCellGPU(const std::vector<std::vector<int>>& A, const std::vector<std::vector<int>>& B, std::vector<std::vector<int>>& C) {
    int n = A.size();
    int size = n * n * sizeof(int);

    int* d_A, * d_B, * d_C;
    cudaError_t err;

    // Allocate memory
    err = cudaMalloc((void**)&d_A, size);
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed for d_A: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    err = cudaMalloc((void**)&d_B, size);
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed for d_B: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A);
        return;
    }
    err = cudaMalloc((void**)&d_C, size);
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed for d_C: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        return;
    }

    // Copy data from host to device
    err = cudaMemcpy(d_A, A[0].data(), size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy to device failed for d_A: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return;
    }
    err = cudaMemcpy(d_B, B[0].data(), size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy to device failed for d_B: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return;
    }

    // Launch the kernel
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((n + 15) / 16, (n + 15) / 16);
    matrixMultiplicationKernel << <blocksPerGrid, threadsPerBlock >> > (d_A, d_B, d_C, n);

    // Wait for the kernel to finish and check for errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel execution failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return;
    }

    // Copy processed data back to host
    err = cudaMemcpy(C[0].data(), d_C, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy to host failed for d_C: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return;
    }

    // Free the GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cells_processed++;
    total_cycles += n * n;
}

// process one cell (matrix multiplication) using CPU
void processCell(const std::vector<std::vector<int>>& A, const std::vector<std::vector<int>>& B, std::vector<std::vector<int>>& C) {
    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(processMatrixMultiplication, i, std::ref(A), std::ref(B), std::ref(C));
    }
    for (auto& thread : threads) {
        thread.join();
    }

    cells_processed++;
    total_cycles += A.size() * A.size();
}

// Grade the CPU based on total cycles
std::string gradeBenchmark(int total_cycles_processed) {
    if (total_cycles_processed >= 25000000) return "SSS";
    else if (total_cycles_processed >= 15000000) return "SS";
    else if (total_cycles_processed >= 11500000) return "S";
    else if (total_cycles_processed >= 8500000) return "A+";
    else if (total_cycles_processed >= 6000000) return "A";
    else if (total_cycles_processed >= 4500000) return "B+";
    else if (total_cycles_processed >= 3500000) return "B";
    else if (total_cycles_processed >= 2250000) return "C+";
    else if (total_cycles_processed >= 1500000) return "C";
    else if (total_cycles_processed >= 1300000) return "D+";
    else if (total_cycles_processed >= 900000) return "D";
    else if (total_cycles_processed >= 500010) return "D-";
    else if (total_cycles_processed == 69420) return "Nice";
    else if (total_cycles_processed >= 0) return "Zero!";
    else return "F";
}

// Grade the GPU based on total cycles
std::string gradeBenchmark2(int total_cycles_processed) {
    if (total_cycles_processed >= 25000000) return "SSS";
    else if (total_cycles_processed >= 13000000) return "SS";
    else if (total_cycles_processed >= 7500000) return "S";
    else if (total_cycles_processed >= 5000000) return "A+";
    else if (total_cycles_processed >= 4000000) return "A";
    else if (total_cycles_processed >= 3000000) return "B+";
    else if (total_cycles_processed >= 2000000) return "B";
    else if (total_cycles_processed >= 1700000) return "C+";
    else if (total_cycles_processed >= 1500000) return "C";
    else if (total_cycles_processed >= 11000000) return "D+";
    else if (total_cycles_processed >= 900000) return "D";
    else if (total_cycles_processed >= 600001) return "D-";
    else if (total_cycles_processed == 69420) return "Nice";
    else if (total_cycles_processed >= 0) return "Zero!";
    else return "F";
}

void startBenchmark(bool useGPU) {
    int n = data_size;
    std::vector<std::vector<int>> A(n, std::vector<int>(n, 1));
    std::vector<std::vector<int>> B(n, std::vector<int>(n, 1));
    std::vector<std::vector<int>> C(n, std::vector<int>(n, 0));

    cells_processed.store(0);
    total_cycles.store(0);

    std::time_t start_time = std::time(nullptr);
    std::cout << "-----> Benchmark started at: " << start_time << std::endl;
    logBenchmark("-----> Benchmark started at: " + std::to_string(start_time));

    std::cout << "Processor: " << (useGPU ? "GPU" : "CPU") << "\n";
    std::cout << "Data Size: " << data_size << " bytes\n";

    auto start = std::chrono::high_resolution_clock::now();

    while (true) {
        auto current_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::seconds>(current_time - start).count();

        if (elapsed_time >= benchmark_duration) {
            break;
        }

        if (useGPU) {
            processCellGPU(A, B, C);
        }
        else {
            processCell(A, B, C);
        }

        int total_cells = cells_processed.load();
        double cells_per_second = static_cast<double>(total_cells) / (elapsed_time + 1); // +1 to avoid division by zero
        int total_cycles_processed = total_cycles.load();
        double cycles_per_second = cells_per_second * n * n;

        std::cout << "\r\033[1;32mTotal Cycles: " << total_cycles_processed
            << " | Cycles/sec: " << static_cast<int>(cycles_per_second)
            << " | Cells/sec: " << static_cast<int>(cells_per_second)
            << " | Total Cells: " << total_cells
            << " | Elapsed Time: " << elapsed_time << "s\033[0m" << std::flush;
    }

    std::time_t end_time = std::time(nullptr);
    std::cout << "\n--> Benchmark completed at: " << end_time << std::endl;
    logBenchmark("--> Benchmark completed at: " + std::to_string(end_time));

    int total_cells = cells_processed.load();
    int total_seconds = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count();
    double cells_per_second = static_cast<double>(total_cells) / total_seconds;
    double cycles_per_second = cells_per_second * n * n;
    int total_cycles_processed = total_cycles.load();

    std::cout << "Cycles Processed per Second: " << static_cast<int>(cycles_per_second) << std::endl;
    std::cout << "Total Cycles Processed: " << total_cycles_processed << std::endl;
    std::cout << "Total Cells Processed: " << total_cells << std::endl;
    std::cout << "Cells Processed per Second (Average): " << static_cast<int>(cells_per_second) << std::endl;

    logBenchmark("Cycles Processed per Second: " + std::to_string(static_cast<int>(cycles_per_second)));
    logBenchmark("Total Cycles Processed: " + std::to_string(total_cycles_processed));
    logBenchmark("Total Cells Processed: " + std::to_string(total_cells));
    logBenchmark("Cells Processed per Second (Average): " + std::to_string(static_cast<int>(cells_per_second)));

    std::string grade = useGPU ? gradeBenchmark2(total_cycles_processed) : gradeBenchmark(total_cycles_processed);
    std::cout << (useGPU ? "GPU" : "CPU") << " Benchmark Grade: " << grade << std::endl;
    logBenchmark((useGPU ? "GPU" : "CPU") + std::string(" Benchmark Grade: ") + grade);

    Sleep(10000);
}

// Display information
void showInfo() {
    std::cout << "\n\n=---=---=---> RBCB - Really Bad CUDA Benchmarking <---=---=---=\n";
    std::cout << "This program benchmarks hardware performance by performing matrix multiplication tasks.\n";
    std::cout << "It can utilize both CPU and GPU for benchmarking.\n";
    std::cout << "The benchmark measures how many matrix multiplication tasks (cells) can be processed in a given time period.\n";
    std::cout << "The program supports multithreaded execution on the CPU and parallel execution on the GPU using CUDA.\n";
    std::cout << "You can configure the data size, number of threads, and benchmark duration.\n";
    std::cout << "Results include metrics such as total cycles and cells processed, cycles and cells per second and FPS on our Graphics Test.\n";
    std::cout << "The program also provides a grade based on the performance of the CPU or GPU.\n";
    std::cout << "=---=---=---> RBCB - Really Bad CUDA Benchmarking <---=---=---=\n";
    std::cout << "[!] Please report any errors and issues to https://github.com/CatchySmile/RBCB\n";
    Sleep(10000);
}


// Change the data size
void changeDataSize() {
    float new_size;
    std::cout << "\nWarning: Proceed with caution, avoid using crazy numbers.\n";
    std::cout << "Enter the new data size (even number): ";
    std::cin >> new_size;

    int rounded_size = static_cast<int>(new_size);
    if (rounded_size % 2 != 0) {
        rounded_size--;
    }

    if (rounded_size < 2) {
        rounded_size = 2;
    }

    data_size = rounded_size;
    std::cout << "Data size updated to " << data_size << " bytes.\n";
    saveConfiguration();
    Sleep(3000);
}

// Change the number of threads
void changeNumThreads() {
    int new_num_threads;
    std::cout << "\nEnter the new number of threads: ";
    std::cin >> new_num_threads;

    if (new_num_threads < 1) {
        new_num_threads = 1;
    }

    num_threads = new_num_threads;
    std::cout << "Number of threads updated to " << num_threads << ".\n";
    saveConfiguration();
    Sleep(3000);
}

// Change the benchmark duration
void changeBenchmarkDuration() {
    int new_duration;
    std::cout << "\nEnter the new benchmark duration (seconds): ";
    std::cin >> new_duration;

    if (new_duration < 1) {
        new_duration = 1;
    }

    benchmark_duration = new_duration;
    std::cout << "Benchmark duration updated to " << benchmark_duration << " seconds.\n";
    saveConfiguration();
    Sleep(3000);
}

// Select the CPU or GPU for benchmarking
void selectProcessor() {
    int choice;
    std::cout << "\nSelect Processor\n";
    std::cout << "[1] CPU\n";
    std::cout << "[2] GPU\n";
    std::cout << "[0] Back\n";
    std::cout << "Enter your choice: ";
    std::cin >> choice;

    switch (choice) {
    case 1:
        std::cout << "CPU selected.\n";
        use_gpu = false;
        saveConfiguration();
        startBenchmark(false);
        break;
    case 2:
        std::cout << "GPU selected.\n";
        use_gpu = true;
        saveConfiguration();
        startBenchmark(true);
        break;
    case 0:
        return;
    default:
        std::cout << "Invalid choice. Please try again.\n";
    }
    Sleep(3000);
}

// Le options
void advancedOptions() {
    int choice;
    while (true) {
        system("cls");
        std::cout << "\n=-------=-------=-------=\n";
        std::cout << "\n\033[1;32mCurrent Settings:\033[0m\n";
        std::cout << "\033[1;32mData Size: " << data_size << " bytes\033[0m\n";
        std::cout << "\033[1;32mThreads per cycle: " << num_threads << "\033[0m\n";
        std::cout << "\033[1;32mBenchmark Duration: " << benchmark_duration << " seconds\033[0m\n";
        std::cout << "\033[1;32mProcessor: " << (use_gpu ? "GPU" : "CPU") << "\033[0m\n";
        std::cout << "\033[1;32mAutosave : On " << "\033[0m\n";
        std::cout << "\n=-------=-------=-------=\n";
        std::cout << "[1] Change Data Size\n";
        std::cout << "[2] Change Number of Threads\n";
        std::cout << "[3] Change Benchmark Duration\n";
        std::cout << "[0] Back to Main Menu\n";
        std::cout << "=-------=-------=-------=\n";
        std::cout << "Enter your choice: ";
        std::cin >> choice;

        switch (choice) {
        case 1:
            changeDataSize();
            break;
        case 2:
            changeNumThreads();
            break;
        case 3:
            changeBenchmarkDuration();
            break;
        case 0:
            return;
        default:
            std::cout << "Invalid choice. Please try again.\n";
        }
    }
}

// using Windows API because im too lazy to use anything else
LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    switch (uMsg) {
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

void runGraphicsTest(bool useGPU) {
    // Define the CLASS_NAME identifier
    const wchar_t CLASS_NAME[] = L"GraphicsRenderingTestWindow";

    // Define the start variable
    auto start = std::chrono::high_resolution_clock::now();

    // Define the n variable
    int n = data_size;

    WNDCLASS wc = {};
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = GetModuleHandle(NULL);
    wc.lpszClassName = CLASS_NAME;

    RegisterClass(&wc);

    HWND hwnd = CreateWindowEx(
        0,
        CLASS_NAME,
        L"Graphics Rendering Test",
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT, 480, 480,
        NULL,
        NULL,
        GetModuleHandle(NULL),
        NULL
    );

    if (hwnd == NULL) {
        std::cerr << "CreateWindowEx failed!" << std::endl;
        return;
    }

    // Set the window to full screen
    SetWindowLong(hwnd, GWL_STYLE, GetWindowLong(hwnd, GWL_STYLE) & ~WS_OVERLAPPEDWINDOW);
    SetWindowPos(hwnd, HWND_TOP, 0, 0, GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN), SWP_NOOWNERZORDER | SWP_FRAMECHANGED);

    ShowWindow(hwnd, SW_SHOW);

    std::vector<std::vector<int>> A(n, std::vector<int>(n, 1));
    std::vector<std::vector<int>> B(n, std::vector<int>(n, 1));
    std::vector<std::vector<int>> C(n, std::vector<int>(n, 0));

    cells_processed.store(0);
    total_cycles.store(0);

    auto last_fps_time = start;
    int frame_count = 0;
    double fps = 0.0;

    MSG msg = {};
    bool running = true;
    while (running) {
        while (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT) {
                running = false;
            }
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }

        if (useGPU) {
            processCellGPU(A, B, C);
        }
        else {
            processCell(A, B, C);
        }

        // Change color every cell
        int r = rand() % 256;
        int g = rand() % 256;
        int b = rand() % 256;
        HBRUSH brush = CreateSolidBrush(RGB(r, g, b));
        HDC hdc = GetDC(hwnd);
        RECT rect;
        GetClientRect(hwnd, &rect);
        FillRect(hdc, &rect, brush);
        DeleteObject(brush);

        frame_count++;
        auto current_time = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(current_time - last_fps_time).count() >= 1) {
            fps = frame_count;
            frame_count = 0;
            last_fps_time = current_time;
        }

        // Calculate metrics
        auto elapsed_time = std::chrono::duration_cast<std::chrono::seconds>(current_time - start).count();
        int total_cells = cells_processed.load();
        double cells_per_second = static_cast<double>(total_cells) / (elapsed_time + 1); // +1 to avoid division by zero
        int total_cycles_processed = total_cycles.load();
        double cycles_per_second = cells_per_second * n * n;

        // Display metrics
        std::wstring metrics = L"FPS: " + std::to_wstring(fps) +
            L"\nTotal Cycles: " + std::to_wstring(total_cycles_processed) +
            L"\nCycles/sec: " + std::to_wstring(static_cast<int>(cycles_per_second)) +
            L"\nTotal Cells: " + std::to_wstring(total_cells) +
            L"\nCells/sec: " + std::to_wstring(static_cast<int>(cells_per_second));

        SetBkColor(hdc, RGB(255, 255, 255));
        SetTextColor(hdc, RGB(0, 0, 0));
        DrawText(hdc, metrics.c_str(), -1, &rect, DT_LEFT | DT_TOP);

        ReleaseDC(hwnd, hdc);

        if (elapsed_time >= benchmark_duration) {
            running = false;
        }
    }

    // Calculate final metrics
    auto end_time = std::chrono::high_resolution_clock::now();
    int total_cells = cells_processed.load();
    int total_seconds = std::chrono::duration_cast<std::chrono::seconds>(end_time - start).count();
    double cells_per_second = static_cast<double>(total_cells) / total_seconds;
    int total_cycles_processed = total_cycles.load();
    double cycles_per_second = cells_per_second * n * n;

    PostMessage(hwnd, WM_CLOSE, 0, 0);

    while (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
        if (msg.message == WM_QUIT) {
            break;
        }
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    std::cout << "\nGraphics Rendering Test completed.\n";
    std::cout << "FPS: " << fps << "\n";
    std::cout << "Cycles Processed per Second: " << static_cast<int>(cycles_per_second) << "\n";
    std::cout << "Total Cycles Processed: " << total_cycles_processed << "\n";
    std::cout << "Total Cells Processed: " << total_cells << "\n";
    std::cout << "Cells Processed per Second (Average): " << static_cast<int>(cells_per_second) << "\n";

    logBenchmark("Graphics Rendering Test completed.");
    logBenchmark("FPS: " + std::to_string(fps));
    logBenchmark("Cycles Processed per Second: " + std::to_string(static_cast<int>(cycles_per_second)));
    logBenchmark("Total Cycles Processed: " + std::to_string(total_cycles_processed));
    logBenchmark("Total Cells Processed: " + std::to_string(total_cells));
    logBenchmark("Cells Processed per Second (Average): " + std::to_string(static_cast<int>(cells_per_second)));

    if (useGPU) {
        std::string grade = gradeBenchmark2(total_cycles_processed);
        std::cout << "GPU Benchmark Grade: " << grade << std::endl;
        logBenchmark("GPU Benchmark Grade: " + grade);
		Sleep(10000);
    }
}



void selectGraphicsProcessor() {
    int choice;
    std::cout << "\nSelect Processor for Graphics Rendering Test\n";
    std::cout << "[1] CPU\n";
    std::cout << "[2] GPU\n";
    std::cout << "[0] Back\n";
    std::cout << "Enter your choice: ";
    std::cin >> choice;

    switch (choice) {
    case 1:
        std::cout << "CPU selected for Graphics Rendering Test.\n";
        runGraphicsTest(false);
        break;
    case 2:
        std::cout << "GPU selected for Graphics Rendering Test.\n";
        runGraphicsTest(true);
        break;
    case 0:
        return;
    default:
        std::cout << "Invalid choice. Please try again.\n";
    }
    Sleep(3000);
}


void startBenchmarkMenu() {
    int choice;
    while (true) {
        system("cls");
        std::cout << "\n=-------=-------=-------=\n";
        std::cout << "[1] Simple Benchmark\n";
        std::cout << "[2] Graphical Rendering\n";
        std::cout << "[0] Back to Main Menu\n";
        std::cout << "=-------=-------=-------=\n";
        std::cout << "Enter your choice: ";
        std::cin >> choice;

        switch (choice) {
        case 1:
            selectProcessor();
            break;
        case 2:
            selectGraphicsProcessor();
            break;
        case 0:
            return;
        default:
            std::cout << "Invalid choice. Please try again.\n";
        }
    }
}

int main() {
    loadConfiguration();
    while (true) {
        system("cls");
        std::cout << R"(
                           _..._                
                        .-'_..._''.             
         /|           .' .'      '.\ /|         
         ||          / .'            ||         
.-,.--.  ||         . '              ||         
|  .-. | ||  __     | |              ||  __     
| |  | | ||/'__ '.  | |              ||/'__ '.  
| |  | | |:/`  '. ' . '              |:/`  '. ' 
| |  '-  ||     | |  \ '.          . ||     | | 
| |      ||\    / '   '. `._____.-'/ ||\    / ' 
| |      |/\'..' /      `-.______ /  |/\'..' /  
|_|      '  `'-'`                `   '  `'-'`   
                                                

)";
        std::cout << "\n=-------=-------=-------=\n";
        std::cout << "RBCB - really bad CUDA benchmarking\n";
        std::cout << "[1] Begin Benchmark\n";
        std::cout << "[2] Info\n";
        std::cout << "[3] Options\n";
        std::cout << "[0] Quit\n";
        std::cout << "=-------=-------=-------=\n";
        std::cout << "Enter your choice: ";

        int choice;
        std::cin >> choice;

        switch (choice) {
        case 1:
            startBenchmarkMenu();
            break;
        case 2:
            showInfo();
            break;
        case 3:
            advancedOptions();
            break;
        case 0:
            std::cout << "Exiting the program. Goodbye!\n";
            return 0;
        default:
            std::cout << "Invalid choice. Please try again.\n";
        }

        // Display current settings
        std::cout << "\n\033[1;32mCurrent Settings:\033[0m\n";
        std::cout << "\033[1;32mData Size: " << data_size << " bytes\033[0m\n";
        std::cout << "\033[1;32mNumber of Threads: " << num_threads << "\033[0m\n";
        std::cout << "\033[1;32mBenchmark Duration: " << benchmark_duration << " seconds\033[0m\n";
        std::cout << "\033[1;32mProcessor: " << (use_gpu ? "GPU" : "CPU") << "\033[0m\n";
        std::cout << "\033[1;32mAutosave : On " << "\033[0m\n";
        std::cout << "[!] Please report any errors and issues to https://github.com/CatchySmile/RBCB\n";
    }
}


