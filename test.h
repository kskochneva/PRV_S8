#ifndef TEST_H
#define TEST_H

#include <iostream>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <random>
#include <chrono>
#include <algorithm>
#include <execution>
#include <numeric>
#include <unordered_map>
#include <iomanip>
#include <cmath>
#include <windows.h>

using namespace std;

// Configuration constants
const int NUM_SENSORS = 100;
const size_t PROCESSING_THRESHOLD = 100;
const int STORAGE_INTERVAL_SEC = 5;
const int RUN_TIME_SEC = 60;

struct SensorData {
    int sensor_id = 0;
    double temperature = 0.0;
    double pressure = 0.0;
    double vibration = 0.0;
    chrono::steady_clock::time_point timestamp = chrono::steady_clock::now();
};

struct ProcessedMetrics {
    double avg_temperature = 0.0;
    double max_pressure = 0.0;
    double total_vibration = 0.0;
    double energy_score = 0.0;
    double prefix_sum_last = 0.0;
    size_t num_readings = 0;
};

// Thread-safe queue
template<typename T>
class SafeQueue {
private:
    queue<T> q;
    mutable mutex mtx;
    condition_variable cv;
public:
    void push(T value) {
        {
            lock_guard<mutex> lock(mtx);
            q.push(move(value));
        }
        cv.notify_one();
    }

    T pop() {
        unique_lock<mutex> lock(mtx);
        cv.wait(lock, [this] { return !q.empty(); });
        T value = move(q.front());
        q.pop();
        return value;
    }
};

// Sensor class for generating readings
class Sensor {
    int id;
    mt19937 rng;
    uniform_real_distribution<> temp_dist{ 20.0, 100.0 };
    uniform_real_distribution<> press_dist{ 0.8, 6.0 };
    uniform_real_distribution<> vib_dist{ 0.0, 5.0 };
public:
    explicit Sensor(int id) : id(id), rng(random_device{}()) {}
    SensorData read();
};

// Result aggregator class
class ResultAggregator {
private:
    mutable mutex metrics_mutex;
    mutable mutex critical_mutex;
    ProcessedMetrics cumulative;
    atomic<size_t> batches_processed{ 0 };
    vector<SensorData> all_critical;
public:
    void add_batch_result(const ProcessedMetrics& m, const vector<SensorData>& critical);
    pair<ProcessedMetrics, vector<SensorData>> get_summary() const;
    size_t get_batches_processed() const { return batches_processed.load(); }
};

// Function declarations
ProcessedMetrics process_batch(const vector<SensorData>& batch);
double calculate_health_index(const SensorData& d);
void sensor_thread(int sensor_id, SafeQueue<SensorData>& queue, atomic<bool>& running);
void worker_thread(SafeQueue<SensorData>& input_queue, ResultAggregator& aggregator, atomic<bool>& running);
void storage_thread(const ResultAggregator& aggregator, atomic<bool>& running);

#endif // TEST_H
