#include "test.h"

// Sensor class implementation
SensorData Sensor::read() {
    return SensorData{ id, temp_dist(rng), press_dist(rng), vib_dist(rng), chrono::steady_clock::now() };
}

// Process batch of sensor data using parallel algorithms
ProcessedMetrics process_batch(const vector<SensorData>& batch) {
    if (batch.empty()) return {};
    ProcessedMetrics m;
    m.num_readings = batch.size();

    vector<double> temps(batch.size()), pressures(batch.size()), vibrations(batch.size());
    for (size_t i = 0; i < batch.size(); ++i) {
        temps[i] = batch[i].temperature;
        pressures[i] = batch[i].pressure;
        vibrations[i] = batch[i].vibration;
    }
    
    // Parallel execution on multiple threads
    m.avg_temperature = reduce(execution::par, temps.begin(), temps.end(), 0.0) / batch.size();
    m.max_pressure = *max_element(execution::par_unseq, pressures.begin(), pressures.end());
    m.total_vibration = reduce(execution::par_unseq, vibrations.begin(), vibrations.end(), 0.0);
    
    // Transform_reduce transforms each element and sums the result
    double energy_sum = transform_reduce(execution::par_unseq, batch.begin(), batch.end(), 0.0,
        plus<double>(), [](const SensorData& d) {
            return d.temperature * d.pressure / (d.vibration + 0.1);
        });
    m.energy_score = energy_sum / batch.size();

    vector<double> prefix(temps.size());
    // Inclusive_scan calculates cumulative sums of temperatures
    inclusive_scan(execution::par_unseq, temps.begin(), temps.end(), prefix.begin(), plus<double>());
    m.prefix_sum_last = prefix.empty() ? 0.0 : prefix.back();

    return m;
}

// Calculate health index for a sensor
double calculate_health_index(const SensorData& d) {
    double tf = max(0.0, 1.0 - (d.temperature - 20.0) / 80.0);
    double pf = max(0.0, 1.0 - (d.pressure - 0.8) / 5.2);
    double vf = max(0.0, 1.0 - d.vibration / 5.0);
    return (tf + pf + vf) / 3.0 * 100.0;
}

// Result aggregator implementation
void ResultAggregator::add_batch_result(const ProcessedMetrics& m, const vector<SensorData>& critical) {
    batches_processed++;
    {
        lock_guard<mutex> lock(metrics_mutex);
        cumulative.avg_temperature = (cumulative.avg_temperature * (batches_processed - 1) + m.avg_temperature) / batches_processed;
        cumulative.max_pressure = max(cumulative.max_pressure, m.max_pressure);
        cumulative.total_vibration += m.total_vibration;
        cumulative.energy_score = (cumulative.energy_score * (batches_processed - 1) + m.energy_score) / batches_processed;
        cumulative.prefix_sum_last += m.prefix_sum_last;
        cumulative.num_readings += m.num_readings;
    }
    if (!critical.empty()) {
        lock_guard<mutex> lock(critical_mutex);
        all_critical.insert(all_critical.end(), critical.begin(), critical.end());
    }
}

pair<ProcessedMetrics, vector<SensorData>> ResultAggregator::get_summary() const {
    lock_guard<mutex> lock_metrics(metrics_mutex);
    lock_guard<mutex> lock_crit(critical_mutex);
    return { cumulative, all_critical };
}

// Sensor thread function
void sensor_thread(int sensor_id, SafeQueue<SensorData>& queue, atomic<bool>& running) {
    Sensor sensor(sensor_id);
    mt19937 delay_rng(random_device{}());
    uniform_int_distribution<> delay_dist(1, 20);
    while (running) {
        this_thread::sleep_for(chrono::milliseconds(delay_dist(delay_rng)));
        queue.push(sensor.read());
    }
}

// Worker thread for processing data
void worker_thread(SafeQueue<SensorData>& input_queue,
    ResultAggregator& aggregator,
    atomic<bool>& running) {
    thread_local unordered_map<int, vector<double>> temp_buffer;
    thread_local unordered_map<int, vector<double>> pressure_buffer;
    thread_local unordered_map<int, vector<double>> vibration_buffer;
    thread_local unordered_map<int, vector<SensorData>> full_buffer;

    while (running) {
        SensorData data = input_queue.pop();
        int sid = data.sensor_id;
        temp_buffer[sid].push_back(data.temperature);
        pressure_buffer[sid].push_back(data.pressure);
        vibration_buffer[sid].push_back(data.vibration);
        full_buffer[sid].push_back(data);

        if (temp_buffer[sid].size() >= PROCESSING_THRESHOLD) {
            // Extract data from thread_local storage
            vector<double> temps = move(temp_buffer[sid]);
            vector<double> pressures = move(pressure_buffer[sid]);
            vector<double> vibrations = move(vibration_buffer[sid]);
            vector<SensorData> full_data = move(full_buffer[sid]);

            ProcessedMetrics metrics;
            metrics.num_readings = full_data.size();

            metrics.avg_temperature = reduce(execution::par, temps.begin(), temps.end(), 0.0) / full_data.size();
            metrics.max_pressure = *max_element(execution::par_unseq, pressures.begin(), pressures.end());
            metrics.total_vibration = reduce(execution::par_unseq, vibrations.begin(), vibrations.end(), 0.0);

            double energy_sum = transform_reduce(execution::par_unseq, full_data.begin(), full_data.end(), 0.0,
                plus<double>(), [](const SensorData& d) {
                    return d.temperature * d.pressure / (d.vibration + 0.1);
                });
            metrics.energy_score = energy_sum / full_data.size();

            vector<double> prefix(temps.size());
            inclusive_scan(execution::par_unseq, temps.begin(), temps.end(), prefix.begin(), plus<double>());
            metrics.prefix_sum_last = prefix.empty() ? 0.0 : prefix.back();

            // Split critical events (temp > 85, pressure > 5, vibration > 4)
            auto mid = partition(execution::par, full_data.begin(), full_data.end(),
                [](const SensorData& d) {
                    return d.temperature > 85.0 || d.pressure > 5.0 || d.vibration > 4.0;
                });
            vector<SensorData> critical(full_data.begin(), mid);

            sort(execution::par, critical.begin(), critical.end(),
                [](const SensorData& a, const SensorData& b) { return a.timestamp < b.timestamp; });

            aggregator.add_batch_result(metrics, critical);
        }
    }
}

// Storage thread for periodic summary output
void storage_thread(const ResultAggregator& aggregator, atomic<bool>& running) {
    while (running) {
        this_thread::sleep_for(chrono::seconds(STORAGE_INTERVAL_SEC));
        auto [cumulative, critical_events] = aggregator.get_summary();
        size_t batches = aggregator.get_batches_processed();

        cout << "\n==================== TELEMETRY SUMMARY ====================" << endl;
        cout << "Batches processed: " << batches << endl;
        cout << "Total readings processed: " << cumulative.num_readings << endl;
        cout << "Critical events recorded: " << critical_events.size() << endl << endl;

        cout << fixed << setprecision(2);
        cout << "Average temperature: " << cumulative.avg_temperature << " C" << endl;
        cout << "Maximum pressure:    " << cumulative.max_pressure << " atm" << endl;
        cout << "Total vibration:     " << cumulative.total_vibration << " mm/s" << endl;
        cout << "Energy score:        " << cumulative.energy_score << endl;
        cout << "Prefix sum (inclusive_scan): " << cumulative.prefix_sum_last << endl;

        if (!critical_events.empty()) {
            cout << "\n--- LAST 5 CRITICAL EVENTS ---" << endl;
            size_t show = min(size_t(5), critical_events.size());
            for (size_t i = critical_events.size() - show; i < critical_events.size(); ++i) {
                const auto& e = critical_events[i];
                cout << "Sensor #" << setw(3) << e.sensor_id
                    << " | T=" << setw(6) << e.temperature << " C"
                    << " | P=" << setw(5) << e.pressure << " atm"
                    << " | V=" << setw(5) << e.vibration << " mm/s"
                    << " | Health=" << setw(3) << setprecision(0)
                    << calculate_health_index(e) << "%" << endl;
            }
        }
        cout << "============================================================" << endl;
    }
}

// Main entry point
int main() {
    SetConsoleOutputCP(1251);
    SetConsoleCP(1251);

    cout << "========================================================" << endl;
    cout << "   REAL-TIME TELEMETRY ANALYSIS SYSTEM v2.0            " << endl;
    cout << "========================================================" << endl;
    cout << "Available logical cores: " << thread::hardware_concurrency() << endl;
    cout << "Processing threshold per sensor: " << PROCESSING_THRESHOLD << " readings" << endl;
    cout << "System will run for " << RUN_TIME_SEC << " seconds..." << endl << endl;

    SafeQueue<SensorData> data_queue;
    ResultAggregator aggregator;
    atomic<bool> running(true);

    // Create sensor threads
    vector<thread> sensor_threads;
    for (int i = 0; i < NUM_SENSORS; ++i)
        sensor_threads.emplace_back(sensor_thread, i, ref(data_queue), ref(running));

    // Create processing pool (number of workers = hardware concurrency)
    unsigned int num_workers = thread::hardware_concurrency();
    if (num_workers == 0) num_workers = 4;
    vector<thread> worker_threads;
    for (unsigned int i = 0; i < num_workers; ++i)
        worker_threads.emplace_back(worker_thread, ref(data_queue), ref(aggregator), ref(running));

    // Create storage thread
    thread storage_t(storage_thread, cref(aggregator), ref(running));

    // Run for specified time
    this_thread::sleep_for(chrono::seconds(RUN_TIME_SEC));
    running = false;

    // Clean shutdown
    for (auto& t : sensor_threads) t.join();
    for (auto& t : worker_threads) t.join();
    storage_t.join();

    cout << "\nSystem terminated gracefully." << endl;
    return 0;
}
