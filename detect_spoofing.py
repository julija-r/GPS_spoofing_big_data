import pandas as pd
import numpy as np
import time
import multiprocessing as mp
import math
from matplotlib import cm
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from tqdm import tqdm

# FUNCTIONS

# 1. Functions for Data Cleaning

def clean_data(df): # for initial data pre-processing
    df = df[~df['Navigational status'].isin(['Moored', 'At anchor', 'Reserved for future use'])]
    df = df.dropna(subset=['MMSI', 'Latitude', 'Longitude', 'Timestamp', 'SOG', 'COG']).copy()
    df = df[(df['Latitude'] >= -90) & (df['Latitude'] <= 90) & (df['Longitude'] >= -180) & (df['Longitude'] <= 180)]
    df = df[df['SOG'] > 0]
    df.loc[:, 'Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    df.loc[:, 'Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
    df.loc[:, 'SOG'] = pd.to_numeric(df['SOG'], errors='coerce')
    df.loc[:, 'COG'] = pd.to_numeric(df['COG'], errors='coerce')
    df = df.dropna(subset=['Latitude', 'Longitude', 'SOG', 'COG'])
    return df

# used for
def process_chunk(chunk): # for more efficient data loading and cleaning
    chunk.columns = chunk.columns.str.strip()
    chunk = chunk.rename(columns={'# Timestamp': 'Timestamp'})
    chunk['Timestamp'] = pd.to_datetime(chunk['Timestamp'], format="%d/%m/%Y %H:%M:%S", errors='coerce')
    return clean_data(chunk)


def parallel_clean_data(file_path, chunksize=100000): # for more efficient data loading and cleaning
    pool = mp.Pool(mp.cpu_count() - 1)
    cleaned_chunks = []

    with pd.read_csv(file_path, chunksize=chunksize) as reader:
        with tqdm(total=sum(1 for _ in open(file_path)) // chunksize + 1, desc="Cleaning Chunks", unit="chunk") as pbar:
            results = [pool.apply_async(process_chunk, (chunk,)) for chunk in reader]
            for r in results:
                cleaned_chunks.append(r.get())
                pbar.update(1)

    pool.close()
    pool.join()

    return pd.concat(cleaned_chunks, ignore_index=True)


# 2. Functions for Calculations and Analysis
def calculate_distance(lat1, lon1, lat2, lon2): #calculates distance between two coordinates
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def great_circle_destination(lon1, lat1, bearing, dist, R=6371): #predict new coordinates
    lat2 = math.asin(math.sin(lat1) * math.cos(dist / R) +
                     math.cos(lat1) * math.sin(dist / R) * math.cos(bearing))
    lon2 = lon1 + math.atan2(math.sin(bearing) * math.sin(dist / R) * math.cos(lat1),
                             math.cos(dist / R) - math.sin(lat1) * math.sin(lat2))
    return lon2, lat2


def predict_next_location(df): # detects spoofing events based on mismatch between predicted and reported coordinates
    spoofed_entries = []

    for mmsi, group in df.groupby('MMSI'): # grouping by vessel
        group = group.sort_values(by='Timestamp').reset_index(drop=True)

        i = 1
        last_valid = group.iloc[0] # to make sure we don't use spoofed coordinates for prediction

        while i < len(group):
            current = group.iloc[i]

            time_diff = (current['Timestamp'] - last_valid['Timestamp']).total_seconds() / 3600
            if time_diff <= 0: # an additional safety check for illogical timestamps
                i += 1
                continue

            estimated_distance = last_valid['SOG'] * time_diff
            bearing = math.radians(last_valid['COG'])
            lat1_rad = math.radians(last_valid['Latitude'])
            lon1_rad = math.radians(last_valid['Longitude'])

            lon_pred_rad, lat_pred_rad = great_circle_destination(
                lon1_rad, lat1_rad, bearing, estimated_distance
            )
            lon_pred = math.degrees(lon_pred_rad)
            lat_pred = math.degrees(lat_pred_rad)

            actual_distance = calculate_distance(
                lat_pred, lon_pred, current['Latitude'], current['Longitude']
            )

            if actual_distance > 18.52: # coordinates are flagged as spoofing if = the difference is nore than 10 NM
                spoofed_entries.append({
                    'Prev_Latitude': last_valid['Latitude'],
                    'Prev_Longitude': last_valid['Longitude'],
                    'MMSI': current['MMSI'],
                    'Timestamp': current['Timestamp'],
                    'Pred_Latitude': lat_pred,
                    'Pred_Longitude': lon_pred,
                    'Reported_Latitude': current['Latitude'],
                    'Reported_Longitude': current['Longitude'],
                    'SOG': last_valid['SOG'],
                    'COG': last_valid['COG'],
                    'Distance_Error_km': actual_distance
                })

                # update last_valid to current + 1 if possible
                if i + 1 < len(group):
                    last_valid = group.iloc[i + 1]
                    i += 2  # skip the spoofed one and move to the one after
                else:
                    break
            else:
                last_valid = current
                i += 1

    return pd.DataFrame(spoofed_entries)


def process_large_file_parallel(file_path, num_workers=None):
    if num_workers is None:
        num_workers = mp.cpu_count() - 1

    print(f" Using {num_workers} worker(s) for parallel processing...")

    start_time = time.time()
    df = parallel_clean_data(file_path)

    spoofed_data = []
    pool = mp.Pool(num_workers)

    grouped = [group for _, group in df.groupby('MMSI')]
    results = [pool.apply_async(predict_next_location, (group,)) for group in grouped]
    #the below is to monitor while code is running :)
    with tqdm(total=len(results), unit='group', desc='Processing vessel groups (Parallel)') as pbar:
        for r in results:
            spoofed_data.append(r.get())
            pbar.update(1)

    pool.close()
    pool.join()
    parallel_time = time.time() - start_time
    print(f" Parallel processing completed in {parallel_time:.2f} seconds.")

    return pd.concat(spoofed_data, ignore_index=True), parallel_time

# the below function is used to test execution time using different chunk sizes for data loading and cleaning
def benchmark_chunk_sizes(file_path, chunk_sizes):
    results = []

    for size in chunk_sizes:
        print(f"\nTesting chunk size: {size}")
        start = time.time()

        df_cleaned = parallel_clean_data(file_path, chunksize=size)
        duration = time.time() - start

        results.append({'Chunk Size': size, 'Time (s)': duration})
        print(f"Time taken for chunk size {size}: {duration:.2f} seconds")

    return pd.DataFrame(results)

def test_parallel_workers(file_path): # used to compare execution time using different number of cpu's

    print("Testing different number of parallel workers...")

    max_workers = mp.cpu_count() - 1

    worker_tests = sorted(set([
        1,
        int(max_workers * 1/3),
        int(max_workers * 2/3),
        max_workers
    ]))

    times = []
    speedups = []

    baseline_time = None

    for workers in worker_tests:
        print(f"\nRunning with {workers} worker(s)...")
        start_time = time.time()

        spoofed_data, parallel_time = process_large_file_parallel(file_path, workers)

        duration = time.time() - start_time
        times.append(duration)

        if workers == 1:
            baseline_time = duration
            print(f"Baseline (1 worker) time: {baseline_time:.2f} seconds")
            speedups.append(1.0)
        else:
            speedup = baseline_time / duration
            speedups.append(speedup)
            print(f"Time: {duration:.2f} seconds, Speedup: {speedup:.2f}")

    # Plot Execution Time
    plt.figure(figsize=(8, 5))
    plt.plot(worker_tests, times, marker='o')
    plt.title("Execution Time vs Number of Workers")
    plt.xlabel("Number of Workers")
    plt.ylabel("Time (seconds)")
    plt.grid()
    plt.tight_layout()
    plt.savefig("execution_time_vs_workers.png")
    plt.close()

    # Plot Speedup
    plt.figure(figsize=(8, 5))
    plt.plot(worker_tests, speedups, marker='o', color='green')
    plt.title("Speedup vs Number of Workers")
    plt.xlabel("Number of Workers")
    plt.ylabel("Speedup")
    plt.grid()
    plt.tight_layout()
    plt.savefig("speedup_vs_workers.png")
    plt.close()


# visualisation functions

def plot_spoofing_events(spoofed_df):
    plt.figure(figsize=(10, 6))
    plt.scatter(spoofed_df['Pred_Longitude'], spoofed_df['Pred_Latitude'], c='green', marker='o', label='Predicted Positions')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Predicted GPS Positions (Spoofing Detection)')
    plt.legend(loc='upper right')
    plt.grid()
    plt.tight_layout()
    plt.savefig('spoofing_events_plot.png')
    plt.close()


def plot_spoofing_distribution_over_time(spoofed_df):
    # Group spoofing events by the hour
    spoofed_df['Hour'] = spoofed_df['Timestamp'].dt.floor('H')
    counts = spoofed_df.groupby('Hour').size()

    # Plot
    plt.figure(figsize=(14, 6))
    plt.bar(counts.index, counts.values, color='red')
    plt.xlabel('Hour of Day')
    plt.ylabel('Number of Spoofing Events')
    plt.title('Distribution of GPS Spoofing Events Over Time')

    # Format x-axis to show every 2 hours
    xticks = pd.date_range(start=counts.index.min(), end=counts.index.max(), freq='2H')
    xticklabels = [t.strftime('%H:%M') for t in xticks]
    plt.xticks(ticks=xticks, labels=xticklabels, rotation=45)

    plt.tight_layout()
    plt.savefig('spoofing_events_distribution_over_time.png')
    plt.close()


def plot_spoofing_clusters(spoofed_df):
    spoofed_df = spoofed_df.sort_values('Timestamp').reset_index(drop=True)
    spoofed_df['ClusterID'] = -1
    cluster_id = 0

    for i in range(len(spoofed_df)):
        if spoofed_df.at[i, 'ClusterID'] != -1:
            continue

        spoofed_df.at[i, 'ClusterID'] = cluster_id
        mmsi_set = {spoofed_df.at[i, 'MMSI']}
        cluster_indices = [i]

        for j in range(i + 1, len(spoofed_df)):
            time_diff = abs((spoofed_df.at[i, 'Timestamp'] - spoofed_df.at[j, 'Timestamp']).total_seconds())
            if time_diff > 600:
                break

            dist_km = calculate_distance(
                spoofed_df.at[i, 'Pred_Latitude'], spoofed_df.at[i, 'Pred_Longitude'],
                spoofed_df.at[j, 'Pred_Latitude'], spoofed_df.at[j, 'Pred_Longitude']
            )
            if dist_km <= 18.52 and spoofed_df.at[j, 'MMSI'] not in mmsi_set:
                mmsi_set.add(spoofed_df.at[j, 'MMSI'])
                cluster_indices.append(j)

        if len(mmsi_set) > 1:
            for idx in cluster_indices:
                spoofed_df.at[idx, 'ClusterID'] = cluster_id
            cluster_id += 1
        else:
            spoofed_df.at[i, 'ClusterID'] = -1

    filtered = spoofed_df[spoofed_df['ClusterID'] != -1].copy()

    norm = Normalize(filtered['Timestamp'].astype(np.int64).min(), filtered['Timestamp'].astype(np.int64).max())
    cmap = plt.colormaps['viridis']
    colors = cmap(norm(filtered['Timestamp'].astype(np.int64)))

    fig, ax = plt.subplots(figsize=(12, 6))
    scatter = ax.scatter(filtered['Pred_Longitude'], filtered['Pred_Latitude'], c=colors, s=30)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Spoofing Clusters (>=2 vessels, within 10min & 10NM)')

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', pad=0.1)
    cbar.set_label('Timestamp')

    start_time = pd.to_datetime(filtered['Timestamp'].min()).strftime('%H:%M:%S')
    end_time = pd.to_datetime(filtered['Timestamp'].max()).strftime('%H:%M:%S')
    cbar.set_ticks([norm.vmin, norm.vmax])
    cbar.set_ticklabels([start_time, end_time])

    ax.grid()
    plt.tight_layout()
    plt.savefig('spoofing_event_clusters_map.png')
    plt.close()


def plot_benchmark_results(df):
    plt.figure(figsize=(8, 5))
    plt.plot(df['Chunk Size'], df['Time (s)'], marker='o')
    plt.xlabel("Chunk Size")
    plt.ylabel("Processing Time (seconds)")
    plt.title("Chunk Size vs Processing Time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("chunk_size_benchmark_plot.png")
    plt.close()


def plot_processing_comparison(sequential_time, parallel_time):
    plt.figure(figsize=(8, 5))
    plt.bar(['Sequential', 'Parallel'], [sequential_time, parallel_time], color=['blue', 'green'])
    plt.ylabel('Processing Time (seconds)')
    plt.title('Sequential vs. Parallel Processing Time')
    plt.tight_layout()
    plt.savefig('processing_comparison.png')
    plt.close()

if __name__ == '__main__':
    file_path = r"C:\Users\Kompiuteris\Desktop\Data Science\2 semester\Big Data\aisdk-2025-03-05.csv"
    #Testing first
    chunk_sizes = [60000, 80000, 100000, 12000, 140000]
    benchmark_results = benchmark_chunk_sizes(file_path, chunk_sizes)
    benchmark_results.to_csv("chunk_size_benchmark.csv", index=False)
    print("\nBenchmark complete! Results saved to chunk_size_benchmark.csv")
    plot_benchmark_results(benchmark_results)

    test_parallel_workers(file_path)

    #Final optimized version
    print("Running Parallel Processing...")
    spoofed_data_parallel, parallel_time = process_large_file_parallel(file_path)

    print(f"Total spoofing events detected (Parallel): {len(spoofed_data_parallel)}")
    spoofed_data_parallel.to_csv("spoofed_data_parallel.csv", index=False)

    plot_spoofing_events(spoofed_data_parallel)
    plot_spoofing_distribution_over_time(spoofed_data_parallel)
    plot_spoofing_clusters(spoofed_data_parallel)


