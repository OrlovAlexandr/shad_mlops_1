import json
import logging
import os
import sys
import time
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer


sys.path.append(os.path.abspath('./src'))
from preprocessing import load_train_data
from preprocessing import run_preproc
from scorer import make_pred


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/service.log'),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def wait_for_file_ready(path, timeout=60, interval=1):
    logger.info(f'Waiting for file to be ready: {path}')
    previous_size = -1
    start_time = time.time()

    while True:
        try:
            current_size = os.path.getsize(path)
        except FileNotFoundError:
            logger.warning("File not found yet...")
            time.sleep(interval)
            continue

        if current_size == previous_size and current_size > 0:
            logger.info("File size stabilized, assuming ready.")
            return

        if time.time() - start_time > timeout:
            raise TimeoutError(f"Timeout waiting for file to finish writing: {path}")

        previous_size = current_size
        time.sleep(interval)


def plot_probabilities_density(y_proba_, filepath):
    plt.figure(figsize=(10, 6))
    sns.kdeplot(y_proba_, fill=True, color='steelblue', alpha=0.6)
    plt.title('Density Plot of Predicted Probabilities', fontsize=14)
    plt.xlabel('Predicted Probability (Class 1)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()


class ProcessingService:
    def __init__(self):
        logger.info('Initializing ProcessingService...')
        self.input_dir = '/app/input'
        self.output_dir = '/app/output'
        self.train = load_train_data()
        logger.info('Service initialized')

    def process_single_file(self, file_path):
        try:
            logger.info('Processing file: %s', file_path)

            wait_for_file_ready(file_path)
            input_df = pd.read_csv(file_path)

            logger.info('Starting preprocessing')
            processed_df = run_preproc(input_df)

            logger.info('Making prediction')
            submission, top_5_features, y_proba = make_pred(processed_df, file_path)

            logger.info('Prepraring submission file')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"predictions_{timestamp}_{os.path.basename(file_path)}"
            submission.to_csv(os.path.join(self.output_dir, output_filename), index=False)
            logger.info('Predictions saved to: %s', output_filename)

            top_5_filename = f'top_5_features_{timestamp}.json'
            top_5_filepath = os.path.join(self.output_dir, top_5_filename)
            with open(top_5_filepath, 'w', encoding='utf-8') as f:
                json.dump(top_5_features, f, indent=4, ensure_ascii=False)
            logger.info('Top 5 features saved to: %s', top_5_filepath)

            plot_name = f'plot_probabilities_density_{timestamp}.png'
            plot_filepath = os.path.join(self.output_dir, plot_name)
            plot_probabilities_density(y_proba, plot_filepath)
            logger.info('Density plot saved to: %s', plot_filepath)

        except Exception as e:
            logger.error('Error processing file %s: %s', file_path, e, exc_info=True)
            return


class FileHandler(FileSystemEventHandler):
    def __init__(self, service):
        self.service = service

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".csv"):
            logger.debug('New file detected: %s', event.src_path)
            self.service.process_single_file(event.src_path)


if __name__ == "__main__":
    logger.info('Starting ML scoring service...')
    service = ProcessingService()
    observer = Observer()
    observer.schedule(FileHandler(service), path=service.input_dir, recursive=False)
    observer.start()
    logger.info('File observer started')
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info('Service stopped by user')
        observer.stop()
    observer.join()
