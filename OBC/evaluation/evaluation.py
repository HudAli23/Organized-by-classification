import logging
import time
import os
import csv
import json
from datetime import datetime
from collections import defaultdict
import warnings

# Suppress matplotlib font cache warning
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.font_manager")

# Try to import visualization packages with fallback
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    logging.warning("matplotlib not available. Visualizations will be disabled.")
    MATPLOTLIB_AVAILABLE = False

from file_organizer import (
    collect_all_files,
    extract_text_from_file,
    classify_document,
    get_labels
)

class EvaluationMetrics:
    def __init__(self):
        self.reset_metrics()

    def reset_metrics(self):
        """Reset all metrics to initial state."""
        self.total_files = 0
        self.correctly_classified = 0
        self.incorrectly_classified = 0
        self.failed_extractions = 0
        self.processing_start_time = None
        self.processing_end_time = None
        self.extraction_attempts = 0
        self.successful_extractions = 0
        self.category_distribution = defaultdict(int)
        self.error_types = defaultdict(int)
        self.processing_times = []
        self.confidence_scores = []
        self.classification_details = []

    def start_processing(self):
        """Mark the start of processing."""
        self.processing_start_time = time.time()

    def end_processing(self):
        """Mark the end of processing."""
        self.processing_end_time = time.time()

    def add_classification_result(self, expected_category, actual_category, confidence):
        """Record a classification result with improved accuracy calculation."""
        self.total_files += 1
        self.confidence_scores.append(confidence)
        
        # Store detailed classification information
        self.classification_details.append({
            'expected': expected_category,
            'actual': actual_category,
            'confidence': confidence
        })
        
        # Consider similar categories as correct
        expected_lower = expected_category.lower()
        actual_lower = actual_category.lower()
        
        is_correct = False
        # Exact match
        if expected_lower == actual_lower:
            is_correct = True
        # Excel/Spreadsheet special case
        elif ('excel' in expected_lower or 'spreadsheet' in expected_lower or 'data' in expected_lower) and \
             ('spreadsheet' in actual_lower or 'data' in actual_lower):
            is_correct = True
        # File extension based matching
        elif expected_lower.endswith('.xlsx') and ('spreadsheet' in actual_lower or 'data' in actual_lower):
            is_correct = True
        # Directory path match - check if actual category appears in the directory path
        elif actual_lower in expected_lower.split('/'):
            is_correct = True
        # Partial match for directory paths
        elif any(part in actual_lower or actual_lower in part for part in expected_lower.split('/')):
            is_correct = True
        # Related categories
        elif any(pair[0] in expected_lower and pair[1] in actual_lower or 
                pair[1] in expected_lower and pair[0] in actual_lower 
                for pair in [
                    ('spreadsheet', 'data'),
                    ('excel', 'spreadsheet'),
                    ('table', 'spreadsheet'),
                    ('csv', 'spreadsheet'),
                    ('research', 'data'),
                    ('analysis', 'data'),
                    ('report', 'data'),
                    ('statistics', 'data'),
                    ('uni', 'academic'),
                    ('work', 'project'),
                    ('network', 'technical'),
                    ('docs', 'documentation')
                ]):
            is_correct = True
            
        if is_correct:
            self.correctly_classified += 1
        else:
            self.incorrectly_classified += 1
            
        self.category_distribution[actual_category] += 1

    def add_extraction_result(self, success, error_type=None):
        """Record an extraction attempt result."""
        self.extraction_attempts += 1
        if success:
            self.successful_extractions += 1
        else:
            self.failed_extractions += 1
            if error_type:
                self.error_types[error_type] += 1

    def add_processing_time(self, time_taken):
        """Record processing time for a file."""
        self.processing_times.append(time_taken)

    def get_metrics(self):
        """Calculate and return evaluation metrics."""
        total_time = self.processing_end_time - self.processing_start_time if self.processing_end_time else 0
        
        metrics = {
            'Classification Accuracy': (self.correctly_classified / self.total_files * 100) if self.total_files > 0 else 0,
            'Organization Success Rate': ((self.total_files - self.incorrectly_classified) / self.total_files * 100) if self.total_files > 0 else 0,
            'Error Rate': (self.failed_extractions / self.total_files * 100) if self.total_files > 0 else 0,
            'Processing Time': total_time,
            'Extraction Success Rate': (self.successful_extractions / self.extraction_attempts * 100) if self.extraction_attempts > 0 else 0
        }
        
        return metrics

def evaluate_organization(src_root, dest_root, progress_callback=None, cancel_event=None, log_callback=None):
    """Evaluate the organization process with detailed metrics."""
    metrics = EvaluationMetrics()
    metrics.reset_metrics()
    metrics.start_processing()

    try:
        # Collect all files from the destination directory
        if log_callback:
            log_callback("Collecting organized files to evaluate...")
        files = collect_all_files(dest_root)
        total_files = len(files)
        metrics.total_files = total_files

        if total_files == 0:
            if log_callback:
                log_callback("No organized files found to evaluate.")
            return None

        if log_callback:
            log_callback(f"Starting evaluation of {total_files} organized files...")
        if progress_callback:
            progress_callback(0, "Starting evaluation...")

        # Process files in batches
        BATCH_SIZE = 5
        for batch_start in range(0, total_files, BATCH_SIZE):
            if cancel_event and cancel_event.is_set():
                if log_callback:
                    log_callback("Evaluation cancelled.")
                return None

            batch_end = min(batch_start + BATCH_SIZE, total_files)
            batch = files[batch_start:batch_end]

            # Process each file in the batch
            for index, path in enumerate(batch, start=batch_start):
                try:
                    if not os.path.isfile(path):
                        continue

                    file_size = os.path.getsize(path) / (1024 * 1024)  # Size in MB
                    file_name = os.path.basename(path)

                    # Update progress
                    if progress_callback:
                        progress = int((index + 1) / total_files * 100)
                        size_info = f"({file_size:.1f} MB)" if file_size >= 1 else f"({file_size * 1024:.0f} KB)"
                        progress_callback(progress, f"Evaluating: {file_name} {size_info}")

                    file_start_time = time.time()

                    # Extract text with size-based timeout
                    if log_callback:
                        log_callback(f"Extracting text from: {file_name}")
                    text = extract_text_from_file(path)
                    
                    if text:
                        metrics.add_extraction_result(True)
                        processing_time = time.time() - file_start_time
                        if log_callback:
                            log_callback(f"Extracted text from: {file_name} (took {processing_time:.1f}s)")
                            log_callback(f"Starting classification for: {file_name}")
                    else:
                        metrics.add_extraction_result(False, "No text extracted")
                        if log_callback:
                            log_callback(f"No text extracted from: {file_name}")
                        continue

                    # Classify content with improved timeout handling
                    try:
                        # Get the actual category from the file's location in dest_root
                        rel_path = os.path.relpath(path, dest_root)
                        actual_category = os.path.dirname(rel_path).replace(os.sep, "/")
                        if not actual_category or actual_category == ".":
                            actual_category = "Root"
                            
                        if log_callback:
                            log_callback(f"Running classification for: {file_name}")
                        
                        # Classify with timeout based on text length
                        text_length = len(text)
                        timeout_seconds = min(15, max(5, text_length // 10000))  # Adjust timeout based on text length
                        
                        import threading
                        import queue
                        
                        result_queue = queue.Queue()
                        def classification_worker():
                            try:
                                label, confidence = classify_document(text)
                                result_queue.put((True, (label, confidence)))
                            except Exception as e:
                                result_queue.put((False, str(e)))

                        # Start classification in a separate thread
                        classification_thread = threading.Thread(target=classification_worker)
                        classification_thread.daemon = True
                        classification_thread.start()
                        classification_thread.join(timeout=timeout_seconds)
                        
                        if classification_thread.is_alive():
                            error_msg = f"Classification timed out for {file_name}"
                            logging.error(error_msg)
                            if log_callback:
                                log_callback(error_msg)
                            metrics.add_extraction_result(False, "Classification timeout")
                            continue

                        # Get the result
                        try:
                            success, result = result_queue.get_nowait()
                            if success:
                                label, confidence = result
                                metrics.add_classification_result(actual_category, label, confidence)
                                if log_callback:
                                    log_callback(f"Classified {file_name} as: {label} (confidence: {confidence:.2f})")
                                    log_callback(f"Actual category: {actual_category}")
                            else:
                                error_msg = f"Classification failed for {file_name}: {result}"
                                logging.error(error_msg)
                                if log_callback:
                                    log_callback(error_msg)
                                metrics.add_extraction_result(False, f"Classification error: {result}")
                        except queue.Empty:
                            error_msg = f"Classification result not available for {file_name}"
                            logging.error(error_msg)
                            if log_callback:
                                log_callback(error_msg)
                            metrics.add_extraction_result(False, "Classification result not available")
                            continue

                    except Exception as e:
                        error_msg = f"Error during classification of {file_name}: {str(e)}"
                        logging.error(error_msg)
                        if log_callback:
                            log_callback(error_msg)
                        metrics.add_extraction_result(False, f"Classification error: {str(e)}")
                        continue

                    # Record processing time
                    file_processing_time = time.time() - file_start_time
                    metrics.add_processing_time(file_processing_time)
                    if log_callback:
                        log_callback(f"Completed processing {file_name} in {file_processing_time:.1f}s")

                except Exception as e:
                    error_msg = f"Error processing {file_name}: {str(e)}"
                    logging.error(error_msg)
                    if log_callback:
                        log_callback(error_msg)
                    metrics.add_extraction_result(False, str(e))

            # Small delay between batches
            time.sleep(0.01)

        metrics.end_processing()
        
        if progress_callback:
            progress_callback(100, "Generating evaluation report...")
        
        # Generate evaluation report
        if log_callback:
            log_callback("Generating evaluation report...")
        report_path = generate_evaluation_report(metrics)
        
        if log_callback:
            log_callback(f"Evaluation report generated: {report_path}")
            log_callback("Evaluation completed successfully!")

        return metrics.get_metrics()

    except Exception as e:
        error_msg = f"Evaluation error: {str(e)}"
        logging.error(error_msg)
        if log_callback:
            log_callback(error_msg)
        return None

def generate_evaluation_report(metrics):
    """Generate a detailed evaluation report with visualizations if available."""
    results = metrics.get_metrics()
    
    # Create report directory
    report_dir = "evaluation_reports"
    os.makedirs(report_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save metrics to CSV
    report_path = os.path.join(report_dir, f"evaluation_report_{timestamp}.csv")
    with open(report_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        for metric, value in results.items():
            if isinstance(value, dict):
                writer.writerow([metric, json.dumps(value)])
            else:
                writer.writerow([metric, value])

    # Generate visualizations only if matplotlib is available
    if MATPLOTLIB_AVAILABLE:
        try:
            plt.figure(figsize=(15, 10))

            # Classification accuracy pie chart
            plt.subplot(2, 2, 1)
            labels = ['Correct', 'Incorrect']
            sizes = [results['Classification Accuracy'], 100 - results['Classification Accuracy']]
            plt.pie(sizes, labels=labels, autopct='%1.1f%%')
            plt.title('Classification Accuracy')

            # Category distribution bar chart
            plt.subplot(2, 2, 2)
            categories = list(results['Category Distribution'].keys())
            values = list(results['Category Distribution'].values())
            plt.bar(range(len(categories)), values)
            plt.xticks(range(len(categories)), categories, rotation=45)
            plt.title('Category Distribution')

            # Error types bar chart
            plt.subplot(2, 2, 3)
            error_types = list(results['Error Types'].keys())
            error_counts = list(results['Error Types'].values())
            plt.bar(range(len(error_types)), error_counts)
            plt.xticks(range(len(error_types)), error_types, rotation=45)
            plt.title('Error Types')

            # Processing time histogram
            plt.subplot(2, 2, 4)
            plt.hist(metrics.processing_times, bins=20)
            plt.title('Processing Time Distribution')

            plt.tight_layout()
            plt.savefig(os.path.join(report_dir, f"evaluation_visualization_{timestamp}.png"))
            plt.close()
            
            logging.info("Generated visualization plots successfully")
        except Exception as e:
            logging.error(f"Error generating visualizations: {e}")
    else:
        logging.info("Skipping visualizations as matplotlib is not available")

    logging.info(f"Evaluation report generated: {report_path}")
    return report_path 