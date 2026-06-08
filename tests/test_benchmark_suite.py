import os
import sys
import shutil

sys.path.append(os.path.abspath("."))

from benchmark_suite import generate_dataset, run_benchmark

def test_benchmark_suite_execution():
    test_dataset_dir = "dataset/test_generated"
    test_output_dir = "output/test_benchmarks"

    # Clean up any previous test artifacts
    if os.path.exists(test_dataset_dir):
        shutil.rmtree(test_dataset_dir)
    if os.path.exists(test_output_dir):
        shutil.rmtree(test_output_dir)

    try:
        # Run dataset generation with repeats=1 (fast)
        gen_callbacks = []
        def gen_callback(curr, total):
            gen_callbacks.append((curr, total))

        manifest = generate_dataset(test_dataset_dir, repeats=1, progress_callback=gen_callback)
        
        assert len(manifest) > 0
        assert os.path.exists(os.path.join(test_dataset_dir, "manifest.csv"))
        assert len(gen_callbacks) > 0
        assert gen_callbacks[-1][0] == gen_callbacks[-1][1]  # Reached 100%

        # Run benchmark
        eval_callbacks = []
        def eval_callback(curr, total):
            eval_callbacks.append((curr, total))

        report = run_benchmark(test_dataset_dir, test_output_dir, progress_callback=eval_callback)

        assert report is not None
        assert "summary" in report
        assert "confusion_matrix_path" in report
        assert "roc_curves_path" in report

        assert os.path.exists(report["confusion_matrix_path"])
        assert os.path.exists(report["roc_curves_path"])
        assert os.path.exists(os.path.join(test_output_dir, "benchmark_summary.json"))
        
        assert len(eval_callbacks) > 0
        assert eval_callbacks[-1][0] == eval_callbacks[-1][1]  # Reached 100%

        print("Benchmark Suite unit test passed successfully!")

    finally:
        # Clean up test directories
        if os.path.exists(test_dataset_dir):
            shutil.rmtree(test_dataset_dir)
        if os.path.exists(test_output_dir):
            shutil.rmtree(test_output_dir)

if __name__ == "__main__":
    test_benchmark_suite_execution()
