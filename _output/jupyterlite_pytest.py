import csv
import os
import pytest
import time
from playwright.sync_api import Page, TimeoutError

NOTEBOOK_NAMES = [
    "01_tabular_data_exploration.ipynb",
    "01_tabular_data_exploration_ex_01.ipynb",
    "01_tabular_data_exploration_sol_01.ipynb",
    "02_numerical_pipeline_cross_validation.ipynb",
    "02_numerical_pipeline_ex_00.ipynb",
    "02_numerical_pipeline_ex_01.ipynb",
    "02_numerical_pipeline_hands_on.ipynb",
    "02_numerical_pipeline_introduction.ipynb",
    "02_numerical_pipeline_scaling.ipynb",
    "02_numerical_pipeline_sol_00.ipynb",
    "02_numerical_pipeline_sol_01.ipynb",
    "03_categorical_pipeline.ipynb",
    "03_categorical_pipeline_column_transformer.ipynb",
    "03_categorical_pipeline_ex_01.ipynb",
    "03_categorical_pipeline_ex_02.ipynb",
    "03_categorical_pipeline_sol_01.ipynb",
    "03_categorical_pipeline_sol_02.ipynb",
    "03_categorical_pipeline_visualization.ipynb",
    "clustering_ex_01.ipynb",
    "clustering_ex_02.ipynb",
    "clustering_hdbscan.ipynb",
    "clustering_kmeans.ipynb",
    "clustering_kmeans_ex_01.ipynb",
    "clustering_kmeans_sol_01.ipynb",
    "clustering_kmeans_sol_02.ipynb",
    "clustering_sol_01.ipynb",
    "clustering_sol_02.ipynb",
    "clustering_supervised_metrics.ipynb",
    "clustering_transformer.ipynb",
    "cross_validation_baseline.ipynb",
    "cross_validation_ex_01.ipynb",
    "cross_validation_ex_02.ipynb",
    "cross_validation_grouping.ipynb",
    "cross_validation_learning_curve.ipynb",
    "cross_validation_nested.ipynb",
    "cross_validation_sol_01.ipynb",
    "cross_validation_sol_02.ipynb",
    "cross_validation_stratification.ipynb",
    "cross_validation_time.ipynb",
    "cross_validation_train_test.ipynb",
    "cross_validation_validation_curve.ipynb",
    "ensemble_adaboost.ipynb",
    "ensemble_bagging.ipynb",
    "ensemble_ex_01.ipynb",
    "ensemble_ex_02.ipynb",
    "ensemble_ex_03.ipynb",
    "ensemble_ex_04.ipynb",
    "ensemble_gradient_boosting.ipynb",
    "ensemble_hist_gradient_boosting.ipynb",
    "ensemble_hyperparameters.ipynb",
    "ensemble_introduction.ipynb",
    "ensemble_random_forest.ipynb",
    "ensemble_sol_01.ipynb",
    "ensemble_sol_02.ipynb",
    "ensemble_sol_03.ipynb",
    "ensemble_sol_04.ipynb",
    "feature_selection_ex_01.ipynb",
    "feature_selection_introduction.ipynb",
    "feature_selection_limitation_model.ipynb",
    "feature_selection_sol_01.ipynb",
    "linear_models_ex_01.ipynb",
    "linear_models_ex_02.ipynb",
    "linear_models_ex_03.ipynb",
    "linear_models_ex_04.ipynb",
    "linear_models_feature_engineering_classification.ipynb",
    "linear_models_regularization.ipynb",
    "linear_models_sol_01.ipynb",
    "linear_models_sol_02.ipynb",
    "linear_models_sol_03.ipynb",
    "linear_models_sol_04.ipynb",
    "linear_regression_in_sklearn.ipynb",
    "linear_regression_non_linear_link.ipynb",
    "linear_regression_without_sklearn.ipynb",
    "logistic_regression.ipynb",
    "metrics_classification.ipynb",
    "metrics_ex_01.ipynb",
    "metrics_ex_02.ipynb",
    "metrics_regression.ipynb",
    "metrics_sol_01.ipynb",
    "metrics_sol_02.ipynb",
    "parameter_tuning_ex_02.ipynb",
    "parameter_tuning_ex_03.ipynb",
    "parameter_tuning_grid_search.ipynb",
    "parameter_tuning_manual.ipynb",
    "parameter_tuning_nested.ipynb",
    "parameter_tuning_parallel_plot.ipynb",
    "parameter_tuning_randomized_search.ipynb",
    "parameter_tuning_sol_02.ipynb",
    "parameter_tuning_sol_03.ipynb",
    "trees_classification.ipynb",
    "trees_dataset.ipynb",
    "trees_ex_01.ipynb",
    "trees_ex_02.ipynb",
    "trees_hyperparameters.ipynb",
    "trees_regression.ipynb",
    "trees_sol_01.ipynb",
    "trees_sol_02.ipynb",
]


def log_to_csv(notebook_name, execution_time, has_errors, has_warnings):
    """Log test results to CSV file."""
    csv_file = "test_results.csv"
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, "a", newline="") as f:
        fieldnames = [
            "notebook_name",
            "execution_time",
            "has_errors",
            "has_warnings",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        # Write header if file is new
        if not file_exists:
            writer.writeheader()

        # Write the test result
        writer.writerow(
            {
                "notebook_name": notebook_name,
                "execution_time": f"{execution_time:.2f}",
                "has_errors": has_errors,
                "has_warnings": has_warnings,
            }
        )


@pytest.fixture(scope="function")
def browser_context_args():
    """Configure browser context with appropriate settings."""
    return {
        "viewport": {"width": 1280, "height": 720},
        "ignore_https_errors": True,
    }


@pytest.mark.parametrize("notebook_name", NOTEBOOK_NAMES)
def test_run_jupyterlite_notebook(page: Page, notebook_name: str):
    """
    Test that runs a JupyterLite notebook using Playwright.

    Steps:
    1. Navigate to the notebook URL
    2. Wait for kernel to initialize
    3. Click on Run menu
    4. Select "Run all cells"
    """

    # Construct URL with the notebook name
    base_url = (
        "https://probabl-static.france-ioi.org/jupyter/v0/notebooks/index.html"
    )
    url = f"{base_url}?sLocale=en&path=notebooks%2F{notebook_name}"
    print(f"\nTesting notebook: {notebook_name}")

    # Navigate to the notebook
    page.goto(url, wait_until="networkidle")

    # Wait for the main notebook interface to be visible
    page.wait_for_selector(".jp-Notebook", state="visible", timeout=30000)

    # Wait for kernel to initialize
    kernel_ready = False
    max_wait_time = 60  # Maximum 60 seconds to wait for kernel
    start_time = time.time()

    while not kernel_ready and (time.time() - start_time) < max_wait_time:
        try:
            # Check for kernel status - looking for "Idle" or ready state
            # JupyterLite may show this in different ways
            kernel_status = page.locator('[data-status="idle"]').first
            if kernel_status.is_visible():
                kernel_ready = True
                break

            # # Alternative: check for the kernel indicator in the toolbar
            # toolbar_kernel = page.locator(".jp-KernelStatus-fade").first
            # if toolbar_kernel.is_visible():
            #     kernel_ready = True
            #     break

            # # Also check if we can see the kernel name in the toolbar
            # kernel_name = page.locator(".jp-KernelName").first
            # if kernel_name.is_visible():
            #     # Give it a bit more time to fully initialize
            #     page.wait_for_timeout(2000)
            #     kernel_ready = True
            #     break

        except (TimeoutError, AttributeError):
            pass

        # Wait a bit before checking again
        page.wait_for_timeout(1000)

    if not kernel_ready:
        print("Warning: Kernel initialization timeout, proceeding anyway...")

    # Additional wait to ensure everything is loaded
    page.wait_for_timeout(2000)

    # Click on the "Run" menu
    run_menu = page.locator('div.lm-MenuBar-itemLabel:has-text("Run")').first
    run_menu.click()

    # Wait for menu to open
    page.wait_for_timeout(500)

    # Click on "Run All Cells" option
    run_all_cells = page.locator(
        'div.lm-Menu-itemLabel:has-text("Run All Cells")'
    ).first
    run_all_cells.click()
    execution_start_time = time.time()

    # Wait for cells to start executing
    page.wait_for_timeout(2000)

    # Get all code cells in the notebook
    code_cells = page.locator(".jp-CodeCell")
    total_code_cells = code_cells.count()

    # Monitor cell execution
    max_execution_time = 300  # 5 minutes max for all cells to execute
    last_executed_count = 0
    stable_count = 0
    start_time = time.time()
    all_cells_complete = False

    while (
        not all_cells_complete
        and (time.time() - start_time) < max_execution_time
    ):
        try:
            # Count executed cells - those with [N] where N is a number
            executed_prompts = page.locator(".jp-InputPrompt").all()
            executed_cells = 0
            running_cells = 0

            for prompt in executed_prompts:
                text = prompt.inner_text().strip()
                if "[*]" in text:
                    running_cells += 1
                elif text and text.startswith("[") and text.endswith("]"):
                    # It's [1], [2], etc.
                    executed_cells += 1

            # Check if execution count is stable
            if executed_cells == last_executed_count:
                stable_count += 1
            else:
                stable_count = 0
                last_executed_count = executed_cells

            # Consider complete if all expected cells executed or stable state
            if executed_cells >= total_code_cells:
                all_cells_complete = True
                execution_time = time.time() - execution_start_time
                print(
                    f"All {executed_cells} cells completed in"
                    f" {execution_time:.2f} seconds!"
                )
                break
            elif running_cells == 0 and stable_count >= 3:
                all_cells_complete = True
                execution_time = time.time() - execution_start_time
                print(
                    f"Execution complete: {total_code_cells} cells in"
                    f" {execution_time:.2f} seconds"
                )
                break

            # Adaptive waiting
            if running_cells > 0:
                page.wait_for_timeout(1000)
            else:
                page.wait_for_timeout(3000)

        except Exception as e:
            print(f"Error checking cell status: {e}")
            page.wait_for_timeout(2000)

    if not all_cells_complete:
        print(
            "Warning: Cell execution timeout after"
            f" {max_execution_time} seconds"
        )
        # Optionally take a debug screenshot
        # page.screenshot(path="execution_timeout.png")
        # print("Timeout screenshot saved as execution_timeout.png")
    else:
        execution_time = time.time() - execution_start_time

    # Final wait to ensure everything has settled
    page.wait_for_timeout(3000)

    # Verify that cells have output
    output_areas = page.locator(".jp-OutputArea")
    output_count = output_areas.count()
    print(f"Found {output_count} output areas")

    # Verify the notebook is still responsive and cells have executed
    assert page.locator(
        ".jp-Notebook"
    ).is_visible(), "Notebook interface is not visible"
    assert (
        output_count > 0 or total_code_cells > 0
    ), "No cells or outputs found"

    # Check for any error and warning outputs
    error_outputs = page.locator(
        ".jp-OutputArea-output[data-mime-type*='error']"
    ).count()
    warning_outputs = page.locator(
        ".jp-OutputArea-output:has-text('Warning')"
    ).count()
    has_errors = 1 if error_outputs > 0 else 0
    has_warnings = 1 if warning_outputs > 0 else 0
    if has_errors:
        pytest.fail("Notebook execution failed with errors")
    elif has_warnings:
        print("⚠️ Notebook has warnings")

    log_to_csv(notebook_name, execution_time, has_errors, has_warnings)
    print(
        f"Total execution time for {notebook_name}:"
        f" {execution_time:.2f} seconds"
    )
