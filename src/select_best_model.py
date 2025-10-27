import os
import json
import yaml
import shutil
import logging
import argparse
import pandas as pd

# --- Logging setup ---
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
logger = logging.getLogger('select_best_model')
logger.setLevel('DEBUG')
log_file_path = os.path.join(log_dir,'select_best_model.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def select_model(config_path: str):
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
            
        train_params = config['train']
        select_params = config['select_best']
        
        reports_dir = train_params['reports_dir']
        models_dir = train_params['models_dir']
        primary_metric = train_params['primary_metric']
        metric_goal = train_params['metric_goal']
        
        final_model_path = select_params['final_model_path']
        summary_path = select_params['best_model_summary_path']

        if not os.path.exists(reports_dir) or not os.listdir(reports_dir):
            logger.error(f"Reports directory is empty or missing: {reports_dir}")
            raise FileNotFoundError("No experiment reports found. Run the train stage first.")

        all_metrics = []
        for exp_name in os.listdir(reports_dir):
            metric_file = os.path.join(reports_dir, exp_name, 'metrics.json')
            if os.path.exists(metric_file):
                with open(metric_file, 'r') as f:
                    metrics = json.load(f)
                    metrics['model_name'] = exp_name
                    all_metrics.append(metrics)
            
        if not all_metrics:
            logger.error("No metrics.json files found in report directories.")
            raise FileNotFoundError("No metrics found.")

        metrics_df = pd.DataFrame(all_metrics)
        logger.info(f"Found {len(metrics_df)} model experiments:\n{metrics_df}")

        # Find the best model
        if metric_goal == 'maximize':
            best_model_row = metrics_df.loc[metrics_df[primary_metric].idxmax()]
        elif metric_goal == 'minimize':
            best_model_row = metrics_df.loc[metrics_df[primary_metric].idxmin()]
        else:
            raise ValueError(f"Invalid metric_goal: {metric_goal}. Must be 'maximize' or 'minimize'.")

        best_model_name = best_model_row['model_name']
        best_metrics = best_model_row.to_dict()
        
        logger.info(f"Best model selected: {best_model_name}")
        logger.info(f"Best metrics: {best_metrics}")

        # Copy the best model to the final path
        best_model_src = os.path.join(models_dir, f"{best_model_name}.pkl")
        os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
        shutil.copy(best_model_src, final_model_path)
        logger.info(f"Best model copied to {final_model_path}")

        # Save summary
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        with open(summary_path, 'w') as f:
            json.dump(best_metrics, f, indent=4)
        logger.info(f"Best model summary saved to {summary_path}")

    except Exception as e:
        logger.error(f"An error occurred during model selection: {e}")
        raise e

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()
    select_model(config_path=args.config)