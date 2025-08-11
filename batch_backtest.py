#!/usr/bin/env python3
"""
Batch backtester for cryptocurrency trading strategies.
Run multiple backtests from configuration files.
"""

import os
import json
import logging
import argparse
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any, Optional

from main import run_backtest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("batch_backtest.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_single_backtest(config_file: str) -> Dict[str, Any]:
    """
    Run a single backtest from a config file
    
    Parameters:
    -----------
    config_file : str
        Path to config file
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary with results and config file name
    """
    config_name = os.path.splitext(os.path.basename(config_file))[0]
    
    try:
        # Read config
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Run backtest
        result = run_backtest(config)
        
        if result is None:
            return {"config": config_name, "status": "failed"}
        
        return {"config": config_name, "status": "success", "result": result}
    
    except Exception as e:
        logger.error(f"Error in backtest {config_file}: {str(e)}")
        return {"config": config_name, "status": "failed", "error": str(e)}

def batch_backtest(configs_dir: str, parallel: bool = False, max_workers: int = None) -> Dict[str, Any]:
    """
    Run multiple backtests from config files in a directory
    
    Parameters:
    -----------
    configs_dir : str
        Directory containing JSON config files
    parallel : bool, optional
        Whether to run backtests in parallel
    max_workers : int, optional
        Maximum number of worker processes for parallel execution
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary with results for each config
    """
    results = {}
    
    # List all JSON files in the configs directory
    config_files = [os.path.join(configs_dir, f) for f in os.listdir(configs_dir) if f.endswith('.json')]
    logger.info(f"Found {len(config_files)} config files")
    
    # Run backtests
    if parallel and len(config_files) > 1:
        logger.info(f"Running backtests in parallel with {max_workers or os.cpu_count()} workers")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all backtests
            future_to_config = {executor.submit(run_single_backtest, config_file): config_file 
                               for config_file in config_files}
            
            # Process results as they complete
            for future in as_completed(future_to_config):
                result = future.result()
                config_name = result["config"]
                results[config_name] = result
                
                if result["status"] == "success":
                    logger.info(f"Completed backtest: {config_name}")
                else:
                    logger.error(f"Failed backtest: {config_name}")
    else:
        logger.info("Running backtests sequentially")
        
        # Run backtest for each config file
        for config_file in config_files:
            result = run_single_backtest(config_file)
            config_name = result["config"]
            results[config_name] = result
            
            if result["status"] == "success":
                logger.info(f"Completed backtest: {config_name}")
            else:
                logger.error(f"Failed backtest: {config_name}")
    
    # Save all results to a combined file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_results_file = f"batch_results_{timestamp}.json"
    
    with open(combined_results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"All backtest results saved to {combined_results_file}")
    return results

def main():
    """CLI entry point for batch backtesting"""
    parser = argparse.ArgumentParser(description="Run multiple backtests from config files")
    
    parser.add_argument("--dir", "-d", required=True, help="Directory containing backtest config files")
    parser.add_argument("--parallel", "-p", action="store_true", help="Run backtests in parallel")
    parser.add_argument("--workers", "-w", type=int, help="Maximum number of worker processes")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.dir):
        print(f"Error: Directory {args.dir} not found.")
        return 1
    
    batch_backtest(args.dir, args.parallel, args.workers)
    return 0

if __name__ == "__main__":
    exit(main())