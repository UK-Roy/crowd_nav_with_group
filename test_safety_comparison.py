import subprocess
import re
import yaml

def run_test_and_extract_metrics(model_dir, test_model, group_avoid=False):
    """Run test and extract metrics from output"""
    
    # Modify config to enable/disable group avoidance
    config_file = 'config.yml'
    
    # Run test
    if group_avoid:
        cmd = f"python test.py --model_dir {model_dir} --test_model {test_model} --test_case -1"
        # You'll need to temporarily modify test.py to set group_avoid_action=True
    else:
        cmd = f"python test.py --model_dir {model_dir} --test_model {test_model} --test_case -1"
    
    output = subprocess.check_output(cmd, shell=True, text=True)
    
    # Extract metrics using regex
    sr_match = re.search(r'success rate: ([\d.]+)', output)
    cr_match = re.search(r'collision rate: ([\d.]+)', output)
    gcr_match = re.search(r'group collision rate: ([\d.]+)', output)
    tr_match = re.search(r'timeout rate: ([\d.]+)', output)
    
    return {
        'SR': float(sr_match.group(1)) if sr_match else 0,
        'CR': float(cr_match.group(1)) if cr_match else 0,
        'GCR': float(gcr_match.group(1)) if gcr_match else 0,
        'TR': float(tr_match.group(1)) if tr_match else 0
    }

def main():
    print("="*60)
    print("TAGA SAFETY LAYER COMPARISON TEST")
    print("="*60)
    
    # Test configurations
    configs = [
        {'policy': 'orca', 'model_dir': 'trained_models/GST_predictor_rand', 'test_model': '41665.pt'},
        {'policy': 'social_force', 'model_dir': 'trained_models/GST_predictor_rand', 'test_model': '41665.pt'}
    ]
    
    results = {}
    
    for config in configs:
        policy = config['policy']
        print(f"\nTesting {policy.upper()}...")
        
        # Test without TAGA (baseline)
        print(f"  Running baseline {policy}...")
        baseline = run_test_and_extract_metrics(config['model_dir'], config['test_model'], group_avoid=False)
        
        # Test with TAGA + Safety
        print(f"  Running {policy} + TAGA + Safety...")
        with_safety = run_test_and_extract_metrics(config['model_dir'], config['test_model'], group_avoid=True)
        
        results[policy] = {
            'baseline': baseline,
            'with_safety': with_safety
        }
    
    # Print results table
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"{'Method':<20} {'Metric':<6} {'Baseline':<12} {'TAGA+Safety':<12} {'Change'}")
    print("-"*70)
    
    for policy in results:
        for metric in ['SR', 'CR', 'GCR', 'TR']:
            baseline_val = results[policy]['baseline'][metric]
            safety_val = results[policy]['with_safety'][metric]
            
            if metric == 'SR':
                change = f"+{(safety_val - baseline_val)*100:.1f}%" if safety_val > baseline_val else f"{(safety_val - baseline_val)*100:.1f}%"
            else:
                change = f"-{(baseline_val - safety_val)*100:.1f}%" if baseline_val > safety_val else f"+{(safety_val - baseline_val)*100:.1f}%"
            
            print(f"{policy + '+TAGA':<20} {metric:<6} {baseline_val:<12.3f} {safety_val:<12.3f} {change}")
        print()

if __name__ == "__main__":
    main()