import subprocess
import re
import numpy as np
import json
import os

def run_single_test(test_case, model_dir, test_model, group_avoid=False):
    """Run a single test case and extract metrics"""
    
    # Build command
    cmd = [
        'python', 'test.py',
        '--model_dir', model_dir,
        '--test_model', test_model,
        '--test_case', str(test_case)
    ]
    
    # Modify test.py temporarily or use a flag to enable/disable TAGA
    if group_avoid:
        # You'll need to add a command line arg to test.py for this
        cmd.append('--group_avoid')
    
    # Run the test
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        
        # Parse the output to extract metrics
        metrics = {}
        
        # Extract success rate
        sr_match = re.search(r'success rate: ([\d.]+)', output)
        metrics['sr'] = float(sr_match.group(1)) if sr_match else 0.0
        
        # Extract collision rate
        cr_match = re.search(r'collision rate: ([\d.]+)', output)
        metrics['cr'] = float(cr_match.group(1)) if cr_match else 0.0
        
        # Extract timeout rate
        tr_match = re.search(r'timeout rate: ([\d.]+)', output)
        metrics['tr'] = float(tr_match.group(1)) if tr_match else 0.0
        
        # Extract GCR
        gcr_match = re.search(r'group intrusion rate \(GCR\): ([\d.]+)%', output)
        metrics['gcr'] = float(gcr_match.group(1)) if gcr_match else 0.0
        
        # Extract path length
        path_match = re.search(r'path length: ([\d.]+)', output)
        metrics['path_len'] = float(path_match.group(1)) if path_match else 0.0
        
        return metrics
        
    except subprocess.CalledProcessError as e:
        print(f"Error running test case {test_case}: {e}")
        return {
            'sr': 0.0,
            'cr': 1.0,
            'tr': 0.0,
            'gcr': 0.0,
            'path_len': 0.0
        }

def test_scenario_range(start, end, scenario_name, model_dir, test_model, group_avoid=False):
    """Test a range of test cases for a specific scenario"""
    
    results = {
        'success': [],
        'collision': [],
        'timeout': [],
        'gcr': [],
        'path_length': []
    }
    
    print(f"  Running test cases {start} to {end-1}...")
    
    for test_case in range(start, end):
        print(f"    Test case {test_case}...", end='')
        metrics = run_single_test(test_case, model_dir, test_model, group_avoid)
        
        results['success'].append(metrics['sr'])
        results['collision'].append(metrics['cr'])
        results['timeout'].append(metrics['tr'])
        results['gcr'].append(metrics['gcr'])
        results['path_length'].append(metrics['path_len'])
        
        print(f" SR={metrics['sr']:.2f}, CR={metrics['cr']:.2f}, GCR={metrics['gcr']:.1f}%")
    
    # Calculate averages
    return {
        'SR': np.mean(results['success']),
        'CR': np.mean(results['collision']),
        'TR': np.mean(results['timeout']),
        'GCR': np.mean(results['gcr']),
        'Path': np.mean(results['path_length']),
        'std_SR': np.std(results['success']),
        'std_CR': np.std(results['collision'])
    }

def main():
    scenarios = [
        ('Dense Groups', 0, 5),      # Test 5 cases per scenario for quick results
        ('Mixed 50-50', 20, 25),
        ('Dynamic Groups', 40, 45),
        ('Crossing Groups', 60, 65),
        ('Static-Dynamic Mix', 80, 85)
    ]
    
    model_dir = 'trained_models/GST_predictor_rand'
    test_model = '41665.pt'
    
    print("\n" + "="*70)
    print("SCENARIO-BASED EVALUATION RESULTS")
    print("="*70)
    
    all_results = {}
    
    for scenario_name, start, end in scenarios:
        print(f"\n{scenario_name} Scenario:")
        print("-"*40)
        
        # Test without TAGA
        print("  Testing baseline (no TAGA)...")
        results_baseline = test_scenario_range(start, end, scenario_name, 
                                              model_dir, test_model, group_avoid=False)
        
        # Test with TAGA
        print("  Testing with TAGA...")
        results_taga = test_scenario_range(start, end, scenario_name,
                                          model_dir, test_model, group_avoid=True)
        
        all_results[scenario_name] = {
            'baseline': results_baseline,
            'taga': results_taga
        }
        
        # Print immediate results
        print(f"\n  {scenario_name} Results:")
        print(f"    Baseline: SR={results_baseline['SR']:.2f}, CR={results_baseline['CR']:.2f}, GCR={results_baseline['GCR']:.1f}%")
        print(f"    TAGA:     SR={results_taga['SR']:.2f}, CR={results_taga['CR']:.2f}, GCR={results_taga['GCR']:.1f}%")
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY BY SCENARIO")
    print("="*80)
    print(f"{'Scenario':<20} {'Method':<10} {'SR ↑':<8} {'CR ↓':<8} {'GCR ↓':<8} {'Path':<8}")
    print("-"*80)
    
    for scenario_name, results in all_results.items():
        print(f"{scenario_name:<20} {'Baseline':<10} "
              f"{results['baseline']['SR']:.2f} "
              f"{results['baseline']['CR']:.2f} "
              f"{results['baseline']['GCR']:.1f}% "
              f"{results['baseline']['Path']:.1f}m")
        print(f"{'':20} {'TAGA':<10} "
              f"{results['taga']['SR']:.2f} "
              f"{results['taga']['CR']:.2f} "
              f"{results['taga']['GCR']:.1f}% "
              f"{results['taga']['Path']:.1f}m")
        
        # Print improvement
        sr_imp = (results['taga']['SR'] - results['baseline']['SR']) / results['baseline']['SR'] * 100
        cr_imp = (results['baseline']['CR'] - results['taga']['CR']) / results['baseline']['CR'] * 100
        gcr_imp = (results['baseline']['GCR'] - results['taga']['GCR']) / results['baseline']['GCR'] * 100
        
        print(f"{'':20} {'Improvement':<10} "
              f"{sr_imp:+.0f}% "
              f"{cr_imp:+.0f}% "
              f"{gcr_imp:+.0f}%")
        print()
    
    # Save results to file
    with open('scenario_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to scenario_results.json")

if __name__ == "__main__":
    main()