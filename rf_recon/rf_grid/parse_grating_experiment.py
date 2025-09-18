import re
from datetime import datetime
import pandas as pd
from typing import Dict, List, Any, Optional

def parse_grating_experiment(file_path: str) -> Dict[str, Any]:
    """
    Parse a grating experiment data file and return structured data.
    
    Args:
        file_path (str): Path to the experiment data file
        
    Returns:
        Dict containing:
        - metadata: Basic experiment information
        - parameters: Experiment parameters and ranges
        - trial_data: DataFrame with all trial data
        - summary: Experiment summary statistics
    """
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Initialize result dictionary
    result = {
        'metadata': {},
        'parameters': {},
        'trial_data': None,
        'summary': {}
    }
    
    # Parse metadata section
    metadata_patterns = {
        'generated': r'Generated: (.+)',
        'animal_id': r'Animal ID: (.+)',
        'experiment_id': r'Experiment ID: (.+)',
        'start_time': r'Start Time: (.+)',
        'end_time': r'End Time: (.+)'
    }
    
    for key, pattern in metadata_patterns.items():
        match = re.search(pattern, content)
        if match:
            value = match.group(1).strip()
            # Convert datetime strings to datetime objects
            if 'time' in key or key == 'generated':
                try:
                    result['metadata'][key] = datetime.strptime(value, '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    result['metadata'][key] = value
            else:
                result['metadata'][key] = value
    
    # Parse experiment parameters
    param_patterns = {
        'total_trials': r'Total Trials: (\d+)',
        'completed_trials': r'Completed Trials: (\d+)',
        'stimulus_duration': r'Stimulus Duration: (.+)',
        'iti_duration': r'ITI Duration: (.+)',
        'gray_screen_duration': r'Gray Screen Duration: (.+)',
        'gray_side_configuration': r'Gray Side Configuration: (.+)',
        'randomized': r'Randomized: (.+)',
        'repeats_per_condition': r'Repeats per condition: (\d+)'
    }
    
    for key, pattern in param_patterns.items():
        match = re.search(pattern, content)
        if match:
            value = match.group(1).strip()
            # Convert numeric values
            if key in ['total_trials', 'completed_trials', 'repeats_per_condition']:
                result['parameters'][key] = int(value)
            elif key == 'randomized':
                result['parameters'][key] = value.lower() == 'true'
            else:
                result['parameters'][key] = value
    
    # Parse parameter ranges
    range_patterns = {
        'left_spatial_frequencies': r'Left Spatial Frequencies: \[(.+?)\]',
        'left_contrasts': r'Left Contrasts: \[(.+?)\]',
        'left_phases': r'Left Phases: \[(.+?)\]',
        'left_orientations': r'Left Orientations: \[(.+?)\]',
        'right_spatial_frequencies': r'Right Spatial Frequencies: \[(.+?)\]',
        'right_contrasts': r'Right Contrasts: \[(.+?)\]',
        'right_phases': r'Right Phases: \[(.+?)\]',
        'right_orientations': r'Right Orientations: \[(.+?)\]'
    }
    
    result['parameters']['ranges'] = {}
    for key, pattern in range_patterns.items():
        match = re.search(pattern, content)
        if match:
            values_str = match.group(1).strip()
            # Parse the list values
            values = [float(x.strip()) for x in values_str.split(',')]
            result['parameters']['ranges'][key] = values
    
    # Parse trial data
    trial_section_match = re.search(r'TRIAL DATA:\s*-+\s*(.+?)\s*SUMMARY:', content, re.DOTALL)
    if trial_section_match:
        trial_content = trial_section_match.group(1).strip()
        lines = trial_content.split('\n')
        
        # Find the header line
        header_line = None
        data_lines = []
        
        for line in lines:
            line = line.strip()
            if line.startswith('Trial\t'):
                header_line = line
            elif line and not line.startswith('Trial\t') and '\t' in line:
                data_lines.append(line)
        
        if header_line and data_lines:
            # Parse header
            headers = header_line.split('\t')
            
            # Parse data
            trial_data = []
            for line in data_lines:
                values = line.split('\t')
                if len(values) == len(headers):
                    row = {}
                    for i, header in enumerate(headers):
                        value = values[i].strip()
                        # Convert data types appropriately
                        if header == 'Trial':
                            row[header] = int(value)
                        elif header in ['Start', 'End']:
                            # Keep as string for now, could convert to datetime if needed
                            row[header] = value
                        elif header in ['Duration', 'L_Orient', 'L_SF', 'L_Contrast', 'L_Phase', 
                                      'R_Orient', 'R_SF', 'R_Contrast', 'R_Phase', 'AnimalX', 
                                      'AnimalZ', 'Heading']:
                            row[header] = float(value)
                        else:
                            row[header] = value
                    trial_data.append(row)
            
            # Create DataFrame
            result['trial_data'] = pd.DataFrame(trial_data)
    
    # Parse summary
    summary_match = re.search(r'SUMMARY:\s*-+\s*(.+?)$', content, re.DOTALL)
    if summary_match:
        summary_content = summary_match.group(1).strip()
        summary_lines = summary_content.split('\n')
        
        for line in summary_lines:
            line = line.strip()
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower().replace(' ', '_')
                value = value.strip()
                
                # Convert values appropriately
                if 'rate' in key and '%' in value:
                    result['summary'][key] = float(value.replace('%', ''))
                elif 'duration' in key:
                    result['summary'][key] = value
                else:
                    result['summary'][key] = value
    
    return result

# Example usage function
def analyze_experiment_data(file_path: str) -> None:
    """
    Example function showing how to use the parsed data.
    """
    data = parse_grating_experiment(file_path)
    
    print("=== EXPERIMENT METADATA ===")
    for key, value in data['metadata'].items():
        print(f"{key}: {value}")
    
    print("\n=== EXPERIMENT PARAMETERS ===")
    for key, value in data['parameters'].items():
        if key != 'ranges':
            print(f"{key}: {value}")
    
    print("\n=== PARAMETER RANGES ===")
    if 'ranges' in data['parameters']:
        for key, value in data['parameters']['ranges'].items():
            print(f"{key}: {value}")
    
    print("\n=== TRIAL DATA OVERVIEW ===")
    if data['trial_data'] is not None:
        print(f"Shape: {data['trial_data'].shape}")
        print(f"Columns: {list(data['trial_data'].columns)}")
        print("\nFirst 5 trials:")
        print(data['trial_data'].head())
        
        print("\nBasic statistics:")
        print(data['trial_data'].describe())
    
    print("\n=== SUMMARY ===")
    for key, value in data['summary'].items():
        print(f"{key}: {value}")

# Additional utility functions
def get_trials_by_orientation(data: Dict[str, Any], orientation: float) -> Optional[pd.DataFrame]:
    """Get all trials for a specific left orientation."""
    if data['trial_data'] is not None:
        return data['trial_data'][data['trial_data']['L_Orient'] == orientation]
    return None

def calculate_position_statistics(data: Dict[str, Any]) -> Dict[str, float]:
    """Calculate basic statistics for animal position."""
    if data['trial_data'] is not None:
        df = data['trial_data']
        return {
            'mean_x': df['AnimalX'].mean(),
            'mean_z': df['AnimalZ'].mean(),
            'std_x': df['AnimalX'].std(),
            'std_z': df['AnimalZ'].std(),
            'mean_heading': df['Heading'].mean(),
            'std_heading': df['Heading'].std()
        }
    return {}