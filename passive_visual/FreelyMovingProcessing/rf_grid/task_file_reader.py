import pandas as pd
import numpy as np
import os
from datetime import datetime
import re

class TaskFileReader:
    """
    Reader for Unity visual stimulus experiment task files.
    Parses the saved .txt files and extracts experiment data and metadata.
    """
    
    def __init__(self, filepath):
        """
        Initialize the reader with a task file path.
        
        Args:
            filepath (str): Path to the .txt task file
        """
        self.filepath = filepath
        self.metadata = {}
        self.experiment_params = {}
        self.grid_settings = {}
        self.trial_data = None
        self.trial_sequence = []
        
        # Load and parse the file
        self._parse_file()
    
    def _parse_file(self):
        """Parse the task file and extract all information."""
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"Task file not found: {self.filepath}")
        
        with open(self.filepath, 'r') as file:
            lines = file.readlines()
        
        # Parse different sections
        data_start_idx = self._parse_header_and_metadata(lines)
        self._parse_trial_data(lines, data_start_idx)
        self._parse_trial_sequence(lines)
    
    def _parse_header_and_metadata(self, lines):
        """Parse header and metadata sections."""
        data_start_idx = 0
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Skip empty lines and section headers
            if not line or line.startswith('# Visual Stimulus') or line.startswith('# Trial Data Columns'):
                continue
            
            # Parse metadata
            if line.startswith('# Generated on:'):
                self.metadata['generated_on'] = line.split(':', 1)[1].strip()
            elif line.startswith('# Animal ID:'):
                self.metadata['animal_id'] = line.split(':', 1)[1].strip()
            elif line.startswith('# Experiment ID:'):
                self.metadata['experiment_id'] = line.split(':', 1)[1].strip()
            elif line.startswith('# Experiment Start Time:'):
                self.metadata['experiment_start_time'] = line.split(':', 1)[1].strip()
            elif line.startswith('# Notes:'):
                self.metadata['notes'] = line.split(':', 1)[1].strip()
            
            # Parse experiment parameters
            elif line.startswith('# Stimulus Duration:'):
                value = line.split(':')[1].strip().split()[0]  # Get number part
                self.experiment_params['stimulus_duration'] = float(value)
            elif line.startswith('# Repetitions Per Cell:'):
                self.experiment_params['repetitions_per_cell'] = int(line.split(':')[1].strip())
            elif line.startswith('# Total Trials:'):
                self.experiment_params['total_trials'] = int(line.split(':')[1].strip())
            elif line.startswith('# Trials Completed:'):
                self.experiment_params['trials_completed'] = int(line.split(':')[1].strip())
            
            # Parse grid settings
            elif line.startswith('# Azimuth Range:'):
                range_text = line.split(':')[1].strip()
                az_min, az_max = re.findall(r'-?\d+\.?\d*', range_text)
                self.grid_settings['azimuth_min'] = float(az_min)
                self.grid_settings['azimuth_max'] = float(az_max)
            elif line.startswith('# Altitude Range:'):
                range_text = line.split(':')[1].strip()
                alt_min, alt_max = re.findall(r'-?\d+\.?\d*', range_text)
                self.grid_settings['altitude_min'] = float(alt_min)
                self.grid_settings['altitude_max'] = float(alt_max)
            elif line.startswith('# Azimuth Divisions:'):
                self.grid_settings['azimuth_divisions'] = int(line.split(':')[1].strip())
            elif line.startswith('# Altitude Divisions:'):
                self.grid_settings['altitude_divisions'] = int(line.split(':')[1].strip())
            
            # Find where trial data starts (first non-comment line)
            elif not line.startswith('#'):
                data_start_idx = i
                break
        
        return data_start_idx
    
    def _parse_trial_data(self, lines, start_idx):
        """Parse the trial data section."""
        # Find the end of trial data (before trial sequence section)
        end_idx = len(lines)
        for i in range(start_idx, len(lines)):
            if lines[i].strip().startswith('#') and 'Original Trial Sequence' in lines[i]:
                end_idx = i
                break
        
        # Extract trial data lines
        data_lines = []
        for i in range(start_idx, end_idx):
            line = lines[i].strip()
            if line and not line.startswith('#'):
                data_lines.append(line)
        
        if not data_lines:
            print("Warning: No trial data found")
            return
        
        # Parse trial data into DataFrame
        columns = ['TrialIndex', 'StimulusIndex', 'TrialStartTime', 'StimulusDuration', 'ITIDuration',
                  'Azimuth', 'Altitude', 'AzGridIndex', 'AltGridIndex']
        
        trial_data = []
        for line in data_lines:
            parts = line.split('\t')
            if len(parts) >= 8:
                trial_data.append([
                    int(parts[0]),      # TrialIndex
                    int(parts[1]),      # StimulusIndex
                    parts[2],           # TrialStartTime (keep as string initially)
                    float(parts[3]),    # StimulusDuration
                    float(parts[4]),    # ITIDuration
                    float(parts[5]),    # Azimuth
                    float(parts[6]),    # Altitude
                    int(parts[7]),      # AzGridIndex
                    int(parts[8])       # AltGridIndex
                ])
        
        self.trial_data = pd.DataFrame(trial_data, columns=columns)
        
        # Convert TrialStartTime to datetime
        if not self.trial_data.empty:
            self.trial_data['TrialStartTime'] = pd.to_datetime(self.trial_data['TrialStartTime'])
    
    def _parse_trial_sequence(self, lines):
        """Parse the original trial sequence."""
        sequence_started = False
        sequence_lines = []
        
        for line in lines:
            line = line.strip()
            if 'Original Trial Sequence' in line:
                sequence_started = True
                continue
            
            if sequence_started and line.startswith('# '):
                # Remove '# ' prefix and collect sequence data
                sequence_part = line[2:].strip()
                if sequence_part:
                    sequence_lines.append(sequence_part)
        
        # Parse the comma-separated sequence
        if sequence_lines:
            sequence_text = ''.join(sequence_lines)
            if sequence_text:
                try:
                    self.trial_sequence = [int(x.strip()) for x in sequence_text.split(',') if x.strip()]
                except ValueError:
                    print("Warning: Could not parse trial sequence")
    
    def get_trial_data(self):
        """
        Get the trial data as a pandas DataFrame.
        
        Returns:
            pd.DataFrame: Trial data with columns for each measurement
        """
        return self.trial_data.copy() if self.trial_data is not None else pd.DataFrame()
    
    def get_metadata(self):
        """
        Get experiment metadata.
        
        Returns:
            dict: Metadata information
        """
        return self.metadata.copy()
    
    def get_experiment_params(self):
        """
        Get experiment parameters.
        
        Returns:
            dict: Experiment parameters
        """
        return self.experiment_params.copy()
    
    def get_grid_settings(self):
        """
        Get grid settings.
        
        Returns:
            dict: Grid configuration
        """
        return self.grid_settings.copy()
    
    def get_trial_sequence(self):
        """
        Get the original trial sequence.
        
        Returns:
            list: Original stimulus sequence
        """
        return self.trial_sequence.copy()
    
    def get_summary(self):
        """
        Get a summary of the experiment.
        
        Returns:
            dict: Summary information
        """
        summary = {
            'file_path': self.filepath,
            'animal_id': self.metadata.get('animal_id', 'Unknown'),
            'experiment_id': self.metadata.get('experiment_id', 'Unknown'),
            'total_trials_planned': self.experiment_params.get('total_trials', 0),
            'trials_completed': self.experiment_params.get('trials_completed', 0),
            'stimulus_duration': self.experiment_params.get('stimulus_duration', 0),
            'grid_size': f"{self.grid_settings.get('azimuth_divisions', 0)}x{self.grid_settings.get('altitude_divisions', 0)}",
            'azimuth_range': f"{self.grid_settings.get('azimuth_min', 0):.1f}° to {self.grid_settings.get('azimuth_max', 0):.1f}°",
            'altitude_range': f"{self.grid_settings.get('altitude_min', 0):.1f}° to {self.grid_settings.get('altitude_max', 0):.1f}°"
        }
        
        if self.trial_data is not None and not self.trial_data.empty:
            summary['actual_trials_recorded'] = len(self.trial_data)
            summary['unique_stimuli'] = self.trial_data['StimulusIndex'].nunique()
            summary['experiment_duration'] = str(self.trial_data['TrialStartTime'].max() - self.trial_data['TrialStartTime'].min())
        
        return summary
    
    def print_summary(self):
        """Print a formatted summary of the experiment."""
        summary = self.get_summary()
        
        print("=" * 60)
        print("VISUAL STIMULUS EXPERIMENT SUMMARY")
        print("=" * 60)
        print(f"File: {summary['file_path']}")
        print(f"Animal ID: {summary['animal_id']}")
        print(f"Experiment ID: {summary['experiment_id']}")
        print()
        print("TRIAL INFORMATION:")
        print(f"  Planned trials: {summary['total_trials_planned']}")
        print(f"  Completed trials: {summary['trials_completed']}")
        if 'actual_trials_recorded' in summary:
            print(f"  Recorded trials: {summary['actual_trials_recorded']}")
        print(f"  Stimulus duration: {summary['stimulus_duration']} seconds")
        print()
        print("GRID SETTINGS:")
        print(f"  Grid size: {summary['grid_size']}")
        print(f"  Azimuth range: {summary['azimuth_range']}")
        print(f"  Altitude range: {summary['altitude_range']}")
        if 'unique_stimuli' in summary:
            print(f"  Unique stimuli: {summary['unique_stimuli']}")
        if 'experiment_duration' in summary:
            print(f"  Experiment duration: {summary['experiment_duration']}")
        print("=" * 60)


def load_task_file(filepath):
    """
    Convenience function to load a task file.
    
    Args:
        filepath (str): Path to the task file
        
    Returns:
        TaskFileReader: Loaded task file reader
    """
    return TaskFileReader(filepath)


# Example usage and demonstration
if __name__ == "__main__":
    # Example of how to use the TaskFileReader
    
    # Load a task file (replace with your actual file path)
    # task_file = load_task_file("mouse001_20250817_rfmapping.txt")
    
    # Example with dummy file path (you'll need to replace this)
    example_file = "example_task_file.txt"
    
    print("TaskFileReader Usage Example")
    print("=" * 40)
    print()
    print("1. Load a task file:")
    print(f"   task_file = load_task_file('{example_file}')")
    print()
    print("2. Get trial data as DataFrame:")
    print("   trial_data = task_file.get_trial_data()")
    print("   print(trial_data.head())")
    print()
    print("3. Access individual columns:")
    print("   stimulus_indices = trial_data['StimulusIndex']")
    print("   azimuth_positions = trial_data['Azimuth']")
    print("   altitude_positions = trial_data['Altitude']")
    print("   trial_times = trial_data['TrialStartTime']")
    print()
    print("4. Get metadata and settings:")
    print("   metadata = task_file.get_metadata()")
    print("   grid_settings = task_file.get_grid_settings()")
    print("   trial_sequence = task_file.get_trial_sequence()")
    print()
    print("5. Print summary:")
    print("   task_file.print_summary()")
    print()
    print("Available DataFrame columns:")
    columns = ['TrialIndex', 'StimulusIndex', 'TrialStartTime', 'StimulusDuration', 
               'Azimuth', 'Altitude', 'AzGridIndex', 'AltGridIndex']
    for col in columns:
        print(f"   - {col}")