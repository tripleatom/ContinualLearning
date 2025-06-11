import numpy as np
from pathlib import Path
import spikeinterface.extractors as se
import spikeinterface.preprocessing as sp
from func import generate_dates, find_session_folders
import pickle
import logging
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
import warnings

# ─── Configuration ────────────────────────────────────────────────────────────

@dataclass
class ProcessingConfig:
    """Configuration parameters for neural signal processing."""
    base_dir: Path = Path(r"/Volumes/xieluanlabs/xl_cl/rf_reconstruction/head_fixed")
    date_start: str = "250307"
    date_end: str = "250601"
    animal_ids: List[str] = None
    shank_ids: List[int] = None
    out_subdir: str = "sortout"
    freq_min: float = 500.0
    freq_max: float = 5000.0
    
    def __post_init__(self):
        if self.animal_ids is None:
            self.animal_ids = ["CnL22", "CnL38", "CnL39", "CnL40", "CnL41"]
        if self.shank_ids is None:
            self.shank_ids = [0, 1, 2, 3]


class NeuralSignalProcessor:
    """Processes neural signal data from NWB files."""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.script_folder = Path(__file__).parent.parent
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Configure logging for the processing pipeline."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('neural_processing.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def process_shank(self, session_folder: Path, session_name: str, 
                     shank_id: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Load NWB file for a shank and compute mean-squared power per channel.
        
        Args:
            session_folder: Path to session directory
            session_name: Name of the session
            shank_id: ID of the shank to process
            
        Returns:
            Tuple of (sorted_power, sorted_depths) or None if processing fails
        """
        nwb_file = session_folder / f"{session_name}sh{shank_id}.nwb"
        
        if not nwb_file.exists():
            self.logger.warning(f"Missing file: {nwb_file.name}")
            return None
        
        rec = None
        rec_filt = None
        
        try:
            # Load and filter recording
            rec = se.NwbRecordingExtractor(str(nwb_file))
            
            # Suppress bandpass filter warnings for cleaner output
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rec_filt = sp.bandpass_filter(
                    rec, 
                    freq_min=self.config.freq_min, 
                    freq_max=self.config.freq_max
                )
            
            # Extract traces and compute power
            traces = rec_filt.get_traces().astype(np.float32)
            power = np.mean(traces ** 2, axis=1)
            
            # Get channel locations and depths
            channel_ids = rec_filt.get_channel_ids()
            locations = np.array([
                rec_filt.get_channel_property(ch, "location")
                for ch in channel_ids
            ])
            depths = locations[:, 1]  # y-coordinate represents depth
            
            # Sort by cortical depth
            depth_order = np.argsort(depths)
            sorted_power = power[depth_order]
            sorted_depths = depths[depth_order]
            
            return sorted_power, sorted_depths
            
        except Exception as e:
            self.logger.error(f"Error processing {nwb_file.name}: {e}")
            return None
            
        finally:
            # Ensure proper cleanup of extractors
            self._cleanup_extractors(rec_filt, rec)
    
    def _cleanup_extractors(self, *extractors) -> None:
        """Safely close extractor objects."""
        for extractor in extractors:
            if extractor is not None:
                try:
                    if hasattr(extractor, 'close'):
                        extractor.close()
                except Exception:
                    pass  # Ignore cleanup errors
    
    def _create_output_directory(self, animal_id: str, date: str, 
                                session_id: str) -> Path:
        """Create and return the output directory path."""
        out_dir = (
            self.script_folder
            / self.config.out_subdir
            / animal_id
            / f"{date}_{session_id}"
            / "VisualLandmark"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir
    
    def _should_skip_session(self, pkl_path: Path, session_name: str) -> bool:
        """Check if session should be skipped (already processed)."""
        if pkl_path.exists():
            self.logger.info(f"Skipping existing session: {session_name}")
            return True
        return False
    
    def process_session(self, session_folder: Path) -> bool:
        """
        Process all shanks in a session.
        
        Args:
            session_folder: Path to the session directory
            
        Returns:
            True if session was processed successfully, False otherwise
        """
        session_name = session_folder.name
        session_parts = session_name.split("_")
        
        if len(session_parts) < 2:
            self.logger.error(f"Invalid session name format: {session_name}")
            return False
            
        session_id = session_parts[-1]
        
        # Extract animal_id and date from session_name or folder structure
        # This assumes session_name contains animal_id and date info
        animal_id = None
        date = None
        
        for aid in self.config.animal_ids:
            if aid in session_name:
                animal_id = aid
                break
                
        if animal_id is None:
            self.logger.error(f"Could not determine animal_id from session: {session_name}")
            return False
        
        # Extract date from session folder or parent directory
        date_candidates = [part for part in session_parts if len(part) == 6 and part.isdigit()]
        if date_candidates:
            date = date_candidates[0]
        else:
            self.logger.error(f"Could not determine date from session: {session_name}")
            return False
        
        self.logger.info(f"Processing session: {session_name}")
        
        # Setup output directory
        out_dir = self._create_output_directory(animal_id, date, session_id)
        pkl_path = out_dir / f"power_{animal_id}_{date}_{session_id}.pkl"
        
        if self._should_skip_session(pkl_path, session_name):
            return True
        
        # Process all shanks
        powers_list = []
        depths_list = []
        valid_shanks = []
        
        total_shanks = len(self.config.shank_ids)
        
        for idx, shank_id in enumerate(self.config.shank_ids):
            self.logger.info(f"  Processing shank {shank_id} ({idx+1}/{total_shanks})")
            
            result = self.process_shank(session_folder, session_name, shank_id)
            if result is None:
                continue
                
            sorted_power, sorted_depths = result
            powers_list.append(sorted_power)
            depths_list.append(sorted_depths)
            valid_shanks.append(shank_id)
        
        if not powers_list:
            self.logger.warning(f"No valid shank data for session: {session_name}")
            return False
        
        # Normalize powers and prepare data
        normalized_powers = [p / np.max(p) if np.max(p) > 0 else p for p in powers_list]
        
        session_data = {
            "shank_ids": np.array(valid_shanks),
            "powers": normalized_powers,
            "depths": depths_list,
            "animal_id": animal_id,
            "date": date,
            "session_id": session_id,
            "processing_config": {
                "freq_min": self.config.freq_min,
                "freq_max": self.config.freq_max,
                "n_valid_shanks": len(valid_shanks)
            }
        }
        
        # Save processed data
        try:
            with open(pkl_path, "wb") as f:
                pickle.dump(session_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            relative_path = pkl_path.relative_to(self.script_folder)
            self.logger.info(f"  → Saved: {relative_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save data for {session_name}: {e}")
            return False
    
    def run_processing_pipeline(self) -> None:
        """Run the complete processing pipeline."""
        self.logger.info("Starting neural signal processing pipeline")
        self.logger.info(f"Config: {self.config}")
        
        dates = generate_dates(self.config.date_start, self.config.date_end)
        total_processed = 0
        total_failed = 0
        
        for date in dates:
            for animal_id in self.config.animal_ids:
                sessions = find_session_folders(self.config.base_dir, date, animal_id)
                
                if not sessions:
                    continue
                
                self.logger.info(f"Found {len(sessions)} session(s) for {animal_id} on {date}")
                
                for session_folder in sessions:
                    try:
                        if self.process_session(session_folder):
                            total_processed += 1
                        else:
                            total_failed += 1
                    except Exception as e:
                        self.logger.error(f"Unexpected error processing {session_folder.name}: {e}")
                        total_failed += 1
        
        self.logger.info(f"Processing complete. Successfully processed: {total_processed}, Failed: {total_failed}")


def main():
    """Main entry point for the neural signal processing script."""
    # You can customize the configuration here
    config = ProcessingConfig(
        # Uncomment and modify these lines to override defaults:
        # base_dir=Path(r"\\10.129.151.108\xieluanlabs\xl_cl\rf_reconstruction\head_fixed"),
        # animal_ids=["CnL22", "CnL38"],  # Process subset of animals
        # freq_min=300.0,  # Lower frequency bound
        # freq_max=6000.0,  # Upper frequency bound
    )
    
    processor = NeuralSignalProcessor(config)
    processor.run_processing_pipeline()


if __name__ == "__main__":
    main()