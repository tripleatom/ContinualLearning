import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from func import generate_dates, find_session_folders
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import warnings

# Suppress matplotlib warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# ─── Configuration ────────────────────────────────────────────────────────────

@dataclass
class PlottingConfig:
    """Configuration parameters for neural signal plotting."""
    base_dir: Path = Path(r"/Volumes/xieluanlabs/xl_cl/rf_reconstruction/head_fixed")
    date_start: str = "250307"
    date_end: str = "250604"
    animal_ids: List[str] = None
    out_subdir: str = "sortout"
    overwrite: bool = True
    
    # Plot styling
    figure_dpi: int = 300
    subplot_width: float = 3.0
    subplot_height: float = 4.0
    max_cols: int = 4
    line_color: str = "black"
    peak_color: str = "red"
    line_width: float = 1.5
    peak_marker_size: int = 10
    font_size_title: int = 12
    font_size_axis_label: int = 9
    font_size_subplot_title: int = 10
    font_size_ticks: int = 8
    
    def __post_init__(self):
        if self.animal_ids is None:
            self.animal_ids = ["CnL22", "CnL38", "CnL39", "CnL40", "CnL41"]


class NeuralSignalPlotter:
    """Handles plotting of neural signal power data from processed pickle files."""
    
    def __init__(self, config: PlottingConfig):
        self.config = config
        self.script_folder = Path(__file__).parent.parent
        self.sortout_dir = self.script_folder / self.config.out_subdir
        self._setup_logging()
        self._setup_matplotlib()
    
    def _setup_logging(self) -> None:
        """Configure logging for the plotting pipeline."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _setup_matplotlib(self) -> None:
        """Configure matplotlib settings for consistent plots."""
        plt.rcParams.update({
            'figure.dpi': self.config.figure_dpi,
            'savefig.dpi': self.config.figure_dpi,
            'font.size': self.config.font_size_ticks,
            'axes.linewidth': 0.8,
            'lines.linewidth': self.config.line_width,
        })
    
    def _load_pickle_data(self, pkl_path: Path) -> Optional[Dict[str, Any]]:
        """
        Safely load pickle data with error handling.
        
        Args:
            pkl_path: Path to the pickle file
            
        Returns:
            Dictionary containing the loaded data or None if loading fails
        """
        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            
            # Validate required keys
            required_keys = ["shank_ids", "powers", "depths", "animal_id", "date", "session_id"]
            missing_keys = [key for key in required_keys if key not in data]
            
            if missing_keys:
                self.logger.error(f"Missing required keys in {pkl_path}: {missing_keys}")
                return None
                
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to load {pkl_path}: {e}")
            return None
    
    def _calculate_subplot_layout(self, num_shanks: int) -> Tuple[int, int]:
        """Calculate optimal subplot layout."""
        cols = min(self.config.max_cols, num_shanks)
        rows = (num_shanks + cols - 1) // cols
        return rows, cols
    
    def _setup_figure_and_axes(self, num_shanks: int) -> Tuple[plt.Figure, np.ndarray]:
        """Create figure and axes with proper layout."""
        rows, cols = self._calculate_subplot_layout(num_shanks)
        
        figsize = (
            self.config.subplot_width * cols, 
            self.config.subplot_height * rows
        )
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        
        # Ensure axes is always iterable
        if num_shanks == 1:
            axes = np.array([axes])
        elif rows == 1 and cols > 1:
            axes = np.array(axes)
        elif rows > 1:
            axes = axes.flatten()
        else:
            axes = np.array([axes])
            
        return fig, axes
    
    def _plot_single_shank(self, ax: plt.Axes, power: np.ndarray, depth: np.ndarray, 
                          shank_id: int, is_bottom_row: bool = False, is_leftmost_col: bool = False) -> None:
        """
        Plot power vs depth for a single shank.
        
        Args:
            ax: Matplotlib axes object
            power: Normalized power values
            depth: Corresponding depth values
            shank_id: ID of the shank being plotted
            is_bottom_row: Whether this is in the bottom row (for x-label)
            is_leftmost_col: Whether this is in the leftmost column (for y-label)
        """
        # Ensure power is normalized
        if np.max(power) > 0:
            norm_power = power / np.max(power)
        else:
            norm_power = power
            self.logger.warning(f"Zero maximum power for shank {shank_id}")
        
        # Find peak
        peak_idx = np.argmax(norm_power)
        peak_power = norm_power[peak_idx]
        peak_depth = depth[peak_idx]
        
        # Plot main line
        ax.plot(norm_power, depth, 
                color=self.config.line_color, 
                linewidth=self.config.line_width,
                alpha=0.8)
        
        # Mark peak
        ax.plot(peak_power, peak_depth, 
                marker='*', 
                color=self.config.peak_color,
                markersize=self.config.peak_marker_size, 
                markeredgecolor='black',
                markeredgewidth=0.5,
                markerfacecolor=self.config.peak_color,
                zorder=5)
        
        # Styling - NO invert_yaxis() since 0 should be at top
        # Only add x-label to bottom row subplots
        if is_bottom_row:
            ax.set_xlabel("Normalized Power\n(500-5000 Hz)", 
                         fontsize=self.config.font_size_axis_label)
        
        # Only add y-label to leftmost column subplot
        if is_leftmost_col:
            ax.set_ylabel("Electrode Site (um)", 
                         fontsize=self.config.font_size_axis_label)
        
        ax.set_title(f"Shank {shank_id}", 
                    fontsize=self.config.font_size_subplot_title, 
                    pad=10)
        
        # Set limits with small margin
        ax.set_xlim(-0.05, 1.05)
        
        # Clean up spines
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        
        # Adjust tick parameters
        ax.tick_params(labelsize=self.config.font_size_ticks, 
                      width=0.8, length=4)
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3, linewidth=0.5)
    
    def _hide_unused_subplots(self, axes: np.ndarray, num_shanks: int) -> None:
        """Hide unused subplot axes."""
        if num_shanks < len(axes):
            for idx in range(num_shanks, len(axes)):
                axes[idx].set_visible(False)
    
    def _generate_output_path(self, pkl_path: Path, animal_id: str, 
                            date: str, session_id: str) -> Path:
        """Generate output PDF path."""
        return pkl_path.with_name(f"power_shanks_{animal_id}_{date}_{session_id}.pdf")
    
    def _should_skip_plotting(self, output_path: Path) -> bool:
        """Check if plotting should be skipped."""
        if output_path.exists() and not self.config.overwrite:
            self.logger.info(f"Skipping {output_path.name} (exists, overwrite=False)")
            return True
        return False
    
    def plot_session_data(self, pkl_path: Path) -> bool:
        """
        Create plots for a single session's data.
        
        Args:
            pkl_path: Path to the pickle file containing session data
            
        Returns:
            True if plotting was successful, False otherwise
        """
        # Load data
        data = self._load_pickle_data(pkl_path)
        if data is None:
            return False
        
        # Extract data
        shank_ids = data["shank_ids"]
        powers = data["powers"]
        depths = data["depths"]
        animal_id = data["animal_id"]
        date = data["date"]
        session_id = data["session_id"]
        
        # Validate data consistency
        if not (len(shank_ids) == len(powers) == len(depths)):
            self.logger.error(f"Data length mismatch in {pkl_path}")
            return False
        
        num_shanks = len(shank_ids)
        if num_shanks == 0:
            self.logger.warning(f"No shank data in {pkl_path}")
            return False
        
        # Check output path
        output_path = self._generate_output_path(pkl_path, animal_id, date, session_id)
        if self._should_skip_plotting(output_path):
            return True
        
        try:
            # Create figure and axes
            fig, axes = self._setup_figure_and_axes(num_shanks)
            rows, cols = self._calculate_subplot_layout(num_shanks)
            
            # Collect all depth values to determine shared y-axis limits
            all_depths = np.concatenate([np.array(depths[idx]) for idx in range(num_shanks)])
            y_min, y_max = np.min(all_depths), np.max(all_depths)
            y_margin = (y_max - y_min) * 0.02  # 2% margin
            # Invert the limits so that smaller values (0) are at the top
            shared_ylim = (y_max + y_margin, y_min - y_margin)
            
            # Plot each shank
            for idx, shank_id in enumerate(shank_ids):
                power = np.array(powers[idx])
                depth = np.array(depths[idx])
                
                if len(power) != len(depth):
                    self.logger.warning(f"Power/depth length mismatch for shank {shank_id}")
                    continue
                
                ax = axes[idx] if num_shanks > 1 else axes[0]
                
                # Determine position in grid
                row = idx // cols
                col = idx % cols
                is_bottom_row = (row == rows - 1) or (idx >= num_shanks - cols)  # Last row or last few plots
                is_leftmost_col = (col == 0)
                
                self._plot_single_shank(ax, power, depth, shank_id, is_bottom_row, is_leftmost_col)
                
                # Set shared y-axis limits (inverted so 0 is at top)
                ax.set_ylim(shared_ylim)
            
            # Hide unused subplots
            self._hide_unused_subplots(axes, num_shanks)
            
            # Add main title
            main_title = f"{animal_id} | {date} | Session {session_id}"
            fig.suptitle(main_title, 
                        fontsize=self.config.font_size_title, 
                        y=0.95, 
                        fontweight='bold')
            
            # Adjust layout
            plt.tight_layout(rect=[0, 0, 1, 0.93])
            
            # Save figure
            plt.savefig(output_path, 
                       bbox_inches="tight", 
                       dpi=self.config.figure_dpi,
                       facecolor='white',
                       edgecolor='none')
            plt.close(fig)
            
            # Log success
            relative_path = output_path.relative_to(self.script_folder)
            self.logger.info(f"Saved plot: {relative_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create plot for {pkl_path}: {e}")
            plt.close('all')  # Clean up any open figures
            return False
    
    def find_and_plot_sessions(self) -> None:
        """Find all session pickle files and create plots."""
        dates = generate_dates(self.config.date_start, self.config.date_end)
        total_processed = 0
        total_failed = 0
        
        self.logger.info("Starting neural signal plotting pipeline")
        
        for date in dates:
            for animal_id in self.config.animal_ids:
                sessions = find_session_folders(self.config.base_dir, date, animal_id)
                
                if not sessions:
                    continue
                
                self.logger.info(f"Found {len(sessions)} session(s) for {animal_id} on {date}")
                
                for session_folder in sessions:
                    try:
                        session_name = session_folder.name
                        session_parts = session_name.split("_")
                        
                        if len(session_parts) < 3:
                            self.logger.warning(f"Invalid session name format: {session_name}")
                            continue
                        
                        session_id = session_parts[-1]
                        
                        # Construct pickle file path
                        pkl_path = (self.sortout_dir / animal_id / 
                                  f"{date}_{session_id}" / "VisualLandmark" / 
                                  f"power_{animal_id}_{date}_{session_id}.pkl")
                        
                        if not pkl_path.exists():
                            self.logger.debug(f"Pickle file not found: {pkl_path}")
                            continue
                        
                        self.logger.info(f"Processing session: {session_name}")
                        
                        if self.plot_session_data(pkl_path):
                            total_processed += 1
                        else:
                            total_failed += 1
                            
                    except Exception as e:
                        self.logger.error(f"Unexpected error processing {session_folder.name}: {e}")
                        total_failed += 1
        
        self.logger.info(f"Plotting complete. Successfully plotted: {total_processed}, Failed: {total_failed}")


def main():
    """Main entry point for the neural signal plotting script."""
    # Customize configuration as needed
    config = PlottingConfig(
        # Uncomment and modify these lines to override defaults:
        # overwrite=False,  # Don't overwrite existing plots
        # figure_dpi=150,   # Lower DPI for faster generation
        # peak_color="blue",  # Different peak marker color
    )
    
    plotter = NeuralSignalPlotter(config)
    plotter.find_and_plot_sessions()


if __name__ == "__main__":
    main()