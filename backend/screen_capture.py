import time
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
from PIL import Image

# Set up clean logging to help track application status and capture performance
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ScreenCapture")

# ==============================================================================
# SECTION 1: DYNAMIC DEPENDENCY RESOLUTION
# ==============================================================================
# We try importing 'mss' (extremely fast screenshot library that directly queries 
# system screen buffers) and 'pygetwindow' (for listing active window positions).
# If they are not installed, the app falls back to PIL's standard ImageGrab.
# This prevents crashes and makes the code highly beginner-friendly!

try:
    import mss
except ImportError:
    logger.warning("mss package not found. Will use standard PIL fallback. Run 'pip install mss' for faster captures.")
    mss = None

try:
    import pygetwindow as gw
except ImportError:
    logger.warning("pygetwindow package not found. Specific window-based capture disabled (Windows-only). Run 'pip install pygetwindow'.")
    gw = None

# Import our configurations
from backend.config import settings


# ==============================================================================
# SECTION 2: SCREEN CAPTURE SERVICE CLASS
# ==============================================================================
class ScreenCaptureService:
    """
    A service class dedicated to capturing and managing screenshots of the user's
    workspace or active lecture window. Highly optimized for low latency.
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        # Set output directory to our shared assets folder
        self.output_dir = output_dir or settings.ASSETS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def list_active_windows(self) -> List[Dict[str, Any]]:
        """
        Lists all active, visible window titles and their bounding box dimensions.
        Useful when you only want to study from a specific slide viewer or PDF reader!
        """
        windows_list = []
        if gw is not None:
            try:
                # Retrieve all active windows running on the OS
                all_windows = gw.getAllWindows()
                for window in all_windows:
                    # Filter out empty names, minimized windows, or 0-size window handles
                    if window.title and window.width > 0 and window.height > 0:
                        windows_list.append({
                            "title": window.title,
                            "box": {
                                "left": window.left,
                                "top": window.top,
                                "width": window.width,
                                "height": window.height
                            }
                        })
            except Exception as e:
                logger.error(f"Failed to scan active windows: {e}")
        else:
            logger.info("Window list is not available on this platform (requires pygetwindow on Windows).")
        return windows_list

    def capture_fullscreen(self) -> Optional[Image.Image]:
        """
        Captures the primary monitor screen contents.
        Optimized to use 'mss' for low-latency capture, with standard PIL as fallback.
        """
        # --- Low Latency Route: mss ---
        if mss is not None:
            try:
                with mss.mss() as sct:
                    # Monitor 1 is typically the primary monitor
                    primary_monitor = sct.monitors[1]
                    raw_screenshot = sct.grab(primary_monitor)
                    # Convert BGRA raw buffer into standard RGB PIL Image
                    img = Image.frombytes("RGB", raw_screenshot.size, raw_screenshot.bgra, "raw", "BGRX")
                    return img
            except Exception as e:
                logger.error(f"MSS capture failed: {e}. Trying fallback.")
        
        # --- Fallback Route: standard PIL ImageGrab ---
        try:
            from PIL import ImageGrab
            # Captures the main screen coordinates automatically
            return ImageGrab.grab()
        except Exception as e:
            logger.error(f"All screenshot capture methods failed: {e}")
            return None

    def capture_window(self, window_title: str) -> Optional[Image.Image]:
        """
        Captures a specific window by finding its exact title on screen,
        focusing it, and cropping the screen buffer to its bounding rectangle.
        """
        # If pygetwindow and mss are available, capture the exact bounding box
        if gw is not None and mss is not None:
            try:
                windows = gw.getWindowsWithTitle(window_title)
                if not windows:
                    logger.warning(f"No window found matching title: '{window_title}'")
                    return None
                
                target_window = windows[0]
                
                # Construct exact coordinate box for mss
                bbox = {
                    "left": target_window.left,
                    "top": target_window.top,
                    "width": target_window.width,
                    "height": target_window.height
                }
                
                # Check for boundary issues (e.g. window is minimized or off-screen)
                if bbox["width"] <= 0 or bbox["height"] <= 0:
                    logger.warning(f"Target window '{window_title}' is minimized or has zero size.")
                    return None
                
                with mss.mss() as sct:
                    raw_screenshot = sct.grab(bbox)
                    img = Image.frombytes("RGB", raw_screenshot.size, raw_screenshot.bgra, "raw", "BGRX")
                    return img
            except Exception as e:
                logger.error(f"Failed direct window capture: {e}. Trying coordinate crop fallback.")

        # --- Fallback Route: Grab full screen and crop to window dimensions ---
        fullscreen = self.capture_fullscreen()
        if fullscreen and gw is not None:
            try:
                windows = gw.getWindowsWithTitle(window_title)
                if windows:
                    target_window = windows[0]
                    # Crop parameters: (left, upper, right, lower)
                    crop_box = (
                        target_window.left,
                        target_window.top,
                        target_window.left + target_window.width,
                        target_window.top + target_window.height
                    )
                    return fullscreen.crop(crop_box)
            except Exception as e:
                logger.error(f"Crop-based fallback window capture failed: {e}")
        
        return None

    def save_capture(self, img: Image.Image, prefix: str = "cap") -> Path:
        """
        Saves a captured PIL Image to the shared assets directory with a clean timestamp.
        Also automatically maintains directory history to avoid using up local storage!
        """
        timestamp = int(time.time())
        filename = f"{prefix}_{timestamp}.jpg"
        save_path = self.output_dir / filename
        
        # Save image using configure-defined quality settings (80% reduces size significantly)
        img.save(save_path, "JPEG", quality=settings.SCREENSHOT_QUALITY)
        logger.info(f"Screenshot saved successfully at: {save_path.name}")
        
        # Automatically clean up older files beyond the max limit
        self._maintain_storage_limit()
        
        return save_path

    def _maintain_storage_limit(self):
        """
        Deletes old screenshots if the folder count exceeds settings.MAX_SCREENSHOT_HISTORY.
        Keeps storage usage minimal for a clean developer machine environment.
        """
        try:
            # Gather all files matching our prefix sorted by their last modification time
            all_captures = sorted(
                self.output_dir.glob("cap_*.jpg"),
                key=lambda file: file.stat().st_mtime
            )
            
            # If we exceeded the limit, purge the oldest files
            if len(all_captures) > settings.MAX_SCREENSHOT_HISTORY:
                excess_count = len(all_captures) - settings.MAX_SCREENSHOT_HISTORY
                to_delete = all_captures[:excess_count]
                
                for file_path in to_delete:
                    file_path.unlink()
                    logger.debug(f"Purged old screenshot to save space: {file_path.name}")
        except Exception as e:
            logger.error(f"Error executing screenshot storage cleanup: {e}")

# ==============================================================================
# SECTION 3: INDEPENDENT MODULE TEST RUNNER
# ==============================================================================
if __name__ == "__main__":
    print("--------------------------------------------------")
    print("ScreenCaptureService Diagnostics")
    print("--------------------------------------------------")
    service = ScreenCaptureService()
    
    # Check what windows are active
    print("[1] Scanning visible windows:")
    active = service.list_active_windows()
    for w in active[:5]:
         print(f"  - Title: {w['title']} | Size: {w['box']['width']}x{w['box']['height']}")
         
    # Perform full screenshot test
    print("\n[2] Attempting to capture main screen...")
    screenshot = service.capture_fullscreen()
    if screenshot:
        saved_file = service.save_capture(screenshot, prefix="cap_test")
        print(f"SUCCESS: Screenshot saved to {saved_file}")
    else:
        print("FAILURE: Could not capture screenshot.")
    print("--------------------------------------------------")
