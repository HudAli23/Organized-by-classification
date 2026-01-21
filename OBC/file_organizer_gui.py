import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
import logging
import json
from utils.filesystem import clean_empty_folders
from core.pipeline import run_pipeline
import webbrowser

CONFIG_FILE = "gui_config.json"

class FileSorterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart File Organizer")
        self.root.geometry("900x700")  # Increased window size
        
        # Configure root window
        self.root.configure(bg='#f0f0f0')  # Light gray background
        self.root.option_add('*Font', ('Segoe UI', 10))  # Modern font
        
        # Configure ttk styles
        self.style = ttk.Style()
        self.style.configure('TFrame', background='#f0f0f0')
        self.style.configure('TButton', padding=5, font=('Segoe UI', 10))
        self.style.configure('TLabel', background='#f0f0f0', font=('Segoe UI', 10))
        self.style.configure('Header.TLabel', font=('Segoe UI', 12, 'bold'))
        self.style.configure('TEntry', padding=5)
        
        self.source_var = tk.StringVar()
        self.dest_var = tk.StringVar()
        self.is_running = False
        self.cancel_event = None
        self.current_thread = None
        
        # Load saved directories
        self.load_config()
        
        self.setup_ui()
        
    def load_config(self):
        """Load saved directory configurations."""
        try:
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, 'r') as f:
                    config = json.load(f)
                    if os.path.exists(config.get('source', '')):
                        self.source_var.set(config['source'])
                    if os.path.exists(config.get('destination', '')):
                        self.dest_var.set(config['destination'])
        except Exception as e:
            logging.warning(f"Could not load config: {e}")
            
    def save_config(self):
        """Save current directory configurations."""
        try:
            config = {
                'source': self.source_var.get(),
                'destination': self.dest_var.get()
            }
            with open(CONFIG_FILE, 'w') as f:
                json.dump(config, f)
        except Exception as e:
            logging.warning(f"Could not save config: {e}")

    def setup_ui(self):
        # Create main container with padding
        container = ttk.Frame(self.root, padding="20", style='TFrame')
        container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Header
        header_label = ttk.Label(container, text="Smart File Organization", style='Header.TLabel')
        header_label.grid(row=0, column=0, columnspan=3, pady=(0, 20), sticky=tk.W)
        
        # Source directory frame
        source_frame = ttk.LabelFrame(container, text="Source", padding="10")
        source_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        ttk.Entry(source_frame, textvariable=self.source_var, width=70).grid(row=0, column=0, padx=(0, 10))
        ttk.Button(source_frame, text="Browse", command=self.browse_source).grid(row=0, column=1)
        
        # Destination directory frame
        dest_frame = ttk.LabelFrame(container, text="Destination", padding="10")
        dest_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 20))
        ttk.Entry(dest_frame, textvariable=self.dest_var, width=70).grid(row=0, column=0, padx=(0, 10))
        ttk.Button(dest_frame, text="Browse", command=self.browse_destination).grid(row=0, column=1)
        
        # Progress section
        progress_frame = ttk.LabelFrame(container, text="Progress", padding="10")
        progress_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 20))
        
        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(progress_frame, length=700, mode='determinate', variable=self.progress_var)
        self.progress.grid(row=0, column=0, columnspan=2, pady=(5, 10))
        
        self.status_var = tk.StringVar(value="Ready to organize files")
        status_label = ttk.Label(progress_frame, textvariable=self.status_var)
        status_label.grid(row=1, column=0, columnspan=2)
        
        # Buttons frame
        button_frame = ttk.Frame(container)
        button_frame.grid(row=4, column=0, columnspan=3, pady=(0, 20))
        
        self.start_button = ttk.Button(button_frame, text="Start Organization", command=self.start_sorting, style='TButton')
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.cancel_button = ttk.Button(button_frame, text="Cancel", command=self.cancel_sorting, state=tk.DISABLED, style='TButton')
        self.cancel_button.pack(side=tk.LEFT, padx=5)
        
        # Log frame
        log_frame = ttk.LabelFrame(container, text="Log", padding="10")
        log_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))
        container.rowconfigure(5, weight=1)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(log_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Configure log text widget with modern styling
        self.log_text = tk.Text(log_frame, height=15, width=80, yscrollcommand=scrollbar.set,
                               font=('Consolas', 9), bg='white', wrap=tk.WORD,
                               borderwidth=1, relief=tk.SOLID)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Configure scrollbar
        scrollbar.config(command=self.log_text.yview)
        
        # Configure window close handler
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def browse_source(self):
        directory = filedialog.askdirectory()
        if directory:
            if not os.access(directory, os.R_OK):
                messagebox.showerror("Permission Error", 
                    "Cannot read from the selected source directory.\n\n"
                    "Please ensure you have read permissions for this directory "
                    "and all its subdirectories.")
                return
            self.source_var.set(directory)
            self.log_message(f"Selected source directory: {directory}")
            self.save_config()
            
    def browse_destination(self):
        directory = filedialog.askdirectory()
        if directory:
            # Check write permissions
            try:
                test_file = os.path.join(directory, '.test_write_permission')
                with open(test_file, 'w') as f:
                    f.write('test')
                os.remove(test_file)
            except (IOError, OSError) as e:
                messagebox.showerror("Permission Error", 
                    f"Cannot write to the selected destination directory.\n\n"
                    f"Error details: {str(e)}\n\n"
                    "Please ensure you have write permissions for this directory.")
                return
                
            self.dest_var.set(directory)
            self.log_message(f"Selected destination directory: {directory}")
            self.save_config()
            
    def start_sorting(self):
        source = self.source_var.get()
        destination = self.dest_var.get()
        
        if not source or not destination:
            messagebox.showerror("Error", "Please select both source and destination directories")
            return
            
        if not os.path.exists(source):
            messagebox.showerror("Error", "Source directory does not exist")
            return
            
        if source == destination:
            messagebox.showerror("Error", "Source and destination directories cannot be the same")
            return
            
        # Check permissions again (they might have changed)
        if not os.access(source, os.R_OK):
            messagebox.showerror("Error", "No read permission for the source directory")
            return
            
        try:
            # Create destination if it doesn't exist
            os.makedirs(destination, exist_ok=True)
            
            # Test write permission
            test_file = os.path.join(destination, '.test_write_permission')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
        except (IOError, OSError) as e:
            messagebox.showerror("Error", f"Cannot write to destination directory: {str(e)}")
            return
        
        # Clear log
        self.log_text.delete(1.0, tk.END)
        self.log_message("Starting file organization...")
        
        # Reset progress
        self.progress_var.set(0)
        
        # Disable start button and enable cancel button
        self.start_button.configure(state=tk.DISABLED)
        self.cancel_button.configure(state=tk.NORMAL)
        
        # Create cancel event
        self.cancel_event = threading.Event()
        
        # Start processing in a separate thread
        self.is_running = True
        self.current_thread = threading.Thread(target=self.run_sorting, args=(source, destination))
        self.current_thread.daemon = True
        self.current_thread.start()
        
    def run_sorting(self, source, destination):
        try:
            self.log_message("Starting content-based file organization with hierarchical structure...\n")
            self.log_message("Phase 1: Analyzing files for semantic relationships...")
            
            # Define progress callback
            def progress_cb(current, total, message):
                percentage = (current / total * 100) if total > 0 else 0
                self.update_progress(percentage, f"[{current}/{total}] {message}")
            
            # Use the pipeline to organize files with dynamic hierarchical folder creation
            stats = run_pipeline(source, destination, progress_callback=progress_cb, use_dynamic_folders=True)
            
            # Log summary
            self.log_message("\n=== Organization Summary ===")
            self.log_message(f"Total files processed: {stats['processed']}")
            self.log_message(f"Files successfully moved: {stats['moved']}")
            self.log_message(f"Files with errors: {stats['failed']}")
            
            # Log hierarchical folder structure
            if stats.get('structure'):
                self.log_message("\n=== Hierarchical Folder Structure Created ===\n")
                
                # Parse and display hierarchical structure
                top_level = {}
                for path, files in stats['structure'].items():
                    parts = path.split('/')
                    parent = parts[0]
                    subfolder = parts[1] if len(parts) > 1 else None
                    
                    if parent not in top_level:
                        top_level[parent] = {}
                    
                    if subfolder:
                        top_level[parent][subfolder] = files
                    else:
                        top_level[parent]['_root'] = files
                
                # Display tree structure
                for parent_folder, subfolders in top_level.items():
                    file_count = sum(len(files) for files in subfolders.values())
                    self.log_message(f"📁 {parent_folder}/ ({file_count} files)")
                    
                    for subfolder_name, files in subfolders.items():
                        if subfolder_name == '_root':
                            # Files directly in parent
                            for file_path in files[:3]:
                                self.log_message(f"   ├─ {os.path.basename(file_path)}")
                            if len(files) > 3:
                                self.log_message(f"   └─ ... and {len(files) - 3} more files")
                        else:
                            # Subfolder with files
                            self.log_message(f"   ├─ 📂 {subfolder_name}/ ({len(files)} files)")
                            for file_path in files[:2]:
                                self.log_message(f"   │  ├─ {os.path.basename(file_path)}")
                            if len(files) > 2:
                                self.log_message(f"   │  └─ ... and {len(files) - 2} more files")
                    
                    self.log_message("")  # Spacing
            
            self.log_message("\n✓ Hierarchical file organization completed successfully!")
            
            self.root.after(0, lambda: self.status_var.set("Organization completed successfully!"))
            self.root.after(0, lambda: messagebox.showinfo(
                "Complete", 
                f"Hierarchical file organization completed!\n\n"
                f"Files processed: {stats['processed']}\n"
                f"Successfully moved: {stats['moved']}\n"
                f"Errors: {stats['failed']}\n\n"
                f"Your files are now organized into:\n"
                f"• Projects with Code/Docs/Resources subfolders\n"
                f"• Job Search with Resume/CoverLetters/Applications\n"
                f"• And other hierarchical structures based on content!"
            ))
                
        except Exception as e:
            error_msg = f"Error during organization: {str(e)}"
            if "permission" in str(e).lower():
                error_msg += "\n\nPlease check that you have appropriate permissions for all directories."
            elif "disk space" in str(e).lower():
                error_msg += "\n\nPlease ensure you have enough disk space in the destination directory."
            elif "file in use" in str(e).lower():
                error_msg += "\n\nSome files may be in use by other applications. Please close them and try again."
            
            logging.error(error_msg)
            self.log_message(f"\nERROR: {error_msg}")
            self.root.after(0, lambda: messagebox.showerror("Error", error_msg))
            self.root.after(0, lambda: self.status_var.set("Error occurred during organization"))
            
        finally:
            self.is_running = False
            self.current_thread = None
            self.root.after(0, lambda: self.start_button.configure(state=tk.NORMAL))
            self.root.after(0, lambda: self.cancel_button.configure(state=tk.DISABLED))
            
    def cancel_sorting(self):
        if self.is_running and self.cancel_event:
            self.cancel_event.set()
            self.status_var.set("Cancelling...")
            self.log_message("Cancelling organization...")
            
    def update_progress(self, percentage, message):
        self.root.after(0, lambda: self.progress_var.set(percentage))
        self.root.after(0, lambda: self.status_var.set(message))
        
    def log_message(self, message):
        self.root.after(0, lambda: self._update_log(message))
        
    def _update_log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        
    def on_closing(self):
        """Handle window closing event."""
        if self.is_running:
            if messagebox.askokcancel("Quit", "Organization is in progress. Do you want to cancel and quit?"):
                self.cancel_sorting()
                # Wait for the thread to finish (with timeout)
                if self.current_thread:
                    self.current_thread.join(timeout=5.0)
                self.root.destroy()
        else:
            self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = FileSorterApp(root)
    root.mainloop()
