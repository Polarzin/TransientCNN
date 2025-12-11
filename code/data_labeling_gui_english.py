import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import numpy as np
import os
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.patches as patches

class DataLabelingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Labeling Tool")
        self.root.geometry("1200x800")
        
        # Data storage
        self.data = None
        self.transposed_data = None
        self.labels = {}  # Store user labels {row_index: label}
        self.current_row = 0
        self.total_rows = 0
        
        # Variables for smoothing feature
        self.smooth_enabled = False
        self.window_size = 5
        
        # Create UI
        self.create_widgets()
        
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top control area
        control_frame = ttk.LabelFrame(main_frame, text="File Control", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # File selection
        ttk.Button(control_frame, text="Select TXT File", command=self.load_file).pack(side=tk.LEFT, padx=(0, 10))
        self.file_label = ttk.Label(control_frame, text="No file selected")
        self.file_label.pack(side=tk.LEFT)
        
        # Save button
        ttk.Button(control_frame, text="Save Labels", command=self.save_labels).pack(side=tk.RIGHT)
        
        # Middle data display area
        data_frame = ttk.LabelFrame(main_frame, text="Data Display", padding=10)
        data_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Create left and right panels
        left_frame = ttk.Frame(data_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        right_frame = ttk.Frame(data_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Left: data table
        table_frame = ttk.LabelFrame(left_frame, text="Data Table", padding=5)
        table_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create Treeview to display data
        self.tree = ttk.Treeview(table_frame, show="headings")
        self.tree.pack(fill=tk.BOTH, expand=True)
        
        # Add scrollbars
        tree_scroll_y = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        tree_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=tree_scroll_y.set)
        
        tree_scroll_x = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL, command=self.tree.xview)
        tree_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.tree.configure(xscrollcommand=tree_scroll_x.set)
        
        # Right: label control
        mark_frame = ttk.LabelFrame(right_frame, text="Label Control", padding=5)
        mark_frame.pack(fill=tk.BOTH, expand=True)
        
        # Current row info
        self.row_info_label = ttk.Label(mark_frame, text="Current Row: 0 / 0")
        self.row_info_label.pack(pady=5)
        
        # Label buttons - ensure visibility
        button_frame = ttk.Frame(mark_frame)
        button_frame.pack(pady=10)
        
        # Label buttons
        ttk.Button(button_frame, text="Label as 0", command=lambda: self.mark_row(0)).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Label as 1", command=lambda: self.mark_row(1)).pack(side=tk.LEFT, padx=5)
        
        # Navigation buttons
        nav_frame = ttk.Frame(mark_frame)
        nav_frame.pack(pady=10)
        
        ttk.Button(nav_frame, text="Previous", command=self.prev_row).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Next", command=self.next_row).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Go To", command=self.goto_row).pack(side=tk.LEFT, padx=5)
        
        # Jump input box
        self.goto_entry = ttk.Entry(nav_frame, width=10)
        self.goto_entry.pack(side=tk.LEFT, padx=5)
        
        # Label statistics
        stats_frame = ttk.LabelFrame(mark_frame, text="Label Statistics", padding=5)
        stats_frame.pack(fill=tk.X, pady=10)
        
        self.stats_label = ttk.Label(stats_frame, text="Labeled: 0 | Label 0: 0 | Label 1: 0")
        self.stats_label.pack()
        
        # Bottom: chart display
        plot_frame = ttk.LabelFrame(main_frame, text="Data Visualization", padding=5)
        plot_frame.pack(fill=tk.BOTH, expand=True)
        
        # Chart control area
        plot_control_frame = ttk.Frame(plot_frame)
        plot_control_frame.pack(fill=tk.X, pady=(0, 5))
        
        # X-axis range control
        ttk.Label(plot_control_frame, text="X Range:").pack(side=tk.LEFT, padx=(0, 5))
        self.x_start_entry = ttk.Entry(plot_control_frame, width=8)
        self.x_start_entry.pack(side=tk.LEFT, padx=2)
        self.x_start_entry.insert(0, "0")
        
        ttk.Label(plot_control_frame, text="to").pack(side=tk.LEFT, padx=2)
        self.x_end_entry = ttk.Entry(plot_control_frame, width=8)
        self.x_end_entry.pack(side=tk.LEFT, padx=2)
        self.x_end_entry.insert(0, "100")
        
        ttk.Button(plot_control_frame, text="Apply Range", command=self.update_plot).pack(side=tk.LEFT, padx=10)
        ttk.Button(plot_control_frame, text="Reset Range", command=self.reset_plot_range).pack(side=tk.LEFT, padx=5)
        
        # Waveform smoothing control
        ttk.Separator(plot_control_frame, orient='vertical').pack(side=tk.LEFT, padx=10, fill=tk.Y)
        
        # Smoothing toggle
        self.smooth_var = tk.BooleanVar()
        self.smooth_checkbutton = ttk.Checkbutton(plot_control_frame, text="Enable Smoothing", 
                                                 variable=self.smooth_var, command=self.update_plot)
        self.smooth_checkbutton.pack(side=tk.LEFT, padx=5)
        
        # Sliding window size setting
        ttk.Label(plot_control_frame, text="Window Size:").pack(side=tk.LEFT, padx=(10, 5))
        self.window_size_var = tk.StringVar(value="5")
        self.window_size_entry = ttk.Entry(plot_control_frame, width=6)
        self.window_size_entry.pack(side=tk.LEFT, padx=2)
        self.window_size_entry.insert(0, "5")
        
        # Add hint label
        ttk.Label(plot_control_frame, text="(odd number recommended)").pack(side=tk.LEFT, padx=2)
        
        ttk.Button(plot_control_frame, text="Apply Smoothing", command=self.update_plot).pack(side=tk.LEFT, padx=5)
        ttk.Button(plot_control_frame, text="Reset Smoothing", command=self.reset_smoothing).pack(side=tk.LEFT, padx=5)
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(10, 4))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Bind events
        self.tree.bind('<<TreeviewSelect>>', self.on_row_select)
        
        # Bind keyboard shortcuts
        self.root.bind('<Right>', lambda event: self.next_row())
        self.root.bind('<Left>', lambda event: self.prev_row())
        self.root.bind('<Up>', lambda event: self.mark_row(1))
        self.root.bind('<Down>', lambda event: self.mark_row(0))
    
    def load_file(self):
        """Load TXT file"""
        file_path = filedialog.askopenfilename(
            title="Select TXT File",
            filetypes=[("TXT files", "*.txt"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
            
        try:
            # Read data
            # data = []
            # with open(file_path, 'r', encoding='utf-8') as f:
            #     for line in f:
            #         # Split line by space
            #         row = [float(x) for x in line.strip().split(',')]
            #         if row:  # Ensure line not empty
            #             data.append(row)
            data = np.loadtxt(file_path, dtype=float, delimiter=',')
            
            if data.size == 0:
                messagebox.showerror("Error", "File is empty or format is incorrect")
                return
                
            # Convert to numpy array
            # self.data = np.array(data)
            self.data = data
            self.transposed_data = self.data  # No transpose
            # self.transposed_data = self.data.T  # Transpose data
            
            self.file_label.config(text=f"Loaded: {os.path.basename(file_path)}")
            self.total_rows = len(self.transposed_data)
            self.current_row = 0
            self.labels = {}  # Clear previous labels
            
            # Reset smoothing settings
            self.reset_smoothing()
            
            # Update data display
            self.update_data_display()
            self.update_stats()
            
            messagebox.showinfo("Success", f"Successfully loaded data with {self.total_rows} rows")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error reading file: {str(e)}")
    
    def update_data_display(self):
        """Update data display"""
        if self.transposed_data is None:
            return
            
        # Clear existing data
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Display current row and nearby rows
        start_row = max(0, self.current_row - 2)
        end_row = min(self.total_rows, self.current_row + 3)
        
        # Set columns - display all columns
        # num_cols = len(self.transposed_data[0]) if self.total_rows > 0 else 0
        # Set columns - 3 columns
        num_cols = 2 if self.total_rows > 0 else 0
        columns = [f"Col{i+1}" for i in range(num_cols)]
        self.tree['columns'] = columns
        
        # Set column headers
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=80, minwidth=60)
        
        # Add multiple rows of data
        for row_idx in range(start_row, end_row):
            row_data = self.transposed_data[row_idx]
            values = [f"{x:.4f}" for x in row_data]
            
            # Set row label colors
            tags = ()
            if row_idx in self.labels:
                if self.labels[row_idx] == 0:
                    tags = ('marked_0',)
                else:
                    tags = ('marked_1',)
            elif row_idx == self.current_row:
                tags = ('current_row',)
            
            # Insert row with row number as ID
            item_id = f"row_{row_idx}"
            self.tree.insert('', 'end', iid=item_id, values=values, tags=tags)
        
        # Set tag colors
        self.tree.tag_configure('marked_0', background='lightblue')
        self.tree.tag_configure('marked_1', background='lightgreen')
        self.tree.tag_configure('current_row', background='yellow')
        
        # Update row info
        self.row_info_label.config(text=f"Current Row: {self.current_row + 1} / {self.total_rows}")
        
        # Update chart
        self.update_plot()
    
    def smooth_data(self, data, window_size):
        """Smooth data using sliding window average"""
        if window_size <= 1:
            return data
        
        # Ensure window size is odd
        if window_size % 2 == 0:
            window_size += 1
        
        # Use numpy convolve for efficient smoothing
        kernel = np.ones(window_size) / window_size
        smoothed = np.convolve(data, kernel, mode='same')
        
        return smoothed
    
    def update_plot(self):
        """Update chart display"""
        if self.transposed_data is None:
            return
            
        self.ax.clear()
        
        # Plot current row data
        row_data = self.transposed_data[self.current_row]
        
        # Get user-specified x-axis range
        try:
            x_start = int(self.x_start_entry.get())
            x_end = int(self.x_end_entry.get())
            
            # Ensure valid range
            if x_start < 0:
                x_start = 0
            if x_end > len(row_data):
                x_end = len(row_data)
            if x_start >= x_end:
                x_start = 0
                x_end = min(100, len(row_data))
                
        except ValueError:
            # Use default range if input invalid
            x_start = 0
            x_end = min(100, len(row_data))
        
        # Update input box display
        self.x_start_entry.delete(0, tk.END)
        self.x_start_entry.insert(0, str(x_start))
        self.x_end_entry.delete(0, tk.END)
        self.x_end_entry.insert(0, str(x_end))
        
        # Plot data in specified range
        x = range(x_start, x_end)
        y = row_data[x_start:x_end]
        
        # Check if smoothing is enabled
        if self.smooth_var.get():
            try:
                window_size = int(self.window_size_entry.get())
                if window_size > 1:
                    y_smoothed = self.smooth_data(y, window_size)
                    # Plot original data (dashed line)
                    self.ax.plot(x, y, 'b--', linewidth=0.8, alpha=0.6, label='Original')
                    # Plot smoothed data (solid line)
                    self.ax.plot(x, y_smoothed, 'r-', linewidth=1.5, label=f'Smoothed (window={window_size})')
                    self.ax.legend()
                else:
                    # Window size too small, show warning and use original data
                    self.ax.plot(x, y, 'b-', linewidth=1)
                    messagebox.showwarning("Warning", "Window size must be greater than 1 for smoothing")
            except ValueError:
                # If window size input invalid, use original data and show error
                self.ax.plot(x, y, 'b-', linewidth=1)
                messagebox.showerror("Error", "Please enter a valid number for window size")
        else:
            self.ax.plot(x, y, 'b-', linewidth=1)
        
        # Set title and labels
        label_text = "Unlabeled"
        if self.current_row in self.labels:
            label_text = f"Labeled as {self.labels[self.current_row]}"
        
        # Build title info
        smooth_status = ""
        if self.smooth_var.get():
            try:
                window_size = int(self.window_size_entry.get())
                smooth_status = f" (Smoothed, window={window_size})"
            except ValueError:
                smooth_status = " (Smoothing Error)"
        
        self.ax.set_title(f"Row {self.current_row + 1} Data (X: {x_start}-{x_end}) - {label_text}{smooth_status}")
        self.ax.set_xlabel("Data Points")
        self.ax.set_ylabel("Values")
        self.ax.grid(True, alpha=0.3)
        
        # Set x-axis range
        self.ax.set_xlim(x_start, x_end)
        
        self.canvas.draw()
    
    def reset_plot_range(self):
        """Reset chart range to default"""
        if self.transposed_data is None:
            return
            
        row_data = self.transposed_data[self.current_row]
        x_start = 0
        x_end = min(100, len(row_data))
        
        self.x_start_entry.delete(0, tk.END)
        self.x_start_entry.insert(0, str(x_start))
        self.x_end_entry.delete(0, tk.END)
        self.x_end_entry.insert(0, str(x_end))
        
        self.update_plot()
    
    def reset_smoothing(self):
        """Reset smoothing to default"""
        self.smooth_var.set(False)
        self.window_size_entry.delete(0, tk.END)
        self.window_size_entry.insert(0, "5")
        self.update_plot()
    
    def mark_row(self, label):
        """Mark current row"""
        if self.transposed_data is None:
            return
            
        self.labels[self.current_row] = label
        self.update_data_display()
        self.update_stats()
    
    def prev_row(self):
        """Jump to previous row"""
        if self.current_row > 0:
            self.current_row -= 1
            self.update_data_display()
    
    def next_row(self):
        """Jump to next row"""
        if self.current_row < self.total_rows - 1:
            self.current_row += 1
            self.update_data_display()
    
    def goto_row(self):
        """Jump to specified row"""
        try:
            row_num = int(self.goto_entry.get()) - 1  # User input starts from 1
            if 0 <= row_num < self.total_rows:
                self.current_row = row_num
                self.update_data_display()
            else:
                messagebox.showerror("Error", f"Row number must be between 1 and {self.total_rows}")
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid row number")
    
    def on_row_select(self, event):
        """Handle row selection event"""
        selection = self.tree.selection()
        if selection:
            # Can add additional selection logic here
            pass
    
    def update_stats(self):
        """Update label statistics"""
        total_marked = len(self.labels)
        marked_0 = sum(1 for label in self.labels.values() if label == 0)
        marked_1 = sum(1 for label in self.labels.values() if label == 1)
        
        self.stats_label.config(
            text=f"Labeled: {total_marked} | Label 0: {marked_0} | Label 1: {marked_1}"
        )
    
    def save_labels(self):
        """Save label results"""
        if not self.labels:
            messagebox.showwarning("Warning", "No labeled data to save")
            return
        
        # Get save filename
        filename = filedialog.asksaveasfilename(
            title="Save Label Results",
            defaultextension=".txt",
            filetypes=[("TXT files", "*.txt"), ("All files", "*.*")]
        )
        
        if not filename:
            return
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                # Write file header
                # f.write("# Data Label Results\n")
                # f.write(f"# Total Rows: {self.total_rows}\n")
                # f.write(f"# Labeled Rows: {len(self.labels)}\n")
                # f.write("# Format: Row Number, Label Value (0 or 1)\n")
                # f.write("# Row numbers start from 0\n\n")
                
                # Write label results sorted by row number
                for row_num in sorted(self.labels.keys()):
                    # f.write(f"{row_num},{self.labels[row_num]}\n")
                    f.write(f"{self.labels[row_num]}\n")
            
            messagebox.showinfo("Success", f"Label results saved to: {filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error saving file: {str(e)}")

def main():
    root = tk.Tk()
    app = DataLabelingGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 