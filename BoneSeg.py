import os
import sys
import types
from pathlib import Path
import threading
import csv
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from nd2reader import ND2Reader
import customtkinter as ctk
from tkinter import filedialog, messagebox
import torch
from skimage.measure import label, regionprops
from skimage.filters import threshold_triangle, threshold_otsu
from scipy.ndimage import binary_dilation, binary_fill_holes
from cellpose import models

# ==========================================
# PERFORMANCE & PYTORCH ENVIRONMENT CONFIGURATION
# ==========================================
mock_modules = [
    'torch._dynamo', 'torch._dynamo.utils', 'torch._dynamo.config',
    'torch._dynamo.convert_frame', 'torch._dynamo.eval_frame',
    'torch._dynamo.resume_execution', 'torch._numpy'
]
for mod_name in mock_modules:
    mock_mod = types.ModuleType(mod_name)
    sys.modules[mod_name] = mock_mod

sys.modules['torch._dynamo.utils'].is_compile_supported = lambda: False
sys.modules['torch._dynamo'].optimize = lambda *args, **kwargs: (lambda x: x)
torch.jit._state.disable()
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# ARCHITECTURAL DEPENDENCY RUNTIME PATH MATCHING
# ==========================================
if getattr(sys, 'frozen', False):
    BASE_DIR = Path(sys._MEIPASS)
else:
    BASE_DIR = Path(__file__).parent.absolute()

SAM2_ROOT = BASE_DIR / "sam2"
if str(SAM2_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM2_ROOT))
os.environ["PYTHONPATH"] = str(SAM2_ROOT)

from hydra.core.global_hydra import GlobalHydra
from hydra import initialize_config_dir
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

GlobalHydra.instance().clear()
config_dir = str(SAM2_ROOT / "sam2" / "configs")
initialize_config_dir(config_dir=config_dir, version_base=None)

# ==========================================
# GRAPHICAL USER INTERFACE CLASS
# ==========================================
class SegmentationApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("BoneSeg - Forensic Bio-Imaging Data Management Pipeline")
        self.geometry("1400x950")
        ctk.set_appearance_mode("dark")
        
        self.left_frame = ctk.CTkFrame(self, width=350, corner_radius=10)
        self.left_frame.pack(side="left", fill="y", padx=10, pady=10)
        
        self.right_frame = ctk.CTkFrame(self, corner_radius=10)
        self.right_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        
        self.bottom_frame = ctk.CTkFrame(self, height=150, corner_radius=10)
        self.bottom_frame.pack(side="bottom", fill="x", padx=10, pady=10)

        self.image_label = ctk.CTkLabel(self.right_frame, text="Load an ND2 microscopy file to initiate pipeline workflow.")
        self.image_label.pack(expand=True)

        self.batch_folder_path = ctk.StringVar(value="No Folder Selected")
        self.file_path = ctk.StringVar()
        self.object_choice = ctk.StringVar(value="lacunae")
        self.channel_choice = ctk.StringVar(value="DAPI")
        self.batch_target_choice = ctk.StringVar(value="All Models") 
        
        self.base_diam_var = ctk.StringVar(value="15") # Default base size optimized for lacunae
        self.macro_diam_var = ctk.StringVar(value="100")
        self.radial_dist_var = ctk.StringVar(value="5, 20, 50, 100") 
        self.export_qa_var = ctk.BooleanVar(value=True)
        
        self.all_channels_data = {} 
        self.current_mask = None
        self.latest_stats = None
        self.current_display_np = None
        self.segmented_image_np = None
        
        self.input_points = []
        self.input_labels = []

        self.build_left_panel()
        self.results_text = ctk.CTkTextbox(self.bottom_frame, height=140)
        self.results_text.pack(fill="both", expand=True, padx=10, pady=10)

    def build_left_panel(self):
        row = 0
        ctk.CTkLabel(self.left_frame, text="--- Manual Mode ---", text_color="cyan").grid(row=row, column=0, pady=(10, 0))
        row += 1
        ctk.CTkButton(self.left_frame, text="Browse Single File (.nd2)", command=self.select_file).grid(row=row, column=0, padx=10, pady=5)
        row += 1
        ctk.CTkLabel(self.left_frame, text="Microscope View:").grid(row=row, column=0, sticky="w", padx=10, pady=(5, 0))
        row += 1
        self.channel_combo = ctk.CTkComboBox(self.left_frame, variable=self.channel_choice, values=["Pending..."], command=self.update_view)
        self.channel_combo.grid(row=row, column=0, sticky="w", padx=10)
        row += 1
        
        ctk.CTkLabel(self.left_frame, text="Target Object Structure:").grid(row=row, column=0, sticky="w", padx=10, pady=(15, 0))
        row += 1
        self.object_combo = ctk.CTkComboBox(self.left_frame, variable=self.object_choice, values=["lacunae", "all objects"])
        self.object_combo.grid(row=row, column=0, sticky="w", padx=10)
        row += 1
        
        self.segment_button = ctk.CTkButton(self.left_frame, text="Run Manual Segmentation", fg_color="green", command=self.run_segmentation)
        self.segment_button.grid(row=row, column=0, sticky="w", padx=10, pady=15)
        row += 1
        self.save_results_button = ctk.CTkButton(self.left_frame, text="Save Manual Results (.csv)", state="disabled", command=self.save_current_results)
        self.save_results_button.grid(row=row, column=0, sticky="w", padx=10, pady=5)
        row += 1

        ctk.CTkLabel(self.left_frame, text="--- Batch Mode ---", text_color="purple").grid(row=row, column=0, pady=(20, 0))
        row += 1
        self.folder_entry = ctk.CTkEntry(self.left_frame, textvariable=self.batch_folder_path, width=280)
        self.folder_entry.grid(row=row, column=0, padx=10, pady=5)
        row += 1
        ctk.CTkButton(self.left_frame, text="Select Batch Folder", command=self.select_folder).grid(row=row, column=0, padx=10, pady=15)
        row += 1
        
        ctk.CTkLabel(self.left_frame, text="Select Batch Target:").grid(row=row, column=0, sticky="w", padx=10, pady=(0, 0))
        row += 1
        self.batch_target_combo = ctk.CTkComboBox(
            self.left_frame, 
            variable=self.batch_target_choice, 
            values=["All Models", "Canals Only (Dark/Bright)", "Lacunae Only"]
        )
        self.batch_target_combo.grid(row=row, column=0, sticky="w", padx=10, pady=(0, 10))
        row += 1

        self.advanced_visible = False
        self.adv_toggle_btn = ctk.CTkButton(self.left_frame, text="⚙️ Show Advanced Options", fg_color="gray30", hover_color="gray40", command=self.toggle_advanced)
        self.adv_toggle_btn.grid(row=row, column=0, padx=10, pady=(5, 10), sticky="ew")
        row += 1

        self.adv_frame = ctk.CTkFrame(self.left_frame, fg_color="transparent")
        self.adv_row_index = row 
        row += 1

        ctk.CTkLabel(self.adv_frame, text="Base Object Size (px):").pack(anchor="w", pady=(0, 0))
        ctk.CTkEntry(self.adv_frame, textvariable=self.base_diam_var).pack(fill="x", pady=(0, 5))
        
        ctk.CTkLabel(self.adv_frame, text="Macro Object Size (px):").pack(anchor="w", pady=(0, 0))
        ctk.CTkEntry(self.adv_frame, textvariable=self.macro_diam_var).pack(fill="x", pady=(0, 5))
        
        ctk.CTkLabel(self.adv_frame, text="Radial Distances (px, comma separated):").pack(anchor="w", pady=(5, 0))
        ctk.CTkEntry(self.adv_frame, textvariable=self.radial_dist_var).pack(fill="x", pady=(0, 5))
        
        ctk.CTkCheckBox(self.adv_frame, text="Export QA Visual Proofs (.png)", variable=self.export_qa_var).pack(anchor="w", pady=(5, 5))

        self.batch_button = ctk.CTkButton(self.left_frame, text="🚀 RUN FULL BATCH", fg_color="purple", height=40, command=self.trigger_batch)
        self.batch_button.grid(row=row, column=0, padx=10, pady=(10, 5), sticky="ew")
        row += 1
        
        ctk.CTkLabel(self.left_frame, text="--- Stitched Macro Pipeline ---", text_color="orange").grid(row=row, column=0, pady=(15, 0))
        row += 1
        self.stitch_button = ctk.CTkButton(self.left_frame, text="🧩 Run Stitched Bone Macro", fg_color="orange", text_color="black", height=40, command=self.trigger_stitch_macro)
        self.stitch_button.grid(row=row, column=0, padx=10, pady=5, sticky="ew")

    def toggle_advanced(self):
        if self.advanced_visible:
            self.adv_frame.grid_forget()
            self.advanced_visible = False
            self.adv_toggle_btn.configure(text="⚙️ Show Advanced Options")
        else:
            self.adv_frame.grid(row=self.adv_row_index, column=0, padx=10, pady=0, sticky="ew")
            self.advanced_visible = True
            self.adv_toggle_btn.configure(text="⚙️ Hide Advanced Options")

    def select_file(self):
        file = filedialog.askopenfilename(title="Open ND2 File", filetypes=[("ND2 files", "*.nd2")])
        if file:
            self.file_path.set(file)
            self.load_and_display_image(file)

    def load_and_display_image(self, path):
        try:
            with ND2Reader(path) as images:
                channels = images.metadata.get('channels', [])
                self.channel_combo.configure(values=channels)
                self.all_channels_data = {}
                for c_name in channels:
                    c_idx = channels.index(c_name)
                    mip = np.max(np.stack([images.get_frame_2D(c=c_idx, z=z) for z in range(images.sizes.get('z', 1))]), axis=0)
                    self.all_channels_data[c_name] = mip.astype(np.float32)
                
                self.current_mask = None
                target_chan = next((c for c in channels if "DAPI" in c.upper() or "405" in str(c)), channels[0])
                self.channel_choice.set(target_chan)
                self.update_view(target_chan)
            self.results_text.insert("end", f"Successfully extracted tensor structures for: {os.path.basename(path)}\n")
        except Exception as e:
            messagebox.showerror("I/O Error", str(e))

    def update_view(self, choice=None):
        if not self.all_channels_data: return
        selected_ch = self.channel_choice.get()
        if selected_ch in self.all_channels_data:
            img2d = self.all_channels_data[selected_ch]
            img_min, img_max = img2d.min(), img2d.max()
            img_uint8 = ((img2d - img_min) / (img_max - img_min + 1e-8) * 255).astype(np.uint8)
            self.current_display_np = img_uint8
            
            if self.current_mask is not None:
                self.apply_mask_overlay(self.current_mask)
            else:
                self.display_image(img_uint8)

    def display_image(self, img_np):
        img_pil = Image.fromarray(img_np)
        self.current_image_tk = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=(800, 800))
        self.image_label.configure(image=self.current_image_tk, text="")

    def run_segmentation(self):
        if not self.all_channels_data: return
        dapi_key = next((k for k in self.all_channels_data.keys() if "DAPI" in k.upper() or "405" in str(k)), None)
        if not dapi_key: return
            
        ai_input = self.all_channels_data[dapi_key]
        img_min, img_max = ai_input.min(), ai_input.max()
        ai_input_8bit = ((ai_input - img_min) / (img_max - img_min + 1e-8) * 255).astype(np.uint8)

        model_cfg = "sam2.1/sam2.1_hiera_b+.yaml"
        ckpt_path = os.path.join(BASE_DIR, "sam2", "checkpoints", "finetuned_checkpoints", "checkpoint_blanc_DAPI.pt")
        
        try:
            self.results_text.insert("end", f"Compiling architecture weights for context target...\n")
            model = build_sam2(model_cfg, ckpt_path, device=device)
            predictor = SAM2ImagePredictor(model)
            rgb = np.stack([ai_input_8bit] * 3, axis=-1)
            predictor.set_image(rgb)
            
            self.enable_click_predictor(rgb, predictor)
            self.results_text.insert("end", " Runtime Predictor Active \n[Left-Click: Append Pixel] | [Right-Click: Extrapolate Void] | [C Key: Clear Pipeline]\n")
        except Exception as e:
            messagebox.showerror("Computer Vision Pipeline Exception", str(e))

    def enable_click_predictor(self, rgb, predictor):
        self.input_points = []
        self.input_labels = []

        def on_left_click(event): add_point(event, 1)
        def on_right_click(event): add_point(event, 0)

        def add_point(event, label_val):
            x = int(event.x * (self.current_display_np.shape[1] / 800))
            y = int(event.y * (self.current_display_np.shape[0] / 800))
            
            self.input_points.append([x, y])
            self.input_labels.append(label_val)
            
            with torch.inference_mode():
                masks, scores, _ = predictor.predict(
                    point_coords=np.array(self.input_points),
                    point_labels=np.array(self.input_labels)
                )
            
            self.current_mask = np.array(masks[np.argmax(scores)], dtype=bool)
            self.apply_mask_overlay(self.current_mask)
            self.compute_and_display_metrics(self.current_mask)

        def clear_points(event=None):
            self.input_points.clear()
            self.input_labels.clear()
            self.current_mask = None
            self.display_image(self.current_display_np)
            self.results_text.insert("end", "🧹 Tracking memory flushed. System cleared.\n")

        self.image_label.unbind("<Button-1>")
        self.image_label.bind("<Button-1>", on_left_click)
        self.image_label.bind("<Button-3>", on_right_click)
        self.image_label.bind("<Button-2>", on_right_click) 
        self.bind("<c>", clear_points)
        self.bind("<C>", clear_points)

    def apply_mask_overlay(self, mask):
        overlay = np.zeros((*mask.shape, 3), dtype=np.uint8)
        overlay[mask > 0] = [255, 0, 0]
        base_rgb = np.stack([self.current_display_np] * 3, axis=-1)
        self.segmented_image_np = (base_rgb * 0.6 + overlay * 0.4).astype(np.uint8)
        self.display_image(self.segmented_image_np)
        self.save_results_button.configure(state="normal")

    def compute_and_display_metrics(self, mask):
        props = regionprops(label(mask))
        if not props: return
        target = max(props, key=lambda r: r.area)
        
        border = binary_dilation(mask, iterations=5) ^ mask
        
        self.latest_stats = {
            "sample": os.path.basename(self.file_path.get()),
            "area": round(target.area, 2),
            "channels": {}
        }
        
        display_text = f"\n--- Quantitative Extraction: {self.latest_stats['sample']} ---\n"
        for ch_name, ch_data in self.all_channels_data.items():
            mean_in = np.mean(ch_data[mask > 0])
            mean_out = np.mean(ch_data[border > 0])
            ratio = mean_in / (mean_out + 1e-8)
            self.latest_stats["channels"][ch_name] = (round(mean_in, 2), round(mean_out, 2), round(ratio, 3))
            display_text += f"{ch_name} (IO/IB Ratio): {round(ratio, 3)}\n"
            
        self.results_text.insert("end", display_text)

    def save_current_results(self):
        if not self.latest_stats: return
        file = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="Manual_Validation_Results.csv")
        if file:
            file_exists = os.path.isfile(file)
            with open(file, "a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    header = ["Sample", "Area"]
                    for ch in self.latest_stats["channels"].keys():
                        header += [f"{ch} In", f"{ch} Border", f"{ch} Ratio"]
                    writer.writerow(header)
                row = [self.latest_stats["sample"], self.latest_stats["area"]]
                for vals in self.latest_stats["channels"].values():
                    row += [vals[0], vals[1], vals[2]]
                writer.writerow(row)
            self.results_text.insert("end", "✅ Observation vector committed directly to spreadsheet architecture storage.\n")

    def select_folder(self):
        folder = filedialog.askdirectory(title="Select Top-Level Experimental Repository Root")
        if folder: self.batch_folder_path.set(folder)

    def trigger_batch(self):
        folder = self.batch_folder_path.get()
        if folder == "No Folder Selected": return
        
        target_workflow = self.batch_target_choice.get()
        export_qa = self.export_qa_var.get()
        
        try:
            base_diam = float(self.base_diam_var.get()) if self.base_diam_var.get().strip() else 30.0
            macro_diam = float(self.macro_diam_var.get()) if self.macro_diam_var.get().strip() else 100.0
            
            dist_str = self.radial_dist_var.get()
            radial_dilations = [int(x.strip()) for x in dist_str.split(",") if x.strip().isdigit()]
            if not radial_dilations: radial_dilations = [5] 
            radial_dilations = sorted(radial_dilations) 
            
        except ValueError:
            messagebox.showerror("Input Error", "Size settings must be numbers.")
            return
        
        model_cfg = "sam2.1/sam2.1_hiera_b+.yaml"
        ckpt_path = os.path.join(BASE_DIR, "sam2", "checkpoints", "finetuned_checkpoints", "checkpoint_blanc_DAPI.pt")
        try:
            model = build_sam2(model_cfg, ckpt_path, device=device)
            predictor = SAM2ImagePredictor(model)
            
            threading.Thread(target=RUN_EMERGENCY_BATCH, args=(predictor, folder, target_workflow, base_diam, macro_diam, export_qa, radial_dilations), daemon=True).start()
        except Exception as e: 
            print(f"Exception triggered in background task scheduling configuration: {e}")

    def trigger_stitch_macro(self):
        folder = self.batch_folder_path.get()
        if folder == "No Folder Selected": 
            messagebox.showerror("Folder Required", "Please select your root experimental folder first.")
            return
        threading.Thread(target=RUN_STITCHED_MACRO, args=(folder,), daemon=True).start()


# ==========================================
# HIGH-THROUGHPUT PIPELINE DATA EXTRACTION SUBPROCESS
# ==========================================
def RUN_EMERGENCY_BATCH(loaded_predictor, data_root, target_workflow, base_diam, macro_diam, export_qa, radial_dilations):
    desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
    output_csv = os.path.join(desktop, "Final_Forensic_Data.csv")
    count_summary_csv = os.path.join(desktop, "Structure_Counts_Summary.csv")
    
    if export_qa:
        qa_visual_dir = os.path.join(desktop, "BoneSeg_QA_Visuals")
        os.makedirs(qa_visual_dir, exist_ok=True)
    
    print(f"\nInitializing Batch Run for: {target_workflow}")
    
    # --- UPGRADED MODEL PATH ROUTING ---
    models_dir = BASE_DIR / "models"
    if not models_dir.exists():
        models_dir = BASE_DIR.parent / "models"
        
    path_lacunae = str(models_dir / "BoneSeg_Lacunae")
    path_dark = str(models_dir / "BoneSeg_Canals_Dark")
    path_bright = str(models_dir / "BoneSeg_Canals_Bright")

    cp_library = {}
    print(f"Loading models from: {models_dir}")
    
    try:
        if target_workflow in ["All Models", "Lacunae Only"]:
            print("Attempting to load Cellpose Lacunae model into GPU...")
            cp_library["lacunae"] = models.CellposeModel(gpu=True, pretrained_model=path_lacunae)
            print("✅ Lacunae model loaded successfully!")
            
        if target_workflow in ["All Models", "Canals Only (Dark/Bright)"]:
            print("Attempting to load Cellpose Canal models into GPU...")
            cp_library["dark"] = models.CellposeModel(gpu=True, pretrained_model=path_dark)
            cp_library["bright"] = models.CellposeModel(gpu=True, pretrained_model=path_bright)
            print("✅ Canal models loaded successfully!")
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR LOADING MODELS: {e}")
        print("Please check your file paths and ensure the GPU has enough memory.")
        return 
    # -----------------------------------
    
    color_palette = [ [0, 255, 0], [255, 255, 0], [255, 165, 0], [255, 0, 255], [0, 255, 255], [255, 255, 255] ]
    channels_list = ["DAPI", "EGFP", "Cy3"]
    
    file_exists = os.path.isfile(output_csv)
    with open(output_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            header = ["Sample_ID", "Region", "Structure", "Full_Filename", "Group", "Category", "Canal_ID", "Area"]
            for ch in channels_list: header.append(f"{ch}_In")
            for d in radial_dilations:
                for ch in channels_list: header.extend([f"{ch}_Out_{d}px", f"{ch}_Ratio_{d}px"])
            for ch in channels_list: header.append(f"Global_{ch}_In")
            for d in radial_dilations:
                for ch in channels_list: header.extend([f"Global_{ch}_Out_{d}px", f"Global_{ch}_Ratio_{d}px"])
            writer.writerow(header)

        count_file_exists = os.path.isfile(count_summary_csv)
        with open(count_summary_csv, "a", newline="", encoding="utf-8") as count_f:
            count_writer = csv.writer(count_f)
            if not count_file_exists:
                count_writer.writerow(["Sample_ID", "Region", "Filename", "Group", "Category", "Total_Objects_Count"])

            for root, dirs, files in os.walk(data_root):
                for file in files:
                    name_lower = file.lower()
                    
                    if not file.endswith(".nd2"): 
                        continue
                    if "stitch" in name_lower or "macro" in name_lower or "stithc" in name_lower: 
                        print(f"Skipping stitched file: {file}")
                        continue
                        
                    print(f"\nProcessing {file}...")
                    
                    file_path = os.path.join(root, file)
                    path_obj = Path(file_path)
                    name = path_obj.stem                             
                    structure_val = name                             
                    region_val = path_obj.parent.name                
                    sample_id = path_obj.parent.parent.name if len(path_obj.parts) >= 3 else "Unknown"
                    
                    group = "ANAT" if "ANAT" in root.upper() or "ANAT" in name.upper() else "KOEK"
                    cat = target_workflow.replace(" Only", "")

                    try:
                        with ND2Reader(file_path) as imgs:
                            channels = imgs.metadata['channels']
                            data = {ch: np.max(np.stack([imgs.get_frame_2D(c=i, z=z) for z in range(imgs.sizes.get('z', 1))]), axis=0).astype(np.float32) for i, ch in enumerate(channels)}
                        
                        dapi_key = next(k for k in data.keys() if "405" in k or "DAPI" in k)
                        img = data[dapi_key]
                        img_8 = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
                        img_rgb = np.stack([img_8]*3, axis=-1)
                        loaded_predictor.set_image(img_rgb)
                        qa_canvas = img_rgb.copy() if export_qa else None
                        
                        models_to_run = []
                        if target_workflow == "Lacunae Only": 
                            models_to_run = [("lacunae", cp_library["lacunae"])]
                        elif target_workflow == "Canals Only (Dark/Bright)": 
                            models_to_run = [("bright", cp_library["bright"]), ("dark", cp_library["dark"])]
                        else: 
                            models_to_run = [("lacunae", cp_library["lacunae"]), ("bright", cp_library["bright"]), ("dark", cp_library["dark"])]

                        global_canal_id = 1
                        total_objects_found = 0
                        
                        approved_sam_masks = []
                        approved_metadata = [] 
                        
                        for model_name, active_cp_model in models_to_run:
                            if model_name == "lacunae":
                                masks_cp, _, _ = active_cp_model.eval(img_rgb, diameter=base_diam)
                            else:
                                masks_standard, _, _ = active_cp_model.eval(img_rgb, diameter=base_diam)
                                masks_large, _, _ = active_cp_model.eval(img_rgb, diameter=macro_diam)
                                max_standard = masks_standard.max()
                                large_mapped = np.where(masks_large > 0, masks_large + max_standard, 0)
                                masks_cp = np.where(masks_standard > 0, masks_standard, large_mapped)
                            
                            unique_canals = np.unique(masks_cp)[1:] 
                            print(f"Sweeps found {len(unique_canals)} unique objects using {model_name} model.")
                            
                            # --- THE FAST BYPASS ---
                            for i, cp_id in enumerate(unique_canals):
                                # Print a progress update every 20 objects so you know it's not frozen
                                if i % 20 == 0:
                                    print(f"  -> Extracting object {i+1} of {len(unique_canals)}...")
                                    
                                single_mask = (masks_cp == cp_id).astype(np.uint8)
                                x, y, w, h = cv2.boundingRect(single_mask)
                                
                                if model_name == "lacunae":
                                    # Cellpose perfectly segments tiny dots. Skip SAM2 to save massive time.
                                    sam_mask = single_mask.astype(bool)
                                else:
                                    # Canals are complex. Use SAM2 to perfectly snap to the edges.
                                    pad = 10
                                    input_box = np.array([max(0, x - pad), max(0, y - pad), min(img_rgb.shape[1], x + w + pad), min(img_rgb.shape[0], y + h + pad)])
                                    sam_masks, _, _ = loaded_predictor.predict(box=input_box[None, :], multimask_output=False)
                                    sam_mask = np.array(sam_masks[0], dtype=bool)
                                
                                is_duplicate = False
                                for prev_mask in approved_sam_masks:
                                    smaller_area = min(sam_mask.sum(), prev_mask.sum())
                                    if smaller_area > 0 and (np.logical_and(sam_mask, prev_mask).sum() / smaller_area) > 0.5:
                                        is_duplicate = True
                                        break
                                if is_duplicate: continue 
                                    
                                approved_sam_masks.append(sam_mask)
                                approved_metadata.append({'id': global_canal_id, 'mask': sam_mask, 'box': (x, y, w, h)})
                                global_canal_id += 1
                                total_objects_found += 1

                        count_writer.writerow([sample_id, region_val, name, group, cat, total_objects_found])
                        count_f.flush()

                        master_canal_mask = np.zeros(img_rgb.shape[:2], dtype=bool)
                        for m in approved_sam_masks: master_canal_mask = master_canal_mask | m

                        global_data = {ch: {"In": 0.0, "Out": {}, "Ratio": {}} for ch in channels_list}
                        if np.any(master_canal_mask):
                            for ch in channels_list:
                                ch_key = next((k for k in data.keys() if ch in k.upper()), None)
                                if ch_key: global_data[ch]["In"] = round(np.mean(data[ch_key][master_canal_mask]), 3)
                                    
                            for d in radial_dilations:
                                g_raw_border = binary_dilation(master_canal_mask, iterations=d)
                                g_clean_border = g_raw_border ^ master_canal_mask
                                
                                for ch in channels_list:
                                    ch_key = next((k for k in data.keys() if ch in k.upper()), None)
                                    if ch_key and np.sum(g_clean_border) > 0:
                                        out_val = np.mean(data[ch_key][g_clean_border])
                                        global_data[ch]["Out"][d] = round(out_val, 3)
                                        global_data[ch]["Ratio"][d] = round(global_data[ch]["In"] / (out_val + 1e-8), 3)
                                    else:
                                        global_data[ch]["Out"][d] = 0.0
                                        global_data[ch]["Ratio"][d] = 0.0
                        else:
                            for d in radial_dilations:
                                for ch in channels_list:
                                    global_data[ch]["Out"][d] = 0.0
                                    global_data[ch]["Ratio"][d] = 0.0

                        for item in approved_metadata:
                            c_id = item['id']
                            sam_mask = item['mask']
                            x, y, w, h = item['box']
                            
                            row = [sample_id, region_val, structure_val, name, group, cat, c_id, np.sum(sam_mask)]
                            
                            ind_in_vals = {}
                            for ch in channels_list:
                                ch_key = next((k for k in data.keys() if ch in k.upper()), None)
                                if ch_key:
                                    val = round(np.mean(data[ch_key][sam_mask]), 3)
                                    ind_in_vals[ch] = val
                                    row.append(val)
                                else:
                                    ind_in_vals[ch] = 0.0
                                    row.append(0.0)
                            
                            dapi_ratio_display = 0.0
                            if export_qa:
                                mask_uint8 = (sam_mask * 255).astype(np.uint8)
                                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                cv2.drawContours(qa_canvas, contours, -1, (255, 0, 0), 2)
                                mask_overlay = qa_canvas.copy()
                                mask_overlay[sam_mask] = [255, 0, 0] 
                                cv2.addWeighted(mask_overlay, 0.5, qa_canvas, 0.5, 0, qa_canvas)
                                border_overlay = qa_canvas.copy()
                            
                            prev_raw_border = np.zeros_like(sam_mask)
                            for i, d in enumerate(radial_dilations):
                                raw_border = binary_dilation(sam_mask, iterations=d) ^ sam_mask
                                clean_border = raw_border & ~master_canal_mask
                                
                                if export_qa:
                                    band_mask = clean_border & ~prev_raw_border
                                    current_color = color_palette[i % len(color_palette)]
                                    border_overlay[band_mask] = current_color
                                
                                prev_raw_border = raw_border 
                                
                                for ch in channels_list:
                                    ch_key = next((k for k in data.keys() if ch in k.upper()), None)
                                    if ch_key and np.sum(clean_border) > 0:
                                        out_val = round(np.mean(data[ch_key][clean_border]), 3)
                                        ratio_calc = round(ind_in_vals[ch] / (out_val + 1e-8), 3)
                                    else:
                                        out_val = 0.0
                                        ratio_calc = 0.0
                                        
                                    row.extend([out_val, ratio_calc])
                                    if ch == "DAPI" and d == radial_dilations[0]: dapi_ratio_display = ratio_calc 
                            
                            for ch in channels_list: row.append(global_data[ch]["In"])
                            for d in radial_dilations:
                                for ch in channels_list: row.extend([global_data[ch]["Out"][d], global_data[ch]["Ratio"][d]])
                            
                            if export_qa:
                                cv2.addWeighted(border_overlay, 0.4, qa_canvas, 0.6, 0, qa_canvas)
                                cv2.rectangle(qa_canvas, (x, y), (x + w, y + h), (0, 255, 255), 2)
                                cv2.putText(qa_canvas, f"ID:{c_id} DAPI:{dapi_ratio_display}", (x, max(y - 6, 15)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
                                    
                            writer.writerow(row)
                            f.flush()
                                
                        print(f"Finished {name}. Total objects extracted: {total_objects_found}.")
                        
                        if export_qa:
                            qa_filename = os.path.join(qa_visual_dir, f"{sample_id}_{region_val}_{name}_QA_Proof.png")
                            cv2.imwrite(qa_filename, cv2.cvtColor(qa_canvas, cv2.COLOR_RGB2BGR))
                        
                    except Exception as e: print(f"Error processing {name}: {e}")
                        
            print("\n✅ BATCH PROCESSING COMPLETE! Check your Desktop for the Final Data and the new Counts Summary CSV.")


# ==========================================
# THE PROFESSOR'S STITCHED BONE MACRO PIPELINE
# ==========================================
def RUN_STITCHED_MACRO(data_root):
    desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
    output_csv = os.path.join(desktop, "Stitched_Bone_Histograms.csv")
    qa_visual_dir = os.path.join(desktop, "BoneSeg_Stitch_Visuals")
    os.makedirs(qa_visual_dir, exist_ok=True)
    
    print(f"\n--- Initiating Stitched Bone Macro ---")
    
    file_exists = os.path.isfile(output_csv)
    with open(output_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            header = ["Sample_ID", "Region", "Filename", "Group", "Processing_Method", "Total_Bone_Area"]
            for i in range(256): header.append(f"Bin_{i}")
            writer.writerow(header)

        all_hists = []
        hist_labels = []
        
        anat_hists = []
        koek_hists = []
        
        cropped_anat_hists = []
        cropped_koek_hists = []

        for root, dirs, files in os.walk(data_root):
            for file in files:
                if not file.endswith(".nd2"): continue
                if "stitch" not in file.lower() and "stithc" not in file.lower(): continue
                
                file_path = os.path.join(root, file)
                path_obj = Path(file_path)
                name = path_obj.stem
                
                region_val = path_obj.parent.name
                sample_id = path_obj.parent.parent.name if len(path_obj.parts) >= 3 else "Unknown"
                group = "ANAT" if "ANAT" in file_path.upper() else "KOEK"

                try:
                    with ND2Reader(file_path) as imgs:
                        raw_channels = imgs.metadata.get('channels', [])
                        if not raw_channels:
                            c_count = imgs.sizes.get('c', 1)
                            raw_channels = [f"Channel_{i}" for i in range(c_count)]
                            
                        data = {ch: np.max(np.stack([imgs.get_frame_2D(c=c_idx, z=z) for z in range(imgs.sizes.get('z', 1))]), axis=0).astype(np.float32) for c_idx, ch in enumerate(raw_channels)}
                    
                    dapi_key = next((k for k in data.keys() if "405" in str(k).upper() or "DAPI" in str(k).upper()), None)
                    if not dapi_key:
                        dapi_key = list(data.keys())[0] 
                        
                    img_data = data[dapi_key]
                    
                    processing_method = "Aggressive_Core_Crop"
                    h, w = img_data.shape
                    
                    # --- THE AGGRESSIVE CROP PARAMETER ---
                    # 0.4 means it will only keep the center 40% of the image, 
                    # completely destroying the borders and background glass.
                    crop_factor = 0.40 
                    
                    start_y = int(h * (1 - crop_factor) / 2)
                    end_y = int(start_y + (h * crop_factor))
                    start_x = int(w * (1 - crop_factor) / 2)
                    end_x = int(start_x + (w * crop_factor))
                    
                    # Create a blank mask and isolate ONLY the core
                    core_mask = np.zeros_like(img_data, dtype=bool)
                    core_mask[start_y:end_y, start_x:end_x] = True
                    
                    core_img_data = img_data[start_y:end_y, start_x:end_x]
                    
                    # Because we removed the glass, Otsu thresholding becomes hyper-precise
                    thresh_val = threshold_otsu(core_img_data)
                    
                    # Apply threshold only within the core
                    binary_mask = (img_data < thresh_val) & core_mask 
                    filled_mask = binary_fill_holes(binary_mask)
                    labeled_mask = label(filled_mask)
                    props = regionprops(labeled_mask)
                    
                    total_area = 0
                    final_mask = np.zeros_like(filled_mask, dtype=bool)
                    
                    if props:
                        # Keep objects inside the core that are larger than a basic noise threshold
                        for prop in props:
                            if prop.area >= 5000:  
                                final_mask[labeled_mask == prop.label] = True
                                total_area += prop.area
                                
                    if total_area == 0:
                        print(f"Skipping {region_val}_{name}: No bone tissue found even in core crop.")
                        continue

                    # --- VISUAL PROOF GENERATION ---
                    img_8bit = ((img_data - img_data.min()) / (img_data.max() - img_data.min() + 1e-8) * 255).astype(np.uint8)
                    img_rgb = cv2.cvtColor(img_8bit, cv2.COLOR_GRAY2BGR)
                    
                    # Draw a yellow box showing exactly what was cropped
                    cv2.rectangle(img_rgb, (start_x, start_y), (end_x, end_y), (0, 255, 255), 15)
                    
                    thresh_rgb = cv2.cvtColor((binary_mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                    filled_rgb = cv2.cvtColor((filled_mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                    
                    final_overlay = img_rgb.copy()
                    final_overlay[final_mask] = [0, 0, 255] 
                    cv2.addWeighted(final_overlay, 0.4, img_rgb, 0.6, 0, final_overlay)
                    
                    def create_diagnostic_panel(img, text_label, target_h=800):
                        h, w = img.shape[:2]
                        scale = target_h / h
                        resized = cv2.resize(img, (int(w * scale), target_h))
                        cv2.putText(resized, text_label, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)
                        return resized

                    panel1 = create_diagnostic_panel(img_rgb, "1. Brightfield (Alg 2 Crop)")
                    panel2 = create_diagnostic_panel(thresh_rgb, f"2. Otsu (Inverted)")
                    panel3 = create_diagnostic_panel(filled_rgb, "3. Filled & Cropped")
                    panel4 = create_diagnostic_panel(final_overlay, "4. Filtered Bone Kept")

                    diagnostic_strip = cv2.hconcat([panel1, panel2, panel3, panel4])
                    
                    qa_filename = os.path.join(qa_visual_dir, f"{sample_id}_{region_val}_{name}_Methodology_Proof.png")
                    cv2.imwrite(qa_filename, diagnostic_strip)
                    
                    plt.style.use('dark_background')
                    plt.figure(figsize=(8, 5))
                    plt.hist(img_data[final_mask], bins=256, range=(1, 5000), color='xkcd:cyan', alpha=0.7)
                    plt.title(f"Bone Matrix Intensity Distribution: {region_val} ({group})", fontsize=14, color='white')
                    plt.xlabel("Pixel Intensity (1 - 5000)", fontsize=12, color='white')
                    plt.ylabel("Frequency (Pixel Count)", fontsize=12, color='white')
                    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
                    
                    hist_graph_filename = os.path.join(qa_visual_dir, f"{sample_id}_{region_val}_{name}_Histogram_Plot.png")
                    plt.savefig(hist_graph_filename, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    hist, bin_edges = np.histogram(img_data[final_mask], bins=256, range=(1, 5000))
                    row = [sample_id, region_val, name, group, processing_method, total_area]
                    row.extend(hist.tolist())
                    writer.writerow(row)
                    f.flush()
                    
                    if np.sum(hist) > 0:
                        hist_norm = (hist / np.sum(hist)) * 100
                    else:
                        hist_norm = hist
                        
                    all_hists.append(hist_norm)
                    hist_labels.append(f"{region_val}_{name} ({group})")
                    
                    if group == "ANAT":
                        anat_hists.append(hist_norm)
                        if processing_method == "Center_Cropped":
                            cropped_anat_hists.append(hist_norm)
                    else:
                        koek_hists.append(hist_norm)
                        if processing_method == "Center_Cropped":
                            cropped_koek_hists.append(hist_norm)
                    
                    print(f"Processed Stitched File: {region_val}_{name} (Group: {group} | Method: {processing_method})")
                    
                except Exception as e: 
                    print(f"Error processing stitched file {region_val}_{name}: {e}")

        def generate_master_plot(anat_list, koek_list, title, filename):
            if not anat_list and not koek_list: return
            plt.style.use('dark_background')
            plt.figure(figsize=(12, 7))
            x_axis = np.linspace(1, 5000, 256)
                
            if anat_list:
                anat_avg = np.mean(anat_list, axis=0)
                plt.plot(x_axis, anat_avg, color='xkcd:red', linewidth=4.0, label=f"ANAT AVERAGE (n={len(anat_list)})")
            if koek_list:
                koek_avg = np.mean(koek_list, axis=0)
                plt.plot(x_axis, koek_avg, color='xkcd:bright cyan', linewidth=4.0, label=f"KOEK AVERAGE (n={len(koek_list)})")
            
            plt.title(title, fontsize=18, color='white', pad=15)
            plt.xlabel("Pixel Intensity (1 - 5000)", fontsize=14, color='white')
            plt.ylabel("Percentage of Total Bone Matrix (%)", fontsize=14, color='white')
            
            plt.legend(loc='upper right', fontsize=14)
            plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
            plt.tight_layout()
            
            master_plot_file = os.path.join(qa_visual_dir, filename)
            plt.savefig(master_plot_file, dpi=300, bbox_inches='tight')
            plt.close()

        generate_master_plot(anat_hists, koek_hists, "Master Histogram: ANAT vs KOEK (All Samples)", "MASTER_Comparison_All_Samples.png")
        generate_master_plot(cropped_anat_hists, cropped_koek_hists, "Master Histogram: ANAT vs KOEK (Center Cropped Only)", "MASTER_Comparison_Cropped_Only.png")
                    
    print("\n✅ STITCHED BATCH COMPLETE! Check your Desktop for the Histograms CSV, Diagnostic Visuals, and Plot Graphs.")


if __name__ == "__main__":
    app = SegmentationApp()
    app.mainloop()