import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk


class MedicalImageComparisonApp:

    def __init__(self, root):

        self.root = root

        self.root.title(
            "Interactive Recovery and Longitudinal Comparison of Archived Medical Images"
        )

        self.root.geometry("1200x760")
        self.root.minsize(950, 650)

        # =========================================================
        # IMAGES
        # =========================================================

        self.earlier_image = None
        self.later_image = None

        self.earlier_cropped = None
        self.later_cropped = None

        # These contain ONLY the selected ROI
        # after pseudocolour highlighting.
        self.earlier_highlighted = None
        self.later_highlighted = None

        self.earlier_roi = None
        self.later_roi = None

        # =========================================================
        # SETTINGS
        # =========================================================

        self.transparency = tk.DoubleVar(value=50)

        # CHANGED PART 1: Highlight display mode; Grayscale is the default.
        self.highlight_mode = tk.StringVar(value="Grayscale")

        self.shift_x = 0
        self.shift_y = 0

        self.zoom_level = 1.0

        # =========================================================
        # CURRENT COMPARISON
        # =========================================================

        self.current_comparison = None
        self.current_comparison_type = None
        self.current_comparison_source = None

        self.comparison_photo = None

        # =========================================================
        # BUILD
        # =========================================================

        self.build_interface()

    # =============================================================
    # MAIN INTERFACE
    # =============================================================

    def build_interface(self):

        # ---------------------------------------------------------
        # TOP BAR
        # ---------------------------------------------------------

        top = tk.Frame(
            self.root,
            bg="#202124",
            height=52
        )

        top.pack(fill="x")
        top.pack_propagate(False)

        tk.Label(
            top,
            text="Medical Image Comparison",
            bg="#202124",
            fg="white",
            font=("Arial", 14, "bold")
        ).pack(
            side="left",
            padx=12
        )

        tk.Label(
            top,
            text="Modality:",
            bg="#202124",
            fg="white"
        ).pack(
            side="left",
            padx=(10, 4)
        )

        self.modality = ttk.Combobox(
            top,
            values=["CT", "X-ray", "MRI"],
            state="readonly",
            width=8
        )

        self.modality.set("CT")
        self.modality.pack(side="left")

        tk.Button(
            top,
            text="ZOOM +",
            command=self.zoom_in,
            width=7
        ).pack(side="left", padx=(12, 2))

        tk.Button(
            top,
            text="ZOOM -",
            command=self.zoom_out,
            width=7
        ).pack(side="left", padx=2)

        tk.Button(
            top,
            text="PAN ←",
            command=lambda: self.pan(-20, 0),
            width=6
        ).pack(side="left", padx=(10, 2))

        tk.Button(
            top,
            text="PAN →",
            command=lambda: self.pan(20, 0),
            width=6
        ).pack(side="left", padx=2)

        tk.Button(
            top,
            text="PAN ↑",
            command=lambda: self.pan(0, -20),
            width=6
        ).pack(side="left", padx=2)

        tk.Button(
            top,
            text="PAN ↓",
            command=lambda: self.pan(0, 20),
            width=6
        ).pack(side="left", padx=2)

        tk.Button(
            top,
            text="RESET VIEW",
            command=self.reset_view,
            width=10
        ).pack(side="left", padx=8)

        # ---------------------------------------------------------
        # MAIN AREA
        # ---------------------------------------------------------

        main = tk.Frame(
            self.root,
            bg="#eeeeee"
        )

        main.pack(
            fill="both",
            expand=True
        )

        # ---------------------------------------------------------
        # LEFT PANEL
        # ---------------------------------------------------------

        left = tk.Frame(
            main,
            bg="#f5f5f5",
            width=400
        )

        left.pack(
            side="left",
            fill="y",
            padx=6,
            pady=6
        )

        left.pack_propagate(False)

        # ---------------------------------------------------------
        # RIGHT PANEL
        # ---------------------------------------------------------

        right = tk.Frame(
            main,
            bg="white"
        )

        right.pack(
            side="right",
            fill="both",
            expand=True,
            padx=(0, 6),
            pady=6
        )

        # =========================================================
        # EARLIER IMAGE
        # =========================================================

        earlier_frame = tk.LabelFrame(
            left,
            text="Earlier Image",
            font=("Arial", 10, "bold"),
            padx=6,
            pady=5
        )

        earlier_frame.pack(
            fill="x",
            pady=3
        )

        tk.Button(
            earlier_frame,
            text="LOAD EARLIER IMAGE",
            command=self.load_earlier,
            width=22
        ).pack(pady=3)

        self.earlier_label = tk.Label(
            earlier_frame,
            text="No image loaded",
            bg="#dddddd",
            width=35,
            height=6
        )

        self.earlier_label.pack(pady=3)

        self.earlier_info = tk.Label(
            earlier_frame,
            text="Earlier Scan\nAge 5 / 2012",
            justify="left",
            anchor="w"
        )

        self.earlier_info.pack(
            fill="x",
            pady=2
        )

        earlier_buttons = tk.Frame(
            earlier_frame
        )

        earlier_buttons.pack()

        tk.Button(
            earlier_buttons,
            text="CROP EARLIER",
            command=lambda: self.start_crop("earlier"),
            width=15
        ).pack(
            side="left",
            padx=2,
            pady=3
        )

        tk.Button(
            earlier_buttons,
            text="SELECT EARLIER REGION",
            command=lambda: self.start_roi("earlier"),
            width=19
        ).pack(
            side="left",
            padx=2,
            pady=3
        )

        # =========================================================
        # LATER IMAGE
        # =========================================================

        later_frame = tk.LabelFrame(
            left,
            text="Later Image",
            font=("Arial", 10, "bold"),
            padx=6,
            pady=5
        )

        later_frame.pack(
            fill="x",
            pady=3
        )

        tk.Button(
            later_frame,
            text="LOAD LATER IMAGE",
            command=self.load_later,
            width=22
        ).pack(pady=3)

        self.later_label = tk.Label(
            later_frame,
            text="No image loaded",
            bg="#dddddd",
            width=35,
            height=6
        )

        self.later_label.pack(pady=3)

        self.later_info = tk.Label(
            later_frame,
            text="Later Scan\nAge 9 / 2016",
            justify="left",
            anchor="w"
        )

        self.later_info.pack(
            fill="x",
            pady=2
        )

        later_buttons = tk.Frame(
            later_frame
        )

        later_buttons.pack()

        tk.Button(
            later_buttons,
            text="CROP LATER",
            command=lambda: self.start_crop("later"),
            width=15
        ).pack(
            side="left",
            padx=2,
            pady=3
        )

        tk.Button(
            later_buttons,
            text="SELECT LATER REGION",
            command=lambda: self.start_roi("later"),
            width=19
        ).pack(
            side="left",
            padx=2,
            pady=3
        )

        # =========================================================
        # SCAN INFORMATION
        # =========================================================

        scan_frame = tk.LabelFrame(
            left,
            text="Scan Information",
            font=("Arial", 10, "bold"),
            padx=6,
            pady=5
        )

        scan_frame.pack(
            fill="x",
            pady=3
        )

        tk.Label(
            scan_frame,
            text="Earlier Scan:"
        ).grid(
            row=0,
            column=0,
            sticky="w",
            pady=2
        )

        self.earlier_scan_entry = tk.Entry(
            scan_frame,
            width=21
        )

        self.earlier_scan_entry.insert(
            0,
            "Age 5 / 2012"
        )

        self.earlier_scan_entry.grid(
            row=0,
            column=1,
            pady=2
        )

        tk.Label(
            scan_frame,
            text="Later Scan:"
        ).grid(
            row=1,
            column=0,
            sticky="w",
            pady=2
        )

        self.later_scan_entry = tk.Entry(
            scan_frame,
            width=21
        )

        self.later_scan_entry.insert(
            0,
            "Age 9 / 2016"
        )

        self.later_scan_entry.grid(
            row=1,
            column=1,
            pady=2
        )

        tk.Button(
            scan_frame,
            text="UPDATE",
            command=self.update_scan_info,
            width=10
        ).grid(
            row=2,
            column=0,
            columnspan=2,
            pady=3
        )

        # =========================================================
        # ROI STATUS
        # =========================================================

        roi_frame = tk.LabelFrame(
            left,
            text="Region Selection Status",
            font=("Arial", 10, "bold"),
            padx=6,
            pady=5
        )

        roi_frame.pack(
            fill="x",
            pady=3
        )

        self.roi_status = tk.Label(
            roi_frame,
            text="Earlier: Not selected | Later: Not selected",
            fg="#444444"
        )

        self.roi_status.pack(
            pady=3
        )

        # =========================================================
        # RIGHT TITLE
        # =========================================================

        tk.Label(
            right,
            text="COMPARISON WORKSPACE",
            font=("Arial", 13, "bold"),
            bg="white"
        ).pack(
            pady=(3, 5)
        )

        # =========================================================
        # SOURCE
        # =========================================================

        source_frame = tk.LabelFrame(
            right,
            text="1. SELECT COMPARISON SOURCE",
            font=("Arial", 10, "bold"),
            bg="white",
            padx=8,
            pady=5
        )

        source_frame.pack(
            fill="x",
            padx=8,
            pady=3
        )

        self.source_var = tk.StringVar(
            value="Original Images"
        )

        tk.Radiobutton(
            source_frame,
            text="ORIGINAL IMAGES",
            variable=self.source_var,
            value="Original Images",
            command=self.source_changed,
            font=("Arial", 10, "bold"),
            bg="white"
        ).pack(
            side="left",
            padx=15
        )

        tk.Radiobutton(
            source_frame,
            text="HIGHLIGHTED IMAGES",
            variable=self.source_var,
            value="Highlighted Images",
            command=self.source_changed,
            font=("Arial", 10, "bold"),
            bg="white"
        ).pack(
            side="left",
            padx=15
        )

        # =========================================================
        # COLOUR HIGHLIGHT
        # =========================================================

        highlight_frame = tk.LabelFrame(
            right,
            text="2. PSEUDOCOLOUR HIGHLIGHT SELECTED REGIONS",
            font=("Arial", 10, "bold"),
            bg="#f7f7f7",
            padx=8,
            pady=6
        )

        highlight_frame.pack(
            fill="x",
            padx=8,
            pady=3
        )

        # CHANGED PART 2: Highlight mode control.
        tk.Label(
            highlight_frame,
            text="Selected region display:",
            bg="#f7f7f7",
            fg="#444444",
            font=("Arial", 9)
        ).pack(
            side="left",
            padx=5
        )

        tk.Button(
            highlight_frame,
            text="COLOUR HIGHLIGHT",
            command=self.create_highlights,
            width=20,
            font=("Arial", 9, "bold")
        ).pack(
            side="left",
            padx=8
        )

        tk.Radiobutton(
            highlight_frame,
            text="Colour",
            variable=self.highlight_mode,
            value="Colour",
            command=self.highlight_mode_changed,
            bg="#f7f7f7",
            font=("Arial", 9)
        ).pack(side="left", padx=3)

        tk.Radiobutton(
            highlight_frame,
            text="Grayscale",
            variable=self.highlight_mode,
            value="Grayscale",
            command=self.highlight_mode_changed,
            bg="#f7f7f7",
            font=("Arial", 9)
        ).pack(side="left", padx=3)

        self.highlighted_status = tk.Label(
            highlight_frame,
            text="Highlighted images not created yet.",
            bg="#f7f7f7",
            fg="#555555"
        )

        self.highlighted_status.pack(
            side="left",
            padx=8
        )

        # =========================================================
        # ORIGINAL COMPARISON
        # =========================================================

        original_frame = tk.LabelFrame(
            right,
            text="3A. ORIGINAL IMAGE COMPARISON",
            font=("Arial", 10, "bold"),
            bg="white",
            padx=6,
            pady=5
        )

        original_frame.pack(
            fill="x",
            padx=8,
            pady=3
        )

        tk.Button(
            original_frame,
            text="ORIGINAL SIDE-BY-SIDE",
            command=lambda: self.run_comparison(
                "Original Images",
                "Side-by-Side"
            ),
            width=19
        ).grid(
            row=0,
            column=0,
            padx=3,
            pady=2
        )

        tk.Button(
            original_frame,
            text="ORIGINAL OVERLAY",
            command=lambda: self.run_comparison(
                "Original Images",
                "Overlay"
            ),
            width=17
        ).grid(
            row=0,
            column=1,
            padx=3,
            pady=2
        )

        tk.Button(
            original_frame,
            text="ORIGINAL DIFFERENCE",
            command=lambda: self.run_comparison(
                "Original Images",
                "Difference"
            ),
            width=19
        ).grid(
            row=0,
            column=2,
            padx=3,
            pady=2
        )

        tk.Button(
            original_frame,
            text="SAVE ORIGINAL",
            command=lambda: self.save_specific_comparison(
                "Original Images"
            ),
            width=15
        ).grid(
            row=0,
            column=3,
            padx=3,
            pady=2
        )

        # =========================================================
        # HIGHLIGHTED COMPARISON
        # =========================================================

        highlighted_frame = tk.LabelFrame(
            right,
            text="3B. HIGHLIGHTED IMAGE COMPARISON — SELECTED REGIONS ONLY",
            font=("Arial", 10, "bold"),
            bg="#f7f7f7",
            padx=6,
            pady=5
        )

        highlighted_frame.pack(
            fill="x",
            padx=8,
            pady=3
        )

        tk.Button(
            highlighted_frame,
            text="HIGHLIGHTED SIDE-BY-SIDE",
            command=lambda: self.run_comparison(
                "Highlighted Images",
                "Side-by-Side"
            ),
            width=20
        ).grid(
            row=0,
            column=0,
            padx=3,
            pady=2
        )

        tk.Button(
            highlighted_frame,
            text="HIGHLIGHTED OVERLAY",
            command=lambda: self.run_comparison(
                "Highlighted Images",
                "Overlay"
            ),
            width=18
        ).grid(
            row=0,
            column=1,
            padx=3,
            pady=2
        )

        tk.Button(
            highlighted_frame,
            text="HIGHLIGHTED DIFFERENCE",
            command=lambda: self.run_comparison(
                "Highlighted Images",
                "Difference"
            ),
            width=20
        ).grid(
            row=0,
            column=2,
            padx=3,
            pady=2
        )

        tk.Button(
            highlighted_frame,
            text="SAVE HIGHLIGHTED",
            command=lambda: self.save_specific_comparison(
                "Highlighted Images"
            ),
            width=17
        ).grid(
            row=0,
            column=3,
            padx=3,
            pady=2
        )

        # =========================================================
        # TRANSPARENCY
        # =========================================================

        transparency_frame = tk.Frame(
            right,
            bg="white"
        )

        transparency_frame.pack(
            fill="x",
            padx=10,
            pady=2
        )

        tk.Label(
            transparency_frame,
            text="Overlay Transparency:",
            bg="white"
        ).pack(
            side="left"
        )

        tk.Scale(
            transparency_frame,
            from_=0,
            to=100,
            orient="horizontal",
            variable=self.transparency,
            command=self.update_overlay,
            bg="white",
            highlightthickness=0
        ).pack(
            side="left",
            fill="x",
            expand=True
        )

        # =========================================================
        # ALIGNMENT
        # =========================================================

        alignment_frame = tk.Frame(
            right,
            bg="white"
        )

        alignment_frame.pack(
            fill="x",
            padx=10,
            pady=2
        )

        tk.Label(
            alignment_frame,
            text="Alignment:"
        ).pack(
            side="left",
            padx=3
        )

        tk.Button(
            alignment_frame,
            text="←",
            command=lambda: self.align(-10, 0),
            width=3
        ).pack(side="left", padx=2)

        tk.Button(
            alignment_frame,
            text="→",
            command=lambda: self.align(10, 0),
            width=3
        ).pack(side="left", padx=2)

        tk.Button(
            alignment_frame,
            text="↑",
            command=lambda: self.align(0, -10),
            width=3
        ).pack(side="left", padx=2)

        tk.Button(
            alignment_frame,
            text="↓",
            command=lambda: self.align(0, 10),
            width=3
        ).pack(side="left", padx=2)

        tk.Button(
            alignment_frame,
            text="RESET ALIGNMENT",
            command=self.reset_alignment,
            width=16
        ).pack(
            side="left",
            padx=7
        )

        # =========================================================
        # CURRENT COMPARISON
        # =========================================================

        display_frame = tk.LabelFrame(
            right,
            text="4. CURRENT COMPARISON",
            font=("Arial", 10, "bold"),
            bg="white"
        )

        display_frame.pack(
            fill="both",
            expand=True,
            padx=8,
            pady=3
        )

        self.current_label = tk.Label(
            display_frame,
            text="No comparison selected",
            bg="#1e1e1e",
            fg="white",
            font=("Arial", 10)
        )

        self.current_label.pack(
            fill="x"
        )

        # CHANGED PART 3: Difference-view quality note. Hidden for non-difference views.
        self.difference_note = tk.Label(
            display_frame,
            text="",
            bg="white",
            fg="#666666",
            font=("Arial", 8),
            anchor="w"
        )
        self.difference_note.pack(
            fill="x",
            padx=6,
            pady=(2, 0)
        )

        self.comparison_canvas = tk.Canvas(
            display_frame,
            bg="#1e1e1e",
            highlightthickness=0
        )

        self.comparison_canvas.pack(
            fill="both",
            expand=True
        )

        # =========================================================
        # CHANGED PART 4: ALWAYS-VISIBLE DISCLAIMER
        # =========================================================

        self.disclaimer = tk.Label(
            self.root,
            text="For learning and viewing only. Not for diagnosis.",
            anchor="center",
            bg="#eeeeee",
            fg="#555555",
            font=("Arial", 8)
        )

        self.disclaimer.pack(
            side="bottom",
            fill="x"
        )

        # =========================================================
        # STATUS
        # =========================================================

        self.status = tk.Label(
            self.root,
            text="Ready",
            anchor="w",
            bg="#dddddd"
        )

        self.status.pack(
            side="bottom",
            fill="x"
        )

    # =============================================================
    # LOAD EARLIER
    # =============================================================

    def load_earlier(self):

        path = filedialog.askopenfilename(
            title="Select Earlier Image",
            filetypes=[
                (
                    "Image files",
                    "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"
                ),
                ("All files", "*.*")
            ]
        )

        if not path:
            return

        image = cv2.imread(path)

        if image is None:
            messagebox.showerror(
                "Error",
                "Could not load the earlier image."
            )
            return

        self.earlier_image = image
        self.earlier_cropped = None
        self.earlier_highlighted = None
        self.earlier_roi = None

        self.display_small_image(
            image,
            self.earlier_label
        )

        self.update_roi_status()
        self.update_highlight_status()

        self.status.config(
            text="Earlier image loaded."
        )

    # =============================================================
    # LOAD LATER
    # =============================================================

    def load_later(self):

        path = filedialog.askopenfilename(
            title="Select Later Image",
            filetypes=[
                (
                    "Image files",
                    "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"
                ),
                ("All files", "*.*")
            ]
        )

        if not path:
            return

        image = cv2.imread(path)

        if image is None:
            messagebox.showerror(
                "Error",
                "Could not load the later image."
            )
            return

        self.later_image = image
        self.later_cropped = None
        self.later_highlighted = None
        self.later_roi = None

        self.display_small_image(
            image,
            self.later_label
        )

        self.update_roi_status()
        self.update_highlight_status()

        self.status.config(
            text="Later image loaded."
        )

    # =============================================================
    # SMALL IMAGE
    # =============================================================

    def display_small_image(
        self,
        image,
        label
    ):

        image = self.to_bgr(image)

        h, w = image.shape[:2]

        scale = min(
            330 / w,
            120 / h,
            1
        )

        nw = max(
            1,
            int(w * scale)
        )

        nh = max(
            1,
            int(h * scale)
        )

        image = cv2.resize(
            image,
            (nw, nh),
            interpolation=cv2.INTER_AREA
        )

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        photo = ImageTk.PhotoImage(
            Image.fromarray(rgb)
        )

        label.config(
            image=photo,
            width=nw,
            height=nh
        )

        label.image = photo

    # =============================================================
    # SCAN INFORMATION
    # =============================================================

    def update_scan_info(self):

        self.earlier_info.config(
            text=(
                "Earlier Scan\n"
                + self.earlier_scan_entry.get()
            )
        )

        self.later_info.config(
            text=(
                "Later Scan\n"
                + self.later_scan_entry.get()
            )
        )

        self.status.config(
            text="Scan information updated."
        )

    # =============================================================
    # CROP
    # =============================================================

    def start_crop(
        self,
        side
    ):

        if side == "earlier":
            image = self.earlier_image
        else:
            image = self.later_image

        if image is None:
            messagebox.showwarning(
                "No Image",
                "Load the image first."
            )
            return

        self.selection_window(
            image,
            side,
            "crop"
        )

    # =============================================================
    # ROI
    # =============================================================

    def start_roi(
        self,
        side
    ):

        if side == "earlier":

            image = self.earlier_cropped

            if image is None:
                image = self.earlier_image

        else:

            image = self.later_cropped

            if image is None:
                image = self.later_image

        if image is None:
            messagebox.showwarning(
                "No Image",
                "Load an image first."
            )
            return

        self.selection_window(
            image,
            side,
            "roi"
        )

    # =============================================================
    # SELECTION WINDOW
    # =============================================================

    def selection_window(
        self,
        image,
        side,
        mode
    ):

        window = tk.Toplevel(
            self.root
        )

        window.title(
            "Select Region"
        )

        window.geometry(
            "850x650"
        )

        tk.Label(
            window,
            text="Drag over the required region and click APPLY.",
            font=("Arial", 11)
        ).pack(
            pady=6
        )

        canvas = tk.Canvas(
            window,
            bg="black",
            cursor="crosshair"
        )

        canvas.pack(
            fill="both",
            expand=True,
            padx=8,
            pady=8
        )

        image = self.to_bgr(image)

        h, w = image.shape[:2]

        scale = min(
            760 / w,
            510 / h,
            1
        )

        dw = max(
            1,
            int(w * scale)
        )

        dh = max(
            1,
            int(h * scale)
        )

        display = cv2.resize(
            image,
            (dw, dh),
            interpolation=cv2.INTER_AREA
        )

        rgb = cv2.cvtColor(
            display,
            cv2.COLOR_BGR2RGB
        )

        photo = ImageTk.PhotoImage(
            Image.fromarray(rgb)
        )

        canvas.create_image(
            0,
            0,
            anchor="nw",
            image=photo
        )

        canvas.image = photo

        selection = {
            "x1": None,
            "y1": None,
            "x2": None,
            "y2": None,
            "rect": None
        }

        def mouse_down(event):

            selection["x1"] = max(
                0,
                min(event.x, dw)
            )

            selection["y1"] = max(
                0,
                min(event.y, dh)
            )

            if selection["rect"]:
                canvas.delete(
                    selection["rect"]
                )

            selection["rect"] = canvas.create_rectangle(
                selection["x1"],
                selection["y1"],
                selection["x1"],
                selection["y1"],
                outline="yellow",
                width=2
            )

        def mouse_drag(event):

            if selection["x1"] is None:
                return

            x = max(
                0,
                min(event.x, dw)
            )

            y = max(
                0,
                min(event.y, dh)
            )

            canvas.coords(
                selection["rect"],
                selection["x1"],
                selection["y1"],
                x,
                y
            )

        def mouse_up(event):

            selection["x2"] = max(
                0,
                min(event.x, dw)
            )

            selection["y2"] = max(
                0,
                min(event.y, dh)
            )

        canvas.bind(
            "<ButtonPress-1>",
            mouse_down
        )

        canvas.bind(
            "<B1-Motion>",
            mouse_drag
        )

        canvas.bind(
            "<ButtonRelease-1>",
            mouse_up
        )

        def apply():

            if (
                selection["x1"] is None
                or selection["x2"] is None
            ):
                messagebox.showwarning(
                    "No Selection",
                    "Select a region first.",
                    parent=window
                )
                return

            x1 = min(
                selection["x1"],
                selection["x2"]
            )

            x2 = max(
                selection["x1"],
                selection["x2"]
            )

            y1 = min(
                selection["y1"],
                selection["y2"]
            )

            y2 = max(
                selection["y1"],
                selection["y2"]
            )

            if (
                x2 - x1 < 5
                or y2 - y1 < 5
            ):
                messagebox.showwarning(
                    "Selection Too Small",
                    "Select a larger region.",
                    parent=window
                )
                return

            ox1 = int(x1 / scale)
            ox2 = int(x2 / scale)
            oy1 = int(y1 / scale)
            oy2 = int(y2 / scale)

            ox1 = max(
                0,
                min(ox1, w - 1)
            )

            ox2 = max(
                1,
                min(ox2, w)
            )

            oy1 = max(
                0,
                min(oy1, h - 1)
            )

            oy2 = max(
                1,
                min(oy2, h)
            )

            if mode == "crop":

                cropped = image[
                    oy1:oy2,
                    ox1:ox2
                ].copy()

                if side == "earlier":

                    self.earlier_cropped = cropped
                    self.earlier_highlighted = None
                    self.earlier_roi = None

                    self.display_small_image(
                        cropped,
                        self.earlier_label
                    )

                else:

                    self.later_cropped = cropped
                    self.later_highlighted = None
                    self.later_roi = None

                    self.display_small_image(
                        cropped,
                        self.later_label
                    )

                self.update_roi_status()
                self.update_highlight_status()

                self.status.config(
                    text=f"{side.capitalize()} image cropped."
                )

            else:

                roi = (
                    ox1,
                    oy1,
                    ox2,
                    oy2
                )

                if side == "earlier":
                    self.earlier_roi = roi
                    self.earlier_highlighted = None
                else:
                    self.later_roi = roi
                    self.later_highlighted = None

                self.update_roi_status()
                self.update_highlight_status()

                self.status.config(
                    text=f"{side.capitalize()} region selected."
                )

            window.destroy()

        tk.Button(
            window,
            text="APPLY",
            command=apply,
            width=18,
            font=("Arial", 10, "bold")
        ).pack(
            pady=6
        )

    # =============================================================
    # ROI STATUS
    # =============================================================

    def update_roi_status(self):

        earlier = (
            "Selected"
            if self.earlier_roi is not None
            else "Not selected"
        )

        later = (
            "Selected"
            if self.later_roi is not None
            else "Not selected"
        )

        self.roi_status.config(
            text=(
                f"Earlier: {earlier} | "
                f"Later: {later}"
            )
        )

    # =============================================================
    # HIGHLIGHT STATUS
    # =============================================================

    def update_highlight_status(self):

        if (
            self.earlier_highlighted is not None
            and self.later_highlighted is not None
        ):

            self.highlighted_status.config(
                text="✓ Selected-region images ready.",
                fg="#087f23"
            )

        else:

            self.highlighted_status.config(
                text="Highlighted regions not created yet.",
                fg="#555555"
            )

    # =============================================================
    # CREATE HIGHLIGHTS
    # =============================================================

    def create_highlights(self):

        if self.earlier_roi is None:
            messagebox.showwarning(
                "Missing Earlier Region",
                "Click SELECT EARLIER REGION and select the region first."
            )
            return

        if self.later_roi is None:
            messagebox.showwarning(
                "Missing Later Region",
                "Click SELECT LATER REGION and select the region first."
            )
            return

        earlier = self.earlier_cropped

        if earlier is None:
            earlier = self.earlier_image

        later = self.later_cropped

        if later is None:
            later = self.later_image

        if earlier is None or later is None:
            return

        self.earlier_highlighted = self.extract_and_highlight_roi(
            earlier,
            self.earlier_roi
        )

        self.later_highlighted = self.extract_and_highlight_roi(
            later,
            self.later_roi
        )

        self.update_highlight_status()

        # CHANGED PART 5: Report the selected highlight mode and ROI-only workflow.
        mode_name = self.highlight_mode.get()
        self.status.config(
            text=f"Selected regions highlighted using {mode_name} mode."
        )

        messagebox.showinfo(
            "Highlight Created",
            f"The selected regions are highlighted using {mode_name} mode.\n\n"
            "The rest of each scan remains visible in grayscale."
        )

    # =============================================================
    # CHANGED PART 6: CALMER COLOUR MAP + GRAYSCALE HIGHLIGHT
    # =============================================================

    def create_pseudocolour_map(
        self,
        gray,
        foreground_mask
    ):

        # Viridis is perceptually uniform and calmer than the previous
        # rainbow-style red/orange/yellow/green map.
        normalized = cv2.normalize(
            gray,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)

        result = cv2.cvtColor(
            gray,
            cv2.COLOR_GRAY2BGR
        )

        colour = cv2.applyColorMap(
            normalized,
            cv2.COLORMAP_VIRIDIS
        )

        result[foreground_mask] = colour[foreground_mask]

        return result

    def create_grayscale_highlight_map(
        self,
        gray,
        foreground_mask
    ):

        # In Grayscale mode the selected ROI remains grayscale but is
        # contrast-normalized so the selected region is still visually
        # highlighted without introducing colour.
        normalized = cv2.normalize(
            gray,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)

        result = cv2.cvtColor(
            gray,
            cv2.COLOR_GRAY2BGR
        )

        enhanced = cv2.cvtColor(
            normalized,
            cv2.COLOR_GRAY2BGR
        )

        result[foreground_mask] = enhanced[foreground_mask]

        return result

    # =============================================================
    # EXTRACT + HIGHLIGHT ROI
    # =============================================================

    def extract_and_highlight_roi(
        self,
        image,
        roi
    ):

        # CHANGED PART 10: ROI-ONLY COMPARISON
        # The selected region is now extracted as the complete working image.
        # The rest of the scan is intentionally not included in comparison views.

        image = self.to_bgr(
            image
        )

        x1, y1, x2, y2 = roi

        h, w = image.shape[:2]

        x1 = max(
            0,
            min(x1, w - 1)
        )

        x2 = max(
            1,
            min(x2, w)
        )

        y1 = max(
            0,
            min(y1, h - 1)
        )

        y2 = max(
            1,
            min(y2, h)
        )

        if x2 <= x1 or y2 <= y1:
            return None

        # CHANGED PART 10: Extract ONLY the selected region.
        # This cropped ROI becomes the image used for comparison.
        region = image[
            y1:y2,
            x1:x2
        ].copy()

        gray_region = cv2.cvtColor(
            region,
            cv2.COLOR_BGR2GRAY
        )

        foreground_mask = np.ones_like(
            gray_region,
            dtype=bool
        )

        if self.highlight_mode.get() == "Colour":
            highlighted_roi = self.create_pseudocolour_map(
                gray_region,
                foreground_mask
            )
        else:
            highlighted_roi = self.create_grayscale_highlight_map(
                gray_region,
                foreground_mask
            )

        # CHANGED PART 10: Return only the selected/highlighted ROI.
        return highlighted_roi

    # =============================================================
    # CHANGED PART 7: UPDATE HIGHLIGHT WHEN MODE IS TOGGLED
    # =============================================================

    def highlight_mode_changed(self):

        # If highlighted images already exist, regenerate them immediately
        # so all comparison views use the newly selected mode.
        if self.earlier_roi is not None and self.later_roi is not None:
            earlier = self.earlier_cropped
            if earlier is None:
                earlier = self.earlier_image

            later = self.later_cropped
            if later is None:
                later = self.later_image

            if earlier is not None and later is not None:
                self.earlier_highlighted = self.extract_and_highlight_roi(
                    earlier, self.earlier_roi
                )
                self.later_highlighted = self.extract_and_highlight_roi(
                    later, self.later_roi
                )
                self.update_highlight_status()

                # Refresh the currently displayed highlighted comparison.
                if self.current_comparison_source == "Highlighted Images" and self.current_comparison_type:
                    self.run_comparison(
                        "Highlighted Images",
                        self.current_comparison_type
                    )
                else:
                    self.status.config(
                        text=f"Highlight mode set to {self.highlight_mode.get()}."
                    )
        else:
            self.status.config(
                text=f"Highlight mode set to {self.highlight_mode.get()}."
            )

    # =============================================================
    # SOURCE CHANGE
    # =============================================================

    def source_changed(self):

        source = self.source_var.get()

        if source == "Highlighted Images":

            if (
                self.earlier_highlighted is None
                or self.later_highlighted is None
            ):

                messagebox.showwarning(
                    "Highlighted Images Not Ready",
                    "Select both regions and click "
                    "COLOUR HIGHLIGHT first."
                )

                self.source_var.set(
                    "Original Images"
                )

                return

            self.status.config(
                text="Highlighted Images selected."
            )

        else:

            self.status.config(
                text="Original Images selected."
            )

    # =============================================================
    # GET COMPARISON IMAGES
    # =============================================================

    def get_images(
        self,
        source
    ):

        # CHANGED PART 10: Use ONLY the selected ROI for comparison.
        # Highlighted Images use the extracted/highlighted ROI.
        # Original Images also use the original extracted ROI, so the
        # comparison workspace never shows the unselected part of the scan
        # once regions have been selected.

        if source == "Highlighted Images":

            if (
                self.earlier_highlighted is None
                or self.later_highlighted is None
            ):

                messagebox.showwarning(
                    "Highlighted Images Not Ready",
                    "Select both regions and click "
                    "COLOUR HIGHLIGHT first."
                )

                return None, None

            return (
                self.earlier_highlighted.copy(),
                self.later_highlighted.copy()
            )

        earlier = self.earlier_cropped

        if earlier is None:
            earlier = self.earlier_image

        later = self.later_cropped

        if later is None:
            later = self.later_image

        if earlier is None or later is None:

            messagebox.showwarning(
                "Images Missing",
                "Load both images first."
            )

            return None, None

        # CHANGED PART 10: If both ROIs are selected, extract those regions
        # and compare only those regions in the Original Images mode.
        if self.earlier_roi is not None and self.later_roi is not None:

            earlier_roi = self.extract_original_roi(
                earlier,
                self.earlier_roi
            )

            later_roi = self.extract_original_roi(
                later,
                self.later_roi
            )

            if earlier_roi is not None and later_roi is not None:
                return (
                    earlier_roi,
                    later_roi
                )

        return (
            earlier.copy(),
            later.copy()
        )

    # =============================================================
    # CHANGED PART 10: EXTRACT ORIGINAL ROI ONLY
    # =============================================================

    def extract_original_roi(
        self,
        image,
        roi
    ):

        image = self.to_bgr(
            image
        )

        x1, y1, x2, y2 = roi

        h, w = image.shape[:2]

        x1 = max(0, min(x1, w - 1))
        x2 = max(1, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(1, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            return None

        return image[
            y1:y2,
            x1:x2
        ].copy()

    # =============================================================
    # RUN COMPARISON
    # =============================================================

    def run_comparison(
        self,
        source,
        mode
    ):

        earlier, later = self.get_images(
            source
        )

        if earlier is None or later is None:
            return

        self.source_var.set(
            source
        )

        self.current_comparison_source = source
        self.current_comparison_type = mode

        if mode == "Side-by-Side":

            result = self.create_side_by_side(
                earlier,
                later
            )

        elif mode == "Overlay":

            result = self.create_overlay(
                earlier,
                later
            )

        else:

            result = self.create_difference(
                earlier,
                later
            )

        self.current_comparison = result

        title = (
            "ORIGINAL"
            if source == "Original Images"
            else "HIGHLIGHTED SELECTED REGIONS"
        )

        self.current_label.config(
            text=f"{title} → {mode}"
        )

        # CHANGED PART 8: Show the difference-view noise/misalignment note only for Difference.
        if mode == "Difference":
            self.difference_note.config(
                text="Differences may be caused by misalignment or scan quality, not only real change."
            )
        else:
            self.difference_note.config(text="")

        self.display_comparison(
            result
        )

        self.status.config(
            text=f"{title} {mode} comparison displayed."
        )

    # =============================================================
    # SIDE-BY-SIDE
    # =============================================================

    def create_side_by_side(
        self,
        earlier,
        later
    ):

        earlier = self.to_bgr(
            earlier
        )

        later = self.to_bgr(
            later
        )

        target_h = max(
            earlier.shape[0],
            later.shape[0]
        )

        earlier = self.resize_keep_ratio(
            earlier,
            target_h
        )

        later = self.resize_keep_ratio(
            later,
            target_h
        )

        label_h = 45

        total_w = (
            earlier.shape[1]
            + later.shape[1]
        )

        canvas = np.ones(
            (
                target_h + label_h,
                total_w,
                3
            ),
            dtype=np.uint8
        ) * 30

        canvas[
            label_h:,
            :earlier.shape[1]
        ] = earlier

        canvas[
            label_h:,
            earlier.shape[1]:
        ] = later

        cv2.putText(
            canvas,
            "EARLIER",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            canvas,
            "LATER",
            (
                earlier.shape[1] + 15,
                30
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        return canvas

    # =============================================================
    # OVERLAY
    # =============================================================

    def create_overlay(
        self,
        earlier,
        later
    ):

        earlier = self.to_bgr(
            earlier
        )

        later = self.to_bgr(
            later
        )

        earlier, later = self.match_size(
            earlier,
            later
        )

        later = self.shift_image(
            later,
            self.shift_x,
            self.shift_y
        )

        alpha = (
            self.transparency.get()
            / 100.0
        )

        beta = 1.0 - alpha

        return cv2.addWeighted(
            earlier,
            beta,
            later,
            alpha,
            0
        )

    # =============================================================
    # DIFFERENCE
    # =============================================================

    def create_difference(
        self,
        earlier,
        later
    ):

        earlier = self.to_gray(
            earlier
        )

        later = self.to_gray(
            later
        )

        earlier, later = self.match_size(
            earlier,
            later
        )

        later = self.shift_image(
            later,
            self.shift_x,
            self.shift_y
        )

        difference = cv2.absdiff(
            earlier,
            later
        )

        difference = cv2.normalize(
            difference,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        )

        return cv2.cvtColor(
            difference,
            cv2.COLOR_GRAY2BGR
        )

    # =============================================================
    # IMAGE HELPERS
    # =============================================================

    def to_bgr(
        self,
        image
    ):

        if len(image.shape) == 2:

            return cv2.cvtColor(
                image,
                cv2.COLOR_GRAY2BGR
            )

        return image.copy()

    def to_gray(
        self,
        image
    ):

        if len(image.shape) == 2:

            return image.copy()

        return cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

    def match_size(
        self,
        first,
        second
    ):

        h = min(
            first.shape[0],
            second.shape[0]
        )

        w = min(
            first.shape[1],
            second.shape[1]
        )

        first = cv2.resize(
            first,
            (w, h),
            interpolation=cv2.INTER_AREA
        )

        second = cv2.resize(
            second,
            (w, h),
            interpolation=cv2.INTER_AREA
        )

        return first, second

    def resize_keep_ratio(
        self,
        image,
        target_h
    ):

        h, w = image.shape[:2]

        scale = target_h / h

        new_w = max(
            1,
            int(w * scale)
        )

        return cv2.resize(
            image,
            (new_w, target_h),
            interpolation=cv2.INTER_AREA
        )

    def shift_image(
        self,
        image,
        dx,
        dy
    ):

        h, w = image.shape[:2]

        matrix = np.float32([
            [1, 0, dx],
            [0, 1, dy]
        ])

        return cv2.warpAffine(
            image,
            matrix,
            (w, h),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

    # =============================================================
    # ALIGNMENT
    # =============================================================

    def align(
        self,
        dx,
        dy
    ):

        if self.current_comparison_source not in [
            "Original Images",
            "Highlighted Images"
        ]:

            self.status.config(
                text="Select a comparison first."
            )

            return

        if self.current_comparison_type not in [
            "Overlay",
            "Difference"
        ]:

            self.status.config(
                text=(
                    "Alignment is available for "
                    "Overlay and Difference comparisons."
                )
            )

            return

        self.shift_x += dx
        self.shift_y += dy

        self.run_comparison(
            self.current_comparison_source,
            self.current_comparison_type
        )

    # =============================================================
    # RESET ALIGNMENT
    # =============================================================

    def reset_alignment(self):

        self.shift_x = 0
        self.shift_y = 0

        if (
            self.current_comparison_source
            and self.current_comparison_type
        ):

            self.run_comparison(
                self.current_comparison_source,
                self.current_comparison_type
            )

        self.status.config(
            text="Alignment reset."
        )

    # =============================================================
    # OVERLAY TRANSPARENCY
    # =============================================================

    def update_overlay(
        self,
        value
    ):

        if (
            self.current_comparison_source
            and self.current_comparison_type
            == "Overlay"
        ):

            self.run_comparison(
                self.current_comparison_source,
                "Overlay"
            )

    # =============================================================
    # ZOOM
    # =============================================================

    def zoom_in(self):

        if self.current_comparison is None:

            self.status.config(
                text="Create a comparison first."
            )

            return

        self.zoom_level *= 1.15

        self.display_comparison(
            self.current_comparison,
            True
        )

        source_name = (
            "Original"
            if self.current_comparison_source
            == "Original Images"
            else "Highlighted"
        )

        self.status.config(
            text=f"{source_name} comparison zoomed in."
        )

    def zoom_out(self):

        if self.current_comparison is None:

            self.status.config(
                text="Create a comparison first."
            )

            return

        self.zoom_level /= 1.15

        if self.zoom_level < 0.25:
            self.zoom_level = 0.25

        self.display_comparison(
            self.current_comparison,
            True
        )

        source_name = (
            "Original"
            if self.current_comparison_source
            == "Original Images"
            else "Highlighted"
        )

        self.status.config(
            text=f"{source_name} comparison zoomed out."
        )

    # =============================================================
    # PAN
    # =============================================================

    def pan(
        self,
        dx,
        dy
    ):

        if self.current_comparison_source not in [
            "Original Images",
            "Highlighted Images"
        ]:

            self.status.config(
                text="Create a comparison first."
            )

            return

        if self.current_comparison_type in [
            "Overlay",
            "Difference"
        ]:

            self.align(
                dx,
                dy
            )

        else:

            self.status.config(
                text=(
                    "Pan/alignment is mainly used "
                    "with Overlay and Difference."
                )
            )

    # =============================================================
    # DISPLAY
    # =============================================================

    def display_comparison(
        self,
        image,
        apply_zoom=False
    ):

        display = image.copy()

        if apply_zoom:

            h, w = display.shape[:2]

            nw = max(
                1,
                int(w * self.zoom_level)
            )

            nh = max(
                1,
                int(h * self.zoom_level)
            )

            display = cv2.resize(
                display,
                (nw, nh),
                interpolation=cv2.INTER_LINEAR
            )

        canvas_w = self.comparison_canvas.winfo_width()
        canvas_h = self.comparison_canvas.winfo_height()

        if canvas_w < 100:
            canvas_w = 700

        if canvas_h < 100:
            canvas_h = 450

        h, w = display.shape[:2]

        scale = min(
            canvas_w / w,
            canvas_h / h,
            1
        )

        if scale < 1:

            nw = max(
                1,
                int(w * scale)
            )

            nh = max(
                1,
                int(h * scale)
            )

            display = cv2.resize(
                display,
                (nw, nh),
                interpolation=cv2.INTER_AREA
            )

        display = self.to_bgr(
            display
        )

        rgb = cv2.cvtColor(
            display,
            cv2.COLOR_BGR2RGB
        )

        photo = ImageTk.PhotoImage(
            Image.fromarray(rgb)
        )

        self.comparison_canvas.delete(
            "all"
        )

        self.comparison_canvas.create_image(
            canvas_w // 2,
            canvas_h // 2,
            anchor="center",
            image=photo
        )

        self.comparison_photo = photo

    # =============================================================
    # SAVE
    # =============================================================

    def save_specific_comparison(
        self,
        source
    ):

        if (
            self.current_comparison is None
            or self.current_comparison_source != source
        ):

            if source == "Highlighted Images":

                messagebox.showwarning(
                    "Nothing to Save",
                    "First click one of the HIGHLIGHTED "
                    "comparison buttons."
                )

            else:

                messagebox.showwarning(
                    "Nothing to Save",
                    "First click one of the ORIGINAL "
                    "comparison buttons."
                )

            return

        mode = self.current_comparison_type

        if source == "Highlighted Images":

            default_name = (
                "highlighted_"
                + mode.lower().replace(
                    "-",
                    "_"
                )
                + ".png"
            )

            title = "Save Highlighted Comparison"

        else:

            default_name = (
                "original_"
                + mode.lower().replace(
                    "-",
                    "_"
                )
                + ".png"
            )

            title = "Save Original Comparison"

        path = filedialog.asksaveasfilename(
            title=title,
            initialfile=default_name,
            defaultextension=".png",
            filetypes=[
                ("PNG Image", "*.png"),
                ("JPEG Image", "*.jpg")
            ]
        )

        if not path:
            return

        success = cv2.imwrite(
            path,
            self.current_comparison
        )

        if success:

            messagebox.showinfo(
                "Saved",
                f"Comparison saved successfully:\n\n{path}"
            )

            self.status.config(
                text=(
                    f"{source} {mode} saved successfully."
                )
            )

        else:

            messagebox.showerror(
                "Save Error",
                "Could not save the comparison."
            )

    # =============================================================
    # RESET VIEW
    # =============================================================

    def reset_view(self):

        self.zoom_level = 1.0
        self.shift_x = 0
        self.shift_y = 0

        if (
            self.current_comparison_source
            and self.current_comparison_type
        ):

            self.run_comparison(
                self.current_comparison_source,
                self.current_comparison_type
            )

        self.status.config(
            text="View reset."
        )


# =============================================================
# RUN APPLICATION
# =============================================================

if __name__ == "__main__":

    root = tk.Tk()

    app = MedicalImageComparisonApp(
        root
    )

    root.mainloop()
