import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import datetime
import pandas as pd
import numpy as np
import math, os, threading, time, sys

# ---------- Optional libraries ----------
# ttkbootstrap: modern themes & widgets (highly recommended)
try:
    import ttkbootstrap as tb
    from ttkbootstrap.constants import *
    USE_TTB = True
except Exception:
    USE_TTB = False
    tb = None

# Matplotlib for charts (optional)
try:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False

# ---------- Helpers ----------
def print_status(msg):
    print(f"[SmartFit] {msg}")

print_status("Starting SmartFit...")
if USE_TTB:
    print_status("Using ttkbootstrap for modern styling.")
else:
    print_status("ttkbootstrap unavailable — falling back to default ttk.")

if not MATPLOTLIB_AVAILABLE:
    print_status("Matplotlib unavailable — charts disabled.")


# -------------------------
# Compact Food DB
# -------------------------
FOOD_DB = [
    {"name": "Oats (1 cup cooked)", "cal": 150, "protein": 5, "carbs": 27, "fat": 3, "serving": "1 cup"},
    {"name": "Egg (large)", "cal": 78, "protein": 6, "carbs": 0.6, "fat": 5, "serving": "1 egg"},
    {"name": "Greek Yogurt (200g)", "cal": 120, "protein": 20, "carbs": 6, "fat": 0, "serving": "200 g"},
    {"name": "Chicken Breast (100g)", "cal": 165, "protein": 31, "carbs": 0, "fat": 3.6, "serving": "100 g"},
    {"name": "Brown Rice (1 cup cooked)", "cal": 215, "protein": 5, "carbs": 45, "fat": 1.8, "serving": "1 cup"},
    {"name": "Broccoli (1 cup)", "cal": 55, "protein": 3.7, "carbs": 11.2, "fat": 0.6, "serving": "1 cup"},
    {"name": "Salmon (100g)", "cal": 208, "protein": 20, "carbs": 0, "fat": 13, "serving": "100 g"},
    {"name": "Almonds (28g)", "cal": 164, "protein": 6, "carbs": 6, "fat": 14, "serving": "28 g"},
    {"name": "Apple (medium)", "cal": 95, "protein": 0.5, "carbs": 25, "fat": 0.3, "serving": "1 medium"},
    {"name": "Peanut Butter (2 tbsp)", "cal": 188, "protein": 8, "carbs": 7, "fat": 16, "serving": "2 tbsp"},
    {"name": "Whole Wheat Bread (1 slice)", "cal": 70, "protein": 3.6, "carbs": 12, "fat": 1, "serving": "1 slice"},
    {"name": "Banana (medium)", "cal": 105, "protein": 1.3, "carbs": 27, "fat": 0.3, "serving": "1 medium"},
    {"name": "Tofu (100g)", "cal": 76, "protein": 8, "carbs": 1.9, "fat": 4.8, "serving": "100 g"},
    {"name": "Olive Oil (1 tbsp)", "cal": 119, "protein": 0, "carbs": 0, "fat": 13.5, "serving": "1 tbsp"},
    {"name": "Quinoa (1 cup cooked)", "cal": 222, "protein": 8, "carbs": 39, "fat": 3.6, "serving": "1 cup"},
]
FOOD_DF = pd.DataFrame(FOOD_DB)


# -------------------------
# Activity & Goals
# -------------------------
ACTIVITY_FACTORS = {
    "Sedentary (little or no exercise)": 1.2,
    "Light (1-3 days/week)": 1.375,
    "Moderate (3-5 days/week)": 1.55,
    "Active (6-7 days/week)": 1.725,
    "Very active (hard exercise & physical work)": 1.9
}
GOAL_ADJUSTMENT = {
    "Lose weight (cut 20%)": 0.80,
    "Maintain weight": 1.00,
    "Gain weight (bulk 15%)": 1.15
}


# -------------------------
# Core Logic
# -------------------------
def calculate_bmr(sex: str, weight_kg: float, height_cm: float, age: int) -> float:
    if str(sex).lower().startswith('m'):
        return 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    return 10 * weight_kg + 6.25 * height_cm - 5 * age - 161

def calculate_tdee_and_targets(sex, weight_kg, height_cm, age, activity_level, goal):
    bmr = calculate_bmr(sex, weight_kg, height_cm, age)
    activity_factor = ACTIVITY_FACTORS.get(activity_level, 1.375)
    tdee = bmr * activity_factor
    target_calories = tdee * GOAL_ADJUSTMENT.get(goal, 1.0)
    protein_g = round(2.0 * weight_kg) if "Lose" in goal else round(1.8 * weight_kg) if "Gain" in goal else round(1.6 * weight_kg)
    fat_cals = 0.25 * target_calories
    fat_g = round(fat_cals / 9)
    protein_cals = protein_g * 4
    remaining_cals = max(0, target_calories - (protein_cals + fat_cals))
    carbs_g = round(remaining_cals / 4)
    return {
        "BMR": round(bmr),
        "TDEE": round(tdee),
        "TargetCalories": round(target_calories),
        "Protein_g": int(protein_g),
        "Carbs_g": int(carbs_g),
        "Fat_g": int(fat_g)
    }

def generate_meal_plan(target_calories, protein_g, carbs_g, fat_g, meals=("Breakfast","Lunch","Dinner","Snack")):
    splits = {"Breakfast":0.25, "Lunch":0.35, "Dinner":0.30, "Snack":0.10}
    available = [m for m in meals if m]
    total_split = sum(splits[m] for m in available if m in splits)
    rows = []
    foods = FOOD_DF.copy()
    for meal in available:
        split = splits.get(meal, 0.15)
        meal_target = target_calories * (split / total_split)
        items = []; cal = prot = carb = fat = 0.0
        cand = foods.sort_values(by="cal") if meal == "Snack" else foods.assign(pdensity=foods["protein"]/(foods["cal"]+1e-6)).sort_values(by="pdensity", ascending=False)
        i = 0
        while cal < meal_target * 0.95 and i < len(cand):
            f = cand.iloc[i].to_dict()
            items.append(f)
            cal += f["cal"]; prot += f["protein"]; carb += f["carbs"]; fat += f["fat"]
            i += 1
        if cal < meal_target * 0.95:
            big = foods.sort_values(by="cal", ascending=False).iloc[0].to_dict()
            items.append(big); cal += big["cal"]; prot += big["protein"]; carb += big["carbs"]; fat += big["fat"]
        rows.append({"Meal": meal, "Items": items, "Calories": round(cal), "Protein_g": round(prot, 1), "Carbs_g": round(carb, 1), "Fat_g": round(fat, 1)})
    return pd.DataFrame(rows)

WORKOUT_TEMPLATES = {
    "beginner": [
        ("Day 1 - Full Body", ["Squats 3x8", "Push-ups 3x8", "Dumbbell Rows 3x8", "Plank 30s"]),
        ("Day 2 - Walk", ["30 min brisk walk"]),
        ("Day 3 - Full Body", ["Lunges 3x10", "Overhead Press 3x8"]),
        ("Day 4 - Recovery", ["Yoga / Mobility 20-30 min"]),
        ("Day 5 - Full Body", ["Goblet Squat 3x10", "Incline Push-ups 3x10"]),
        ("Day 6 - Cardio", ["20-30 min intervals"]),
        ("Day 7 - Rest", ["Rest"])
    ],
    "intermediate": [
        ("Day 1 - Upper Push", ["Bench Press 4x6-8", "Incline DB 3x8"]),
        ("Day 2 - Lower", ["Back Squat 4x6-8", "Deadlift 3x5"]),
        ("Day 3 - Pull/Core", ["Pull-ups 4x6", "Barbell Row 4x6"]),
        ("Day 4 - Recovery", ["Mobility"]),
        ("Day 5 - Push Hypertrophy", ["DB Press 4x10"]),
        ("Day 6 - Lower Hypertrophy", ["Lunges 3x12"]),
        ("Day 7 - Rest", ["Light walk"])
    ],
    "advanced": [
        ("Day 1 - Power", ["Power Cleans 5x3", "Box Jumps 4x5"]),
        ("Day 2 - Conditioning", ["HIIT 20 min"]),
        ("Day 3 - Strength", ["Deadlift 5x5", "Front Squat 4x6"]),
        ("Day 4 - Mobility", ["Yoga/Mobility 30 min"]),
        ("Day 5 - Speed/Agility", ["Sprints 8x60m"]),
        ("Day 6 - Mixed Strength", ["Bench 5x5", "Rows 4x6"]),
        ("Day 7 - Active Recovery", ["Light swim / walk"])
    ]
}

def generate_workout_plan(level, goal):
    template = WORKOUT_TEMPLATES.get(level, WORKOUT_TEMPLATES["beginner"])
    adapted = []
    for day, exs in template:
        ex = list(exs)
        if "Lose" in goal and not any(kw in e.lower() for e in ex for kw in ["cardio", "walk"]):
            ex.append("10-15 min cardio finisher")
        if "Gain" in goal:
            ex.append("Progressive overload: increase weight over weeks")
        adapted.append((day, ex))
    return adapted

def ai_diet_suggestions(targets, last_plan_df):
    tips = []
    tcal = targets.get("TargetCalories")
    if tcal and tcal < 1600:
        tips.append("Target calories are low — prioritize protein and nutrient-dense foods.")
    if targets and targets.get("Protein_g", 0) < 1:
        tips.append("Distribute protein across meals (20-30g per meal).")
    if last_plan_df is not None:
        for _, r in last_plan_df.iterrows():
            if any("Peanut Butter" in it["name"] and r["Meal"].lower().startswith("sn") for it in r["Items"]):
                tips.append("Swap some peanut-butter snacks for Greek yogurt + berries.")
                break
    tips.extend([
        "Include colored vegetables for vitamins and fiber.",
        "Stay hydrated (2-3 L/day depending on activity).",
        "For medical conditions, consult a dietitian."
    ])
    return tips


# -------------------------
# UI: Splash + App
# -------------------------

class SplashScreen:
    """3-second gradient teal splash with progress animation"""
    def __init__(self, master, duration_ms=3000, on_finish=None):
        self.on_finish = on_finish
        self.duration = max(1000, int(duration_ms))
        self.win = tk.Toplevel(master)
        self.win.overrideredirect(True)
        sw, sh = self.win.winfo_screenwidth(), self.win.winfo_screenheight()
        size = int(min(sw, sh) * 0.45); h = int(size * 0.45)
        x, y = (sw - size) // 2, (sh - h) // 2
        self.win.geometry(f"{size}x{h}+{x}+{y}")
        self.win.configure(bg="#042227")
        self._build_ui()
        self._drag_data = {"x": 0, "y": 0}
        self.win.bind("<ButtonPress-1>", self._start_drag)
        self.win.bind("<B1-Motion>", self._do_drag)

    def _start_drag(self, ev): self._drag_data.update(x=ev.x, y=ev.y)
    def _do_drag(self, ev):
        nx = self.win.winfo_x() + ev.x - self._drag_data["x"]
        ny = self.win.winfo_y() + ev.y - self._drag_data["y"]
        self.win.geometry(f"+{nx}+{ny}")

    def _build_ui(self):
        frame = tk.Frame(self.win, bg="#052f33", bd=0)
        frame.pack(fill="both", expand=True, padx=12, pady=12)
        glow = tk.Label(frame, text="SmartFit", font=("Helvetica", 34, "bold"), fg="#64fff4", bg="#052f33")
        glow.place(relx=0.5, rely=0.22, anchor="n")
        main = tk.Label(frame, text="SmartFit", font=("Helvetica", 34, "bold"), fg="#00CFE6", bg="#052f33")
        main.place(relx=0.5, rely=0.18, anchor="n")
        tag = tk.Label(frame, text="Your AI fitness companion", font=("Helvetica", 11), fg="#dffaff", bg="#052f33")
        tag.place(relx=0.5, rely=0.52, anchor="n")
        self.progress = ttk.Progressbar(frame, orient="horizontal", mode="determinate", length=400)
        self.progress.place(relx=0.5, rely=0.72, anchor="n")
        self.status = tk.Label(frame, text="Loading SmartFit...", font=("Helvetica", 10), fg="#bff3f0", bg="#052f33")
        self.status.place(relx=0.5, rely=0.86, anchor="n")

    def start(self):
        steps = 100; delay = max(5, int(self.duration / steps))
        def step(i=0):
            if i > steps:
                self.win.after(200, self.finish); return
            frac = i / steps
            self.progress["value"] = frac * 100
            self.status.config(text=[
                "Loading SmartFit...", "Preparing dashboard...", "Generating personalized plans...", "Finalizing..."
            ][min(int(frac * 4), 3)])
            self.win.update_idletasks()
            self.win.after(delay, lambda: step(i + 1))
        step(0)

    def finish(self):
        try: self.win.destroy()
        except: pass
        if callable(self.on_finish): self.on_finish()


class SmartFitApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SmartFit")
        self.root.geometry("1400x900")
        try: self.root.state("zoomed")
        except: pass
        self.last_plan_df = self.last_targets = None

        # Professional, modern color palettes (default: dark)
        self.PALETTES = {
            "dark": {
                "bg": "#0f172a",          # Deep slate blue-gray
                "card": "#1e293b",        # Elevated card
                "card_hover": "#334155",
                "text": "#f1f5f9",        # Clean white
                "muted": "#94a3b8",       # Soft gray
                "accent": "#22d3ee",      # Vibrant cyan (primary)
                "accent2": "#06b6d4",     # Slightly deeper cyan
                "success": "#34d399",      # Fresh green
                "warning": "#fbbf24",     # Warm amber
                "border": "#334155",
                "input_bg": "#334155",
                "input_fg": "#f8fafc"
            },
            "light": {
                "bg": "#f1f5f9",          # Soft light gray-blue
                "card": "#ffffff",        # Crisp white cards
                "card_hover": "#e2e8f0",
                "text": "#1e293b",        # Deep slate
                "muted": "#64748b",       # Muted blue-gray
                "accent": "#0891b2",      # Rich cyan (primary)
                "accent2": "#0e7490",     # Deeper cyan
                "success": "#10b981",      # Emerald green
                "warning": "#f59e0b",     # Amber
                "border": "#cbd5e1",
                "input_bg": "#ffffff",
                "input_fg": "#1e293b"
            }
        }
        self.palette_key = "dark"  # Default to dark
        self.palette = self.PALETTES[self.palette_key]

        self._setup_style()
        self._build_ui()
        self._apply_palette()

    def _setup_style(self):
        if USE_TTB:
            base_theme = "cyborg" if self.palette_key == "dark" else "cosmo"
            self.style = tb.Style(theme=base_theme)
            # Custom card style with subtle border and rounded corners
            self.style.configure("Card.TFrame", background=self.palette["card"], relief="flat", borderwidth=1)
            self.style.configure("Card.TLabelframe", background=self.palette["card"], foreground=self.palette["accent"], borderwidth=1)
            self.style.configure("Card.TLabelframe.Label", font=("Segoe UI", 13, "bold"), foreground=self.palette["accent"])
            self.style.configure("TEntry", fieldbackground=self.palette["input_bg"], foreground=self.palette["input_fg"])
            self.style.configure("TCombobox", fieldbackground=self.palette["input_bg"], foreground=self.palette["input_fg"])
        else:
            self.style = ttk.Style()

    def _build_ui(self):
        self.container = tk.Frame(self.root, bg=self.palette["bg"])
        self.container.pack(fill="both", expand=True, padx=16, pady=16)

        # Sidebar
        self.sidebar = ttk.Frame(self.container, style="Card.TFrame", width=380)
        self.sidebar.pack(side="left", fill="y", padx=(0, 12))
        self.sidebar.pack_propagate(False)
        self._build_sidebar()

        # Main content
        self.main_area = tk.Frame(self.container, bg=self.palette["bg"])
        self.main_area.pack(side="left", fill="both", expand=True)

        self.canvas = tk.Canvas(self.main_area, bg=self.palette["bg"], highlightthickness=0)
        self.vsb = ttk.Scrollbar(self.main_area, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vsb.set)
        self.vsb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.content_frame = tk.Frame(self.canvas, bg=self.palette["bg"])
        self.canvas.create_window((0, 0), window=self.content_frame, anchor="nw")
        self.content_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        self._build_content()

    def _on_mousewheel(self, event):
        try: self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        except:
            try:
                self.canvas.yview_scroll(-1 if event.num == 4 else 1, "units")
            except: pass

    def _build_sidebar(self):
        # Logo Card
        logo_card = ttk.Frame(self.sidebar, style="Card.TFrame")
        logo_card.pack(fill="x", pady=(0, 12), padx=12)
        ttk.Label(logo_card, text="SmartFit", font=("Segoe UI", 22, "bold"), foreground=self.palette["accent"]).pack(anchor="w", padx=16, pady=(16, 4))
        ttk.Label(logo_card, text="AI-Powered Nutrition & Training", font=("Segoe UI", 11), foreground=self.palette["muted"]).pack(anchor="w", padx=16, pady=(0, 16))

        # Profile Card
        profile_card = ttk.Labelframe(self.sidebar, text="User Profile", style="Card.TLabelframe")
        profile_card.pack(fill="x", pady=(0, 12), padx=12, ipadx=12, ipady=12)

        grid = ttk.Frame(profile_card)
        grid.pack(fill="x", padx=8, pady=4)

        labels = ["Name", "Age", "Sex", "Experience", "Weight (kg)", "Height (cm)", "Activity Level", "Goal"]
        vars = [tk.StringVar(value="Aditya"), tk.IntVar(value=22), tk.StringVar(value="Male"), tk.StringVar(value="beginner"),
                tk.DoubleVar(value=70.0), tk.DoubleVar(value=172.0), tk.StringVar(value=list(ACTIVITY_FACTORS.keys())[2]),
                tk.StringVar(value=list(GOAL_ADJUSTMENT.keys())[1])]
        self.vars = dict(zip(["name", "age", "sex", "exp", "weight", "height", "activity", "goal"], vars))

        row = 0
        for label, key in zip(labels, self.vars.keys()):
            ttk.Label(grid, text=label + ":", foreground=self.palette["muted"], width=16, anchor="w").grid(row=row, column=0, sticky="w", pady=3, padx=(0, 8))
            if "Sex" in label or "Experience" in label or "Activity" in label or "Goal" in label:
                values = ["Male","Female"] if "Sex" in label else ["beginner","intermediate","advanced"] if "Experience" in label else list(ACTIVITY_FACTORS.keys()) if "Activity" in label else list(GOAL_ADJUSTMENT.keys())
                widget = ttk.Combobox(grid, textvariable=self.vars[key], values=values, state="readonly", width=20)
            else:
                widget = ttk.Entry(grid, textvariable=self.vars[key], width=22)
            widget.grid(row=row, column=1, sticky="ew", pady=3)
            row += 1
        grid.columnconfigure(1, weight=1)

        # Action Buttons Card
        btn_card = ttk.Frame(self.sidebar, style="Card.TFrame")
        btn_card.pack(fill="x", pady=(0, 12), padx=12)

        self.gen_nut_btn = ttk.Button(btn_card, text="Generate Nutrition Plan", style="success.TButton", command=self.on_generate)
        self.gen_nut_btn.pack(fill="x", pady=(12, 6), padx=16)

        self.gen_wkt_btn = ttk.Button(btn_card, text="Generate Workout Plan", style="info.TButton", command=self.on_generate_workout)
        self.gen_wkt_btn.pack(fill="x", pady=(0, 6), padx=16)

        self.export_btn = ttk.Button(btn_card, text="Export Plan to CSV", style="warning.TButton", command=self.export_plan)
        self.export_btn.pack(fill="x", pady=(0, 12), padx=16)

        # Stats Card
        stats_card = ttk.Labelframe(self.sidebar, text="Quick Stats", style="Card.TLabelframe")
        stats_card.pack(fill="x", pady=(0, 12), padx=12, ipadx=12, ipady=8)

        self.stat_labels = {}
        for key, text in [("BMR", "BMR"), ("TDEE", "TDEE"), ("TargetCalories", "Target Calories")]:
            frame = ttk.Frame(stats_card)
            frame.pack(fill="x", padx=12, pady=4)
            ttk.Label(frame, text=text + ":", foreground=self.palette["muted"], width=14).pack(side="left")
            self.stat_labels[key] = ttk.Label(frame, text="—", font=("Segoe UI", 10, "bold"), foreground=self.palette["accent"])
            self.stat_labels[key].pack(side="right")

        # Theme Toggle (with icon-like style)
        theme_frame = ttk.Frame(self.sidebar)
        theme_frame.pack(fill="x", pady=(8, 0), padx=12)
        self.theme_var = tk.BooleanVar(value=False)  # False = dark
        theme_check = ttk.Checkbutton(
            theme_frame,
            text="Light Theme",
            variable=self.theme_var,
            command=self._toggle_theme,
            style="Roundtoggle.Toolbutton"
        )
        theme_check.pack(side="left")

        # Footer
        footer = ttk.Label(self.sidebar, text="v1.5 • Guidelines only", font=("Segoe UI", 8), foreground=self.palette["muted"])
        footer.pack(side="bottom", pady=12)

    def _build_content(self):
        # Summary Card
        summary_card = ttk.Labelframe(self.content_frame, text="Nutrition Plan Summary", style="Card.TLabelframe")
        summary_card.pack(fill="x", pady=(0, 12), padx=12, ipadx=12, ipady=12)
        self.results_text = tk.Text(summary_card, height=5, wrap="word", font=("Segoe UI", 10), relief="flat", bg=self.palette["card"], fg=self.palette["text"], insertbackground=self.palette["text"])
        self.results_text.pack(fill="x", padx=12, pady=(0, 12))

        # Charts Row
        chart_row = ttk.Frame(self.content_frame)
        chart_row.pack(fill="x", pady=(0, 12), padx=12)

        macro_card = ttk.Labelframe(chart_row, text="Macronutrient Split", style="Card.TLabelframe")
        macro_card.pack(side="left", fill="both", expand=True, padx=(0, 6))
        self.macro_canvas = None
        if MATPLOTLIB_AVAILABLE:
            self.fig_macro = Figure(figsize=(5, 4), dpi=100, facecolor=self.palette["card"])
            self.ax_macro = self.fig_macro.add_subplot(111)
            self.macro_canvas = FigureCanvasTkAgg(self.fig_macro, macro_card)
            self.macro_canvas.get_tk_widget().pack(fill="both", expand=True, padx=12, pady=12)
        else:
            ttk.Label(macro_card, text="Charts disabled (matplotlib not found)", foreground=self.palette["muted"]).pack(pady=20)

        meal_card = ttk.Labelframe(chart_row, text="Calories per Meal", style="Card.TLabelframe")
        meal_card.pack(side="right", fill="both", expand=True, padx=(6, 0))
        self.meal_canvas = None
        if MATPLOTLIB_AVAILABLE:
            self.fig_meal = Figure(figsize=(5, 4), dpi=100, facecolor=self.palette["card"])
            self.ax_meal = self.fig_meal.add_subplot(111)
            self.meal_canvas = FigureCanvasTkAgg(self.fig_meal, meal_card)
            self.meal_canvas.get_tk_widget().pack(fill="both", expand=True, padx=12, pady=12)

        # Meal Table Card
        table_card = ttk.Labelframe(self.content_frame, text="Detailed Meal Plan", style="Card.TLabelframe")
        table_card.pack(fill="x", pady=(0, 12), padx=12, ipadx=12, ipady=8)
        cols = ("Meal", "Calories", "Protein_g", "Carbs_g", "Fat_g")
        self.tree = ttk.Treeview(table_card, columns=cols, show="headings", height=6)
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, anchor="center", width=130)
        self.tree.pack(fill="x", padx=12, pady=8)

        # Workout Card
        workout_card = ttk.Labelframe(self.content_frame, text="Weekly Workout Plan", style="Card.TLabelframe")
        workout_card.pack(fill="x", pady=(0, 12), padx=12, ipadx=12, ipady=12)
        self.workout_text = tk.Text(workout_card, height=10, wrap="word", font=("Segoe UI", 10), relief="flat", bg=self.palette["card"], fg=self.palette["text"], insertbackground=self.palette["text"])
        self.workout_text.pack(fill="x", padx=12, pady=(0, 12))

        # AI Suggestions Card
        suggest_card = ttk.Labelframe(self.content_frame, text="AI Diet Suggestions", style="Card.TLabelframe")
        suggest_card.pack(fill="x", pady=(0, 12), padx=12, ipadx=12, ipady=12)
        self.suggestions_text = tk.Text(suggest_card, height=6, wrap="word", font=("Segoe UI", 10), relief="flat", bg=self.palette["card"], fg=self.palette["text"], insertbackground=self.palette["text"])
        self.suggestions_text.pack(fill="x", padx=12, pady=(0, 12))

    def _apply_palette(self):
        p = self.palette
        for widget in [self.root, self.container, self.main_area, self.canvas, self.content_frame]:
            try: widget.configure(bg=p["bg"])
            except: pass
        # Reconfigure text widgets
        for txt in [self.results_text, self.workout_text, self.suggestions_text]:
            txt.configure(bg=p["card"], fg=p["text"], insertbackground=p["text"])
        self._setup_style()  # Reapply style with new palette

    def _toggle_theme(self):
        self.palette_key = "light" if self.theme_var.get() else "dark"
        self.palette = self.PALETTES[self.palette_key]
        self._apply_palette()

    # ---------- Actions ----------
    def _display_plan(self, df, targets):
        self.last_plan_df, self.last_targets = df, targets
        for key in ["BMR", "TDEE", "TargetCalories"]:
            self.stat_labels[key].config(text=targets.get(key, "—"))
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        header = f"Nutrition Plan — {now}\nTarget: {targets['TargetCalories']} kcal | Protein: {targets['Protein_g']}g | Carbs: {targets['Carbs_g']}g | Fat: {targets['Fat_g']}g\n\n"
        self.results_text.delete("1.0", tk.END); self.results_text.insert(tk.END, header)
        for i in self.tree.get_children(): self.tree.delete(i)
        for _, r in df.iterrows():
            self.tree.insert("", tk.END, values=(r["Meal"], r["Calories"], r["Protein_g"], r["Carbs_g"], r["Fat_g"]))
        if MATPLOTLIB_AVAILABLE:
            self._draw_macro_chart(targets)
            self._draw_meal_chart(df)
        tips = ai_diet_suggestions(targets, df)
        self.suggestions_text.delete("1.0", tk.END)
        self.suggestions_text.insert(tk.END, "\n".join(f"• {t}" for t in tips))

    def _draw_macro_chart(self, targets):
        self.fig_macro.clf(); ax = self.fig_macro.add_subplot(111)
        vals = [targets["Protein_g"]*4, targets["Carbs_g"]*4, targets["Fat_g"]*9]
        labels = ["Protein", "Carbs", "Fat"]
        colors = [self.palette["accent"], self.palette["accent2"], self.palette["success"]]
        ax.pie(vals, labels=labels, autopct="%1.1f%%", startangle=90, colors=colors, wedgeprops=dict(edgecolor=self.palette["card"]))
        ax.set_title("Macronutrient Distribution", color=self.palette["text"], fontsize=12)
        self.fig_macro.set_facecolor(self.palette["card"])
        self.macro_canvas.draw()

    def _draw_meal_chart(self, df):
        self.fig_meal.clf(); ax = self.fig_meal.add_subplot(111)
        names, cals = df["Meal"].tolist(), df["Calories"].tolist()
        colors = [self.palette["accent"] if i % 2 == 0 else self.palette["accent2"] for i in range(len(names))]
        bars = ax.bar(names, cals, color=colors)
        ax.set_title("Calories per Meal", color=self.palette["text"], fontsize=12)
        ax.set_ylabel("Calories", color=self.palette["text"])
        for b in bars:
            h = b.get_height()
            ax.annotate(f"{int(h)}", xy=(b.get_x() + b.get_width()/2, h), xytext=(0, 4), textcoords="offset points", ha="center", color=self.palette["muted"])
        self.fig_meal.set_facecolor(self.palette["card"])
        self.meal_canvas.draw()

    def on_generate(self):
        def worker():
            try:
                sex = self.vars["sex"].get(); weight = float(self.vars["weight"].get())
                height = float(self.vars["height"].get()); age = int(self.vars["age"].get())
                activity = self.vars["activity"].get(); goal = self.vars["goal"].get()
            except Exception as e:
                messagebox.showerror("Input Error", f"Invalid input: {e}"); return
            time.sleep(0.5)
            targets = calculate_tdee_and_targets(sex, weight, height, age, activity, goal)
            df = generate_meal_plan(targets["TargetCalories"], targets["Protein_g"], targets["Carbs_g"], targets["Fat_g"])
            self.root.after(0, lambda: self._display_plan(df, targets))
        threading.Thread(target=worker, daemon=True).start()

    def on_generate_workout(self):
        def worker():
            level = self.vars["exp"].get(); goal = self.vars["goal"].get()
            time.sleep(0.3)
            plan = generate_workout_plan(level, goal)
            self.root.after(0, lambda: self._show_workout(plan))
        threading.Thread(target=worker, daemon=True).start()

    def _show_workout(self, plan):
        self.workout_text.delete("1.0", tk.END)
        for day, exs in plan:
            self.workout_text.insert(tk.END, f"{day}:\n")
            for e in exs:
                self.workout_text.insert(tk.END, f"  - {e}\n")
            self.workout_text.insert(tk.END, "\n")

    def export_plan(self):
        if not self.last_plan_df or not self.last_targets:
            messagebox.showinfo("No Plan", "Generate a plan first."); return
        rows = []
        for _, r in self.last_plan_df.iterrows():
            items_str = "; ".join(f"{it['name']} ({it['serving']})" for it in r["Items"])
            rows.append({"Meal": r["Meal"], "Items": items_str, "Calories": r["Calories"], "Protein_g": r["Protein_g"], "Carbs_g": r["Carbs_g"], "Fat_g": r["Fat_g"]})
        export_df = pd.DataFrame(rows)
        fpath = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")], initialfile="smartfit_plan.csv")
        if not fpath: return
        with open(fpath, "w", encoding="utf-8") as fh:
            fh.write("# SmartFit - Personalized Plan\n")
            for k, v in self.last_targets.items():
                fh.write(f"# {k}: {v}\n")
            export_df.to_csv(fh, index=False)
        messagebox.showinfo("Exported", f"Plan saved to {fpath}")


# ---------- Runner ----------
def start_app_with_splash():
    root = tk.Tk()
    root.withdraw()
    def on_splash_finish():
        root.deiconify()
        app = SmartFitApp(root)
        app.on_generate()  # Auto-generate on launch
    splash = SplashScreen(root, duration_ms=3000, on_finish=on_splash_finish)
    splash.start()
    root.mainloop()

if __name__ == "__main__":
    start_app_with_splash()
