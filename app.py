from flask import Flask, render_template_string, request, jsonify
import pandas as pd
import numpy as np
import io, base64, os, pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

# ML imports
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

app = Flask(__name__)

# -------------------------------------------------
# File paths for persisted models / metrics images
# -------------------------------------------------
CAL_REG_PKL = "calorie_predictor.pkl"
WK_REC_PKL  = "workout_recommender.pkl"
CONF_IMG_PNG = "confusion_matrix.png"

# -------------------------
# Data and Logic
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
    {"name": "Banana (medium)", "cal": 105, "protein": 1.3, "carbs": 27, "fat": 0.3, "serving": "1 medium"},
    {"name": "Sweet Potato (medium)", "cal": 112, "protein": 2, "carbs": 26, "fat": 0.1, "serving": "1 medium"},
    {"name": "Peanut Butter (2 tbsp)", "cal": 188, "protein": 8, "carbs": 7, "fat": 16, "serving": "2 tbsp"},
    {"name": "Quinoa (1 cup cooked)", "cal": 222, "protein": 8, "carbs": 39, "fat": 4, "serving": "1 cup"},
    {"name": "Avocado (half)", "cal": 120, "protein": 1.5, "carbs": 6, "fat": 11, "serving": "1/2 avocado"},
]

WORKOUT_TEMPLATES = {
    "beginner": [
        ("Day 1 - Full Body", ["Bodyweight Squats 3x10", "Push-ups 3x8", "Mountain Climbers 3x15", "Plank 30s"]),
        ("Day 2 - Cardio", ["30 min brisk walk or light jog"]),
        ("Day 3 - Core & Flexibility", ["Sit-ups 3x15", "Leg Raises 3x10", "Stretching 15 min"]),
        ("Day 4 - Active Recovery", ["Yoga or light mobility work 20-30 min"]),
        ("Day 5 - Upper Body", ["Shoulder Press 3x10", "Dumbbell Rows 3x10", "Tricep Dips 3x8"]),
        ("Day 6 - Cardio", ["20-30 min moderate intensity intervals"]),
        ("Day 7 - Rest", ["Complete rest or gentle stretching"])
    ],
    "intermediate": [
        ("Day 1 - Upper Push", ["Bench Press 4x8", "Incline Dumbbell Press 3x10", "Overhead Press 3x10", "Lateral Raises 3x12"]),
        ("Day 2 - Lower Body", ["Back Squat 4x8", "Romanian Deadlift 3x10", "Leg Press 3x12", "Calf Raises 4x15"]),
        ("Day 3 - Core & Cardio", ["Weighted Sit-ups 3x20", "Russian Twists 3x30", "Mountain Climbers 3x20", "20 min HIIT"]),
        ("Day 4 - Active Recovery", ["Swimming, cycling, or yoga 30-40 min"]),
        ("Day 5 - Upper Pull", ["Pull-ups 4x6-8", "Barbell Rows 4x8", "Face Pulls 3x15", "Bicep Curls 3x12"]),
        ("Day 6 - Lower Hypertrophy", ["Front Squat 4x10", "Walking Lunges 3x12 each", "Leg Curls 3x12", "Glute Bridges 3x15"]),
        ("Day 7 - Rest", ["Light walk or complete rest"])
    ],
    "advanced": [
        ("Day 1 - Power", ["Clean & Press 5x3", "Box Jumps 4x5", "Power Snatch 5x3", "Medicine Ball Slams 4x10"]),
        ("Day 2 - Conditioning", ["HIIT Sprints 8x200m", "Battle Ropes 5x30s", "Burpees 5x15"]),
        ("Day 3 - Max Strength", ["Deadlift 5x5", "Front Squat 4x6", "Weighted Pull-ups 4x5"]),
        ("Day 4 - Core & Accessory", ["Weighted Sit-ups 4x25", "Hanging Leg Raises 4x12", "Planks 3x60s", "Farmers Walks 4x50m"]),
        ("Day 5 - Speed & Agility", ["Sprint Intervals 10x60m", "Agility Ladder Drills 15 min", "Jump Rope 5x2min"]),
        ("Day 6 - Strength Hypertrophy", ["Bench Press 5x5", "Barbell Rows 5x5", "Overhead Press 4x8", "Weighted Dips 4x8"]),
        ("Day 7 - Active Recovery", ["Light swim, bike ride, or mobility work 30-45 min"])
    ]
}

def calculate_bmr(sex, weight, height, age):
    """Calculate Basal Metabolic Rate using Mifflin-St Jeor Equation"""
    return 10 * weight + 6.25 * height - 5 * age + (5 if str(sex).lower().startswith('m') else -161)

def calculate_targets(sex, weight, height, age, activity, goal):
    """Calculate daily calorie and macro targets"""
    bmr = calculate_bmr(sex, weight, height, age)
    tdee = bmr * activity
    target = tdee * goal
    protein = round(1.8 * weight)
    fat = round((0.25 * target) / 9)
    carbs = round((target - (protein * 4 + fat * 9)) / 4)
    return {
        "bmr": round(bmr),
        "tdee": round(tdee),
        "target_calories": round(target),
        "protein": int(protein),
        "carbs": int(carbs),
        "fat": int(fat)
    }

def generate_meal_plan(target_cal):
    """Generate meal distribution based on target calories"""
    meals = []
    distribution = {"Breakfast": 0.25, "Lunch": 0.35, "Dinner": 0.30, "Snacks": 0.10}
    for meal, split in distribution.items():
        meals.append({"meal": meal, "calories": int(target_cal * split)})
    return meals

def generate_workout(level, goal):
    """Get workout template based on fitness level"""
    return WORKOUT_TEMPLATES.get(level, WORKOUT_TEMPLATES["beginner"])

# -------------------------------------------------
# ML: Synthetic data generation and model training
# -------------------------------------------------
def make_synthetic_calorie_dataset(n=800, random_state=42):
    rng = np.random.default_rng(random_state)
    sex = rng.integers(0, 2, n)  # 0=female,1=male
    age = rng.integers(16, 65, n)
    height = rng.normal(170, 10, n).clip(145, 200)  # cm
    weight = rng.normal(70, 12, n).clip(45, 120)    # kg
    activity = rng.choice([1.2, 1.375, 1.55, 1.725, 1.9], n)
    goal = rng.choice([0.8, 0.9, 1.0, 1.1, 1.2], n)

    # Ground-truth via Mifflin-St Jeor + noise
    bmr = 10*weight + 6.25*height - 5*age + np.where(sex==1, 5, -161)
    tdee = bmr * activity
    target = tdee * goal
    noise = rng.normal(0, 80, n)  # small measurement noise
    y = (target + noise).clip(1200, 4200)

    X = np.column_stack([sex, age, height, weight, activity, goal])
    cols = ["sex_male", "age", "height_cm", "weight_kg", "activity", "goal"]
    df = pd.DataFrame(X, columns=cols)
    df["target_calories"] = y
    return df, cols

def make_synthetic_workout_dataset(n=800, random_state=1337):
    rng = np.random.default_rng(random_state)
    # Features: BMI, goal, experience_level
    height_m = rng.normal(1.70, 0.1, n).clip(1.5, 2.0)
    weight_kg = rng.normal(70, 12, n).clip(45, 120)
    bmi = weight_kg / (height_m**2)
    goal = rng.choice([0.8, 1.0, 1.2], n)  # 0.8=FatLoss,1.0=Balanced,1.2=MuscleGain
    exp_map = {"beginner":0, "intermediate":1, "advanced":2}
    exp = rng.choice([0,1,2], n, p=[0.45, 0.4, 0.15])

    # Label rules (create separable pattern)
    # 0=Cardio focus, 1=Balanced, 2=Strength focus
    y = []
    for i in range(n):
        if goal[i] == 0.8 or bmi[i] >= 27:
            y.append(0)  # Cardio
        elif goal[i] == 1.2 or (bmi[i] < 22 and exp[i] >= 1):
            y.append(2)  # Strength
        else:
            y.append(1)  # Balanced
    y = np.array(y)

    X = np.column_stack([bmi, goal, exp])
    cols = ["bmi", "goal", "experience_level"]
    df = pd.DataFrame(X, columns=cols)
    df["label"] = y
    return df, cols

def train_or_load_models():
    # Calorie Regression
    if os.path.exists(CAL_REG_PKL):
        with open(CAL_REG_PKL, "rb") as f:
            cal_reg = pickle.load(f)
    else:
        cal_df, cal_cols = make_synthetic_calorie_dataset()
        X_cal = cal_df[cal_cols].values
        y_cal = cal_df["target_calories"].values
        cal_reg = LinearRegression()
        cal_reg.fit(X_cal, y_cal)
        with open(CAL_REG_PKL, "wb") as f:
            pickle.dump(cal_reg, f)

    # Workout Classifier
    need_metrics = False
    if os.path.exists(WK_REC_PKL):
        with open(WK_REC_PKL, "rb") as f:
            wk_clf = pickle.load(f)
        # If confusion image missing, retrain once to regenerate metrics
        need_metrics = not os.path.exists(CONF_IMG_PNG)
    else:
        need_metrics = True
        wk_clf = None

    if need_metrics:
        wk_df, wk_cols = make_synthetic_workout_dataset()
        X = wk_df[wk_cols].values
        y = wk_df["label"].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=7, stratify=y)
        clf = DecisionTreeClassifier(max_depth=5, random_state=7)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        # Save model
        with open(WK_REC_PKL, "wb") as f:
            pickle.dump(clf, f)
        wk_clf = clf
        # Save confusion matrix image
        save_confusion_matrix(y_test, y_pred, labels=["Cardio(0)","Balanced(1)","Strength(2)"])

        # Also store metrics to a small json-like text if needed later (optional)

    return cal_reg, wk_clf

def save_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5,4), dpi=120)
    im = ax.imshow(cm, cmap='Blues')
    ax.set_title('Workout Classifier - Confusion Matrix')
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=15, ha='right')
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='black', fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(CONF_IMG_PNG, bbox_inches='tight')
    plt.close(fig)

# Load or train once at startup
CAL_REG_MODEL, WK_REC_MODEL = train_or_load_models()

# -------------------------
# API Routes
# -------------------------
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/generate-workout', methods=['POST'])
def gen_workout():
    data = request.json
    try:
        plan = generate_workout(data['level'], float(data.get('goal', 1.0)))
        return jsonify({'success': True, 'plan': plan})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/generate-nutrition', methods=['POST'])
def gen_nutrition():
    data = request.json
    try:
        result = calculate_targets(
            data['sex'],
            float(data['weight']),
            float(data['height']),
            int(data['age']),
            float(data['activity']),
            float(data['goal'])
        )
        meals = generate_meal_plan(result['target_calories'])
        return jsonify({'success': True, 'targets': result, 'meals': meals})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/food-database', methods=['GET'])
def get_food_db():
    return jsonify({'success': True, 'foods': FOOD_DB})

# -------------------------
# NEW: ML Endpoints
# -------------------------
@app.route('/api/predict-calories', methods=['POST'])
def api_predict_calories():
    """
    Inputs: sex (male/female), age, height, weight, activity, goal
    Returns: predicted_calories (regression) + formula target for comparison
    """
    try:
        data = request.json
        sex = data['sex']
        sex_num = 1 if str(sex).lower().startswith('m') else 0
        age = float(data['age'])
        height = float(data['height'])
        weight = float(data['weight'])
        activity = float(data['activity'])
        goal = float(data['goal'])

        X = np.array([[sex_num, age, height, weight, activity, goal]])
        pred = float(CAL_REG_MODEL.predict(X)[0])

        # Also compute formula-based target for side-by-side view
        formula = calculate_targets(sex, weight, height, int(age), activity, goal)['target_calories']

        return jsonify({'success': True, 'predicted_calories': int(round(pred)), 'formula_target': int(formula)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/recommend-workout', methods=['POST'])
def api_recommend_workout():
    """
    Inputs: height (cm), weight (kg), goal (0.8/1.0/1.2), level (beginner/intermediate/advanced)
    Output: label {0:Cardio,1:Balanced,2:Strength}
    """
    try:
        data = request.json
        height_cm = float(data['height'])
        weight_kg = float(data['weight'])
        goal = float(data['goal'])
        level = str(data['level']).lower()
        level_map = {"beginner":0, "intermediate":1, "advanced":2}
        exp = float(level_map.get(level, 0))

        bmi = weight_kg / ((height_cm/100.0)**2)
        X = np.array([[bmi, goal, exp]])
        label = int(WK_REC_MODEL.predict(X)[0])
        label_name = {0:"Cardio Focus", 1:"Balanced", 2:"Strength Focus"}[label]
        return jsonify({'success': True, 'recommendation': label_name, 'label': label, 'bmi': round(bmi,2)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/ml-metrics', methods=['GET'])
def api_ml_metrics():
    """
    Returns: confusion matrix image (base64) + example metrics computed on a fresh test split
    """
    try:
        # Recreate evaluation quickly to get metrics (uses same random_state for reproducibility)
        wk_df, wk_cols = make_synthetic_workout_dataset()
        X = wk_df[wk_cols].values
        y = wk_df["label"].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=7, stratify=y)
        y_pred = WK_REC_MODEL.predict(X_test)

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, average='weighted', zero_division=0))
        rec = float(recall_score(y_test, y_pred, average='weighted', zero_division=0))
        f1 = float(f1_score(y_test, y_pred, average='weighted', zero_division=0))

        # If confusion image file exists, serve it; else regenerate
        if not os.path.exists(CONF_IMG_PNG):
            save_confusion_matrix(y_test, y_pred, labels=["Cardio(0)","Balanced(1)","Strength(2)"])

        with open(CONF_IMG_PNG, "rb") as f:
            b64img = base64.b64encode(f.read()).decode()

        return jsonify({
            'success': True,
            'metrics': {
                'accuracy': round(acc,4),
                'precision_weighted': round(prec,4),
                'recall_weighted': round(rec,4),
                'f1_weighted': round(f1,4),
            },
            'confusion_image_base64': b64img
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

# -------------------------
# HTML Template (added ML buttons)
# -------------------------
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SmartFit - Fitness & Nutrition Planner (with ML)</title>
    <style>
        * { margin:0; padding:0; box-sizing:border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0A0E27 0%, #1a1f3a 100%);
            color:#fff; min-height:100vh; padding:20px;
        }
        .container { max-width:1200px; margin:0 auto; }
        header { text-align:center; padding:40px 0; border-bottom:2px solid rgba(99,102,241,.3); margin-bottom:30px; }
        h1 { font-size:3em; background:linear-gradient(135deg,#6366f1 0%,#8b5cf6 100%); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
        .subtitle { color:#9ca3af; }
        .tabs { display:flex; gap:10px; margin:20px 0 30px; flex-wrap:wrap; }
        .tab { padding:12px 30px; background:rgba(99,102,241,.1); border:2px solid rgba(99,102,241,.3); border-radius:10px; cursor:pointer; color:#9ca3af; font-weight:600; transition:.2s; }
        .tab.active { background:linear-gradient(135deg,#6366f1 0%,#8b5cf6 100%); color:#fff; border-color:#6366f1; }
        .tab:hover { border-color:rgba(99,102,241,.6); }
        .tab-content { display:none; }
        .tab-content.active { display:block; animation:fadeIn .25s; }
        @keyframes fadeIn { from{opacity:0; transform:translateY(8px);} to{opacity:1; transform:translateY(0);} }
        .card { background:rgba(30,41,59,.5); border:1px solid rgba(99,102,241,.2); border-radius:15px; padding:24px; margin-bottom:20px; backdrop-filter:blur(10px); }
        .grid-2 { display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:20px; }
        .form-group { margin-bottom:16px; }
        label { display:block; margin-bottom:8px; color:#e5e7eb; font-weight:500; }
        input, select { width:100%; padding:12px; background:rgba(15,23,42,.8); border:1px solid rgba(99,102,241,.3); border-radius:8px; color:#fff; font-size:16px; }
        .btn { padding:14px 20px; background:linear-gradient(135deg,#6366f1 0%,#8b5cf6 100%); border:none; border-radius:10px; color:#fff; font-weight:700; cursor:pointer; width:100%; margin-top:8px; }
        .btn.secondary { background:linear-gradient(135deg,#10b981 0%, #34d399 100%); }
        .btn.ghost { background:transparent; border:2px dashed rgba(99,102,241,.5); }
        .results { margin-top:20px; }
        .result-card { background:linear-gradient(135deg, rgba(99,102,241,.1) 0%, rgba(139,92,246,.1) 100%); border:1px solid rgba(99,102,241,.3); border-radius:12px; padding:16px; margin-bottom:12px; }
        .macro-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; margin-top:12px; }
        .macro-item { background:rgba(15,23,42,.5); padding:12px; border-radius:10px; text-align:center; }
        .macro-label { color:#9ca3af; font-size:.9em; }
        .macro-value { font-size:1.6em; font-weight:800; color:#6366f1; }
        .workout-day { background:rgba(15,23,42,.5); padding:12px; border-radius:10px; margin-bottom:12px; border-left:4px solid #6366f1; }
        .loading { text-align:center; padding:12px; color:#9ca3af; }
        .img-box { text-align:center; margin-top:10px; }
        img { max-width:100%; border-radius:10px; border:1px solid rgba(99,102,241,.3); }
    </style>
</head>
<body>
<div class="container">
    <header>
        <h1>💪 SmartFit</h1>
        <p class="subtitle">Fitness & Nutrition Planning — now with ML Predictions</p>
    </header>

    <div class="tabs">
        <div class="tab active" onclick="switchTab('nutrition')">🥗 Nutrition Plan</div>
        <div class="tab" onclick="switchTab('workout')">🏋️ Workout Plan</div>
        <div class="tab" onclick="switchTab('ml')">🧠 ML Predictions</div>
        <div class="tab" onclick="switchTab('food')">🍎 Food Database</div>
    </div>

    <!-- Nutrition -->
    <div id="nutrition" class="tab-content active">
        <div class="card">
            <h2>Generate Your Nutrition Plan</h2>
            <form id="nutritionForm">
                <div class="grid-2">
                    <div class="form-group">
                        <label>Sex</label>
                        <select name="sex" required>
                            <option value="male">Male</option>
                            <option value="female">Female</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Age (years)</label>
                        <input type="number" name="age" min="15" max="100" required>
                    </div>
                    <div class="form-group">
                        <label>Weight (kg)</label>
                        <input type="number" name="weight" step="0.1" min="30" max="300" required>
                    </div>
                    <div class="form-group">
                        <label>Height (cm)</label>
                        <input type="number" name="height" min="100" max="250" required>
                    </div>
                    <div class="form-group">
                        <label>Activity Level</label>
                        <select name="activity" required>
                            <option value="1.2">Sedentary (little/no exercise)</option>
                            <option value="1.375">Lightly Active (1-3 days/week)</option>
                            <option value="1.55">Moderately Active (3-5 days/week)</option>
                            <option value="1.725">Very Active (6-7 days/week)</option>
                            <option value="1.9">Extremely Active (athlete)</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Goal</label>
                        <select name="goal" required>
                            <option value="0.8">Weight Loss (20% deficit)</option>
                            <option value="0.9">Mild Weight Loss (10% deficit)</option>
                            <option value="1.0">Maintain Weight</option>
                            <option value="1.1">Mild Weight Gain (10% surplus)</option>
                            <option value="1.2">Weight Gain (20% surplus)</option>
                        </select>
                    </div>
                </div>
                <button type="submit" class="btn">Generate Nutrition Plan</button>
            </form>
            <div id="nutritionResults" class="results"></div>
        </div>
    </div>

    <!-- Workout -->
    <div id="workout" class="tab-content">
        <div class="card">
            <h2>Generate Your Workout Plan</h2>
            <form id="workoutForm">
                <div class="form-group">
                    <label>Fitness Level</label>
                    <select name="level" required>
                        <option value="beginner">Beginner (0-1 year)</option>
                        <option value="intermediate">Intermediate (1-3 years)</option>
                        <option value="advanced">Advanced (3+ years)</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Primary Goal</label>
                    <select name="goal" required>
                        <option value="1.0">General Fitness</option>
                        <option value="0.8">Fat Loss</option>
                        <option value="1.2">Muscle Gain</option>
                    </select>
                </div>
                <button type="submit" class="btn">Generate Workout Plan</button>
            </form>
            <div id="workoutResults" class="results"></div>
        </div>
    </div>

    <!-- ML -->
    <div id="ml" class="tab-content">
        <div class="card">
            <h2>🧠 ML: Predict Calories & Recommend Workout</h2>
            <p class="subtitle" style="margin:8px 0 16px;">Use trained models (Linear Regression & Decision Tree)</p>
            <form id="mlForm">
                <div class="grid-2">
                    <div class="form-group">
                        <label>Sex</label>
                        <select name="sex" required>
                            <option value="male">Male</option>
                            <option value="female">Female</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Age (years)</label>
                        <input type="number" name="age" min="15" max="100" required>
                    </div>
                    <div class="form-group">
                        <label>Weight (kg)</label>
                        <input type="number" name="weight" step="0.1" min="30" max="300" required>
                    </div>
                    <div class="form-group">
                        <label>Height (cm)</label>
                        <input type="number" name="height" min="100" max="250" required>
                    </div>
                    <div class="form-group">
                        <label>Activity</label>
                        <select name="activity" required>
                            <option value="1.2">Sedentary</option>
                            <option value="1.375">Light (1-3 days/wk)</option>
                            <option value="1.55">Moderate (3-5 days/wk)</option>
                            <option value="1.725">Active (6-7 days/wk)</option>
                            <option value="1.9">Very Active</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Goal</label>
                        <select name="goal" required>
                            <option value="0.8">Fat Loss</option>
                            <option value="1.0">Maintain</option>
                            <option value="1.2">Muscle Gain</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Experience Level</label>
                        <select name="level" required>
                            <option value="beginner">Beginner</option>
                            <option value="intermediate">Intermediate</option>
                            <option value="advanced">Advanced</option>
                        </select>
                    </div>
                </div>
                <button type="button" class="btn secondary" onclick="predictCalories()">🔮 Predict Calories (ML)</button>
                <button type="button" class="btn" onclick="recommendWorkout()">🤖 Recommend Workout (ML)</button>
                <button type="button" class="btn ghost" onclick="loadMetrics()">📈 Show Classifier Metrics</button>
            </form>
            <div id="mlResults" class="results"></div>
            <div id="metricsBox" class="results"></div>
        </div>
    </div>

    <!-- Food DB -->
    <div id="food" class="tab-content">
        <div class="card">
            <h2>Food Database</h2>
            <p style="color:#9ca3af; margin-bottom:12px;">Browse our nutritional dataset</p>
            <div id="foodList"></div>
        </div>
    </div>
</div>

<script>
    function switchTab(tabName) {
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        event.target.classList.add('active');
        document.getElementById(tabName).classList.add('active');
        if (tabName === 'food') loadFoodDatabase();
    }

    // Nutrition
    document.getElementById('nutritionForm').addEventListener('submit', async (e) => {
        e.preventDefault();
        const data = Object.fromEntries(new FormData(e.target));
        const resultsDiv = document.getElementById('nutritionResults');
        resultsDiv.innerHTML = '<div class="loading">⏳ Generating your personalized nutrition plan...</div>';
        try {
            const res = await fetch('/api/generate-nutrition', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(data)});
            const result = await res.json();
            if (result.success) displayNutritionResults(result.targets, result.meals);
            else resultsDiv.innerHTML = `<div class="result-card">❌ ${result.error}</div>`;
        } catch (err) { resultsDiv.innerHTML = `<div class="result-card">❌ ${err.message}</div>`; }
    });

    function displayNutritionResults(targets, meals) {
        const html = `
            <div class="result-card">
                <h3>📊 Your Daily Targets</h3>
                <div class="macro-grid">
                    <div class="macro-item"><div class="macro-label">BMR</div><div class="macro-value">${targets.bmr}</div></div>
                    <div class="macro-item"><div class="macro-label">TDEE</div><div class="macro-value">${targets.tdee}</div></div>
                    <div class="macro-item"><div class="macro-label">Target (kcal)</div><div class="macro-value">${targets.target_calories}</div></div>
                    <div class="macro-item"><div class="macro-label">Protein (g)</div><div class="macro-value">${targets.protein}</div></div>
                    <div class="macro-item"><div class="macro-label">Carbs (g)</div><div class="macro-value">${targets.carbs}</div></div>
                    <div class="macro-item"><div class="macro-label">Fat (g)</div><div class="macro-value">${targets.fat}</div></div>
                </div>
            </div>
            <div class="result-card">
                <h3>🍽️ Meal Distribution</h3>
                ${meals.map(m => `<div class="workout-day"><strong>${m.meal}</strong><div>${m.calories} calories</div></div>`).join('')}
            </div>`;
        document.getElementById('nutritionResults').innerHTML = html;
    }

    // Workout (template)
    document.getElementById('workoutForm').addEventListener('submit', async (e) => {
        e.preventDefault();
        const data = Object.fromEntries(new FormData(e.target));
        const box = document.getElementById('workoutResults');
        box.innerHTML = '<div class="loading">⏳ Generating your workout plan...</div>';
        try {
            const res = await fetch('/api/generate-workout', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(data)});
            const result = await res.json();
            if (result.success) {
                const html = `
                    <div class="result-card">
                        <h3>📅 Your 7-Day Workout Plan</h3>
                        ${result.plan.map(([day, exs]) => `
                            <div class="workout-day">
                                <strong>${day}</strong>
                                <ul>${exs.map(ex => `<li>✓ ${ex}</li>`).join('')}</ul>
                            </div>`).join('')}
                    </div>`;
                box.innerHTML = html;
            } else box.innerHTML = `<div class="result-card">❌ ${result.error}</div>`;
        } catch (err) { box.innerHTML = `<div class="result-card">❌ ${err.message}</div>`; }
    });

    // Food DB
    async function loadFoodDatabase() {
        const target = document.getElementById('foodList');
        target.innerHTML = '<div class="loading">⏳ Loading...</div>';
        try {
            const res = await fetch('/api/food-database');
            const result = await res.json();
            if (result.success) {
                target.innerHTML = result.foods.map(food => `
                    <div class="workout-day">
                        <strong>${food.name}</strong>
                        <div style="color:#9ca3af;">Serving: ${food.serving}</div>
                        <div class="macro-grid" style="margin-top:8px;">
                            <div><div class="macro-label">Calories</div><div class="macro-value">${food.cal}</div></div>
                            <div><div class="macro-label">Protein</div><div class="macro-value">${food.protein}g</div></div>
                            <div><div class="macro-label">Carbs</div><div class="macro-value">${food.carbs}g</div></div>
                            <div><div class="macro-label">Fat</div><div class="macro-value">${food.fat}g</div></div>
                        </div>
                    </div>`).join('');
            }
        } catch (err) { target.innerHTML = `<div class="result-card">❌ ${err.message}</div>`; }
    }

    // ML: Predict Calories
    async function predictCalories() {
        const form = document.getElementById('mlForm');
        const data = Object.fromEntries(new FormData(form));
        const box = document.getElementById('mlResults');
        box.innerHTML = '<div class="loading">⏳ Predicting calories with ML...</div>';
        try {
            const res = await fetch('/api/predict-calories', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(data)});
            const result = await res.json();
            if (result.success) {
                box.innerHTML = `
                    <div class="result-card">
                        <h3>🔮 Calorie Prediction (ML)</h3>
                        <div class="macro-grid">
                            <div class="macro-item"><div class="macro-label">ML Predicted</div><div class="macro-value">${result.predicted_calories}</div></div>
                            <div class="macro-item"><div class="macro-label">Formula Target</div><div class="macro-value">${result.formula_target}</div></div>
                        </div>
                    </div>`;
            } else box.innerHTML = `<div class="result-card">❌ ${result.error}</div>`;
        } catch (err) { box.innerHTML = `<div class="result-card">❌ ${err.message}</div>`; }
    }

    // ML: Recommend Workout
    async function recommendWorkout() {
        const form = document.getElementById('mlForm');
        const data = Object.fromEntries(new FormData(form));
        const box = document.getElementById('mlResults');
        box.innerHTML = '<div class="loading">⏳ Getting ML workout recommendation...</div>';
        try {
            const res = await fetch('/api/recommend-workout', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(data)});
            const result = await res.json();
            if (result.success) {
                box.innerHTML = `
                    <div class="result-card">
                        <h3>🤖 Workout Recommendation (ML)</h3>
                        <div>BMI: <strong>${result.bmi}</strong></div>
                        <div>Suggested Focus: <strong>${result.recommendation}</strong></div>
                    </div>`;
            } else box.innerHTML = `<div class="result-card">❌ ${result.error}</div>`;
        } catch (err) { box.innerHTML = `<div class="result-card">❌ ${err.message}</div>`; }
    }

    // ML: Metrics (Confusion Matrix)
    async function loadMetrics() {
        const target = document.getElementById('metricsBox');
        target.innerHTML = '<div class="loading">⏳ Loading classifier metrics...</div>';
        try {
            const res = await fetch('/api/ml-metrics');
            const result = await res.json();
            if (result.success) {
                const img = "data:image/png;base64," + result.confusion_image_base64;
                target.innerHTML = `
                    <div class="result-card">
                        <h3>📈 Classifier Metrics (Workout Recommender)</h3>
                        <div>Accuracy: <strong>${result.metrics.accuracy}</strong></div>
                        <div>Precision (weighted): <strong>${result.metrics.precision_weighted}</strong></div>
                        <div>Recall (weighted): <strong>${result.metrics.recall_weighted}</strong></div>
                        <div>F1 (weighted): <strong>${result.metrics.f1_weighted}</strong></div>
                        <div class="img-box"><img src="${img}" alt="Confusion Matrix" /></div>
                    </div>`;
            } else target.innerHTML = `<div class="result-card">❌ ${result.error}</div>`;
        } catch (err) { target.innerHTML = `<div class="result-card">❌ ${err.message}</div>`; }
    }
</script>
</body>
</html>
"""

# -------------------------
# Run App
# -------------------------
if __name__ == '__main__':
    print("=" * 70)
    print("🚀 SmartFit Flask Application (with ML)")
    print("=" * 70)
    print("📍 http://localhost:5000")
    print("✨ Features:")
    print("   • Nutrition planner (formula)")
    print("   • Workout schedules (templates)")
    print("   • ML: Calorie prediction (Linear Regression)")
    print("   • ML: Workout recommender (Decision Tree)")
    print("   • Confusion matrix & metrics endpoint")
    print("=" * 70)
    app.run(debug=True, port=5000)
