import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neural_network import MLPClassifier

# ── Training data (57 rows — all 11 classes represented) ──────────────────────
sunlight = [
    "Full sun","Full sun","Partial sun","Full sun","Full sun","Partial sun",
    "Full sun","Full sun","Partial sun","Full sun","Full sun","Full sun",
    "Partial sun","Full sun","Full sun","Partial sun","Full sun","Full sun",
    "Full sun","Partial sun","Full sun","Full sun","Partial sun","Full sun",
    "Full sun","Partial sun","Full sun","Full sun","Partial sun","Full sun",
    "Full sun","Partial sun","Full sun","Full sun","Partial sun","Full sun",
    "Full sun","Partial sun",
    "Full sun","Full sun","Partial sun","Full sun",
    "Full sun","Full sun","Partial sun",
    "Full sun","Partial sun","Full sun",
    "Partial sun","Partial sun","Partial sun",
    "Partial sun","Partial sun","Full sun",
    "Partial sun","Partial sun","Partial sun",
]
soil = [
    "Loamy","Loamy","Clayey","Moist Loam","Sandy","Loamy","Clayey","Sandy",
    "Moist Loam","Loamy","Sandy","Wet soil","Loamy","Loamy","Sandy","Moist Loam",
    "Loamy","Wet soil","Sandy","Moist Loam","Moist Loam","Loamy","Sandy","Sandy",
    "Loamy","Clayey","Loamy","Sandy","Loamy","Loamy","Loamy","Moist Loam","Sandy",
    "Sandy","Moist Loam","Moist Loam","Sandy","Moist Loam",
    "Sandy","Sandy","Sandy","Loamy",
    "Loamy","Moist Loam","Loamy",
    "Sandy","Sandy","Moist Loam",
    "Moist Loam","Loamy","Clayey",
    "Moist Loam","Moist Loam","Moist Loam",
    "Moist Loam","Loamy","Sandy",
]
habitat = [
    "Garden","Open field","Pathside","Flower bed","Lawn","Garden","Pathside",
    "Open field","Pond","Garden","Lawn","Pond","Garden","Garden","Open field",
    "Flower bed","Lawn","Pond","Open field","Garden","Pond","Pond","Pathside",
    "Open field","Garden","Flower bed","Garden","Lawn","Garden","Garden","Garden",
    "Lawn","Open field","Flower bed","Garden","Garden","Open field","Flower bed",
    "Garden","Pathside","Garden","Flower bed",
    "Garden","Garden","Pathside",
    "Lawn","Open field","Lawn",
    "Pathside","Garden","Pathside",
    "Garden","Pond","Garden",
    "Garden","Garden","Garden",
]
temperature = [
    "20-35","20-40","20-35","20-35","20-35","20-40","20-35","25-35","25-40","18-35",
    "20-35","20-35","20-40","24-35","24-35","20-40","22-35","20-35","20-35","20-35",
    "18-30","18-35","20-35","20-38","20-35","25-35","22-35","20-35","22-35","20-35",
    "20-35","18-30","25-40","20-35","18-30","24-35","20-35","20-35",
    "25-40","25-40","20-38","20-35",
    "20-35","22-35","20-35",
    "20-35","20-40","18-35",
    "20-35","22-35","20-35",
    "18-30","18-30","20-30",
    "18-30","20-30","22-32",
]
growth = [
    "Tree","Tree","Shrub","Tree","Tree","Shrub","Climber","Tree","Aquatic Herb",
    "Tree","Grass","Aquatic Herb","Tree","Tree","Tree","Tree","Tree","Tree",
    "Tree","Tree","Aquatic Herb","Aquatic Herb","Herb","Vine","Shrub","Herb",
    "Tree","Herb","Tree","Tree","Tree","Palm","Palm","Shrub","Fern","Tree",
    "Tree","Epiphyte",
    "Succulent","Succulent","Succulent","Succulent",
    "Vine","Vine","Vine",
    "Grass","Grass","Grass",
    "Climber","Climber","Climber",
    "Fern","Fern","Fern",
    "Epiphyte","Epiphyte","Epiphyte",
]

# ── All valid classes ──────────────────────────────────────────────────────────
ALL_SUNLIGHT    = ["Full sun", "Partial sun"]
ALL_SOIL        = ["Loamy", "Clayey", "Moist Loam", "Wet soil", "Sandy"]
ALL_HABITAT     = ["Pathside", "Garden", "Open field", "Flower bed", "Lawn", "Pond"]
ALL_GROWTH_FORM = [
    "Tree", "Palm", "Shrub", "Climber", "Grass",
    "Aquatic Herb", "Herb", "Vine", "Succulent", "Fern", "Epiphyte"
]

def predict(input_data):
    # ── Validate input ─────────────────────────────────────────────────────────
    errors = []
    if input_data["Sunlight"] not in ALL_SUNLIGHT:
        errors.append(f"Invalid Sunlight '{input_data['Sunlight']}'. Choose from: {ALL_SUNLIGHT}")
    if input_data["Soil"] not in ALL_SOIL:
        errors.append(f"Invalid Soil '{input_data['Soil']}'. Choose from: {ALL_SOIL}")
    if input_data["Habitat"] not in ALL_HABITAT:
        errors.append(f"Invalid Habitat '{input_data['Habitat']}'. Choose from: {ALL_HABITAT}")
    if errors:
        raise ValueError("\n".join(errors))

    df = pd.DataFrame({
        "Sunlight":    sunlight,
        "Soil":        soil,
        "Habitat":     habitat,
        "Temperature": temperature,
        "Growth Form": growth
    })

    # ── Convert temperature range → midpoint ───────────────────────────────────
    df["Temperature"] = df["Temperature"].apply(
        lambda x: (int(x.split('-')[0]) + int(x.split('-')[1])) / 2
    )

    # ── Fit encoders on ALL known classes ─────────────────────────────────────
    le_sun    = LabelEncoder(); le_sun.fit(ALL_SUNLIGHT)
    le_soil   = LabelEncoder(); le_soil.fit(ALL_SOIL)
    le_hab    = LabelEncoder(); le_hab.fit(ALL_HABITAT)
    le_target = LabelEncoder(); le_target.fit(ALL_GROWTH_FORM)

    df["Sunlight"]   = le_sun.transform(df["Sunlight"])
    df["Soil"]       = le_soil.transform(df["Soil"])
    df["Habitat"]    = le_hab.transform(df["Habitat"])
    df["Growth_enc"] = le_target.transform(df["Growth Form"])

    # ── Scale all 4 features ───────────────────────────────────────────────────
    scaler = MinMaxScaler()
    df[["Sunlight","Soil","Habitat","Temperature"]] = scaler.fit_transform(
        df[["Sunlight","Soil","Habitat","Temperature"]]
    )

    X = df[["Sunlight","Soil","Habitat","Temperature"]]
    y = df["Growth_enc"]

    # ── Train MLP ──────────────────────────────────────────────────────────────
    mlp = MLPClassifier(hidden_layer_sizes=(10, 10), activation='relu',
                        solver='lbfgs', max_iter=1000, random_state=42)
    mlp.fit(X, y)

    # ── Transform user input ───────────────────────────────────────────────────
    input_sun  = le_sun.transform([input_data["Sunlight"]])[0]
    input_soil = le_soil.transform([input_data["Soil"]])[0]
    input_hab  = le_hab.transform([input_data["Habitat"]])[0]

    temp_str = str(input_data.get("Temperature", "")).strip()
    try:
        parts = temp_str.split("-")
        if len(parts) != 2:
            raise ValueError("Invalid format")
        input_temp = (float(parts[0].strip()) + float(parts[1].strip())) / 2
    except Exception:
        raise ValueError(
            f"Temperature must be in 'min-max' format (e.g. '20-35'); got: '{temp_str}'"
        )

    user_row = pd.DataFrame(
        [[input_sun, input_soil, input_hab, input_temp]],
        columns=["Sunlight","Soil","Habitat","Temperature"]
    )
    user_row_scaled = pd.DataFrame(
        scaler.transform(user_row),
        columns=["Sunlight","Soil","Habitat","Temperature"]
    )

    pred_enc = mlp.predict(user_row_scaled)[0]
    return le_target.inverse_transform([pred_enc])[0]


if __name__ == "__main__":
    input_data = {
        "Sunlight":    "Full sun",
        "Soil":        "Loamy",
        "Habitat":     "Garden",
        "Temperature": "20-35"
    }
    result = predict(input_data)
    print(f"Predicted growth form: {result}")