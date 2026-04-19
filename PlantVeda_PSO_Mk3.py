import numpy as np
from pyswarm import pso

# -----------------------------
# DATA (unchanged)
# -----------------------------
plants = [
    "Banyan Tree","Peepal Tree","Rain Tree","Gulmohar","Ashoka Tree",
    "Mango Tree","Jackfruit Tree","Neem Tree","Arjun Tree","Kadam Tree",
    "Coconut Palm","Date Palm","Areca Palm",
    "Water Hyacinth","Lotus","Water Lily","Papyrus",
    "Indian Grass","Elephant Grass",
    "Touch-me-not","Wild Spinach","Tulsi","Periwinkle","Mint",
    "Morning Glory","Money Plant","Bougainvillea",
    "Hibiscus","Ixora","Lantana","Jasmine",
    "Aloe Vera","Canna Lily","Spider Lily",
    "Boston Fern","Bamboo","Orchid",
    "Marigold","Portulaca","Vinca",
]

habitat = [
    "Open field","Pathside","Pathside","Pathside","Open field",
    "Open field","Open field","Pathside","Pathside","Garden",
    "Garden","Garden","Garden",
    "Pond","Pond","Pond","Pond",
    "Open field","Open field",
    "Garden","Open field","Garden","Garden","Garden",
    "Garden","Garden","Pathside",
    "Garden","Garden","Garden","Pathside",
    "Garden","Flower bed","Flower bed",
    "Garden","Open field","Garden",
    "Flower bed","Flower bed","Flower bed",
]

soil = [
    "Loamy","Loamy","Sandy","Sandy","Clayey",
    "Loamy","Clayey","Loamy","Loamy","Clayey",
    "Loamy","Sandy","Sandy",
    "Wet soil","Wet soil","Wet soil","Wet soil",
    "Loamy","Sandy",
    "Loamy","Loamy","Loamy","Loamy","Loamy",
    "Loamy","Loamy","Loamy",
    "Loamy","Loamy","Loamy","Loamy",
    "Sandy","Sandy","Sandy",
    "Moist Loam","Loamy","Moist Loam",
    "Loamy","Sandy","Sandy",
]

growth_form = [
    "Tree","Tree","Tree","Tree","Tree",
    "Tree","Tree","Tree","Tree","Tree",
    "Palm","Palm","Palm",
    "Aquatic Herb","Aquatic Herb","Aquatic Herb","Aquatic Herb",
    "Grass","Grass",
    "Herb","Herb","Herb","Herb","Herb",
    "Climber","Climber","Climber",
    "Shrub","Shrub","Shrub","Shrub",
    "Succulent","Herb","Herb",
    "Fern","Tree","Epiphyte",
    "Herb","Succulent","Herb",
]

carbon = np.array([
    250,110,180,220,120,90,150,140,
    170,150,160,150,60,50,55,
    8,10,12,25,5,
    15,18,20,12,6,
    20,5,5,15,4,
    3,4,6,5,8,
    7,6,9,5,4
])

cost = np.array([
    5000,2500,3200,3500,2500,2000,3000,2500,
    3000,2800,2500,2500,2500,2000,2000,
    500,600,700,1200,200,
    900,800,500,700,300,
    900,300,300,400,250,
    200,300,400,350,600,
    500,800,700,300,250
])

water = np.array([
    200,70,150,150,90,60,120,100,
    120,80,120,100,60,40,50,
    30,35,40,45,10,
    5,6,5,4,3,
    8,2,4,8,2,
    2,3,4,3,5,
    4,6,5,2,2
])

area = np.array([
    200,50,70,80,50,20,60,70,
    60,60,50,50,25,6,20,
    1.2,1.5,1,3,0.2,
    4,3,2,2,0.5,
    3,0.5,1,0.2,0.3,
    0.3,0.2,1,0.5,0.8,
    0.6,0.5,1,0.4,0.3
])

# -----------------------------
# FALLBACK FILTER
# -----------------------------
def get_valid_indices(user_habitat, user_soil, predicted_growth_form):

    # LEVEL 1
    idx = [
        i for i in range(len(plants))
        if habitat[i].lower() == user_habitat.lower()
        and soil[i].lower() == user_soil.lower()
        and growth_form[i].lower() == predicted_growth_form.lower()
    ]
    if len(idx) >= 5:
        print("\n✅ Using strict filtering")
        return idx

    # LEVEL 2
    idx = [
        i for i in range(len(plants))
        if habitat[i].lower() == user_habitat.lower()
        and soil[i].lower() == user_soil.lower()
    ]
    if len(idx) >= 5:
        print("\n⚠️ Relaxed growth form condition")
        return idx

    # LEVEL 3
    idx = [
        i for i in range(len(plants))
        if habitat[i].lower() == user_habitat.lower()
    ]
    if len(idx) >= 5:
        print("\n⚠️ Relaxed soil condition")
        return idx

    # LEVEL 4
    print("\n⚠️ Using global fallback (all plants)")
    return list(range(len(plants)))


# -----------------------------
# MAIN FUNCTION
# -----------------------------
def recommend(total_area, total_cost, user_habitat, user_soil, predicted_growth_form):

    valid_idx = get_valid_indices(user_habitat, user_soil, predicted_growth_form)

    carbon_f = carbon[valid_idx]
    cost_f   = cost[valid_idx]
    water_f  = water[valid_idx]
    area_f   = area[valid_idx]
    plants_f = [plants[i] for i in valid_idx]

    # OBJECTIVE
    def objective(n):
        n = np.abs(n)
        n_safe = np.where(n == 0, 1e-6, n)
        return np.sum((cost_f + water_f) / n_safe) + abs(np.sum(n * cost_f) - total_cost)

    # CONSTRAINTS
    def constraints(n):
        return [
            total_area - np.sum(n * area_f),
            total_cost - np.sum(n * cost_f),
        ]

    # PSO
    lb = [0] * len(plants_f)
    ub = [50] * len(plants_f)

    best_n, _ = pso(objective, lb, ub, f_ieqcons=constraints,
                    swarmsize=50, maxiter=200, debug=False)

    # SCORING
    scores = carbon_f / (cost_f + water_f)

    k = min(5, len(plants_f))
    top_idx = np.argsort(scores)[::-1][:k]

    results = []
    for i in top_idx:
        results.append({
            "name": plants_f[i],
            "count": max(1, int(round(best_n[i]))),
            "score": round(float(scores[i]), 4),
            "carbon": int(carbon_f[i]),
            "cost_per_unit": int(cost_f[i]),
            "water_per_unit": int(water_f[i]),
        })

    return results