import os
from datetime import datetime

# Import Sub-Modules
import PlantVeda_KNN_Mk3
import PlantVeda_LR_Mk3
import PlantVeda_MLP_Mk3
import PlantVeda_NB_Mk3
import PlantVeda_SVM_Mk3
import PlantVeda_VotingSystem_Mk3 as voting
import PlantVeda_PSO_Mk3
import PlantVeda_PDF_Mk3

def run_pipeline(area, budget, soil, habitat, temp_range, sunlight):
    print(f"--- Pipeline Started for {habitat} ---")
    ml_input = {
        "Sunlight": sunlight,
        "Soil": soil,
        "Habitat": habitat,
        "Temperature": temp_range
    }

    predictions = {
        "KNN": PlantVeda_KNN_Mk3.predict(ml_input)[0],
        "LR": PlantVeda_LR_Mk3.predict(ml_input)[0],
        "MLP": PlantVeda_MLP_Mk3.predict(ml_input)[0],
        "NB": PlantVeda_NB_Mk3.predict(ml_input)[0],
        "SVM": PlantVeda_SVM_Mk3.predict(ml_input)[0]
    }

    final_growth_form = voting.vote(predictions)
    print(f"ML Prediction: {final_growth_form}")

    recommendations = PlantVeda_PSO_Mk3.recommend(
        total_area=area,
        total_cost=budget,
        user_habitat=habitat,
        user_soil=soil,
        predicted_growth_form=final_growth_form
    )
    print(f"PSO found {len(recommendations)} plants.")

    buffer = PlantVeda_PDF_Mk3.generate_pdf(  # ← gets buffer back
        recommended_plants=recommendations,
        growth_form=final_growth_form,
        habitat=habitat,
        soil=soil,
        total_area=area,
        total_cost=budget
    )

    return buffer  # ← pass buffer up to app.py