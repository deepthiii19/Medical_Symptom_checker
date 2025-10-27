from flask import Flask, render_template, request, jsonify
from flask_ngrok import run_with_ngrok

import pickle
import numpy as np

app = Flask(__name__)
run_with_ngrok(app)


# Disease to doctor specialist mapping
disease_specialist_mapping = {
    'Fungal infection': 'Dermatologist',
    'Allergy': 'Allergist/Immunologist',
    'GERD': 'Gastroenterologist',
    'Chronic cholestasis': 'Hepatologist',
    'Drug Reaction': 'Allergist',
    'Peptic ulcer diseae': 'Gastroenterologist',
    'AIDS': 'Infectious Disease Specialist',
    'Diabetes': 'Endocrinologist',
    'Gastroenteritis': 'Gastroenterologist',
    'Bronchial Asthma': 'Pulmonologist',
    'Hypertension': 'Cardiologist',
    'Migraine': 'Neurologist',
    'Cervical spondylosis': 'Orthopedist',
    'Paralysis': 'Neurologist',
    'Jaundice': 'Hepatologist',
    'Malaria': 'General Physician',
    'Chicken pox': 'Dermatologist',
    'Dengue': 'General Physician',
    'Typhoid': 'General Physician',
    'hepatitis A': 'Hepatologist',
    'Hepatitis B': 'Hepatologist',
    'Hepatitis C': 'Hepatologist',
    'Hepatitis D': 'Hepatologist',
    'Hepatitis E': 'Hepatologist',
    'Alcoholic hepatitis': 'Hepatologist',
    'Tuberculosis': 'Pulmonologist',
    'Common Cold': 'General Physician',
    'Pneumonia': 'Pulmonologist',
    'Dimorphic hemmorhoids(piles)': 'Proctologist',
    'Heart attack': 'Cardiologist',
    'Varicose veins': 'Vascular Surgeon',
    'Hypothyroidism': 'Endocrinologist',
    'Hyperthyroidism': 'Endocrinologist',
    'Hypoglycemia': 'Endocrinologist',
    'Osteoarthristis': 'Rheumatologist',
    'Arthritis': 'Rheumatologist',
    'Acne': 'Dermatologist',
    'Urinary tract infection': 'Urologist',
    'Psoriasis': 'Dermatologist',
    'Impetigo': 'Dermatologist'
}

# Symptom list (you can expand this)
symptoms_list = ['itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing',
                 'shivering', 'chills', 'joint_pain', 'stomach_pain', 'acidity',
                 'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 'burning_micturition',
                 'spotting_urination', 'fatigue', 'weight_gain', 'anxiety',
                 'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness',
                 'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough',
                 'high_fever', 'sunken_eyes', 'breathlessness', 'sweating', 'dehydration',
                 'indigestion', 'headache', 'yellowish_skin', 'dark_urine', 'nausea',
                 'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 'constipation',
                 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine',
                 'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload',
                 'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise',
                 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation',
                 'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion',
                 'chest_pain', 'weakness_in_limbs', 'fast_heart_rate',
                 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool',
                 'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps',
                 'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels',
                 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails',
                 'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts',
                 'drying_and_tingling_lips', 'slurred_speech', 'knee_pain', 'hip_joint_pain',
                 'muscle_weakness', 'stiff_neck', 'swelling_joints', 'movement_stiffness',
                 'spinning_movements', 'loss_of_balance', 'unsteadiness',
                 'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort',
                 'foul_smell_of_urine', 'continuous_feel_of_urine', 'passage_of_gases',
                 'internal_itching', 'toxic_look_(typhos)', 'depression', 'irritability',
                 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain',
                 'abnormal_menstruation', 'dischromic_patches', 'watering_from_eyes',
                 'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum',
                 'rusty_sputum', 'lack_of_concentration', 'visual_disturbances',
                 'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma',
                 'stomach_bleeding', 'distention_of_abdomen', 'history_of_alcohol_consumption',
                 'fluid_overload', 'blood_in_sputum', 'prominent_veins_on_calf',
                 'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads',
                 'scurring', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails',
                 'inflammatory_nails', 'blister', 'red_sore_around_nose', 'yellow_crust_ooze']


@app.route('/')
def home():
    return render_template('index.html', symptoms=symptoms_list)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get patient details
        name = request.form.get('name')
        age = request.form.get('age')
        gender = request.form.get('gender')

        # Get selected symptoms
        selected_symptoms = request.form.getlist('symptoms')

        if not selected_symptoms:
            return render_template('result.html',
                                   error="Please select at least one symptom",
                                   name=name)

        # Simple rule-based prediction (you can replace with ML model)
        predicted_disease = predict_disease(selected_symptoms)

        # Get specialist recommendation
        specialist = disease_specialist_mapping.get(predicted_disease, 'General Physician')

        # Generate gentle message
        result_message = generate_gentle_message(predicted_disease, specialist)

        return render_template('result.html',
                               name=name,
                               age=age,
                               gender=gender,
                               symptoms=selected_symptoms,
                               disease=predicted_disease,
                               specialist=specialist,
                               message=result_message)

    except Exception as e:
        return render_template('result.html', error=str(e))


def predict_disease(symptoms):
    """
    Simple symptom matching logic
    Replace this with your trained ML model for better accuracy
    """
    # Example mapping (expand based on your dataset)
    symptom_disease_map = {
        ('itching', 'skin_rash', 'nodal_skin_eruptions'): 'Fungal infection',
        ('continuous_sneezing', 'shivering', 'chills'): 'Allergy',
        ('stomach_pain', 'acidity', 'vomiting'): 'GERD',
        ('cough', 'high_fever', 'breathlessness'): 'Pneumonia',
        ('headache', 'chest_pain', 'dizziness'): 'Migraine',
        ('joint_pain', 'neck_pain', 'knee_pain'): 'Arthritis',
        ('high_fever', 'headache', 'nausea'): 'Malaria',
        ('itching', 'skin_rash', 'pus_filled_pimples'): 'Acne',
        ('burning_micturition', 'bladder_discomfort', 'foul_smell_of_urine'): 'Urinary tract infection',
    }

    # Check for matching symptom combinations
    symptoms_set = set(symptoms)
    for key_symptoms, disease in symptom_disease_map.items():
        if symptoms_set.intersection(set(key_symptoms)):
            return disease

    return 'Common Cold'  # Default prediction


def generate_gentle_message(disease, specialist):
    """Generate a gentle, non-alarming message"""
    messages = {
        'default': f"Based on your symptoms, you might be experiencing {disease}. This is a common condition that can be effectively managed with proper care."
    }

    base_message = messages.get(disease, messages['default'])
    recommendation = f" We recommend consulting a {specialist} for a proper evaluation and personalized treatment plan."

    return base_message + recommendation


if __name__ == '__main__':
    app.run()
