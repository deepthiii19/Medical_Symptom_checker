from flask import Flask, render_template, request
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

app = Flask(__name__)

# Disease to doctor specialist mapping
disease_specialist_mapping = {
    'Fungal infection': 'Dermatologist',
    'Allergy': 'Allergist/Immunologist',
    'GERD': 'Gastroenterologist',
    'Chronic cholestasis': 'Hepatologist',
    'Drug Reaction': 'Allergist',
    'Peptic ulcer disease': 'Gastroenterologist',
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
    'Hepatitis A': 'Hepatologist',
    'Hepatitis B': 'Hepatologist',
    'Hepatitis C': 'Hepatologist',
    'Hepatitis D': 'Hepatologist',
    'Hepatitis E': 'Hepatologist',
    'Hepatitis': 'Hepatologist',
    'Alcoholic hepatitis': 'Hepatologist',
    'Tuberculosis': 'Pulmonologist',
    'Pneumonia': 'Pulmonologist',
    'Hemorrhoids': 'Proctologist',
    'Heart attack': 'Cardiologist',
    'Varicose veins': 'Vascular Surgeon',
    'Hypothyroidism': 'Endocrinologist',
    'Hyperthyroidism': 'Endocrinologist',
    'Hypoglycemia': 'Endocrinologist',
    'Osteoarthritis': 'Rheumatologist',
    'Arthritis': 'Rheumatologist',
    'Acne': 'Dermatologist',
    'Urinary tract infection': 'Urologist',
    'Psoriasis': 'Dermatologist',
    'Impetigo': 'Dermatologist',
    'Menstrual Pain': 'Gynecologist',
    'Endometriosis': 'Gynecologist',
    'Anxiety Attack': 'Psychiatrist',
    'Panic Attack': 'Psychiatrist',
    'Food Poisoning': 'General Physician',
    'Asthma Attack': 'Pulmonologist',
    'Liver Disease': 'Hepatologist',
}

# Symptom list
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
                 'swollen_extremeties', 'excessive_hunger', 'drying_and_tingling_lips',
                 'slurred_speech', 'knee_pain', 'hip_joint_pain',
                 'muscle_weakness', 'stiff_neck', 'swelling_joints', 'movement_stiffness',
                 'spinning_movements', 'loss_of_balance', 'unsteadiness',
                 'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort',
                 'foul_smell_of_urine', 'continuous_feel_of_urine', 'passage_of_gases',
                 'internal_itching', 'depression', 'irritability',
                 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain',
                 'abnormal_menstruation', 'dischromic_patches', 'watering_from_eyes',
                 'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum',
                 'rusty_sputum', 'lack_of_concentration', 'visual_disturbances',
                 'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma',
                 'stomach_bleeding', 'distention_of_abdomen', 'history_of_alcohol_consumption',
                 'blood_in_sputum', 'prominent_veins_on_calf',
                 'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads',
                 'scurring', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails',
                 'inflammatory_nails', 'blister', 'red_sore_around_nose', 'yellow_crust_ooze']


def extract_symptoms_from_text(text):
    """Extract symptoms from natural language text using NLP"""
    if not text or text.strip() == "":
        return []

    text = text.lower()

    # Comprehensive symptom keywords dictionary
    symptom_keywords = {
        'headache': ['headache', 'head pain', 'head ache', 'migraine', 'head hurts', 'head is hurting'],
        'high_fever': ['fever', 'high fever', 'temperature', 'hot', 'burning up', 'feverish'],
        'stomach_pain': ['stomach pain', 'stomach ache', 'tummy pain', 'belly pain', 'stomach hurts',
                         'stomach is hurting'],
        'vomiting': ['vomiting', 'vomit', 'throwing up', 'puke', 'puking', 'throw up'],
        'cough': ['cough', 'coughing', 'coughs'],
        'breathlessness': ['breathlessness', 'breathless', 'shortness of breath', 'cant breathe',
                           'breathing difficulty', 'hard to breathe', 'difficulty breathing', 'breathing problem'],
        'dizziness': ['dizziness', 'dizzy', 'lightheaded', 'vertigo', 'feeling dizzy'],
        'nausea': ['nausea', 'nauseous', 'feel sick', 'queasy', 'feeling sick'],
        'chest_pain': ['chest pain', 'chest ache', 'chest hurts', 'pain in chest'],
        'joint_pain': ['joint pain', 'joints hurt', 'joint ache', 'arthritis pain'],
        'back_pain': ['back pain', 'backache', 'back ache', 'back hurts'],
        'abdominal_pain': ['abdominal pain', 'belly pain', 'abdomen hurts'],
        'diarrhoea': ['diarrhea', 'diarrhoea', 'loose motion', 'loose stool', 'watery stool', 'loose stools'],
        'constipation': ['constipation', 'constipated', 'cant poop', 'hard stool'],
        'acidity': ['acidity', 'acid', 'heartburn', 'acid reflux', 'burning sensation'],
        'throat_irritation': ['throat irritation', 'sore throat', 'throat pain', 'scratchy throat', 'throat hurts'],
        'runny_nose': ['runny nose', 'running nose', 'nasal discharge', 'dripping nose'],
        'congestion': ['congestion', 'blocked nose', 'stuffy nose', 'nose blocked'],
        'itching': ['itching', 'itch', 'itchy', 'scratchy'],
        'skin_rash': ['rash', 'skin rash', 'spots', 'red spots'],
        'continuous_sneezing': ['sneezing', 'sneeze', 'sneezes'],
        'shivering': ['shivering', 'shiver', 'shaking', 'trembling'],
        'chills': ['chills', 'cold', 'feeling cold', 'freezing'],
        'fatigue': ['fatigue', 'tired', 'exhausted', 'weakness', 'weak', 'feeling weak'],
        'weight_loss': ['weight loss', 'losing weight', 'lost weight'],
        'restlessness': ['restlessness', 'restless', 'anxious', 'agitated'],
        'sweating': ['sweating', 'sweat', 'perspiration', 'perspiring'],
        'dehydration': ['dehydration', 'dehydrated', 'dry mouth'],
        'indigestion': ['indigestion', 'digestion problem', 'upset stomach'],
        'yellowish_skin': ['yellow skin', 'yellowish skin', 'jaundice'],
        'dark_urine': ['dark urine', 'yellow urine', 'orange urine'],
        'loss_of_appetite': ['loss of appetite', 'no appetite', 'dont want to eat', 'cant eat', 'not hungry'],
        'mild_fever': ['mild fever', 'slight fever', 'low fever'],
        'muscle_weakness': ['muscle weakness', 'weak muscles'],
        'fast_heart_rate': ['fast heart', 'heart racing', 'palpitations', 'rapid heartbeat'],
        'neck_pain': ['neck pain', 'stiff neck', 'neck ache'],
        'knee_pain': ['knee pain', 'knee ache', 'knee hurts'],
        'hip_joint_pain': ['hip pain', 'hip joint pain', 'hip hurts'],
        'muscle_pain': ['muscle pain', 'body pain', 'muscle ache', 'body ache'],
        'depression': ['depression', 'depressed', 'sad', 'hopeless'],
        'irritability': ['irritable', 'irritability', 'angry', 'mood swings'],
        'red_spots_over_body': ['red spots', 'spots on body', 'red patches'],
        'belly_pain': ['belly pain', 'tummy ache'],
        'anxiety': ['anxiety', 'anxious', 'worried', 'panic', 'nervous'],
        'lethargy': ['lethargy', 'lethargic', 'sluggish', 'drowsy'],
        'pain_behind_the_eyes': ['pain behind eyes', 'eye pain', 'eyes hurt'],
        'phlegm': ['phlegm', 'mucus', 'sputum'],
        'redness_of_eyes': ['red eyes', 'bloodshot eyes', 'eyes red'],
        'sinus_pressure': ['sinus pressure', 'sinus pain', 'sinusitis'],
        'cramps': ['cramps', 'cramping', 'muscle cramps', 'period cramps', 'menstrual cramps'],
        'loss_of_smell': ['loss of smell', 'cant smell', 'no smell'],
        'bladder_discomfort': ['bladder discomfort', 'bladder pain'],
        'abnormal_menstruation': ['period pain', 'menstrual pain', 'painful periods', 'heavy periods', 'period cramps'],
        'yellowing_of_eyes': ['yellowing of eyes', 'yellow eyes', 'jaundiced eyes', 'eyes yellow'],
        'swelling_of_stomach': ['swelling of stomach', 'stomach swelling', 'belly swelling', 'abdominal swelling',
                                'bloated stomach'],
        'fluid_overload': ['fluid overload', 'water retention', 'swelling all over', 'edema'],
        'blood_in_sputum': ['blood in sputum', 'coughing blood', 'blood in cough'],
        'malaise': ['malaise', 'general discomfort', 'not feeling well', 'overall illness'],
    }

    detected_symptoms = []

    for symptom, keywords in symptom_keywords.items():
        for keyword in keywords:
            if keyword in text:
                if symptom not in detected_symptoms:
                    detected_symptoms.append(symptom)
                break

    return detected_symptoms


def predict_disease(symptoms, gender=None):
    """
    Predict multiple possible diseases based on symptoms
    Returns a list of possible diseases ranked by probability
    """
    symptom_disease_map = {
        'Gastroenteritis': {
            'symptoms': ['stomach_pain', 'vomiting', 'diarrhoea', 'nausea', 'abdominal_pain', 'swelling_of_stomach'],
            'score_needed': 2
        },
        'GERD': {
            'symptoms': ['stomach_pain', 'acidity', 'vomiting', 'chest_pain', 'throat_irritation'],
            'score_needed': 2
        },
        'Hepatitis': {
            'symptoms': ['yellowish_skin', 'dark_urine', 'yellowing_of_eyes', 'abdominal_pain', 'nausea', 'fatigue'],
            'score_needed': 2
        },
        'Jaundice': {
            'symptoms': ['yellowish_skin', 'yellowing_of_eyes', 'dark_urine', 'swelling_of_stomach'],
            'score_needed': 2
        },
        'Liver Disease': {
            'symptoms': ['yellowish_skin', 'yellowing_of_eyes', 'swelling_of_stomach', 'fluid_overload', 'fatigue'],
            'score_needed': 2
        },
        'Pneumonia': {
            'symptoms': ['cough', 'high_fever', 'breathlessness', 'chest_pain', 'fatigue'],
            'score_needed': 2
        },
        'Asthma Attack': {
            'symptoms': ['breathlessness', 'cough', 'chest_pain'],
            'score_needed': 1
        },
        'Anxiety Attack': {
            'symptoms': ['breathlessness', 'chest_pain', 'dizziness', 'fast_heart_rate', 'sweating', 'anxiety'],
            'score_needed': 2
        },
        'Panic Attack': {
            'symptoms': ['breathlessness', 'fast_heart_rate', 'chest_pain', 'sweating', 'anxiety'],
            'score_needed': 2
        },
        'Menstrual Pain': {
            'symptoms': ['stomach_pain', 'abdominal_pain', 'back_pain', 'cramps', 'nausea', 'fatigue',
                         'abnormal_menstruation'],
            'score_needed': 2,
            'gender_specific': 'female'
        },
        'Endometriosis': {
            'symptoms': ['stomach_pain', 'abdominal_pain', 'back_pain', 'abnormal_menstruation'],
            'score_needed': 2,
            'gender_specific': 'female'
        },
        'Migraine': {
            'symptoms': ['headache', 'nausea', 'dizziness', 'vomiting'],
            'score_needed': 2
        },
        'Arthritis': {
            'symptoms': ['joint_pain', 'neck_pain', 'knee_pain', 'stiff_neck', 'swelling_joints'],
            'score_needed': 2
        },
        'Urinary Tract Infection': {
            'symptoms': ['bladder_discomfort', 'stomach_pain', 'back_pain', 'burning_micturition'],
            'score_needed': 2
        },
        'Food Poisoning': {
            'symptoms': ['stomach_pain', 'vomiting', 'diarrhoea', 'nausea', 'cramps'],
            'score_needed': 3
        },
        'Heart Attack': {
            'symptoms': ['chest_pain', 'breathlessness', 'fast_heart_rate', 'sweating', 'nausea'],
            'score_needed': 3
        },
        'Malaria': {
            'symptoms': ['high_fever', 'chills', 'sweating', 'headache', 'nausea', 'vomiting'],
            'score_needed': 3
        },
        'Bronchial Asthma': {
            'symptoms': ['breathlessness', 'cough', 'chest_pain'],
            'score_needed': 2
        },
        'Tuberculosis': {
            'symptoms': ['cough', 'high_fever', 'blood_in_sputum', 'chest_pain', 'fatigue', 'weight_loss'],
            'score_needed': 2
        },
        'Typhoid': {
            'symptoms': ['high_fever', 'headache', 'stomach_pain', 'vomiting', 'malaise'],
            'score_needed': 2
        },
        'Dengue': {
            'symptoms': ['high_fever', 'joint_pain', 'muscle_pain', 'headache', 'redness_of_eyes'],
            'score_needed': 2
        },
        'Chicken Pox': {
            'symptoms': ['high_fever', 'skin_rash', 'itching', 'red_spots_over_body'],
            'score_needed': 2
        },
        'Acne': {
            'symptoms': ['pus_filled_pimples', 'skin_rash', 'itching'],
            'score_needed': 2
        },
        'Psoriasis': {
            'symptoms': ['skin_rash', 'itching', 'scurring', 'skin_peeling'],
            'score_needed': 2
        },
    }

    symptoms_set = set(symptoms)
    disease_scores = []

    # Calculate scores for each disease
    for disease, data in symptom_disease_map.items():
        # Check gender-specific conditions
        if 'gender_specific' in data:
            if not gender or gender.lower() != data['gender_specific']:
                continue

        # Count matching symptoms
        matching = len(symptoms_set.intersection(set(data['symptoms'])))

        if matching >= data['score_needed']:
            # Calculate confidence percentage
            confidence = (matching / len(data['symptoms'])) * 100
            disease_scores.append({
                'disease': disease,
                'matches': matching,
                'confidence': round(confidence, 1)
            })

    # Sort by number of matches (descending)
    disease_scores.sort(key=lambda x: x['matches'], reverse=True)

    # Return top 3 possible diseases
    if disease_scores:
        return disease_scores[:3]
    else:
        # Return "Unable to Determine" instead of Common Cold
        return [{'disease': 'Unable to Determine', 'matches': 0, 'confidence': 0}]


def generate_multiple_disease_message(disease_list):
    """Generate message for multiple possible diseases"""
    if not disease_list or disease_list[0]['disease'] == 'Unable to Determine':
        return "The symptoms you provided don't match our common disease database. Please consult a healthcare professional for proper diagnosis."

    if len(disease_list) == 1:
        disease = disease_list[0]['disease']
        return f"Based on your symptoms, you might be experiencing {disease}. This is a common condition that can be effectively managed with proper care."

    # Multiple diseases
    diseases = [d['disease'] for d in disease_list]

    if len(diseases) == 2:
        disease_str = f"{diseases[0]} or {diseases[1]}"
    else:
        disease_str = ", ".join(diseases[:-1]) + f", or {diseases[-1]}"

    return f"Based on your symptoms, you might be experiencing {disease_str}. These conditions share similar symptoms. A medical professional can provide an accurate diagnosis through proper examination."


@app.route('/')
def home():
    return render_template('index.html', symptoms=symptoms_list)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get patient details
        name = request.form.get('name', 'Patient')
        age = request.form.get('age', 'Not specified')
        gender = request.form.get('gender', 'Not specified')

        # Get selected symptoms
        selected_symptoms = request.form.getlist('symptoms')

        # Get text-based symptoms (NLP)
        symptom_text = request.form.get('symptom_text', '').strip()

        # Extract symptoms from text
        text_symptoms = []
        if symptom_text:
            text_symptoms = extract_symptoms_from_text(symptom_text)

        # Combine symptoms
        all_symptoms = list(set(selected_symptoms + text_symptoms))

        if not all_symptoms:
            return render_template('result.html',
                                   error="Please either type your symptoms or select them from the list.",
                                   name=name)

        # Predict multiple possible diseases
        predicted_diseases = predict_disease(all_symptoms, gender)

        # Get specialists for all diseases
        specialists = []
        for pred in predicted_diseases:
            if pred['disease'] != 'Unable to Determine':
                specialist = disease_specialist_mapping.get(pred['disease'], 'General Physician')
                if specialist not in specialists:
                    specialists.append(specialist)

        if not specialists:
            specialists = ['General Physician']

        return render_template('result.html',
                               name=name,
                               age=age,
                               gender=gender,
                               symptoms=all_symptoms,
                               text_input=symptom_text,
                               extracted_count=len(text_symptoms),
                               predicted_diseases=predicted_diseases,
                               specialists=specialists,
                               message=generate_multiple_disease_message(predicted_diseases))

    except Exception as e:
        return render_template('result.html', error=f"Error: {str(e)}")


if __name__ == '__main__':
    app.run(debug=True)
