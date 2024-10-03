import streamlit as st
import pandas as pd
import ast
import nltk

try:
    nltk.load('punkt')
except:
    nltk.download('punkt')
try:
    nltk.load('stopwords')
except:
    nltk.download('stopwords')
try:
    nltk.load('punkt_tab')
except:
    nltk.download('punkt_tab')



from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string



# Job relevance keywords
job_keywords = [
    'data', 'data science', 'machine learning', 'deep learning',
    'predictive analytics', 'statistical', 'modeling', 'analysis',
    'artificial intelligence', 'python', 'sql', 'visualization',
    'big data', 'natural language processing', 'nlp', 'neural networks',
    'analytics', 'clustering', 'classification', 'regression'
]

# Scoring functions
def calculate_proficiency_match_score(candidate_skills, required_skills, weight):
    total_score = 0
    max_possible_score = 5 * len(required_skills)
    
    for skill, required_proficiency in required_skills.items():
        candidate_proficiency = candidate_skills.get(skill, 0)
        if candidate_proficiency >= required_proficiency:
            skill_score = 5
        else:
            skill_score = (candidate_proficiency / required_proficiency) * 5
        skill_score = min(skill_score, 5)
        total_score += skill_score
    
    normalized_score = (total_score / max_possible_score) * 5
    weighted_score = (normalized_score / 5) * weight
    return weighted_score

def calculate_skill_match_score(candidate_skills, required_skills, weight):
    candidate_skill_set = set(candidate_skills.keys())
    required_skill_set = set(required_skills.keys())
    intersection = candidate_skill_set.intersection(required_skill_set)
    
    if len(intersection) == len(required_skill_set):
        score = 5  # Full Match
    elif len(intersection) >= len(required_skill_set) / 2:
        score = 3  # Partial Match
    else:
        score = 1  # Minimal Match
    
    weighted_score = (score / 5) * weight
    return weighted_score

def calculate_experience_score(candidate_experience_years, required_experience_years, job_position, experience_description, weight):
    # Combine job position and experience description
    text = f"{job_position} {experience_description}".lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    # Match keywords
    matched_keywords = set(filtered_tokens).intersection(set(job_keywords))
    
    # Calculate relevance score
    relevance_ratio = len(matched_keywords) / len(set(job_keywords))
    relevance_score = relevance_ratio * 5  # Scale to 0-5
    relevance_score = min(relevance_score, 5)
    
    # Calculate experience difference
    max_difference = 5  # Define the acceptable maximum difference
    difference = abs(candidate_experience_years - required_experience_years)
    difference = min(difference, max_difference)
    
    # Calculate experience match score
    experience_match_score = (max_difference - difference) / max_difference * 5  # Scale to 0-5
    
    # Combine relevance score and experience match score (you can adjust the weights here)
    final_experience_score = (relevance_score + experience_match_score) / 2  # Average the two scores
    
    # Apply weight
    weighted_score = (final_experience_score / 5) * weight
    return weighted_score

def calculate_cultural_fit_score(cultural_fit, weight):
    weighted_score = (cultural_fit / 5) * weight
    return weighted_score

def calculate_video_ai_score(video_ai_score, weight):
    weighted_score = (video_ai_score / 5) * weight
    return weighted_score

def format_skills(skills_dict):
    return ', '.join([f"{skill} ({level})" for skill, level in skills_dict.items()])

def main():
    st.title('Data Science Candidate Ranking Model')

    # Load candidate data
    df = pd.read_csv('candidates.csv')
    df['candidate_skills'] = df['candidate_skills'].apply(ast.literal_eval)

    # Define required skills with required proficiency levels
    st.sidebar.header('Required Skills and Proficiency Levels')
    required_skills = {}
    default_required_skills = {
        'Python': 5,
        'SQL': 4,
        'Machine Learning': 5,
        'Data Visualization': 4,
        'Statistics': 4
    }

    for skill in default_required_skills.keys():
        proficiency = st.sidebar.slider(f'Required proficiency for {skill}', 1, 5, default_required_skills[skill])
        required_skills[skill] = proficiency

    # Required Years of Experience
    st.sidebar.header('Experience Requirements')
    required_experience_years = st.sidebar.number_input('Required Years of Experience', min_value=0, max_value=50, value=3)

    # Define weights
    st.sidebar.header('Weights for Each Criterion')
    total_weight = 0
    weights = {}
    weights['technical_skill_proficiency'] = st.sidebar.number_input('Technical Skill Proficiency (%)', min_value=0, max_value=100, value=50)
    weights['skill_match'] = st.sidebar.number_input('Skill Match (%)', min_value=0, max_value=100, value=15)
    weights['experience'] = st.sidebar.number_input('Experience (%)', min_value=0, max_value=100, value=15)
    weights['cultural_fit'] = st.sidebar.number_input('Cultural Fit (%)', min_value=0, max_value=100, value=10)
    weights['video_ai'] = st.sidebar.number_input('Video AI Assessment (%)', min_value=0, max_value=100, value=10)

    total_weight = sum(weights.values())
    if total_weight != 100:
        st.error('Total weight must sum up to 100%. Please adjust the weights.')
        return

    # Calculate scores for each candidate
    total_scores = []
    for index, row in df.iterrows():
        candidate_skills = row['candidate_skills']
        proficiency_score = calculate_proficiency_match_score(candidate_skills, required_skills, weights['technical_skill_proficiency'])
        skill_match_score = calculate_skill_match_score(candidate_skills, required_skills, weights['skill_match'])
        experience_score = calculate_experience_score(
            row['experience_years'],
            required_experience_years,
            row['job_position'],
            row['experience_description'],
            weights['experience']
        )
        cultural_fit_score = calculate_cultural_fit_score(row['cultural_fit'], weights['cultural_fit'])
        video_ai_score = calculate_video_ai_score(row['video_ai_score'], weights['video_ai'])

        total_score = proficiency_score + skill_match_score + experience_score + cultural_fit_score + video_ai_score

        total_scores.append({
            'candidate_name': row['candidate_name'],
            'proficiency_score': round(proficiency_score, 2),
            'skill_match_score': round(skill_match_score, 2),
            'experience_score': round(experience_score, 2),
            'cultural_fit_score': round(cultural_fit_score, 2),
            'video_ai_score': round(video_ai_score, 2),
            'total_score': round(total_score, 2),
            'candidate_skills': row['candidate_skills'],
            'job_position': row['job_position'],
            'experience_description': row['experience_description'],
            'skills_and_proficiency': format_skills(row['candidate_skills']),
            'years_of_experience': row['experience_years']
        })

    # Create a DataFrame of the results
    results_df = pd.DataFrame(total_scores)
    results_df = results_df.sort_values(by='total_score', ascending=False)

    # Display the results
    st.header('Candidate Rankings')

    # Select columns to display
    display_columns = [
        'candidate_name',
        'total_score',
        'skills_and_proficiency',
        'job_position',
        'experience_description',
        'years_of_experience'
    ]

    # Display the DataFrame
    st.dataframe(results_df[display_columns].reset_index(drop=True))

    # Optionally display detailed scores
    if st.checkbox('Show Detailed Scores'):
        st.subheader('Detailed Scores')
        detailed_columns = [
            'candidate_name',
            'proficiency_score',
            'skill_match_score',
            'experience_score',
            'cultural_fit_score',
            'video_ai_score',
            'total_score',
            'skills_and_proficiency',
            'job_position',
            'experience_description',
            'years_of_experience'
        ]
        st.dataframe(results_df[detailed_columns].reset_index(drop=True))

    # Download results as CSV
    if st.button('Download Results as CSV'):
        # Include all relevant columns in the CSV
        csv_columns = display_columns + [
            'proficiency_score',
            'skill_match_score',
            'experience_score',
            'cultural_fit_score',
            'video_ai_score',
            'candidate_skills'
        ]
        results_csv = results_df[csv_columns].to_csv(index=False)
        st.download_button(
            label='Download CSV',
            data=results_csv,
            file_name='candidate_rankings.csv',
            mime='text/csv'
        )

if __name__ == '__main__':
    main()
