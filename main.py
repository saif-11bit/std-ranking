import streamlit as st
import pandas as pd
import ast

# Scoring functions (from previous section)
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

def calculate_experience_score(experience_years, experience_relevance, weight):
    relevance_scores = {'high': 5, 'medium': 3, 'low': 1}
    base_score = relevance_scores.get(experience_relevance.lower(), 1)
    
    years_factor = min(experience_years / 5, 1)
    experience_score = base_score * years_factor
    experience_score = min(experience_score, 5)
    
    weighted_score = (experience_score / 5) * weight
    return weighted_score

def calculate_cultural_fit_score(cultural_fit, weight):
    weighted_score = (cultural_fit / 5) * weight
    return weighted_score

def calculate_video_ai_score(video_ai_score, weight):
    weighted_score = (video_ai_score / 5) * weight
    return weighted_score

# Main function
def main():
    st.title('Data Science Candidate Ranking Model')
    
    # Load candidate data
    df = pd.read_csv('candidates.csv')
    
    # Convert candidate_skills from string to dictionary
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
        experience_score = calculate_experience_score(row['experience_years'], row['experience_relevance'], weights['experience'])
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
            'total_score': round(total_score, 2)
        })
    
    # Create a DataFrame of the results
    results_df = pd.DataFrame(total_scores)
    results_df = results_df.sort_values(by='total_score', ascending=False)
    
    # Display the results
    st.header('Candidate Rankings')
    st.dataframe(results_df[['candidate_name', 'total_score']].reset_index(drop=True))
    
    # Optionally display detailed scores
    if st.checkbox('Show Detailed Scores'):
        st.subheader('Detailed Scores')
        st.dataframe(results_df.reset_index(drop=True))
    
    # Download results as CSV
    if st.button('Download Results as CSV'):
        results_csv = results_df.to_csv(index=False)
        st.download_button('Download CSV', data=results_csv, file_name='candidate_rankings.csv', mime='text/csv')

if __name__ == '__main__':
    main()
