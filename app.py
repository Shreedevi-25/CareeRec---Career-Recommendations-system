import streamlit as st
import requests
import json
from typing import List, Dict, Any

# --- Page Configuration ---
st.set_page_config(
    page_title="Career Recommender",
    page_icon="🚀",
    layout="wide"
)

# --- API URL ---
API_URL = "http://127.0.0.1:8000/recommend"

# --- Skill Lists ---
TECH_SKILLS = [
    'Python', 'Java', 'C++', 'C#', '.NET', 'JavaScript', 'TypeScript',
    'React', 'Angular', 'Vue.js', 'Node.js', 'Express.js',
    'HTML', 'CSS', 'SQL', 'MySQL', 'PostgreSQL', 'MongoDB', 'Redis',
    'Pandas', 'NumPy', 'Scikit-learn', 'TensorFlow', 'PyTorch',
    'Machine Learning', 'Data Analysis', 'Data Visualization', 'Deep Learning',
    'AWS', 'Azure', 'Google Cloud (GCP)', 'Docker', 'Kubernetes',
    'Git', 'CI/CD', 'Agile', 'Scrum', 'JIRA',
    'UI Design', 'UX Design', 'Figma', 'Adobe XD',
    'SEO', 'Digital Marketing', 'Content Writing'
]
TECH_SKILLS.sort()

SOFT_SKILLS_INTERESTS = [
    'Communication', 'Teamwork', 'Problem Solving', 'Leadership',
    'Project Management', 'Time Management', 'Creativity',
    'Public Speaking', 'Writing', 'Negotiation', 'Critical Thinking',
    'Finance', 'Marketing', 'Startups', 'Gaming', 'Art & Design'
]
SOFT_SKILLS_INTERESTS.sort()

# --- Location Dictionary ---
LOCATIONS = {
    "India": "in",
    "USA": "us",
    "United Kingdom": "gb",
    "Canada": "ca",
    "Australia": "au",
    "Singapore": "sg",
    "Germany": "de"
}

# # --- Sidebar ---
# st.sidebar.title("🚀 About This Project")
# st.sidebar.info(
#     """
#     **Project:** Real-Time Career Recommendation System
#     **By:** [Your Name Here]
    
#     This app uses an ensemble of machine learning models to predict your ideal career based on your skills.
    
#     It then fetches live, relevant job postings from the Adzuna API.
#     """
# )
# st.sidebar.title("Contact")
# st.sidebar.markdown(
#     """
#     [LinkedIn](https://www.linkedin.com/)
#     [GitHub](https://github.com/)
#     """
# )

# --- Main Application ---
st.title("Career Recommendation System")
st.write("Fill in your details below, and our AI will recommend the best career path for you.")

# --- Use columns for a cleaner layout ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Tell us about yourself:")
    # --- Input Form ---
    with st.form("user_input_form"):
        
        location_name = st.selectbox(
            "Select Job Location:",
            options=LOCATIONS.keys(),
            index=0 
        )

        exp_level_options = ('Fresher', 'Entry Level', 'Mid-Level', 'Experienced', 'Senior')
        exp_level = st.selectbox(
            "Experience Level:",
            exp_level_options
        )

        years_exp = st.slider(
            "Years of Experience:",
            min_value=0.0,
            max_value=20.0,
            value=1.0, 
            step=0.5
        )

        with st.expander("Select Your Skills"):
            selected_tech_skills = st.multiselect(
                "Technical Skills:",
                options=TECH_SKILLS,
                default=['Python', 'SQL', 'Data Analysis']
            )
            
            selected_soft_skills = st.multiselect(
                "Soft Skills & Interests:",
                options=SOFT_SKILLS_INTERESTS,
                default=['Communication', 'Problem Solving']
            )

        # Submit Button
        submitted = st.form_submit_button("Get Recommendations")

# --- Results Area ---
with col2:
    st.subheader("Your Results:")

    # --- Form Submission Logic ---
    if submitted:
        
        all_skills_list = selected_tech_skills + selected_soft_skills
        final_master_text = " ".join(all_skills_list)

        with st.spinner("Finding your career AND scraping live job listings..."):
            
            payload = {
                "ExperienceLevel": exp_level,
                "YearsOfExperience": years_exp,
                "master_text": final_master_text,
                "location_code": LOCATIONS[location_name] 
            }
            
            # --- THIS IS THE BLOCK TO CHECK ---
            # The 'try' and the 'except' blocks below
            # MUST have the same indentation level.
            try:
                response = requests.post(API_URL, json=payload, timeout=180) 
                
                if response.status_code == 200:
                    data = response.json()
                    st.success("Here are your top recommendations!")

                    tab1, tab2 = st.tabs(["🏆 Your Top 3 Recommendations", "📄 Live Job Listings"])

                    with tab1:
                        st.header("Your Top Career Matches:")
                        icons = ["🏆", "🥈", "🥉"] 
                        
                        for rec, icon in zip(data['recommendations'], icons):
                            title = rec['title']
                            st.markdown(f"## {icon} **{title}**")
                            st.divider()

                    with tab2:
                        st.header(f"Live Job Listings in {location_name}")
                        for job_group in data['job_listings']:
                            st.subheader(f"Listings for: {job_group['recommended_title']}")
                            
                            if not job_group['job_listings']:
                                st.write("No live job listings found for this role.")
                                continue
                            
                            # --- JOB CARDS WITH ICONS ---
                            for job in job_group['job_listings']:
                                with st.container(border=True):
                                    st.markdown(f"**{job['title']}**")
                                    st.markdown(f"🏢 **Company:** {job['company']}")
                                    st.markdown(f"📍 **Location:** {job['location']}")
                                    st.link_button("View Job", job['link'])
                            
                        st.info("Job listings are provided in real-time by the Adzuna API.")

                else:
                    st.error("Error from API: " + response.text)

            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the API server. Is it running?")
            except requests.exceptions.Timeout:
                st.error("The request timed out. The Adzuna API is likely taking too long.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
            # --- END OF THE TRY/EXCEPT BLOCK ---

    else:
        st.info("Please fill out the form on the left to see your results.")