import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- Model and Preprocessor Loading ---

@st.cache_resource
def load_assets():
    try:
        # Load the model
        with open('cricket_predictor_model.pkl', 'rb') as f:
            model = pickle.load(f)
            
        # Load the preprocessor (ColumnTransformer)
        with open('model_preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)

        # Extract categories from the preprocessor for the UI
        # The first transformer ('cat') is the OneHotEncoder
        ohe_categories = preprocessor.named_transformers_['cat'].categories_
        
        venue_list = ohe_categories[0].tolist()
        team_list_1 = ohe_categories[1].tolist()
        team_list_2 = ohe_categories[2].tolist()
        
        # Combine all team lists and sort unique teams
        teams = sorted(list(set(team_list_1 + team_list_2)))
        venues = sorted(venue_list)
        
        return model, preprocessor, teams, venues
        
    except FileNotFoundError:
        st.error("Error: Model files (cricket_predictor_model.pkl or model_preprocessor.pkl) not found.")
        st.info("Please run `python ml.py` first to train the model and generate the required PKL files.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model components: {e}")
        st.stop()

# --- Metrics Loading ---
@st.cache_resource
def load_metrics():
    try:
        with open('model_metrics.pkl', 'rb') as f:
            metrics = pickle.load(f)
        return metrics
    except FileNotFoundError:
        st.error("Error: Model metrics file (model_metrics.pkl) not found.")
        st.info("Please run `python ml.py` first to generate the metrics file.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model metrics: {e}")
        st.stop()

model, preprocessor, teams, venues = load_assets()
ml_metrics = load_metrics() # Dynamic loading of metrics!


# --- Custom Functions for Live Prediction  ---

def make_prediction(batting_team, bowling_team, target_score, current_score, wickets_down, overs_completed, venue):
    
    
    # Convert overs_completed (e.g., 10.1) to total balls faced (10 * 6 + 1 = 61)
    total_balls_faced = int(overs_completed) * 6 + int(round((overs_completed - int(overs_completed)) * 10))
    
    balls_remaining = 120 - total_balls_faced
    balls_remaining = max(0, balls_remaining) # Ensure no negative balls

    runs_to_get = target_score - current_score
    innings_wickets = wickets_down
    
    overs_faced = 120 - balls_remaining
    
    # Calculate Current Run Rate (CRR)
    current_run_rate = (current_score / overs_faced) * 6 if overs_faced > 0 else 0
    
    # Calculate Required Run Rate (RRR)
    req_run_rate = (runs_to_get / balls_remaining) * 6 if balls_remaining > 0 else 0

    # 2. Create input DataFrame for prediction
    # Feature order MUST match the features used in ml.py: 
    # ['Venue', 'Bat First', 'Bat Second', 'Runs to Get', 'Balls Remaining', 'Innings Wickets', 'current_run_rate', 'req_run_rate']
    
    input_data = pd.DataFrame({
        'Venue': [venue],
        'Bat First': [bowling_team],
        'Bat Second': [batting_team],
        'Runs to Get': [runs_to_get],
        'Balls Remaining': [balls_remaining],
        'Innings Wickets': [innings_wickets],
        'current_run_rate': [current_run_rate],
        'req_run_rate': [req_run_rate]
    })

    # 3. Preprocess the input data using the ColumnTransformer
    processed_input = preprocessor.transform(input_data)
    
    # 4. Make Prediction (model is the Logistic Regression estimator)
    # predict_proba returns [[P(Fail), P(Success)]]
    prediction_proba = model.predict_proba(processed_input)[0][1]
    
    # Return probability, runs_needed, CRR, RRR
    return prediction_proba, runs_to_get, current_run_rate, req_run_rate


# --- Streamlit Application Layout ---

st.set_page_config(page_title="Live Cricket Winner Predictor", layout="wide")

st.title("🏏 Live Cricket Winner Predictor")
st.markdown("A Machine Learning project for the Introduction to Data Science course.")

# --- Introduction Section ---
st.header("1. Introduction")
st.write("""
This application predicts the live winning probability of the **chasing team** in a T20 cricket match. 
It uses a **Logistic Regression** model trained on ball-by-ball data, considering the dynamic match situation (score, wickets, overs) and contextual factors (teams, venue).
""")

# --- Model Section ---
st.header("2. Model Section: Logistic Regression Predictor")
st.write("""
**Model Used:** Logistic Regression (chosen for its speed, simplicity, and natural output of win probability).
**Preprocessing:** Data was filtered to the 2nd innings. Features like **Required Run Rate (RRR)** and **Current Run Rate (CRR)** were engineered. Categorical variables (Teams, Venue) were One-Hot Encoded using a ColumnTransformer, and all features were passed to the model.
""")

st.subheader("Model Performance Metrics (Test Set)")
metrics_html = f'''
<div style="display: flex; justify-content: space-around; background-color:#black; padding: 15px; border-radius: 10px;">
    <div style="text-align: center;">
        <h3 style="margin-bottom: 0px; color: #1f77b4;">Accuracy</h3>
        <p style="font-size: 24px; font-weight: bold;">{ml_metrics['Accuracy']:.4f}</p>
    </div>
    <div style="text-align: center;">
        <h3 style="margin-bottom: 0px; color: #ff7f0e;">Precision</h3>
        <p style="font-size: 24px; font-weight: bold;">{ml_metrics['Precision']:.4f}</p>
    </div>
    <div style="text-align: center;">
        <h3 style="margin-bottom: 0px; color: #2ca02c;">F1-Score</h3>
        <p style="font-size: 24px; font-weight: bold;">{ml_metrics['F1-Score']:.4f}</p>
    </div>
    <div style="text-align: center;">
        <h3 style="margin-bottom: 0px; color: #d62728;">ROC-AUC</h3>
        <p style="font-size: 24px; font-weight: bold;">{ml_metrics['ROC-AUC']:.4f}</p>
    </div>
</div>
'''
st.markdown(metrics_html, unsafe_allow_html=True)


st.subheader("Live Prediction at Runtime")
st.markdown("Adjust the match parameters below to see the predicted winning probability for the **Batting Team**.")

# --- Runtime Prediction Interface ---
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Context")
    # Setting default indices safely
    default_batting_index = teams.index('India') if 'India' in teams else 0
    default_bowling_index = teams.index('Pakistan') if 'Pakistan' in teams else 1
    default_venue_index = venues.index('SuperSport Park') if 'SuperSport Park' in venues else 0
    
    batting_team = st.selectbox("Batting Team (Chasing)", teams, index=default_batting_index)
    bowling_team = st.selectbox("Bowling Team (Fielding)", teams, index=default_bowling_index)
    venue = st.selectbox("Venue", venues, index=default_venue_index)

    # Simple validation
    if batting_team == bowling_team:
        st.error("Batting and Bowling teams cannot be the same.")
        st.stop()

with col2:
    st.markdown("### First Innings Score")
    target_score = st.number_input("Target Score (1st Innings Score + 1)", min_value=50, max_value=300, value=170, step=1)
    
with col3:
    st.markdown("### Current Match State")
    current_score = st.number_input("Current Score", min_value=0, max_value=target_score, value=80, step=1)
    wickets_down = st.slider("Wickets Down", min_value=0, max_value=10, value=3)
    overs = st.number_input("Overs Completed (0-19)", min_value=0, max_value=19, value=10, step=1)
    balls = st.slider("Balls of Current Over (0-5)", min_value=0, max_value=5, value=0, step=1)
    
   
# Prediction Button
if st.button("Predict Winning Probability", type="primary"):
    
   # Calculate the Over.Ball decimal format needed by the prediction function
    overs_completed = overs + (balls / 10)
    
    runs_needed = target_score - current_score
    # Calculate balls left based on over.ball format
    total_balls_faced = int(overs_completed) * 6 + int(round((overs_completed - int(overs_completed)) * 10))
    balls_left = max(0, 120 - total_balls_faced)
    
    if runs_needed <= 0 and balls_left >= 0:
        st.balloons()
        st.success(f"**{batting_team} wins!** (Target {target_score}, Current Score {current_score})")
    elif runs_needed > 0 and balls_left == 0:
        st.error(f"**{bowling_team} wins!** (Match over, {batting_team} scored {current_score} against Target {target_score})")
    else:
        # Perform prediction
        win_prob, _, crr, rrr = make_prediction(
            batting_team, bowling_team, target_score, current_score, wickets_down, overs_completed, venue
        )
        
        prob_percent = win_prob * 100
        loss_prob_percent = 100 - prob_percent
        
        st.subheader(f"Prediction Result for {batting_team}")
        
        # Color coding for probability
        if prob_percent > 60:
            st.balloons()
            st.success(f"**HIGH CHANCE!** The model predicts {batting_team} is likely to win.")
        elif prob_percent > 40:
            st.info(f"**EVENLY MATCHED!** The match is closely contested.")
        else:
            st.warning(f"**LOW CHANCE!** The model predicts {bowling_team} is likely to win.")

        st.progress(prob_percent/100)
        
        
        result_cols = st.columns(2)
        with result_cols[0]:
            st.metric(label=f"{batting_team} Win Probability", value=f"{prob_percent:.2f} %")
            st.metric(label="Runs Needed", value=runs_needed)
            st.metric(label="Balls Remaining", value=balls_left)

        with result_cols[1]:
            st.metric(label=f"{bowling_team} Win Probability", value=f"{loss_prob_percent:.2f} %")
            st.metric(label="Current Run Rate (CRR)", value=f"{crr:.2f}")
            st.metric(label="Required Run Rate (RRR)", value=f"{rrr:.2f}")
# --- EDA Insights Section (Moved after Model/Prediction) ---
st.header("3. Exploratory Data Analysis (EDA) Insights")
st.write("Key insights derived from the dataset:")

col_eda1, col_eda2, col_eda3 = st.columns(3)

with col_eda1:
    st.image("eda_chase_outcome.png", caption="Match Outcomes (Successful Chase)")
    st.write("Distribution of successful vs. failed run chases.")
with col_eda2:
    st.image("eda_venue_avg_score.png", caption="Top 10 Venues by Average 1st Innings Score")
    st.write("Identifies high and low scoring grounds, which impacts required run rate.")
with col_eda3:
    st.image("eda_correlation_heatmap.png", caption="Correlation Matrix of 2nd Innings Features")
    st.write("The strongest predictors for 'Chased Successfully' are **Runs to Get** (negative correlation) and **Balls Remaining** (positive correlation).")

st.image("eda_win_prob_trend.png", caption="Win Probability Trend vs. Overs Completed")
st.write("**Trend Insight:** The chasing team's average win probability generally sees key fluctuations during the powerplay and death overs, heavily influenced by wickets taken and run rate control.")


# --- Conclusion Section (Moved after EDA) ---
st.header("4. Conclusion")
st.write(f"""
The Logistic Regression model, leveraging engineered features like RRR and CRR, successfully classifies the likely winner of a T20 match with high accuracy (around **{ml_metrics['Accuracy']:.4f}**).
The project demonstrates the end-to-end data science process:
- **EDA** revealed the criticality of score, wickets, and overs.
- **Preprocessing** transformed ball-by-ball data into live-game states.
- **ML Model** provides a probabilistic prediction.
- **Streamlit App** makes the model interactive for runtime predictions.
""")

st.markdown("---")
st.caption("Zoha Anjum | IDS Project | Live Winner Predictor in Cricket")
