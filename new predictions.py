import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import streamlit as st


def load_ipl_data():
    """Load the IPL dataset from CSV"""
    try:
        # Read the CSV file
        df = pd.read_csv('deliveries.csv')  # Make sure the file is in your directory
        print(f"✅ Dataset loaded successfully!")
        print(f"📊 Shape: {df.shape}")
        print(f"📋 Columns: {df.columns.tolist()}")
        return df
    except FileNotFoundError:
        st.error("❌ File 'deliveries.csv' not found! Please make sure it's in your directory.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading file: {e}")
        return None

class IPLDataPreprocessor:
    def __init__(self, df):
        self.df = df
        self.match_data = None
        
    def preprocess_data(self):
        """Main preprocessing pipeline"""
        print("🔄 Preprocessing data...")
        # Create match summaries
        self._create_match_summaries()
        # Engineer features
        self._engineer_features()
        # Handle missing values
        self._handle_missing_values()
        print(f"✅ Preprocessing complete! {len(self.match_data)} matches processed")
        return self.match_data
    
    def _create_match_summaries(self):
        """Create match-level summaries from ball-by-ball data"""
        print("📈 Creating match summaries...")
        match_summaries = []
        
        for match_id in self.df['match_id'].unique():
            match_df = self.df[self.df['match_id'] == match_id]
            
            # Get basic match info
            innings_1 = match_df[match_df['inning'] == 1]
            innings_2 = match_df[match_df['inning'] == 2]
            
            if len(innings_1) == 0 or len(innings_2) == 0:
                continue
                
            # Team names
            team1 = innings_1['batting_team'].iloc[0]
            team2 = innings_1['bowling_team'].iloc[0]
            
            # Innings 1 summary
            runs_1 = innings_1['total_runs'].sum()
            wickets_1 = innings_1['is_wicket'].sum()
            extras_1 = innings_1['extra_runs'].sum()
            
            # Innings 2 summary
            runs_2 = innings_2['total_runs'].sum()
            wickets_2 = innings_2['is_wicket'].sum()
            extras_2 = innings_2['extra_runs'].sum()
            
            # Determine winner
            if runs_1 > runs_2:
                winner = team1
            elif runs_2 > runs_1:
                winner = team2
            else:
                winner = 'Draw'
            
            match_summary = {
                'match_id': match_id,
                'team1': team1,
                'team2': team2,
                'runs_1': runs_1,
                'wickets_1': wickets_1,
                'extras_1': extras_1,
                'runs_2': runs_2,
                'wickets_2': wickets_2,
                'extras_2': extras_2,
                'winner': winner,
                'total_runs': runs_1 + runs_2
            }
            match_summaries.append(match_summary)
        
        self.match_data = pd.DataFrame(match_summaries)
    
    def _engineer_features(self):
        """Create advanced features for prediction"""
        print("⚙️ Engineering features...")
        # Basic match features
        self.match_data['run_difference'] = abs(self.match_data['runs_1'] - self.match_data['runs_2'])
        self.match_data['total_wickets'] = self.match_data['wickets_1'] + self.match_data['wickets_2']
        self.match_data['total_extras'] = self.match_data['extras_1'] + self.match_data['extras_2']
        
        # Team strength features (historical performance)
        self._add_team_strength_features()
        
        # Match context features
        self._add_match_context_features()
        
        # Target variable
        self.match_data['target'] = (self.match_data['winner'] == self.match_data['team1']).astype(int)
    
    def _add_team_strength_features(self):
        """Add features based on team historical performance"""
        team_wins = {}
        team_matches = {}
        
        # Calculate win percentages
        for _, row in self.match_data.iterrows():
            for team in [row['team1'], row['team2']]:
                team_matches[team] = team_matches.get(team, 0) + 1
                if row['winner'] == team:
                    team_wins[team] = team_wins.get(team, 0) + 1
        
        # Add features
        self.match_data['team1_win_pct'] = self.match_data['team1'].map(
            lambda x: team_wins.get(x, 0) / team_matches.get(x, 1)
        )
        self.match_data['team2_win_pct'] = self.match_data['team2'].map(
            lambda x: team_wins.get(x, 0) / team_matches.get(x, 1)
        )
        self.match_data['win_pct_diff'] = self.match_data['team1_win_pct'] - self.match_data['team2_win_pct']
    
    def _add_match_context_features(self):
        """Add match-specific context features"""
        # High scoring match indicator
        self.match_data['high_scoring'] = (self.match_data['total_runs'] > 180).astype(int)
        
        # Close match indicator
        self.match_data['close_match'] = (self.match_data['run_difference'] < 10).astype(int)
        
        # Wicket-heavy match
        self.match_data['wicket_heavy'] = (self.match_data['total_wickets'] > 12).astype(int)
    
    def _handle_missing_values(self):
        """Handle any missing values"""
        self.match_data.fillna(0, inplace=True)

class IPLVisualizer:
    def __init__(self, match_data):
        self.match_data = match_data
    
    def plot_team_performance(self):
        """Plot team performance statistics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Team win counts
        team_wins = pd.concat([
            self.match_data[self.match_data['winner'] == self.match_data['team1']]['team1'],
            self.match_data[self.match_data['winner'] == self.match_data['team2']]['team2']
        ]).value_counts()
        
        axes[0,0].bar(team_wins.index, team_wins.values, color='skyblue')
        axes[0,0].set_title('Total Wins by Team', fontsize=14, fontweight='bold')
        axes[0,0].tick_params(axis='x', rotation=45)
        axes[0,0].grid(axis='y', alpha=0.3)
        
        # Run distribution
        axes[0,1].hist(self.match_data['total_runs'], bins=20, alpha=0.7, color='lightcoral')
        axes[0,1].set_title('Distribution of Total Runs per Match', fontsize=14, fontweight='bold')
        axes[0,1].set_xlabel('Total Runs')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].grid(alpha=0.3)
        
        # Win percentage by team
        win_pct_data = []
        for team in self.match_data['team1'].unique():
            wins = len(self.match_data[self.match_data['winner'] == team])
            matches = len(self.match_data[(self.match_data['team1'] == team) | (self.match_data['team2'] == team)])
            win_pct = (wins / matches) * 100 if matches > 0 else 0
            win_pct_data.append({'Team': team, 'Win Percentage': win_pct})
        
        win_pct_df = pd.DataFrame(win_pct_data).sort_values('Win Percentage', ascending=False)
        axes[1,0].bar(win_pct_df['Team'], win_pct_df['Win Percentage'], color='lightgreen')
        axes[1,0].set_title('Win Percentage by Team', fontsize=14, fontweight='bold')
        axes[1,0].tick_params(axis='x', rotation=45)
        axes[1,0].grid(axis='y', alpha=0.3)
        
        # Match outcome types
        outcome_types = ['High Scoring', 'Close Matches', 'Wicket Heavy']
        outcome_counts = [
            self.match_data['high_scoring'].sum(),
            self.match_data['close_match'].sum(),
            self.match_data['wicket_heavy'].sum()
        ]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        axes[1,1].bar(outcome_types, outcome_counts, color=colors)
        axes[1,1].set_title('Match Type Distribution', fontsize=14, fontweight='bold')
        axes[1,1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_feature_importance(self, model, feature_names):
        """Plot feature importance from trained model"""
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        else:
            # For logistic regression, use absolute coefficients
            importance = np.abs(model.coef_[0])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        indices = np.argsort(importance)[::-1]
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(importance)))
        bars = ax.bar(range(len(importance)), importance[indices], color=colors)
        
        ax.set_title('Feature Importance', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(importance)))
        ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        return fig
    
    def plot_correlation_heatmap(self, features):
        """Plot correlation heatmap of features"""
        fig, ax = plt.subplots(figsize=(12, 8))
        correlation_matrix = features.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax, 
                   fmt='.2f', linewidths=0.5)
        ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
        return fig

class IPLPredictor:
    def __init__(self):
        self.models = {}
        self.feature_columns = [
            'runs_1', 'wickets_1', 'extras_1', 'runs_2', 'wickets_2', 'extras_2',
            'run_difference', 'total_wickets', 'total_extras',
            'team1_win_pct', 'team2_win_pct', 'win_pct_diff',
            'high_scoring', 'close_match', 'wicket_heavy'
        ]
    
    def prepare_features(self, match_data):
        """Prepare features for model training"""
        X = match_data[self.feature_columns]
        y = match_data['target']
        return X, y
    
    def train_models(self, X, y):
        """Train multiple models"""
        print("🤖 Training machine learning models...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Model definitions
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        results = {}
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'predictions': y_pred
            }
            
            print(f"✅ {name} Accuracy: {accuracy:.3f}")
        
        return results, X_test, y_test
    
    def predict_match(self, team1, team2, match_data, features_dict):
        """Predict match outcome for new data"""
        # Get team historical data
        team1_data = match_data[(match_data['team1'] == team1) | (match_data['team2'] == team1)]
        team2_data = match_data[(match_data['team1'] == team2) | (match_data['team2'] == team2)]
        
        team1_win_pct = team1_data[team1_data['winner'] == team1].shape[0] / team1_data.shape[0] if team1_data.shape[0] > 0 else 0.5
        team2_win_pct = team2_data[team2_data['winner'] == team2].shape[0] / team2_data.shape[0] if team2_data.shape[0] > 0 else 0.5
        
        # Prepare feature vector
        features = [
            features_dict['runs_1'], features_dict['wickets_1'], features_dict['extras_1'],
            features_dict['runs_2'], features_dict['wickets_2'], features_dict['extras_2'],
            features_dict['run_difference'], features_dict['total_wickets'], features_dict['total_extras'],
            team1_win_pct, team2_win_pct, team1_win_pct - team2_win_pct,
            features_dict['high_scoring'], features_dict['close_match'], features_dict['wicket_heavy']
        ]
        
        predictions = {}
        for name, model in self.models.items():
            pred_proba = model.predict_proba([features])[0]
            pred_class = model.predict([features])[0]

            predictions[name] = {
                'team1_win_prob': pred_proba[1],
                'team2_win_prob': pred_proba[0],
                'predicted_winner': team1 if pred_class == 1 else team2,
                'confidence': pred_proba[1] if pred_class == 1 else pred_proba[0]
            }

        return predictions
    

# Streamlit App
def main():
    st.set_page_config(page_title="IPL Match Predictor", page_icon="🏏", layout="wide")
    
    st.title("🏏 IPL Match Winner Prediction System")
    st.markdown("### Predict IPL match outcomes using machine learning! 🤖")
    
    # Load data
    with st.spinner("Loading IPL data..."):
        df = load_ipl_data()
    
    if df is None:
        st.error("Please make sure 'deliveries.csv' is in your directory!")
        return
    
    # Preprocess data
    with st.spinner("Preprocessing data..."):
        preprocessor = IPLDataPreprocessor(df)
        match_data = preprocessor.preprocess_data()
    
    # Initialize components
    visualizer = IPLVisualizer(match_data)
    predictor = IPLPredictor()
    
    # Train models (only once)
    if 'models_trained' not in st.session_state:
        with st.spinner("Training machine learning models..."):
            X, y = predictor.prepare_features(match_data)
            results, X_test, y_test = predictor.train_models(X, y)
            predictor.models = {name: results[name]['model'] for name in results}
            st.session_state.results = results
            st.session_state.models_trained = True
            st.session_state.best_model = max(results, key=lambda x: results[x]['accuracy'])
            st.session_state.predictor = predictor
    else:
        # Restore predictor from session state
        predictor.models = {name: st.session_state.results[name]['model'] for name in st.session_state.results}
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", [
        "🏠 Data Overview", 
        "📊 Visualizations", 
        "🔮 Match Prediction",
        "🤖 Model Performance"
    ])
    
    if page == "🏠 Data Overview":
        st.header("📊 Dataset Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Match Data Sample")
            st.dataframe(match_data.head(10), use_container_width=True)
            
            st.subheader("📈 Dataset Info")
            st.metric("Total Matches", len(match_data))
            st.metric("Total Teams", len(match_data['team1'].unique()))
            st.metric("Features", len(predictor.feature_columns))
        
        with col2:
            st.subheader("📋 Basic Statistics")
            st.dataframe(match_data[['runs_1', 'runs_2', 'total_runs', 'total_wickets']].describe())
            
            st.subheader("🏆 Team Performance Summary")
            team_stats = []
            for team in match_data['team1'].unique():
                wins = len(match_data[match_data['winner'] == team])
                matches = len(match_data[(match_data['team1'] == team) | (match_data['team2'] == team)])
                win_pct = (wins / matches) * 100
                team_stats.append({
                    'Team': team, 
                    'Matches': matches, 
                    'Wins': wins, 
                    'Win %': f"{win_pct:.1f}%"
                })
            
            team_stats_df = pd.DataFrame(team_stats).sort_values('Win %', ascending=False)
            st.dataframe(team_stats_df, use_container_width=True)
    
    elif page == "📊 Visualizations":
        st.header("📈 Data Visualizations")
        
        tab1, tab2, tab3 = st.tabs(["Team Performance", "Feature Importance", "Correlations"])
        
        with tab1:
            st.subheader("Team Performance Analysis")
            fig = visualizer.plot_team_performance()
            st.pyplot(fig)
            
            # Additional insights
            col1, col2 = st.columns(2)
            with col1:
                avg_runs = match_data['total_runs'].mean()
                st.metric("Average Total Runs", f"{avg_runs:.1f}")
            with col2:
                close_matches = match_data['close_match'].mean() * 100
                st.metric("Close Matches (%)", f"{close_matches:.1f}%")
        
        with tab2:
            st.subheader("Feature Importance")
            if 'best_model' in st.session_state:
                best_model = st.session_state.results[st.session_state.best_model]['model']
                fig = visualizer.plot_feature_importance(best_model, predictor.feature_columns)
                st.pyplot(fig)
            else:
                st.info("Train models first to see feature importance")
        
        with tab3:
            st.subheader("Feature Correlations")
            X, _ = predictor.prepare_features(match_data)
            fig = visualizer.plot_correlation_heatmap(X)
            st.pyplot(fig)
    
    elif page == "🔮 Match Prediction":
        st.header("🔮 Match Prediction")
        st.info("Adjust the match parameters below to predict the winner!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏏 Team 1 Batting")
            team1 = st.selectbox("Select Team 1", sorted(match_data['team1'].unique()), key="team1")
            
            runs_1 = st.slider("Runs Scored", 50, 250, 160, key="runs1")
            wickets_1 = st.slider("Wickets Lost", 0, 10, 6, key="wickets1")
            extras_1 = st.slider("Extras", 0, 30, 8, key="extras1")
            
            st.metric("Team 1 Score", f"{runs_1}/{wickets_1}")
        
        with col2:
            st.subheader("🎯 Team 2 Batting")
            team2 = st.selectbox("Select Team 2", 
                               sorted([t for t in match_data['team1'].unique() if t != team1]), 
                               key="team2")
            
            runs_2 = st.slider("Runs Scored", 50, 250, 150, key="runs2")
            wickets_2 = st.slider("Wickets Lost", 0, 10, 7, key="wickets2")
            extras_2 = st.slider("Extras", 0, 30, 7, key="extras2")
            
            st.metric("Team 2 Score", f"{runs_2}/{wickets_2}")
        
        # Calculate derived features
        run_difference = abs(runs_1 - runs_2)
        total_wickets = wickets_1 + wickets_2
        total_extras = extras_1 + extras_2
        
        # Match context
        st.subheader("🎪 Match Context")
        col1, col2, col3 = st.columns(3)
        with col1:
            high_scoring = st.checkbox("High Scoring Match", value=(runs_1 + runs_2) > 180)
        with col2:
            close_match = st.checkbox("Close Match", value=run_difference < 10)
        with col3:
            wicket_heavy = st.checkbox("Wicket Heavy", value=total_wickets > 12)
        
        if st.button("🎯 Predict Winner", type="primary", use_container_width=True):
            with st.spinner("Analyzing match data..."):
                # Prepare features dictionary
                features_dict = {
                    'runs_1': runs_1, 'wickets_1': wickets_1, 'extras_1': extras_1,
                    'runs_2': runs_2, 'wickets_2': wickets_2, 'extras_2': extras_2,
                    'run_difference': run_difference,
                    'total_wickets': total_wickets,
                    'total_extras': total_extras,
                    'high_scoring': int(high_scoring),
                    'close_match': int(close_match),
                    'wicket_heavy': int(wicket_heavy)
                }
                
                try:
                    # Make prediction
                    predictions = predictor.predict_match(team1, team2, match_data, features_dict)
                    
                    # Store in session state
                    st.session_state.predictions = predictions
                    st.session_state.show_predictions = True
                    st.session_state.pred_team1 = team1
                    st.session_state.pred_team2 = team2
                    
                except Exception as e:
                    st.error(f"Error making prediction: {e}")
                    st.write("Debug info:")
                    st.write(f"Team 1: {team1}, Team 2: {team2}")
                    st.write(f"Features: {features_dict}")
        
        # Display results outside the button block
        if 'show_predictions' in st.session_state and st.session_state.show_predictions:
            predictions = st.session_state.predictions
            pred_team1 = st.session_state.pred_team1
            pred_team2 = st.session_state.pred_team2
            
            st.markdown("---")
            st.subheader("🎊 Prediction Results")
            
            # Create columns for each model
            for model_name, pred in predictions.items():
                st.markdown(f"### {model_name}")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Team 1 win probability
                    team1_prob = float(pred['team1_win_prob'])
                    st.metric(f"{pred_team1} Win Probability", f"{team1_prob*100:.1f}%")
                    st.progress(min(1.0, max(0.0, team1_prob)))
                
                with col2:
                    # Team 2 win probability
                    team2_prob = float(pred['team2_win_prob'])
                    st.metric(f"{pred_team2} Win Probability", f"{team2_prob*100:.1f}%")
                    st.progress(min(1.0, max(0.0, team2_prob)))
                
                with col3:
                    # Predicted winner
                    winner = pred['predicted_winner']
                    confidence = pred['confidence'] * 100
                    winner_emoji = "🏆" if winner == pred_team1 else "🎯"
                    
                    st.metric(
                        "Predicted Winner", 
                        f"{winner_emoji} {winner}"
                    )
                    st.metric("Confidence", f"{confidence:.1f}%")
                
                st.markdown("---")
            
            # Summary
            st.success("✅ Prediction Complete!")
    
    elif page == "🤖 Model Performance":
        st.header("🤖 Model Performance")
        
        if 'results' in st.session_state:
            st.subheader("Model Comparison")
            
            # Create performance comparison
            model_results = []
            for name, result in st.session_state.results.items():
                model_results.append({
                    'Model': name,
                    'Accuracy': f"{result['accuracy']:.3f}",
                    'Accuracy %': f"{result['accuracy']*100:.1f}%"
                })
            
            results_df = pd.DataFrame(model_results)
            st.dataframe(results_df, use_container_width=True)
            
            # Best model info
            best_model_name = st.session_state.best_model
            best_accuracy = st.session_state.results[best_model_name]['accuracy']
            
            st.success(f"🎯 **Best Model**: {best_model_name} (Accuracy: {best_accuracy*100:.1f}%)")
            
            # Feature importance visualization
            st.subheader("Feature Importance - Best Model")
            best_model = st.session_state.results[best_model_name]['model']
            fig = visualizer.plot_feature_importance(best_model, predictor.feature_columns)
            st.pyplot(fig)
            
            # Model insights
            st.subheader("📋 Model Insights")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Models", len(st.session_state.results))
            with col2:
                st.metric("Best Accuracy", f"{best_accuracy*100:.1f}%")
            with col3:
                avg_accuracy = np.mean([r['accuracy'] for r in st.session_state.results.values()])
                st.metric("Average Accuracy", f"{avg_accuracy*100:.1f}%")
        else:
            st.info("Please train the models first in the Match Prediction section!")

if __name__ == "__main__":
    main()
