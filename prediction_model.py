import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, classification_report
import glob
import pickle

def load_data():
    all_files = glob.glob('data/team_stats_*.csv')
    df_list = []
    
    for file in all_files:
        df = pd.read_csv(file)
        year = int(file.split('_')[-1].replace('.csv', ''))
        df['Year'] = year
        df_list.append(df)
    
    combined = pd.concat(df_list, ignore_index=True)
    return combined

def clean_seed_columns(df):
    seed_col = [col for col in df.columns if 'Seed' in col or 'SRS' in col]
    
    if 'Seed' in df.columns:
        df['Seed_Number'] = df['Seed'].str.extract(r'(\d+)').astype(float)
        df['Region'] = df['Seed'].str.extract(r'([WXYZ])')
        
    return df

def load_tournament_data():
    """
    Load tournament game results with seeds
    """
    try:
        results = pd.read_csv('kaggle-data/MNCAATourneyCompactResults.csv')
        seeds = pd.read_csv('kaggle-data/MNCAATourneySeeds.csv')
        teams = pd.read_csv('kaggle-data/MTeams.csv')
        
        # Merge seeds and team names
        merged = (results
            .merge(seeds, left_on=['Season', 'WTeamID'], 
                   right_on=['Season', 'TeamID'], how='left')
            .rename(columns={'Seed': 'WSeed'})
            .drop(columns=['TeamID'])
            
            .merge(seeds, left_on=['Season', 'LTeamID'], 
                   right_on=['Season', 'TeamID'], how='left')
            .rename(columns={'Seed': 'LSeed'})
            .drop(columns=['TeamID'])
            
            .merge(teams, left_on='WTeamID', right_on='TeamID', how='left')
            .rename(columns={'TeamName': 'WinningTeam'})
            .drop(columns=['TeamID', 'FirstD1Season', 'LastD1Season'])
            
            .merge(teams, left_on='LTeamID', right_on='TeamID', how='left')
            .rename(columns={'TeamName': 'LosingTeam'})
            .drop(columns=['TeamID', 'FirstD1Season', 'LastD1Season'])
        )
        
        # Extract numeric seeds
        merged['WSeed_Numeric'] = merged['WSeed'].str.extract(r'(\d+)').astype(float)
        merged['LSeed_Numeric'] = merged['LSeed'].str.extract(r'(\d+)').astype(float)
        
        merged = merged.rename(columns={'Season': 'Year'})
        
        print(f"Loaded {len(merged)} tournament games with seeds")
        
        return merged  # Make sure you're only returning the DataFrame
        
    except FileNotFoundError:
        print("Kaggle data not found.")
        return None

def align_tournament_data(tourament_games, team_stats):
    available_years = team_stats['Year'].unique()
    
    aligned_games = tourament_games[tourament_games['Year'].isin(available_years)]
    return aligned_games, available_years

def create_features(team_a_stats, team_b_stats, feature_columns):
    features = {}
    
    for col in feature_columns:
        if col in team_a_stats and col in team_b_stats:
            # Handle potential missing values
            val_a = team_a_stats[col] if pd.notna(team_a_stats[col]) else 0
            val_b = team_b_stats[col] if pd.notna(team_b_stats[col]) else 0
            
            try:
                features[f'{col}_diff'] = float(val_a) - float(val_b)
            except (ValueError, TypeError):
                # Skip non-numeric columns
                continue
    return features
    
def prepare_training_data(team_stats, tournament_games):
    tournament_games, available_years = align_tournament_data(tournament_games, team_stats)
    
    feature_columns = [
        'Overall_W-L%',
        'Overall_SRS',
        'Overall_SOS',
        'Points_Tm.',
        'Points_Opp.',
        'Totals_FG%',
        'Totals_3P%',
        'Totals_FT%',
        'Totals_TRB',
        'Totals_AST',
        'Totals_STL',
        'Totals_BLK',
        'Totals_TOV',
        'ORtg',
        'Pace',
        'FTr',
        'eFG%',
        'TS%',
        'TRB%',
        'TOV%']
    
    school_col = [col for col in team_stats.columns if 'School' in col][0]
    available_features = [col for col in feature_columns if col in team_stats.columns]
    
    matchups = []
    skipped = 0
    
    for idx, game in tournament_games.iterrows():
        year = game['Year']
        team_a_name = game['WinningTeam']
        team_b_name = game['LosingTeam']
        
        team_a_stats = team_stats[(team_stats[school_col] == team_a_name) & (team_stats['Year'] == year)]
        team_b_stats = team_stats[(team_stats[school_col] == team_b_name) & (team_stats['Year'] == year)]
        
        if len(team_a_stats) == 1 and len(team_b_stats) == 1:
            features = create_features(team_a_stats.iloc[0], team_b_stats.iloc[0], available_features)
            features['team_a_won'] = 1
            matchups.append(features)
            
            features_reverse = create_features(
                team_b_stats.iloc[0],
                team_a_stats.iloc[0],
                available_features
            )
            features_reverse['team_a_won'] = 0  # Team A (now team B) lost
            matchups.append(features_reverse)
        else:
            skipped += 1
            
    return pd.DataFrame(matchups), available_features

def train_model(X_train, y_train, X_test, y_test):
    print("\n" + "="*60)
    print("Training Logistic Regression Model")
    print("="*60)
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    y_pred_proba_train = model.predict_proba(X_train)[:,1]
    y_pred_proba_test = model.predict_proba(X_test)[:,1]
    
    print("\nTraining Performance:")
    print(f"  Accuracy: {accuracy_score(y_train, y_pred_train):.4f}")
    print(f"  Log Loss: {log_loss(y_train, y_pred_proba_train):.4f}")
    
    print("\nTest Performance:")
    print(f"  Accuracy: {accuracy_score(y_test, y_pred_test):.4f}")
    print(f"  Log Loss: {log_loss(y_test, y_pred_proba_test):.4f}")
    
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_pred_test, target_names=['Team B Wins', 'Team A Wins']))
    
    print("\nTop 10 Most Important Features:")
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'coefficient': model.coef_[0]
    }).sort_values('coefficient', key=abs, ascending=False)
    
    print(feature_importance.head(10).to_string(index=False))
    
    return model
    
def save_model(model, feature_names, filename='models/march_madness_model.pkl'):
    import os
    os.makedirs('models', exist_ok=True)
        
    model_data = {'model': model, 'feature_names': feature_names}
        
    with open(filename, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\nModel saved to {filename}")

def main():
    print("="*60)
    print("MARCH MADNESS PREDICTION MODEL - TRAINING PIPELINE")
    print("="*60)
    
    # Load data
    print("\n[1/5] Loading data...")
    team_stats = load_data()
    tournament_games = load_tournament_data()
    
    if tournament_games is None:
        print("Cannot proceed without tournament data.")
        return
    
    # Prepare training data
    print("\n[2/5] Preparing training data...")
    training_data, feature_names = prepare_training_data(team_stats, tournament_games)
    
    if len(training_data) == 0:
        print("No training data created. Check team name matching.")
        return
    
    # Split features and target
    print("\n[3/5] Splitting data...")
    X = training_data[[col for col in training_data.columns if col != 'team_a_won']]
    y = training_data['team_a_won']
    
    # Handle missing values
    X = X.fillna(0)
    
    print(f"Features shape: {X.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Train-test split (by year to avoid data leakage if possible)
    # For now, random split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)} examples")
    print(f"Test set: {len(X_test)} examples")
    
    # Train model
    print("\n[4/5] Training model...")
    model = train_model(X_train, y_train, X_test, y_test)
    
    # Save model
    print("\n[5/5] Saving model...")
    save_model(model, X.columns.tolist())
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Use this model to predict tournament games")
    print("2. Run bracket simulations")
    print("3. Tune hyperparameters or try different models")

if __name__ == "__main__":
    main()
