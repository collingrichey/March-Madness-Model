import pandas as pd
import numpy as np
import pickle
from collections import Counter

"""
Go to line 38. Please read the commment.
"""

def load_model(filename='models/march_madness_model.pkl'):
    with open(filename, 'rb') as f:
        model_data = pickle.load(f)
    return model_data['model'], model_data['feature_names']

def load_team_stats(year):
    df = pd.read_csv(f'data/team_stats_{year}.csv')
    df['Year'] = year
    return df

def load_bracket(year):
    try:
        seeds = pd.read_csv('kaggle-data/MNCAATourneySeeds.csv')
        teams = pd.read_csv('kaggle-data/MTeams.csv')
        
        year_seeds = seeds[seeds['Season'] == year].copy()
        
        year_seeds = year_seeds.merge(teams, on='TeamID')
        
        year_seeds['Region'] = year_seeds['Seed'].str.extract(r'([WXYZ])')
        year_seeds['Seed_Num'] = year_seeds['Seed'].str.extract(r'(\d+)').astype(int)
        year_seeds['Is_PlayIn'] = year_seeds['Seed'].str.contains(r'[ab]$', regex=True)
        year_seeds = year_seeds[~year_seeds['Seed'].str.endswith('b')]
        return year_seeds[['TeamName', 'Region', 'Seed_Num', 'Seed', 'Is_PlayIn']]
    except FileNotFoundError:
        print("Kaggle data not found. You'll need to manually create bracket data.")
        return None


"""
This function is to map names between the scraped data and the team ID and team name data. 
If it says "Could not find data for (team1) or (team2)", add a mapping of the team names in this format:
'(MTeams table)' : '(team_stats table)' 
"""    
def normalize_team_name(name):
    
    mappings = {
        # Kaggle format → Scraped format
        'St Mary\'s CA': 'Saint Mary\'s (CA)',
        'St Peter\'s': 'Saint Peter\'s',
        'St John\'s': 'St. John\'s (NY)',
        'St Bonaventure': 'St. Bonaventure',
        'Mt St Mary\'s': 'Mount St. Mary\'s',
        'St Francis NY': 'St. Francis (NY)',
        'St Francis PA': 'St. Francis (PA)',
        'Alabama St': 'Alabama State',
        'Alcorn St': 'Alcorn State',
        'Appalachian St': 'Appalachian State',
        'Arizona St': 'Arizona State',
        'Arkansas St': 'Arkansas State',
        'Ball St': 'Ball State',
        'Boise St': 'Boise State',
        'Cleveland St': 'Cleveland State',
        'Colorado St': 'Colorado State',
        'Coppin St': 'Coppin State',
        'Delaware St': 'Delaware State',
        'Florida St': 'Florida State',
        'Fresno St': 'Fresno State',
        'Georgia St': 'Georgia State',
        'Idaho St': 'Idaho State',
        'Illinois St': 'Illinois State',
        'Indiana St': 'Indiana State',
        'Iowa St': 'Iowa State',
        'Jackson St': 'Jackson State',
        'Jacksonville St': 'Jacksonville State',
        'Kansas St': 'Kansas State',
        'Kennesaw St': 'Kennesaw State',
        'Kent St': 'Kent State',
        'Long Beach St': 'Long Beach State',
        'McNeese St': 'McNeese State',
        'Michigan St': 'Michigan State',
        'Mississippi St': 'Mississippi State',
        'Mississippi Val': 'Mississippi Valley State',
        'Missouri St': 'Missouri State',
        'Montana St': 'Montana State',
        'Morehead St': 'Morehead State',
        'Morgan St': 'Morgan State',
        'Murray St': 'Murray State',
        'N Dakota St': 'North Dakota State',
        'New Mexico St': 'New Mexico State',
        'Nicholls St': 'Nicholls State',
        'Norfolk St': 'Norfolk State',
        'Northwestern St': 'Northwestern State',
        'Ohio St': 'Ohio State',
        'Oklahoma St': 'Oklahoma State',
        'Oregon St': 'Oregon State',
        'Penn St': 'Penn State',
        'Portland St': 'Portland State',
        'S Carolina St': 'South Carolina State',
        'S Dakota St': 'South Dakota State',
        'Sacramento St': 'Sacramento State',
        'Sam Houston St': 'Sam Houston State',
        'San Diego St': 'San Diego State',
        'San Jose St': 'San Jose State',
        'Savannah St': 'Savannah State',
        'Tennessee St': 'Tennessee State',
        'Texas St': 'Texas State',
        'Utah St': 'Utah State',
        'Weber St': 'Weber State',
        'Wichita St': 'Wichita State',
        'Wright St': 'Wright State',
        'Youngstown St': 'Youngstown State',
        
        # Other common mismatches
        'SUNY Albany': 'Albany (NY)',
        'American Univ': 'American',
        'Ark Pine Bluff': 'Arkansas-Pine Bluff',
        'Boston Univ': 'Boston University',
        'BYU': 'Brigham Young',
        'C Michigan': 'Central Michigan',
        'Cent Arkansas': 'Central Arkansas',
        'Central Conn': 'Central Connecticut State',
        'Charleston So': 'Charleston Southern',
        'Coastal Car': 'Coastal Carolina',
        'Col Charleston': 'Charleston',
        'CS Bakersfield': 'Cal State Bakersfield',
        'CS Fullerton': 'Cal State Fullerton',
        'CS Northridge': 'Cal State Northridge',
        'CS Sacramento': 'Sacramento State',
        'Detroit': 'Detroit Mercy',
        'E Illinois': 'Eastern Illinois',
        'E Kentucky': 'Eastern Kentucky',
        'E Michigan': 'Eastern Michigan',
        'E Washington': 'Eastern Washington',
        'ETSU': 'East Tennessee State',
        'F Dickinson': 'FDU',
        'FL Atlantic': 'Florida Atlantic',
        'FGCU': 'Florida Gulf Coast',
        'Florida Intl': 'Florida International',
        'G Washington': 'George Washington',
        'Ga Southern': 'Georgia Southern',
        'Ga State': 'Georgia State',
        'Gardner Webb': 'Gardner-Webb',
        'Green Bay': 'Wisconsin-Green Bay',
        'Houston Bap': 'Houston Baptist',
        'LIU Brooklyn': 'Long Island University',
        'Loyola Chicago': 'Loyola (IL)',
        'Loyola MD': 'Loyola (MD)',
        'Loy Marymount': 'Loyola Marymount',
        'LSU': 'Louisiana State',
        'Miami FL': 'Miami (FL)',
        'Miami OH': 'Miami (OH)',
        'Middle Tenn': 'Middle Tennessee',
        'Milwaukee': 'Wisconsin-Milwaukee',
        'N Colorado': 'Northern Colorado',
        'N Illinois': 'Northern Illinois',
        'N Iowa': 'Northern Iowa',
        'NC A&T': 'North Carolina A&T',
        'NC Central': 'North Carolina Central',
        'Northern Ariz': 'Northern Arizona',
        'Northern Ky': 'Northern Kentucky',
        'Omaha': 'Nebraska Omaha',
        'Penn': 'Pennsylvania',
        'Prairie View': 'Prairie View A&M',
        'S Illinois': 'Southern Illinois',
        'S Miss': 'Southern Mississippi',
        'S Utah': 'Southern Utah',
        'SIUE': 'SIU-Edwardsville',
        'SMU': 'Southern Methodist',
        'South Fla': 'South Florida',
        'Southeast Mo St': 'Southeast Missouri State',
        'Southeastern La': 'Southeastern Louisiana',
        'Southern Ill': 'Southern Illinois',
        'Stephen F Austin': 'Stephen F. Austin',
        'TCU': 'Texas Christian',
        'TN Martin': 'Tennessee-Martin',
        'Tenn Tech': 'Tennessee Tech',
        'Texas A&M Corp': 'Texas A&M-Corpus Christi',
        'UAB': 'Alabama-Birmingham',
        'UC Davis': 'UC Davis',
        'UC Irvine': 'UC Irvine',
        'UC Riverside': 'UC Riverside',
        'UC Santa Barbara': 'UC Santa Barbara',
        'UCF': 'Central Florida',
        'UConn': 'Connecticut',
        'UIC': 'Illinois-Chicago',
        'UMBC': 'Maryland-Baltimore County',
        'UMES': 'Maryland-Eastern Shore',
        'UMass Lowell': 'Massachusetts-Lowell',
        'UNC Asheville': 'North Carolina-Asheville',
        'UNC Greensboro': 'North Carolina-Greensboro',
        'UNC Wilmington': 'North Carolina-Wilmington',
        'UNLV': 'Nevada-Las Vegas',
        'USC': 'Southern California',
        'USC Upstate': 'South Carolina Upstate',
        'UT Arlington': 'Texas-Arlington',
        'UTEP': 'Texas-El Paso',
        'UTRGV': 'Texas-Rio Grande Valley',
        'UTSA': 'Texas-San Antonio',
        'VCU': 'Virginia Commonwealth',
        'Western Caro': 'Western Carolina',
        'Western Ill': 'Western Illinois',
        'Western Ky': 'Western Kentucky',
        'Western Mich': 'Western Michigan',
    }
    
    # Check if we have a direct mapping
    if name in mappings:
        return mappings[name]
    
    return name

def find_team_in_stats(team_name, team_stats, school_col):
    """
    Try to find a team in stats with fuzzy matching
    """
    # First try exact match
    exact_match = team_stats[team_stats[school_col] == team_name]
    if len(exact_match) > 0:
        return exact_match.iloc[0]
    
    # Try normalized name
    normalized = normalize_team_name(team_name)
    exact_match = team_stats[team_stats[school_col] == normalized]
    if len(exact_match) > 0:
        return exact_match.iloc[0]
    
    # Try partial match (case insensitive)
    partial_match = team_stats[team_stats[school_col].str.contains(team_name, case=False, na=False)]
    if len(partial_match) > 0:
        return partial_match.iloc[0]
    
    # Try searching for key words
    # Remove common words and try again
    search_term = team_name.replace('State', '').replace('University', '').replace('College', '').strip()
    partial_match = team_stats[team_stats[school_col].str.contains(search_term, case=False, na=False)]
    if len(partial_match) > 0:
        return partial_match.iloc[0]
    
    return None

def resolve_play_in_games(bracket_data, team_stats, model, feature_names):
    """
    Simulate play-in games to determine which teams make the main bracket
    """
    print(f"\n{'='*50}")
    print(f"  FIRST FOUR (PLAY-IN GAMES)")
    print(f"{'='*50}\n")
    
    # Find play-in matchups (same region and seed number)
    play_in_teams = bracket_data[bracket_data['Is_PlayIn'] == True]
    
    resolved_teams = []
    
    # Group by region and seed to find matchups
    for (region, seed_num), group in play_in_teams.groupby(['Region', 'Seed_Num']):
        if len(group) == 2:
            team_a = group.iloc[0]
            team_b = group.iloc[1]
            
            # Simulate the play-in game
            prob_a_wins = predict_game(
                team_a['TeamName'], 
                team_b['TeamName'], 
                team_stats, 
                model, 
                feature_names
            )
            
            # Pick the winner (higher probability)
            if prob_a_wins > 0.5:
                winner = team_a
                loser = team_b
                print(f"({seed_num}) {team_a['TeamName']:25s} over {team_b['TeamName']:25s} [{prob_a_wins:.1%}]")
            else:
                winner = team_b
                loser = team_a
                print(f"({seed_num}) {team_b['TeamName']:25s} over {team_a['TeamName']:25s} [{1-prob_a_wins:.1%}]")
            
            # Add winner to resolved teams
            winner_dict = winner.to_dict()
            winner_dict['Is_PlayIn'] = False  # No longer a play-in team
            resolved_teams.append(winner_dict)
        else:
            # If only one team or odd number, just include them all
            for _, team in group.iterrows():
                team_dict = team.to_dict()
                team_dict['Is_PlayIn'] = False
                resolved_teams.append(team_dict)
    
    # Combine resolved play-in teams with main bracket teams
    main_bracket = bracket_data[bracket_data['Is_PlayIn'] == False]
    resolved_df = pd.DataFrame(resolved_teams)
    
    final_bracket = pd.concat([main_bracket, resolved_df], ignore_index=True)
    
    print(f"\nFinal bracket: {len(final_bracket)} teams")
    
    return final_bracket

def create_matchup_features(team_a_stats, team_b_stats, feature_names):
    features = {}
    
    for feature in feature_names:
        col = feature.replace('_diff', '')
        
        if col in team_a_stats.index and col in team_b_stats.index:
            val_a = team_a_stats[col] if pd.notna(team_a_stats[col]) else 0
            val_b = team_b_stats[col] if pd.notna(team_b_stats[col]) else 0

            try:
                features[feature] = float(val_a) - float(val_b)
            except (ValueError, TypeError):
                features[feature] = 0
        else:
            features[feature] = 0
            
    return features

def predict_game(team_a_name, team_b_name, team_stats, model, feature_names):
    school_col = [col for col in team_stats.columns if 'School' in col][0]
    
    team_a_stats = find_team_in_stats(team_a_name, team_stats, school_col)
    team_b_stats = find_team_in_stats(team_b_name, team_stats, school_col)
    
    if team_a_stats is None or team_b_stats is None:
        print(f"Warning: Could not find stats for {team_a_name} or {team_b_name}")
        return 0.5
    
    team_a_stats.iloc[0]
    team_b_stats.iloc[0]
    
    features = create_matchup_features(team_a_stats, team_b_stats, feature_names)
    
    X = pd.DataFrame([features])[feature_names]
    X = X.fillna(0)
    
    prob_a_wins = model.predict_proba(X)[0][1]
    
    return prob_a_wins

def create_bracket_structure(bracket_data):
    regions = {}
    
    matchup_order = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]
    
    for region in ['W', 'X', 'Y', 'Z']:
        region_teams = bracket_data[bracket_data['Region'] == region].copy()
        
        # Sort by seed to get all teams in order
        region_teams = region_teams.sort_values('Seed_Num')
        
        # Create a dict for easy lookup
        seed_to_team = {}
        for _, row in region_teams.iterrows():
            seed_to_team[row['Seed_Num']] = (row['Seed_Num'], row['TeamName'])
        
        # Build the bracket in the correct matchup order
        ordered_teams = []
        for seed in matchup_order:
            if seed in seed_to_team:
                ordered_teams.append(seed_to_team[seed])
            else:
                print(f"Warning: Seed {seed} not found in region {region}")
        
        regions[region] = ordered_teams
    
    return regions

def simulate_region(region_teams, team_stats, model, feature_names, region_name):
    print(f"\n{'='*50}")
    print(f"  {region_name} REGION")
    print(f"{'='*50}")
    
    current_round = region_teams.copy()
    round_num = 1
    round_names = ["Round of 64", "Round of 32", "Sweet 16", "Elite Eight"]
    
    while len(current_round) > 1:
        print(f"\n{round_names[round_num-1]}:")
        next_round = []
        
        # Pair up teams (1 vs 16, 8 vs 9, etc.)
        for i in range(0, len(current_round), 2):
            seed_a, team_a = current_round[i]
            seed_b, team_b = current_round[i+1]
            
            # Predict game
            prob_a_wins = predict_game(team_a, team_b, team_stats, model, feature_names)
            
            # Simulate outcome
            if np.random.random() < prob_a_wins:
                winner = (seed_a, team_a)
                print(f"  ({seed_a}) {team_a:25s} defeats ({seed_b}) {team_b:25s} [{prob_a_wins:.1%}]")
            else:
                winner = (seed_b, team_b)
                print(f"  ({seed_b}) {team_b:25s} defeats ({seed_a}) {team_a:25s} [{1-prob_a_wins:.1%}]")
            
            next_round.append(winner)
        
        current_round = next_round
        round_num += 1
    
    return current_round[0]
    
def simulate_final_four(regional_winners, team_stats, model, feature_names):
    """
    Simulate Final Four and Championship
    """
    print(f"\n{'='*50}")
    print(f"  FINAL FOUR")
    print(f"{'='*50}")
    
    # Semifinal 1
    seed_a, team_a = regional_winners[0]
    seed_b, team_b = regional_winners[1]
    prob_a_wins = predict_game(team_a, team_b, team_stats, model, feature_names)
    
    if np.random.random() < prob_a_wins:
        finalist_1 = (seed_a, team_a)
        print(f"Semifinal 1: ({seed_a}) {team_a:25s} defeats ({seed_b}) {team_b:25s} [{prob_a_wins:.1%}]")
    else:
        finalist_1 = (seed_b, team_b)
        print(f"Semifinal 1: ({seed_b}) {team_b:25s} defeats ({seed_a}) {team_a:25s} [{1-prob_a_wins:.1%}]")
    
    # Semifinal 2
    seed_c, team_c = regional_winners[2]
    seed_d, team_d = regional_winners[3]
    prob_c_wins = predict_game(team_c, team_d, team_stats, model, feature_names)
    
    if np.random.random() < prob_c_wins:
        finalist_2 = (seed_c, team_c)
        print(f"Semifinal 2: ({seed_c}) {team_c:25s} defeats ({seed_d}) {team_d:25s} [{prob_c_wins:.1%}]")
    else:
        finalist_2 = (seed_d, team_d)
        print(f"Semifinal 2: ({seed_d}) {team_d:25s} defeats ({seed_c}) {team_c:25s} [{1-prob_c_wins:.1%}]")
    
    # Championship
    print(f"\n{'='*50}")
    print(f"  CHAMPIONSHIP")
    print(f"{'='*50}")
    
    seed_1, team_1 = finalist_1
    seed_2, team_2 = finalist_2
    prob_1_wins = predict_game(team_1, team_2, team_stats, model, feature_names)
    
    if np.random.random() < prob_1_wins:
        champion = (seed_1, team_1)
        print(f"({seed_1}) {team_1:25s} defeats ({seed_2}) {team_2:25s} [{prob_1_wins:.1%}]")
    else:
        champion = (seed_2, team_2)
        print(f"({seed_2}) {team_2:25s} defeats ({seed_1}) {team_1:25s} [{1-prob_1_wins:.1%}]")
    
    print(f"\n CHAMPION: ({champion[0]}) {champion[1]} \n")
    
    return champion

def simulate_single_bracket(bracket_data, team_stats, model, feature_names, verbose=True):
    """
    Simulate one complete tournament bracket
    """
    # Create bracket structure
    regions = create_bracket_structure(bracket_data)
    
    # Simulate each region
    regional_winners = []
    region_names = {'X': 'WEST', 'W': 'EAST', 'Z': 'SOUTH', 'Y': 'MIDWEST'}
    
    for region_code in ['X', 'W', 'Z', 'Y']:
        if verbose:
            winner = simulate_region(regions[region_code], team_stats, model, feature_names, region_names[region_code])
        else:
            # Silent simulation
            current_round = regions[region_code]
            while len(current_round) > 1:
                next_round = []
                for i in range(0, len(current_round), 2):
                    seed_a, team_a = current_round[i]
                    seed_b, team_b = current_round[i+1]
                    prob_a_wins = predict_game(team_a, team_b, team_stats, model, feature_names)
                    winner = current_round[i] if np.random.random() < prob_a_wins else current_round[i+1]
                    next_round.append(winner)
                current_round = next_round
            winner = current_round[0]
        
        regional_winners.append(winner)
    
    # Simulate Final Four
    if verbose:
        champion = simulate_final_four(regional_winners, team_stats, model, feature_names)
    else:
        # Silent Final Four
        # Semifinal 1
        prob = predict_game(regional_winners[0][1], regional_winners[1][1], team_stats, model, feature_names)
        finalist_1 = regional_winners[0] if np.random.random() < prob else regional_winners[1]
        
        # Semifinal 2
        prob = predict_game(regional_winners[2][1], regional_winners[3][1], team_stats, model, feature_names)
        finalist_2 = regional_winners[2] if np.random.random() < prob else regional_winners[3]
        
        # Championship
        prob = predict_game(finalist_1[1], finalist_2[1], team_stats, model, feature_names)
        champion = finalist_1 if np.random.random() < prob else finalist_2
    
    return champion

def run_monte_carlo_simulation(bracket_data, team_stats, model, feature_names, n_simulations=10000):
    """
    Run many bracket simulations to get probabilities
    """
    print(f"\n{'='*60}")
    print(f"Running {n_simulations:,} Monte Carlo Simulations")
    print(f"{'='*60}\n")
    
    champions = []
    
    for i in range(n_simulations):
        if (i + 1) % 1000 == 0:
            print(f"Completed {i+1:,} / {n_simulations:,} simulations...")
        
        seed, team = simulate_single_bracket(bracket_data, team_stats, model, feature_names, verbose=False)
        champions.append(team)
    
    # Count champions
    champion_counts = Counter(champions)
    
    print(f"\n{'='*60}")
    print(f"CHAMPIONSHIP PROBABILITIES (Top 20)")
    print(f"{'='*60}\n")
    
    results = []
    for team, count in champion_counts.most_common(20):
        prob = count / n_simulations
        print(f"{team:30s} {prob:6.2%}  ({count:,} / {n_simulations:,})")
        results.append({'Team': team, 'Championships': count, 'Probability': prob})
    
    return pd.DataFrame(results)

def predict_most_likely_bracket(bracket_data, team_stats, model, feature_names):
    """
    Create the most likely bracket by always picking the team with higher win probability
    This maximizes expected correct picks
    """
    print(f"\n{'='*60}")
    print(f"MOST LIKELY BRACKET (Deterministic Predictions)")
    print(f"{'='*60}")
    print("Picking the higher probability team in each matchup\n")
    
    # bracket_data = resolve_play_in_games(bracket_data, team_stats, model, feature_names)
    # Create bracket structure
    regions = create_bracket_structure(bracket_data)
    region_names = {'X': 'WEST', 'W': 'EAST', 'Z': 'SOUTH', 'Y': 'MIDWEST'}
    
    all_predictions = []
    regional_winners = []
    
    # Simulate each region
    for region_code in ['X', 'W', 'Z', 'Y']:
        print(f"\n{'='*50}")
        print(f"  {region_names[region_code]} REGION")
        print(f"{'='*50}")
        
        current_round = regions[region_code].copy()
        round_num = 1
        round_names = ["Round of 64", "Round of 32", "Sweet 16", "Elite Eight"]
        
        while len(current_round) > 1:
            print(f"\n{round_names[round_num-1]}:")
            next_round = []
            
            for i in range(0, len(current_round), 2):
                seed_a, team_a = current_round[i]
                seed_b, team_b = current_round[i+1]
                
                # Predict game
                prob_a_wins = predict_game(team_a, team_b, team_stats, model, feature_names)
                
                # Pick higher probability team
                if prob_a_wins > 0.5:
                    winner = (seed_a, team_a)
                    print(f"  ({seed_a}) {team_a:25s} over ({seed_b}) {team_b:25s} [{prob_a_wins:.1%}]")
                    all_predictions.append({
                        'Round': round_names[round_num-1],
                        'Region': region_names[region_code],
                        'Winner': team_a,
                        'Winner_Seed': seed_a,
                        'Loser': team_b,
                        'Loser_Seed': seed_b,
                        'Win_Probability': prob_a_wins
                    })
                else:
                    winner = (seed_b, team_b)
                    print(f"  ({seed_b}) {team_b:25s} over ({seed_a}) {team_a:25s} [{1-prob_a_wins:.1%}]")
                    all_predictions.append({
                        'Round': round_names[round_num-1],
                        'Region': region_names[region_code],
                        'Winner': team_b,
                        'Winner_Seed': seed_b,
                        'Loser': team_a,
                        'Loser_Seed': seed_a,
                        'Win_Probability': 1 - prob_a_wins
                    })
                
                next_round.append(winner)
            
            current_round = next_round
            round_num += 1
        
        regional_winners.append(current_round[0])
        print(f"\n {region_names[region_code]} CHAMPION: ({current_round[0][0]}) {current_round[0][1]}")
    
    # Final Four
    print(f"\n{'='*50}")
    print(f"  FINAL FOUR")
    print(f"{'='*50}\n")
    
    # Semifinal 1
    seed_a, team_a = regional_winners[0]
    seed_b, team_b = regional_winners[1]
    prob_a_wins = predict_game(team_a, team_b, team_stats, model, feature_names)
    
    if prob_a_wins > 0.5:
        finalist_1 = (seed_a, team_a)
        print(f"Semifinal 1: ({seed_a}) {team_a:25s} over ({seed_b}) {team_b:25s} [{prob_a_wins:.1%}]")
        all_predictions.append({
            'Round': 'Final Four',
            'Region': 'National',
            'Winner': team_a,
            'Winner_Seed': seed_a,
            'Loser': team_b,
            'Loser_Seed': seed_b,
            'Win_Probability': prob_a_wins
        })
    else:
        finalist_1 = (seed_b, team_b)
        print(f"Semifinal 1: ({seed_b}) {team_b:25s} over ({seed_a}) {team_a:25s} [{1-prob_a_wins:.1%}]")
        all_predictions.append({
            'Round': 'Final Four',
            'Region': 'National',
            'Winner': team_b,
            'Winner_Seed': seed_b,
            'Loser': team_a,
            'Loser_Seed': seed_a,
            'Win_Probability': 1 - prob_a_wins
        })
    
    # Semifinal 2
    seed_c, team_c = regional_winners[2]
    seed_d, team_d = regional_winners[3]
    prob_c_wins = predict_game(team_c, team_d, team_stats, model, feature_names)
    
    if prob_c_wins > 0.5:
        finalist_2 = (seed_c, team_c)
        print(f"Semifinal 2: ({seed_c}) {team_c:25s} over ({seed_d}) {team_d:25s} [{prob_c_wins:.1%}]")
        all_predictions.append({
            'Round': 'Final Four',
            'Region': 'National',
            'Winner': team_c,
            'Winner_Seed': seed_c,
            'Loser': team_d,
            'Loser_Seed': seed_d,
            'Win_Probability': prob_c_wins
        })
    else:
        finalist_2 = (seed_d, team_d)
        print(f"Semifinal 2: ({seed_d}) {team_d:25s} over ({seed_c}) {team_c:25s} [{1-prob_c_wins:.1%}]")
        all_predictions.append({
            'Round': 'Final Four',
            'Region': 'National',
            'Winner': team_d,
            'Winner_Seed': seed_d,
            'Loser': team_c,
            'Loser_Seed': seed_c,
            'Win_Probability': 1 - prob_c_wins
        })
    
    # Championship
    print(f"\n{'='*50}")
    print(f"  CHAMPIONSHIP")
    print(f"{'='*50}\n")
    
    seed_1, team_1 = finalist_1
    seed_2, team_2 = finalist_2
    prob_1_wins = predict_game(team_1, team_2, team_stats, model, feature_names)
    
    if prob_1_wins > 0.5:
        champion = (seed_1, team_1)
        print(f"({seed_1}) {team_1:25s} over ({seed_2}) {team_2:25s} [{prob_1_wins:.1%}]")
        all_predictions.append({
            'Round': 'Championship',
            'Region': 'National',
            'Winner': team_1,
            'Winner_Seed': seed_1,
            'Loser': team_2,
            'Loser_Seed': seed_2,
            'Win_Probability': prob_1_wins
        })
    else:
        champion = (seed_2, team_2)
        print(f"({seed_2}) {team_2:25s} over ({seed_1}) {team_1:25s} [{1-prob_1_wins:.1%}]")
        all_predictions.append({
            'Round': 'Championship',
            'Region': 'National',
            'Winner': team_2,
            'Winner_Seed': seed_2,
            'Loser': team_1,
            'Loser_Seed': seed_1,
            'Win_Probability': 1 - prob_1_wins
        })
    
    print(f"\nPREDICTED CHAMPION: ({champion[0]}) {champion[1]} \n")
    
    return pd.DataFrame(all_predictions), champion

def calculate_bracket_confidence(predictions_df):
    """
    Calculate overall bracket confidence and identify risky picks
    """
    print(f"\n{'='*60}")
    print(f"BRACKET CONFIDENCE ANALYSIS")
    print(f"{'='*60}\n")
    
    # Overall confidence
    avg_confidence = predictions_df['Win_Probability'].mean()
    print(f"Average Pick Confidence: {avg_confidence:.1%}")
    
    # By round
    print(f"\nConfidence by Round:")
    round_confidence = predictions_df.groupby('Round')['Win_Probability'].agg(['mean', 'min', 'max'])
    print(round_confidence)
    
    # Risky picks (close to 50-50)
    print(f"\n{'='*60}")
    print(f"RISKIEST PICKS (Closest to 50-50)")
    print(f"{'='*60}\n")
    
    predictions_df['Confidence_Score'] = abs(predictions_df['Win_Probability'] - 0.5)
    risky_picks = predictions_df.nsmallest(10, 'Confidence_Score')
    
    for _, pick in risky_picks.iterrows():  
        print(f"{pick['Round']:15s} | ({pick['Winner_Seed']}) {pick['Winner']:20s} over ({pick['Loser_Seed']}) {pick['Loser']:20s} [{pick['Win_Probability']:.1%}]")
    
    
    # Upset picks (lower seed predicted to win)
    print(f"\n{'='*60}")
    print(f"UPSET PREDICTIONS")
    print(f"{'='*60}\n")
    
    upsets = predictions_df[predictions_df['Winner_Seed'] > predictions_df['Loser_Seed']]
    
    if len(upsets) > 0:
        for _, pick in upsets.iterrows():
            print(f"{pick['Round']:15s} | ({pick['Winner_Seed']}) {pick['Winner']:20s} over ({pick['Loser_Seed']}) {pick['Loser']:20s} [{pick['Win_Probability']:.1%}]")
    else:
        print("No upsets predicted (all favorites advance)")
        
def simulate_game_monte_carlo(team_a_name, team_b_name, team_stats, model, feature_names, n_simulations=1000):
    """
    Simulate a single game many times and return win percentage for team_a
    """
    prob_a_wins = predict_game(team_a_name, team_b_name, team_stats, model, feature_names)
    
    # Simulate the game n_simulations times
    team_a_wins = 0
    for _ in range(n_simulations):
        if np.random.random() < prob_a_wins:
            team_a_wins += 1
    
    win_percentage = team_a_wins / n_simulations
    
    return win_percentage, prob_a_wins

def predict_bracket_with_game_simulations(bracket_data, team_stats, model, feature_names, n_simulations_per_game=1000):
    """
    Create bracket by simulating each game many times and picking the team that wins most often
    """
    print(f"\n{'='*60}")
    print(f"BRACKET WITH GAME-BY-GAME MONTE CARLO")
    print(f"{'='*60}")
    print(f"Running {n_simulations_per_game:,} simulations per game\n")
    
    # Create bracket structure
    regions = create_bracket_structure(bracket_data)
    region_names = {'X': 'WEST', 'W': 'EAST', 'Z': 'SOUTH', 'Y': 'MIDWEST'}
    
    all_predictions = []
    regional_winners = []
    
    # Simulate each region
    for region_code in ['X', 'W', 'Z', 'Y']:
        print(f"\n{'='*50}")
        print(f"  {region_names[region_code]} REGION")
        print(f"{'='*50}")
        
        current_round = regions[region_code].copy()
        round_num = 1
        round_names = ["Round of 64", "Round of 32", "Sweet 16", "Elite Eight"]
        
        while len(current_round) > 1:
            print(f"\n{round_names[round_num-1]}:")
            next_round = []
            
            for i in range(0, len(current_round), 2):
                seed_a, team_a = current_round[i]
                seed_b, team_b = current_round[i+1]
                
                # Simulate this game many times
                win_percentage, model_prob = simulate_game_monte_carlo(
                    team_a, team_b, team_stats, model, feature_names, n_simulations_per_game
                )
                
                # Pick team that won more simulations
                if win_percentage > 0.5:
                    winner = (seed_a, team_a)
                    print(f"  ({seed_a}) {team_a:25s} over ({seed_b}) {team_b:25s} [Won {win_percentage:.1%} of {n_simulations_per_game} sims | Model: {model_prob:.1%}]")
                    all_predictions.append({
                        'Round': round_names[round_num-1],
                        'Region': region_names[region_code],
                        'Winner': team_a,
                        'Winner_Seed': seed_a,
                        'Loser': team_b,
                        'Loser_Seed': seed_b,
                        'Simulation_Win_Pct': win_percentage,
                        'Model_Probability': model_prob,
                        'Simulations': n_simulations_per_game
                    })
                else:
                    winner = (seed_b, team_b)
                    print(f"  ({seed_b}) {team_b:25s} over ({seed_a}) {team_a:25s} [Won {1-win_percentage:.1%} of {n_simulations_per_game} sims | Model: {1-model_prob:.1%}]")
                    all_predictions.append({
                        'Round': round_names[round_num-1],
                        'Region': region_names[region_code],
                        'Winner': team_b,
                        'Winner_Seed': seed_b,
                        'Loser': team_a,
                        'Loser_Seed': seed_a,
                        'Simulation_Win_Pct': 1 - win_percentage,
                        'Model_Probability': 1 - model_prob,
                        'Simulations': n_simulations_per_game
                    })
                
                next_round.append(winner)
            
            current_round = next_round
            round_num += 1
        
        regional_winners.append(current_round[0])
        print(f"\n🏆 {region_names[region_code]} CHAMPION: ({current_round[0][0]}) {current_round[0][1]}")
    
    # Final Four
    print(f"\n{'='*50}")
    print(f"  FINAL FOUR")
    print(f"{'='*50}\n")
    
    # Semifinal 1
    seed_a, team_a = regional_winners[0]
    seed_b, team_b = regional_winners[1]
    win_percentage, model_prob = simulate_game_monte_carlo(
        team_a, team_b, team_stats, model, feature_names, n_simulations_per_game
    )
    
    if win_percentage > 0.5:
        finalist_1 = (seed_a, team_a)
        print(f"Semifinal 1: ({seed_a}) {team_a:25s} over ({seed_b}) {team_b:25s} [Won {win_percentage:.1%} of {n_simulations_per_game} sims]")
        all_predictions.append({
            'Round': 'Final Four',
            'Region': 'National',
            'Winner': team_a,
            'Winner_Seed': seed_a,
            'Loser': team_b,
            'Loser_Seed': seed_b,
            'Simulation_Win_Pct': win_percentage,
            'Model_Probability': model_prob,
            'Simulations': n_simulations_per_game
        })
    else:
        finalist_1 = (seed_b, team_b)
        print(f"Semifinal 1: ({seed_b}) {team_b:25s} over ({seed_a}) {team_a:25s} [Won {1-win_percentage:.1%} of {n_simulations_per_game} sims]")
        all_predictions.append({
            'Round': 'Final Four',
            'Region': 'National',
            'Winner': team_b,
            'Winner_Seed': seed_b,
            'Loser': team_a,
            'Loser_Seed': seed_a,
            'Simulation_Win_Pct': 1 - win_percentage,
            'Model_Probability': 1 - model_prob,
            'Simulations': n_simulations_per_game
        })
    
    # Semifinal 2
    seed_c, team_c = regional_winners[2]
    seed_d, team_d = regional_winners[3]
    win_percentage, model_prob = simulate_game_monte_carlo(
        team_c, team_d, team_stats, model, feature_names, n_simulations_per_game
    )
    
    if win_percentage > 0.5:
        finalist_2 = (seed_c, team_c)
        print(f"Semifinal 2: ({seed_c}) {team_c:25s} over ({seed_d}) {team_d:25s} [Won {win_percentage:.1%} of {n_simulations_per_game} sims]")
        all_predictions.append({
            'Round': 'Final Four',
            'Region': 'National',
            'Winner': team_c,
            'Winner_Seed': seed_c,
            'Loser': team_d,
            'Loser_Seed': seed_d,
            'Simulation_Win_Pct': win_percentage,
            'Model_Probability': model_prob,
            'Simulations': n_simulations_per_game
        })
    else:
        finalist_2 = (seed_d, team_d)
        print(f"Semifinal 2: ({seed_d}) {team_d:25s} over ({seed_c}) {team_c:25s} [Won {1-win_percentage:.1%} of {n_simulations_per_game} sims]")
        all_predictions.append({
            'Round': 'Final Four',
            'Region': 'National',
            'Winner': team_d,
            'Winner_Seed': seed_d,
            'Loser': team_c,
            'Loser_Seed': seed_c,
            'Simulation_Win_Pct': 1 - win_percentage,
            'Model_Probability': 1 - model_prob,
            'Simulations': n_simulations_per_game
        })
    
    # Championship
    print(f"\n{'='*50}")
    print(f"  CHAMPIONSHIP")
    print(f"{'='*50}\n")
    
    seed_1, team_1 = finalist_1
    seed_2, team_2 = finalist_2
    win_percentage, model_prob = simulate_game_monte_carlo(
        team_1, team_2, team_stats, model, feature_names, n_simulations_per_game
    )
    
    if win_percentage > 0.5:
        champion = (seed_1, team_1)
        print(f"({seed_1}) {team_1:25s} over ({seed_2}) {team_2:25s} [Won {win_percentage:.1%} of {n_simulations_per_game} sims]")
        all_predictions.append({
            'Round': 'Championship',
            'Region': 'National',
            'Winner': team_1,
            'Winner_Seed': seed_1,
            'Loser': team_2,
            'Loser_Seed': seed_2,
            'Simulation_Win_Pct': win_percentage,
            'Model_Probability': model_prob,
            'Simulations': n_simulations_per_game
        })
    else:
        champion = (seed_2, team_2)
        print(f"({seed_2}) {team_2:25s} over ({seed_1}) {team_1:25s} [Won {1-win_percentage:.1%} of {n_simulations_per_game} sims]")
        all_predictions.append({
            'Round': 'Championship',
            'Region': 'National',
            'Winner': team_2,
            'Winner_Seed': seed_2,
            'Loser': team_1,
            'Loser_Seed': seed_1,
            'Simulation_Win_Pct': 1 - win_percentage,
            'Model_Probability': 1 - model_prob,
            'Simulations': n_simulations_per_game
        })
    
    print(f"\nPREDICTED CHAMPION: ({champion[0]}) {champion[1]} \n")
    
    return pd.DataFrame(all_predictions), champion

def analyze_simulation_results(predictions_df):
    """
    Analyze the simulation results and confidence
    """
    print(f"\n{'='*60}")
    print(f"SIMULATION ANALYSIS")
    print(f"{'='*60}\n")
    
    # Games where simulation differed from model probability
    predictions_df['Simulation_Agrees_With_Model'] = (
        (predictions_df['Simulation_Win_Pct'] > 0.5) == (predictions_df['Model_Probability'] > 0.5)
    )
    
    disagreements = predictions_df[~predictions_df['Simulation_Agrees_With_Model']]
    
    if len(disagreements) > 0:
        print(f"{len(disagreements)} games where simulations picked differently than model probability:\n")
        for _, pick in disagreements.iterrows():
            print(f"{pick['Round']:15s} | ({pick['Winner_Seed']}) {pick['Winner']:20s} over ({pick['Loser_Seed']}) {pick['Loser']:20s}")
            print(f"                  Simulation: {pick['Simulation_Win_Pct']:.1%} | Model: {pick['Model_Probability']:.1%}\n")
    else:
        print("All simulation picks agree with model probabilities")
    
    # Confidence by round
    print(f"\n{'='*60}")
    print(f"CONFIDENCE BY ROUND")
    print(f"{'='*60}\n")
    
    round_order = ["Round of 64", "Round of 32", "Sweet 16", "Elite Eight", "Final Four", "Championship"]
    predictions_df['Round'] = pd.Categorical(predictions_df['Round'], categories=round_order, ordered=True)
    
    round_stats = predictions_df.groupby('Round')['Simulation_Win_Pct'].agg(['mean', 'min', 'max', 'std'])
    print(round_stats)
    
    # Most confident picks
    print(f"\n{'='*60}")
    print(f"MOST CONFIDENT PICKS (Highest Win %)")
    print(f"{'='*60}\n")
    
    confident = predictions_df.nlargest(10, 'Simulation_Win_Pct')
    for _, pick in confident.iterrows():
        print(f"{pick['Round']:15s} | ({pick['Winner_Seed']}) {pick['Winner']:20s} over ({pick['Loser_Seed']}) {pick['Loser']:20s} [{pick['Simulation_Win_Pct']:.1%}]")
    
    # Least confident picks
    print(f"\n{'='*60}")
    print(f"LEAST CONFIDENT PICKS (Closest to 50-50)")
    print(f"{'='*60}\n")
    
    predictions_df['Distance_From_50'] = abs(predictions_df['Simulation_Win_Pct'] - 0.5)
    uncertain = predictions_df.nsmallest(10, 'Distance_From_50')
    
    for _, pick in uncertain.iterrows():
        print(f"{pick['Round']:15s} | ({pick['Winner_Seed']}) {pick['Winner']:20s} over ({pick['Loser_Seed']}) {pick['Loser']:20s} [{pick['Simulation_Win_Pct']:.1%}]")

def main():
    print("="*60)
    print("March Madness Bracket Simulation")
    print("="*60)
    
    YEAR = 2012
    N_SIMULATIONS = 10000
    SIMULATIONS_PER_GAME = 1000
    
    print("\nLoading model...")
    model, feature_names = load_model()
    print(f"Model loaded with {len(feature_names)} features.")
    
    print(f"Loading {YEAR} team stats...")
    team_stats = load_team_stats(YEAR)
    print(f"Loaded stats for {len(team_stats)} teams.")
    
    print(f"Loading {YEAR} tournament bracket.")
    bracket_data = load_bracket(YEAR)
    
    if bracket_data is None:
        print("Cannot simulate without bracket data")
        return
    
    print(f"Loaded bracket with {len(bracket_data)} teams")
    
    print("\n" + "="*60)
    print("SELECT SIMULATION MODE")
    print("="*60)
    print("1. Most Likely Bracket (pick higher probability in each game)")
    print("2. Game-by-Game Monte Carlo (simulate each game 1000x, pick winner)")
    print("3. Single Full Bracket Simulation (random outcomes)")
    print("4. Full Tournament Monte Carlo (10,000 complete brackets)")
    print("5. Compare All Methods")
    
    choice = input("\n Enter choice (1-5): ")
    
    if choice in ['1', '5']:
        print("\n" + "="*60)
        print("Method 1: Most Likely Bracket")
        print("="*60)
        predictions_df, champion = predict_most_likely_bracket(bracket_data, team_stats, model, feature_names)
        calculate_bracket_confidence(predictions_df)
        predictions_df.to_csv(f'results/most_likely_bracket_{YEAR}.csv', index=False)
        print(f"Saved bracket to results/most_likely_bracket_{YEAR}.csv")

    if choice in ['2', '5']:
        print("\n" + "="*60)
        print("METHOD 2: GAME-BY-GAME MONTE CARLO")
        print("="*60)
        predictions_df, champion = predict_bracket_with_game_simulations(bracket_data, team_stats, model, feature_names, SIMULATIONS_PER_GAME)
        analyze_simulation_results(predictions_df)
        predictions_df.to_csv(f'results/game_by_game_bracket_{YEAR}.csv', index=False)
        print(f"\n Saved to results/game_by_game_bracket_{YEAR}.csv")
    
    if choice in ['3', '5']:
        print("\n" + "="*60)
        print("METHOD 3: SINGLE BRACKET SIMULATION")
        print("="*60)
        results = simulate_single_bracket(bracket_data, team_stats, model, feature_names, verbose=True)
        # results.to_csv(f'results/single_game_bracket_{YEAR}.csv')
        # print(f"\nSaved to results/single_game_bracket_{YEAR}.csv")
    
    if choice in ['4', '5']:
        print("\n" + "="*60)
        print("METHOD 4: FULL TOURNAMENT MONTE CARLO")
        print("="*60)
        results = run_monte_carlo_simulation(bracket_data, team_stats, model, feature_names, 10000)
        results.to_csv(f'results/championship_probablities_{YEAR}.csv')
        print(f"\n Saved to results/championship_probabilities_{YEAR}.csv")
    
    # Add to your main() after loading everything

if __name__ == "__main__":
    import os
    os.makedirs('results', exist_ok=True)
    main()
    