# March Madness Prediction Model

A machine learning model that predicts NCAA March Madness tournament outcomes using logistic regression trained on historical team statistics and tournament results.

This is a work in progress model. 

## Features

- **Data Scraping**: Automatically scrapes team statistics from Sports Reference
- **Model Training**: Trains a logistic regression model on historical tournament games
- **Bracket Simulation**: Three simulation modes:
  - Most Likely Bracket (deterministic predictions)
  - Game-by-Game Monte Carlo (simulates each game 1000x)
  - Full Tournament Monte Carlo (10,000 complete brackets)
- **Performance Analysis**: Confidence analysis, upset predictions, and feature importance

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Internet connection (for scraping data)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/march-madness-model.git
cd march-madness-model
```

### 2. Set Up Virtual Environment (Recommended)

**On Mac/Linux:**
```bash
python3 -m venv march_madness_env
source march_madness_env/bin/activate
```

**On Windows:**
```bash
python -m venv march_madness_env
march_madness_env\Scripts\activate
```

### 3. Install Required Libraries

**Option A: Using requirements.txt (recommended)**
```bash
pip install -r requirements.txt
```

**Option B: Using the install script**
```bash
chmod +x install_requirements.sh
./install_requirements.sh
```

### 4. Download Kaggle Tournament Data

1. Go to [Kaggle's March Madness Competition](https://www.kaggle.com/competitions/march-machine-learning-mania-2024/data)
2. Download these files:
   - `MNCAATourneyCompactResults.csv`
   - `MNCAATourneySeeds.csv`
   - `MTeams.csv`
3. Create a `kaggle-data/` folder in the project directory
4. Place the downloaded files in `kaggle-data/`

Your directory structure should look like:
```
march-madness-model/
├── data_scraper.py
├── prediction_model.py
├── bracket_sim.py
├── requirements.txt
├── install_requirements.sh
├── kaggle-data/
│   ├── MNCAATourneyCompactResults.csv
│   ├── MNCAATourneySeeds.csv
│   └── MTeams.csv
└── data/ (will be created by scraper)
```

#### Simulating the Current Year's Tournament

If you want to simulate the **current year's tournament** (e.g., 2025), the Kaggle data won't have the bracket yet. You'll need to manually add the seeds:

1. **Wait for Selection Sunday** to find out the tournament bracket
2. **Open** `kaggle-data/MNCAATourneySeeds.csv` in a text editor or Excel
3. **Add entries** for the current year following this format:

```csv
Season,Seed,TeamID
2025,W01,1181
2025,W16,1234
2025,W08,1437
...
```

**Format explanation:**
- `Season`: The year (e.g., 2025)
- `Seed`: Region letter + seed number (e.g., `W01` = West #1 seed, `X16` = East #16 seed)
  - **W** = West region
  - **X** = East region  
  - **Y** = South region
  - **Z** = Midwest region
  - Seeds 01-16 for each region
  - Play-in teams use letters: `W16a` and `W16b` for the two 16-seeds playing in
- `TeamID`: The team's ID from `MTeams.csv`

**Example:** If Duke (TeamID 1181) is the #1 seed in the West region:
```csv
2025,W01,1181
```

**Quick method to find TeamIDs:**
```bash
# Search for a team in MTeams.csv
grep -i "Duke" kaggle-data/MTeams.csv
# Output: 1181,Duke,1985,2025
```

**Complete example for West region:**
```csv
Season,Seed,TeamID
2025,W01,1181
2025,W16a,1234
2025,W16b,1456
2025,W08,1437
2025,W09,1246
2025,W05,1314
2025,W12,1159
2025,W04,1104
2025,W13,1345
2025,W06,1233
2025,W11a,1277
2025,W11b,1388
2025,W03,1112
2025,W14,1401
2025,W07,1417
2025,W10,1283
2025,W02,1166
2025,W15,1423
```

Repeat for all four regions (W, X, Y, Z) with all 68 teams (64 main bracket + 4 play-in teams).

## Usage

### Step 1: Scrape Team Statistics

Scrape historical team statistics from Sports Reference:

```bash
python data_scraper.py
```

**What this does:**
- Scrapes both basic and advanced team statistics
- Combines them into a single file per year
- Saves to `data/team_stats_YYYY.csv`

**Important:** 
- The script includes 3-5 second delays between requests to be respectful to the server
- Edit the `years` range in `data_scraper.py` to choose which years to scrape
- Default scrapes years 2000-2004 (update to your desired range)

**Recommended:** Scrape at least 10 years of data (e.g., 2013-2023) for good model performance.

### Step 2: Train the Model

Train the prediction model on historical tournament games:

```bash
python prediction_model.py
```

**What this does:**
- Loads scraped team statistics
- Loads tournament results from Kaggle data
- Creates training examples with differential features
- Trains a logistic regression model
- Saves the model to `models/march_madness_model.pkl`

**Output includes:**
- Training and test accuracy
- Log loss scores
- Top 10 most important features
- Classification report

### Step 3: Simulate Brackets

Run bracket simulations for any year:

```bash
python bracket_sim.py
```

**Simulation Modes:**
1. **Most Likely Bracket**: Picks the higher probability team in each game
2. **Game-by-Game Monte Carlo**: Simulates each game 1000 times, picks winner
3. **Single Bracket Simulation**: Random outcomes based on probabilities
4. **Full Tournament Monte Carlo**: Runs 10,000 complete bracket simulations

**Before running:**
- Edit the `YEAR` variable in `bracket_sim.py` to set which tournament to simulate
- Make sure you have scraped data for that year
- Make sure the Kaggle data includes seeds for that year
  - **For current year**: You must manually add the bracket to `MNCAATourneySeeds.csv` (see Installation step 4)
  - **For past years**: Seeds should already be in the Kaggle data

**Note:** The model can only simulate years where you have both:
1. Scraped team statistics (from `data_scraper.py`)
2. Tournament seeds (from Kaggle data or manually added)

## Configuration

### Scraping Different Years

Edit `data_scraper.py`:
```python
years = range(2010, 2025)  # Scrape 2010-2024
```

### Changing Model Features

Edit `prediction_model.py`:
```python
feature_columns = [
    'Overall_W-L%',
    'Overall_SRS',
    'Overall_SOS',
    # Add or remove features here
]
```

### Adjusting Simulation Parameters

Edit `bracket_sim.py`:
```python
YEAR = 2024  # Tournament year to simulate
SIMULATIONS_PER_GAME = 1000  # Monte Carlo simulations per game
```

## Project Structure

```
march-madness-model/
├── data_scraper.py           # Scrapes team statistics from Sports Reference
├── prediction_model.py       # Trains the prediction model
├── bracket_sim.py           # Runs bracket simulations
├── requirements.txt         # Python dependencies
├── install_requirements.sh  # Installation script
├── README.md               # This file
├── data/                   # Scraped team statistics (created by scraper)
│   ├── team_stats_2015.csv
│   ├── team_stats_2016.csv
│   └── ...
├── kaggle-data/           # Kaggle tournament data (you must download)
│   ├── MNCAATourneyCompactResults.csv
│   ├── MNCAATourneySeeds.csv
│   └── MTeams.csv
├── models/                # Trained models (created by training)
│   └── march_madness_model.pkl
└── results/              # Simulation results (created by simulation)
    ├── most_likely_bracket_2024.csv
    ├── game_by_game_bracket_2024.csv
    └── championship_probabilities_2024.csv
```

## Troubleshooting

### SSL Certificate Errors
The scraper handles SSL certificate issues automatically. If you still encounter errors, the code includes SSL verification bypass for Sports Reference.

### Team Name Mismatches
If you see "Warning: Could not find stats for [team]", the team names between Kaggle and scraped data don't match. The code includes comprehensive team name mappings, but you may need to add more in the `normalize_team_name()` function in `bracket_sim.py`.

### "Cannot find model file"
Make sure you've run `prediction_model.py` to train and save the model before running `bracket_sim.py`.

### Empty Predictions (All 50-50)
This usually means:
1. The model wasn't trained properly, or
2. Feature names don't match between training and prediction

Check that you're using the same years of data and that column names are consistent.

## Model Performance

Typical performance metrics:
- **Accuracy**: 65-70% on test set
- **Log Loss**: 0.55-0.65
- **Most Important Features**: Win-Loss %, SRS (Simple Rating System), SOS (Strength of Schedule)

## Advanced Usage

### Using Different Models

You can experiment with other models by editing `prediction_model.py`:

```python
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Replace LogisticRegression with:
model = RandomForestClassifier(n_estimators=100, random_state=42)
# or
model = XGBClassifier(n_estimators=100, random_state=42)
```

### Adding More Features

Common features to consider:
- Offensive/Defensive ratings (ORtg, DRtg)
- Four Factors (eFG%, TOV%, ORB%, FTr)
- Pace statistics
- Recent performance metrics

## Contributing

Contributions are welcome! Some ideas:
- Add more sophisticated models (neural networks, ensemble methods)
- Implement player-level statistics
- Add momentum/recency weighting
- Create visualization tools for brackets
- Improve team name matching

## License

MIT License - feel free to use for personal or educational purposes.

## Acknowledgments

- Data from [Sports Reference](https://www.sports-reference.com/cbb/)
- Tournament data from [Kaggle's March Madness Competition](https://www.kaggle.com/competitions/march-machine-learning-mania-2024)
- Inspired by the annual quest to predict the unpredictable

## Contact

Questions or issues? Open an issue on GitHub or contact [your email/contact info]

---

**Note:** This model is for educational and entertainment purposes. March Madness is notoriously unpredictable - use at your own risk! 🏀
