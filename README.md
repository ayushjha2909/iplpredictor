# IPL Match Winner Probability Prediction

A mini machine learning project that predicts IPL match winner probability with a Random Forest classifier.

The upgraded model predicts the probability of `team1` beating `team2`, then shows the probability for both teams. This is better than a raw multi-team classifier because the output is always limited to the two teams actually playing.

## Project Structure

```text
.
|-- app.py
|-- data/
|   `-- sample_ipl_matches.csv
|-- models/
|   `-- .gitkeep
|-- requirements.txt
`-- src/
    |-- __init__.py
    |-- features.py
    |-- predict.py
    `-- train_model.py
```

## Features Used

Raw match inputs:

- `season`
- `city`
- `venue`
- `team1`
- `team2`
- `toss_winner`
- `toss_decision`
- `pitch_type`
- `expected_first_innings_score`
- `pace_assist`
- `spin_assist`
- `dew_factor`

Engineered factors:

- home advantage for team1 and team2
- neutral venue flag
- previous match result for both teams
- recent win rate from last 5 matches
- overall historical win rate
- head-to-head win rate
- venue win rate
- pitch-type win rate
- differences between both teams for form, head-to-head, venue, and pitch record

Target:

- `team1_win`, where `1` means `team1` won and `0` means `team2` won

## Setup in VS Code

Open this folder in VS Code:

```text
C:\Users\xande\OneDrive\Pictures\Documents\New project
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Train the Model

```powershell
python src\train_model.py
```

This saves the trained model at:

```text
models/ipl_winner_random_forest.joblib
```

## Predict from Command Line

```powershell
python src\predict.py --team1 "Mumbai Indians" --team2 "Chennai Super Kings" --city "Mumbai" --venue "Wankhede Stadium" --toss-winner "Mumbai Indians" --toss-decision "bat" --season 2024 --pitch-type batting --expected-score 184 --pace-assist 0.60 --spin-assist 0.35 --dew-factor 0.75
```

To see the engineered factors used by the model:

```powershell
python src\predict.py --team1 "Mumbai Indians" --team2 "Chennai Super Kings" --city "Mumbai" --venue "Wankhede Stadium" --toss-winner "Mumbai Indians" --toss-decision "bat" --season 2024 --show-factors
```

## Run the Streamlit App

```powershell
streamlit run app.py
```

Open the local URL shown by Streamlit, usually:

```text
http://localhost:8501
```

## Using a Real Dataset

The included CSV is a small demo dataset so the project runs immediately. For a stronger project, replace `data/sample_ipl_matches.csv` with a larger historical IPL dataset.

Minimum required columns:

```text
season,city,venue,team1,team2,toss_winner,toss_decision,winner
```

Optional pitch columns:

```text
pitch_type,expected_first_innings_score,pace_assist,spin_assist,dew_factor
```

If pitch columns are not present, the code fills them from venue profiles in `src/features.py`.

## Important Note

This project is now structurally better, but the sample CSV is still small. To get realistic predictions, train on a full IPL dataset with many seasons and add richer columns such as playing XI strength, injuries, batting first score trends, player form, and recent match dates.
