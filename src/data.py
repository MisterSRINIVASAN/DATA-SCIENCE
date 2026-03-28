import pandas as pd
import numpy as np

def generate_tn_election_data(num_samples: int = 1000, random_seed: int = 42) -> pd.DataFrame:
    """
    Generates a synthetic dataset modeling Tamil Nadu 2026 Assembly Election predictions.
    
    Features:
    - dmk_strength (0-100)
    - admk_strength (0-100)
    - tvk_strength (0-100)
    - ntk_strength (0-100)
    - pmk_strength (0-100)
    - bjp_strength (0-100)
    - inc_strength (0-100)
    - literacy_rate (50-100 %)
    - urban_population_percentage (10-100 %)
    - anti_incumbency_factor (0-10)
    - campaign_spending_index (1-10)
    
    Target:
    - winner: 0 (DMK), 1 (ADMK), 2 (TVK), 3 (NTK), 4 (PMK), 5 (BJP), 6 (INC)
    """
    np.random.seed(random_seed)
    
    # Generate features
    dmk_strength = np.random.uniform(30, 90, num_samples)
    admk_strength = np.random.uniform(30, 90, num_samples)
    tvk_strength = np.random.uniform(10, 80, num_samples)
    ntk_strength = np.random.uniform(5, 40, num_samples)
    pmk_strength = np.random.uniform(0, 50, num_samples) # PMK is highly concentrated, so 0 in many places, 50 in Vanniyar belts
    bjp_strength = np.random.uniform(5, 60, num_samples)
    inc_strength = np.random.uniform(5, 40, num_samples)
    
    literacy_rate = np.random.uniform(60, 98, num_samples)
    urban_population_percentage = np.random.uniform(15, 95, num_samples)
    anti_incumbency_factor = np.random.uniform(1, 10, num_samples)
    campaign_spending_index = np.random.uniform(2, 10, num_samples)
    
    # Create synthetic scores based on realistic 2026 political landscape
    
    # DMK (Ruling Party): Strong cadre base, high spending power, but faces anti-incumbency (10 years rule potentially or 5 years). 
    # Strong in both urban and rural, but anti-incumbency is the biggest headwind.
    dmk_score = (
        dmk_strength * 1.5 + 
        (campaign_spending_index * 4) +
        (urban_population_percentage * 0.2) - 
        (anti_incumbency_factor * 12)
    )
    
    # ADMK (Main Opposition): Benefits directly from anti-incumbency. Strong rural and semi-urban base.
    admk_score = (
        admk_strength * 1.4 + 
        (campaign_spending_index * 3) + 
        ((100 - urban_population_percentage) * 0.3) + # Slight rural advantage
        (anti_incumbency_factor * 8)
    )

    # TVK (New Challenger - Vijay): High youth/literacy appeal, very high anti-incumbency absorption. Strong in urban.
    tvk_score = (
        tvk_strength * 1.3 + 
        (urban_population_percentage * 0.4) + 
        (literacy_rate * 0.3) +
        (anti_incumbency_factor * 7)
    )
    
    # NTK (Naam Tamilar Katchi): Ideological core, appeals to youth and Tamil nationalism.
    # Benefits from anti-incumbency, less reliant on campaign spending, more organic ground appeal.
    ntk_score = (
        ntk_strength * 1.6 + 
        (anti_incumbency_factor * 5) +
        (literacy_rate * 0.2) -
        (campaign_spending_index * 1) # Often outspent
    )
    
    # PMK (Pattali Makkal Katchi): Highly regionalized (northern TN). High strength means high chance of winning there.
    # Usually alliances matter, but assuming independent here.
    pmk_score = (
        pmk_strength * 2.0 + # Highly dependent on base strength
        ((100 - urban_population_percentage) * 0.4) # Stronger in rural/semi-urban belts
    )
    
    # BJP (Bharatiya Janata Party): Trying to grow. High spending, relies on urban pockets and specific demographics.
    bjp_score = (
        bjp_strength * 1.2 +
        (urban_population_percentage * 0.5) +
        (campaign_spending_index * 5) - # Huge spending impact
        (anti_incumbency_factor * 2) # Often seen as incumbent at the center
    )
    
    # INC (Indian National Congress): Legacy votes, usually dependent on alliances.
    inc_score = (
        inc_strength * 1.1 +
        (campaign_spending_index * 2) +
        (literacy_rate * 0.1)
    )
    
    # Add random noise
    noise_dmk = np.random.normal(0, 15, num_samples)
    noise_admk = np.random.normal(0, 15, num_samples)
    noise_tvk = np.random.normal(0, 15, num_samples)
    noise_ntk = np.random.normal(0, 10, num_samples)
    noise_pmk = np.random.normal(0, 15, num_samples)
    noise_bjp = np.random.normal(0, 15, num_samples)
    noise_inc = np.random.normal(0, 10, num_samples)
    
    final_dmk = dmk_score + noise_dmk
    final_admk = admk_score + noise_admk
    final_tvk = tvk_score + noise_tvk
    final_ntk = ntk_score + noise_ntk
    final_pmk = pmk_score + noise_pmk
    final_bjp = bjp_score + noise_bjp
    final_inc = inc_score + noise_inc
    
    # Determine winner: argmax of scores
    scores = np.vstack([final_dmk, final_admk, final_tvk, final_ntk, final_pmk, final_bjp, final_inc])
    winner = np.argmax(scores, axis=0)
    
    data = {
        'dmk_strength': np.round(dmk_strength, 2),
        'admk_strength': np.round(admk_strength, 2),
        'tvk_strength': np.round(tvk_strength, 2),
        'ntk_strength': np.round(ntk_strength, 2),
        'pmk_strength': np.round(pmk_strength, 2),
        'bjp_strength': np.round(bjp_strength, 2),
        'inc_strength': np.round(inc_strength, 2),
        'literacy_rate': np.round(literacy_rate, 2),
        'urban_population_percentage': np.round(urban_population_percentage, 2),
        'anti_incumbency_factor': np.round(anti_incumbency_factor, 2),
        'campaign_spending_index': np.round(campaign_spending_index, 2),
        'winner': winner
    }
    
    df = pd.DataFrame(data)
    return df

def get_train_test_split(test_size: float = 0.2, random_seed: int = 42):
    """
    Generates the dataset and returns a train/test split.
    """
    from sklearn.model_selection import train_test_split
    
    # Generate a larger dataset to ensure we get samples of all 7 classes
    df = generate_tn_election_data(num_samples=5000, random_seed=random_seed)
    
    X = df.drop(columns=['winner'])
    y = df['winner']
    
    # We use stratify to make sure all classes are represented in train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    df = generate_tn_election_data(10)
    print("Sample generated TN Election Data:")
    print(df.head())
    print("\nClass distribution:")
    print(df['winner'].value_counts())

