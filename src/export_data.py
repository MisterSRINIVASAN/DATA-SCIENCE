import pandas as pd
from data import generate_tn_election_data

def export_to_csv():
    print("Generating synthetic TN Election data for PowerBI/Tableau...")
    df = generate_tn_election_data(num_samples=2000)
    
    # Add a descriptive label column for the target
    df['Winner_Label'] = df['winner'].map({
        0: 'DMK', 1: 'ADMK', 2: 'TVK', 
        3: 'NTK', 4: 'PMK', 5: 'BJP', 6: 'INC'
    })
    
    output_path = "tn_election_data.csv"
    df.to_csv(output_path, index=False)
    print(f"Data successfully exported to {output_path}")

if __name__ == "__main__":
    export_to_csv()
