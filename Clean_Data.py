import pandas as pd
import os

def clean_and_summarize_data():
    print("Initiating Data Cleaning Protocol...")
    
    # Setup paths directly to your Desktop
    desktop = os.path.join(os.environ['USERPROFILE'], 'Desktop')
    input_csv = os.path.join(desktop, "Final_Forensic_Data.csv")
    cleaned_csv = os.path.join(desktop, "Forensic_Data_Cleaned.csv")
    summary_csv = os.path.join(desktop, "Forensic_Data_Averages.csv")

    # Safety check
    if not os.path.exists(input_csv):
        print(f"❌ Error: Could not find {input_csv}")
        return

    # 1. Load the raw data
    try:
        df = pd.read_csv(input_csv)
        original_count = len(df)
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return

    # 2. Remove all exact duplicate rows caused by testing loops
    df_clean = df.drop_duplicates()
    new_count = len(df_clean)
    removed_count = original_count - new_count
    
    # Save the cleaned, object-level data
    df_clean.to_csv(cleaned_csv, index=False)
    print(f"✅ Scrubbed {removed_count} duplicate rows!")
    print(f"✅ Saved cleaned full dataset: {cleaned_csv}")

    # 3. Create a high-level summary (Averages per Image)
    print("\nCalculating microstructural averages...")
    
    # Group by the image and calculate the mean for the numbers
    summary = df_clean.groupby(['Sample', 'Group', 'Category']).agg({
        'Canal_ID': 'count',       # This counts how many canals were found in the image
        'Area': 'mean',            # Average size of the canals
        'DAPI_Ratio': 'mean',      # Average DAPI ratio
        'EGFP_Ratio': 'mean',      # Average EGFP ratio
        'Cy3_Ratio': 'mean'        # Average Cy3 ratio
    }).reset_index()

    # Rename the count column so it makes sense
    summary.rename(columns={'Canal_ID': 'Total_Canals_Found', 'Area': 'Average_Area'}, inplace=True)
    
    # Round the numbers to 3 decimal places for a clean look
    summary = summary.round(3)

    # Save the summary data
    summary.to_csv(summary_csv, index=False)
    print(f"✅ Saved averaged summary dataset: {summary_csv}")
    print("\n🚀 Data cleaning complete. You are ready for analysis!")

if __name__ == "__main__":
    clean_and_summarize_data()