import glob
import pandas as pd
from pmi_data_prep import get_df

def combine_and_preprocess_jsons() -> pd.DataFrame:
    """
    Combines and preprocesses JSON files from the 'jsons' directory.
    """
    json_files = glob.glob('jsons/*.json')
    if not json_files:
        print("No JSON files found in the 'jsons' directory.")
        return

    print(f"Found {len(json_files)} JSON files to process.")
    
    combined_df = get_df(json_files)
    
    return combined_df

    # output_filename = 'combined_data.csv'
    # combined_df.to_csv(output_filename, index=False)
    
    # print(f"Successfully combined and preprocessed data into {output_filename}")

def concatenate_dataframes(df1:pd.DataFrame, df2:pd.DataFrame) -> pd.DataFrame:
    '''
    Docstring for concatenate_datatframes
    
    :param df1: Original dataframe
    :param df2: Dataframe with additional data

    This function checks for the intersection and only keeps the new rows.
    '''
    if df1['Date'].max() > df2['Date'].max():
        df1, df2 = df2, df1

    combined_df = pd.concat([df1, df2], ignore_index=True)
    final_df = combined_df[df2.columns].drop_duplicates()

    return final_df

if __name__ == '__main__':
    new_df = combine_and_preprocess_jsons()
    previous_df = pd.read_csv('data/PMI_Res_Transaction_w_STN.csv')
    df = concatenate_dataframes(previous_df, new_df)
