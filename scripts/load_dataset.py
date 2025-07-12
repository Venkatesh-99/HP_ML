import pandas as pd

def load_dataset(filepath, sheet_name):
    """
    Load a dataset from an Excel file.

    Parameters:
    - filepath: str, path to the Excel file.
    - sheet_name: str or None, name of the sheet to load. If None, loads the first sheet.

    Returns:
    - DataFrame containing the loaded data.
    """
    try:
        df = pd.read_excel(filepath, sheet_name=sheet_name)
        # return df.copy()

        cols = df.columns.tolist()
        index = df.columns.get_loc("Phenotype")
        new_order = cols[:index] + cols[index+1:] + [cols[index]]
        df = df[new_order]

        # Fix continent names
        df['Continent'] = df['Continent'].replace('America', 'South America')

        # Standardize Phenotype values
        df['Phenotype'] = df['Phenotype'].replace({
            'Gastric cancer ' : 'Gastric cancer',
            'Non-gastric cancer ' : 'Non-gastric cancer',
            'non-gastric cancer ' : 'Non-gastric cancer'
        })

        # Capitalize Sex values
        df['Sex'] = df['Sex'].str.title()

        # Remove rows with Sex as "Not Applicable"
        df = df[df['Sex'] != 'Not Applicable']

        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

# def clean_data(df):
#     """
#     Clean the dataset by fixing continent names, standardizing labels, make 'Sex' column same case, and
#     remove rows which has Sex as 'Not Applicable'.

#     Parameters:
#     - df: DataFrame, the dataset to clean.
#     Returns:
#     - DataFrame, the cleaned dataset.
#     """
#     # Move the label column "Phenotype" to the end
#     cols = df.columns.tolist()
#     index = df.columns.get_loc("Phenotype")
#     new_order = cols[:index] + cols[index+1:] + [cols[index]]
#     df = df[new_order]

#     # Fix continent names
#     df['Continent'] = df['Continent'].replace('America', 'South America')

#     # Standardize Phenotype values
#     df['Phenotype'] = df['Phenotype'].replace({
#         'Gastric cancer ' : 'Gastric cancer',
#         'Non-gastric cancer ' : 'Non-gastric cancer',
#         'non-gastric cancer ' : 'Non-gastric cancer'
#     })

#     # Capitalize Sex values
#     df['Sex'] = df['Sex'].str.title()

#     # Remove rows with Sex as "Not Applicable"
#     df = df[df['Sex'] != 'Not Applicable']

#     return df
    
