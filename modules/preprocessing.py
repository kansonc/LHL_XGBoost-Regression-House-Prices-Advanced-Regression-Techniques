"""Houses all used defined functions using in preprocessing
"""
# Imports
import itertools
import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
from scipy.stats import levene, shapiro


def set_working_directory():
    """Sets the directory to the parent folder"""
    os.path.dirname(os.path.abspath(__file__))


def extract_column_descriptions(column_names=list) -> dict:
    """
    returns a pairing of column name an its description. 
    """
    with open("data/data_description.txt", "r", encoding='UTF-8') as file:
        file_content = file.read()

    result = {}
    current_column = None

    for line in file_content.split('\n'):
        line = line.strip()
        if ":" in line:
            current_column = line.split(":")[0].strip()
            result[current_column] = [line.split(":")[1].strip()]
        elif current_column is not None and line:
            result[current_column].append(f'\n {line}')

    # Join the lines under each column into a single description
    for column, lines in result.items():
        result[column] = ' '.join(lines)

    # Filter based on the provided list of column names
    filtered_result = {column: description for column,
                       description in result.items() if column in column_names}
    return filtered_result


def print_column_descriptions(column_names=list):
    """takes a  list of columns and prints out descriptions"""
    output_dict = extract_column_descriptions(column_names)
    for column, description in output_dict.items():
        print(f"{column}: {description}\n")


def filter_quantitative_columns(dataframe):
    """
    Filter quantitative columns from the given dataframe.

    Parameters:
    - dataframe (pd.DataFrame): Input dataframe.

    Returns:
    pd.DataFrame: Dataframe containing only quantitative columns.
    list: List of columns that should be qualitative.
    """
    # Filter for quanitative values
    quantitative_columns_df = dataframe.map(
        lambda x: x if isinstance(x, (int, float)) else None)
    quantitative_columns_df = dataframe.select_dtypes(
        include=['float64', 'int64'])

    # Retrieve list of unfiltered columns (the ones that should be qualitative)
    missing_columns_both = missing_columns_from_dataframes(
        dataframe, quantitative_columns_df)
    return quantitative_columns_df, missing_columns_both


def missing_columns_from_dataframes(df1, df2):
    """returns a list of columns that are not present in both dataframes"""
    return list(set(df1.columns).symmetric_difference(set(df2.columns)))


def quant_cols_dtypes(quantitative_columns_df):
    """
    Store column names and datatypes from the quantitative columns dataframe.

    Parameters:
    quantitative_columns_df (pd.DataFrame): Training dataframe.

    Returns:
    dict: Dictionary containing column names as keys and their datatypes as values.
    """
    # Store column names and dtypes in list
    quant_columns_dtypes_dict = dict(
        zip(quantitative_columns_df.columns, quantitative_columns_df.dtypes))
    # Save the list of quantitative columns and their datatypes
    joblib.dump(quant_columns_dtypes_dict,
                'modules/joblib_preprocessing/quant_columns_dtypes_dict.joblib')
    return


def drop_columns_from_list(quant_df, columns_to_drop):
    """
    Drop columns from a DataFrame based on a given list.

    Parameters:
        df (pd.DataFrame): Data.
        columns_to_drop (list): List of column names to be dropped.

    Returns:
        pd.DataFrame: DataFrame with specified columns dropped.
    """
    if not columns_to_drop:
        return quant_df
    else:
        return quant_df.drop(columns=columns_to_drop, inplace=False)


def add_columns_to_dataframe(org_df, quant_df, columns_to_add):
    """
    Add columns to a DataFrame if the list is not empty.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        columns_to_add (list): List of columns to be added.

    Returns:
        pd.DataFrame: DataFrame with specified columns added.
    """
    if columns_to_add:
        return pd.concat([quant_df, org_df[columns_to_add]], axis=1)
    else:
        return quant_df


def select_columns_func(dataframe, quant_columns_dtypes_dict):
    """
    Filter the dataframe to return the columns and their
    datatypes as specified

    Parameters:
    - dataframe: dataframe of values
    - quant_columns_dtypes_dict: dictionary of column name and dtype pair

    Returns:
    pd.DataFrame: Test dataframe with only the specified columns and corresponding datatypes.
    """

    columns = list(quant_columns_dtypes_dict.keys())
    datatypes = list(quant_columns_dtypes_dict.values())

    # Create a copy of the test dataframe with selected columns
    df = dataframe[columns].copy()

    # Set datatypes for each column in the result dataframe
    for col, dtype in zip(columns, datatypes):
        if dtype == np.int64:
            df[col] = pd.to_numeric(
                df[col], errors='coerce', downcast='integer')
        elif dtype == np.float64:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(dtype)
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

    return df


def check_negative_values(df=pd.DataFrame):
    """ 
    Returns the dataframe with all negative values changed to np.NaN
    """
    # Check if any value in any column is less than 0
    df_no_neg = df.apply(lambda x: x.apply(lambda y: np.nan if y < 0 else y))
    return df_no_neg


def replace_zeros_with_nan(df=pd.DataFrame, columns_to_transform=list):
    """
    Replace all zero values in specified columns with np.NaN and
    return a new DataFrame along with a DataFrame containing
    original and new minimum values for each transformed column.
    """
    # Copy the input dataframe to keep the original unchanged
    transformed_df = df.copy()

    # Replace all zero values in specified columns with np.NaN
    transformed_df[columns_to_transform] = transformed_df[columns_to_transform].replace(
        0, np.NaN)

    # Create a dictionary to store the original and new minimum values
    min_dict = {"column name": [],
                "original min value": [], "new min value": []}

    # Populate the dictionary with the minimum values for each column
    for col in columns_to_transform:
        min_dict["column name"].append(col)
        min_dict["original min value"].append(df[col].min())
        min_dict["new min value"].append(transformed_df[col].min())

    # Create a DataFrame from the dictionary
    min_df = pd.DataFrame(data=min_dict)

    return transformed_df, min_df


def count_differences(df1, df2):
    """
    Compare two Pandas DataFrames and count the number of differences in each column.

    Parameters:
    - df1 (pd.DataFrame): The first DataFrame for comparison.
    - df2 (pd.DataFrame): The second DataFrame for comparison.

    Returns:
    dict: A dictionary where keys are column names and values are the count of differences
          between corresponding columns in df1 and df2.

    Raises:
    ValueError: If the DataFrames do not have the same shape.
    """
    # Check if the DataFrames have the same shape
    if df1.shape != df2.shape:
        raise ValueError("DataFrames must have the same shape")

    # Use a dictionary comprehension for counting differences
    differences_by_column = {
        column: (df1[column] != df2[column]).sum()
        for column in df1.columns
    }

    return differences_by_column


def replace_values_based_on_another_column(org_df, work_df, trans_df):
    """
    Replace values in one column based on the values of another column in the given DataFrame.
    """
    unique_qual_col_ref_list = trans_df['qual_col_ref'].unique()
    # Append reference columns to working df
    temp_df = pd.concat([work_df.reset_index(
        drop=True), org_df[unique_qual_col_ref_list].reset_index(drop=True)], axis=1)
    dataframe = temp_df.copy()
    # DataFrame to store counts of values changed to np.NaN per column
    changes_count_df = pd.DataFrame(
        columns=['function', 'column', 'num_val_changed'])
    # Use zip() to iterate over multiple lists simultaneously
    for col_name_to_trans, col_name_to_ref, value_to_ref in zip(
        trans_df['quant_col_to_trans'],
        trans_df['qual_col_ref'],
        trans_df['qual_col_and_val_dict']
    ):
        mask = (dataframe[[col_name_to_ref]] == value_to_ref).any(axis=1)
        if mask.any().item():
            dataframe.loc[mask, col_name_to_trans] = np.NaN
            # Count and store the number of values changed to np.NaN
            values_changed_count = mask.sum()
            changes_count_df = changes_count_df.append({
                'function': 'replace_values_based_on_another_column',
                'column': col_name_to_trans,
                'num_val_changed': values_changed_count
            }, ignore_index=True
            )
    dataframe.drop(columns=unique_qual_col_ref_list, inplace=True)
    return dataframe, changes_count_df


def outliers_iqr(data):
    """
    Removes outliers from the data using the IQR (Interquartile Range) method.
    Returns a DataFrame with outliers removed (replaced with NaN).
    """
    q1 = np.nanpercentile(data, 25)
    q3 = np.nanpercentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)

    lower_outliers_count = 0
    upper_outliers_count = 0

    data_no_outliers = pd.Series(dtype=data.dtype)

    for i, value in enumerate(data):
        if value < lower_bound or value > upper_bound:
            data_no_outliers[i] = np.NaN
            if value < lower_bound:
                lower_outliers_count += 1
            else:
                upper_outliers_count += 1
        else:
            data_no_outliers[i] = value

    return data_no_outliers, lower_outliers_count, upper_outliers_count


def plot_histogram(data, ax, colour):
    """
    Plots a histogram of the data.
    """
    ax.hist(data, bins=10, alpha=0.5, color=colour)
    ax.set_xlabel('Values')
    ax.set_ylabel('Frequency')
    ax.set_title('Outliers Included')


def plot_outlier_summary(data, column_name, ax, colour):
    """
    Plots a summary of outliers using a boxplot using Seaborn.
    """
    sns.boxplot(data=data, ax=ax, orient='h', color=colour)
    _, lower_outliers_count, upper_outliers_count = outliers_iqr(data)
    # Add text near the top right corner
    top_right_text = f'Upper Outliers: {upper_outliers_count}'
    ax.text(0.95, 0.95, top_right_text, transform=ax.transAxes, ha='right', va='top',
            bbox=dict(facecolor='white', edgecolor='white', boxstyle='round,pad=0.5'))

    # Add text near the bottom right corner
    bottom_right_text = f'Lower Outliers: {lower_outliers_count}'
    ax.text(0.95, 0.05, bottom_right_text, transform=ax.transAxes, ha='right', va='bottom',
            bbox=dict(facecolor='white', edgecolor='white', boxstyle='round,pad=0.5'))

    ax.set_xlabel('Values')
    ax.set_title(f'{column_name}')


def plot_histogram_with_outliers_removed(data, ax, colour):
    """
    Plots a histogram with outliers removed (replaced with NaN).
    """
    ax.hist(data.dropna(), bins=10, alpha=0.5, color=colour)
    ax.set_xlabel('Values')
    ax.set_ylabel('Frequency')
    ax.set_title('Outliers Removed')


def plot_subplots_for_columns(df):
    """
    Creates horizontal subplots for each column in the DataFrame.
    The subplots include a histogram, a summary of outliers, 
    and a histogram with outliers removed.
    """
    num_columns = df.shape[1]

    # Define a bold color cycler with at least 10 different colors using itertools.cycle
    colors = ['#FF0000', '#00FF00', '#0000FF', '#FF00FF', '#FFFF00',
              '#00FFFF', '#FF8000', '#8000FF', '#0080FF', '#FF0080']
    bold_color_cycler = itertools.cycle(colors)

    _, axes = plt.subplots(nrows=num_columns, ncols=3,
                           figsize=(12, num_columns * 4))

    for i, column in enumerate(df.columns):
        data = df[column]
        colour = next(bold_color_cycler)

        # Plot histogram
        plot_histogram(data, axes[i, 0], colour)

        # Plot outlier summary
        plot_outlier_summary(data, column, axes[i, 1], colour)

        # Plot histogram with outliers removed
        data_no_outliers, _, _ = outliers_iqr(data)
        plot_histogram_with_outliers_removed(
            data_no_outliers, axes[i, 2], colour)

        # Set the same y-axis for the first and third subplots
        axes[i, 2].set_ylim(axes[i, 0].get_ylim())

    plt.tight_layout(pad=3.0)

    # establish the path for viewing
    pdf_path = 'visualizations/data_exploration_outliers_histo_box.pdf'

    # Save the plots
    plt.savefig(pdf_path, bbox_inches='tight')


def remove_outliers(dataframe: pd.DataFrame, cols_not_to_remove_outliers: list) -> pd.DataFrame:
    """Removes Outliers based on the IQR formula

    Args:
        dataframe (pd.DataFrame): Tabular Data
        cols_not_to_remove_outliers (list): Columns specified not to have outliers removed

    Returns:
        pd.DataFrame: Tabular Data with outliers removed
    """
    # Remove columns not for transformation
    df1 = dataframe.drop(columns=cols_not_to_remove_outliers)
    # Create Dataframe to store transformations
    df2 = pd.DataFrame()
    # House the number of outliers removed
    outliers_removed = pd.DataFrame(
        columns=['column_name', 'num_outliers_removed'])
    # Transform remaining columns
    for _, column in enumerate(df1.columns):
        # Store transformations
        df2[column], upper_outlier_count, lower_outlier_count = outliers_iqr(
            df1[column])
        # Total the count of removed outliers
        sum_outliers_removed = upper_outlier_count + lower_outlier_count
        # Print if outliers removed
        if sum_outliers_removed > 0:
            # store changes
            data_dict = {'column_name': column,
                         'num_outliers_removed': sum_outliers_removed}
            outliers_removed = pd.concat(
                [outliers_removed, pd.DataFrame(data_dict, index=[0])], ignore_index=True)

    # Return transformed and non-transformed columns.
    return_df = pd.concat([df2.reset_index(drop=True),
                           dataframe[cols_not_to_remove_outliers].reset_index(drop=True)], axis=1)
    return return_df, outliers_removed


def convert_int64_to_float64(df1, df2):
    """
    Check if two dataframes have different numeric datatypes for the same columns.
    If numeric datatypes are different and one is int64, convert to float64.

    Parameters:
    - df1: First dataframe.
    - df2: Second dataframe.

    Returns:
    - df1_updated: First dataframe with possible datatype conversions.
    - df2_updated: Second dataframe with possible datatype conversions.
    """
    # Get common columns
    common_columns = set(df1.columns) & set(df2.columns)

    # Iterate through common columns
    for col in common_columns:
        dtype1 = df1[col].dtype
        dtype2 = df2[col].dtype

        # Check if datatypes are different and one is int64
        if dtype1 != dtype2 and ('int' in str(dtype1) or 'int' in str(dtype2)):
            # Convert to float64
            if 'int' in str(dtype1):
                df1[col] = df1[col].astype('float64')
            if 'int' in str(dtype2):
                df2[col] = df2[col].astype('float64')

    col_order = df2.columns
    temp_df = df1.copy()
    temp_df = temp_df.drop(columns=col_order)
    df1_1 = df1[col_order]
    df1_2 = pd.concat([df1_1, temp_df], axis=1)

    return df1_2, df2


def ordinal_hierarchy(org_df: pd.DataFrame, quant_df: pd.DataFrame, ordinal_list: list):
    '''
    Takes the dataset and a list with each list element formatted with 
    the column name [0] and a dictionary [1] containing the ordinal hierarchy.  
    Returns the transformed df with columns transformed with the 
    suffix "_ordinal".
    '''
    for item in ordinal_list:
        quant_df[item[0]] = org_df[item[0]].replace(item[1], regex=True)
        quant_df[item[0]] = pd.to_numeric(quant_df[item[0]], errors='coerce')
    return quant_df


def ordinal_hierarchy_enforce(quant_df: pd.DataFrame, coerce_list: list):
    '''
    Cleans ordinal columns to enforce ranges and integer values

    Args:
        quant_df: DataFrame containing the ordinal columns to be enforced.
        coerce_list: List of tuples where each tuple contains the column name and a dictionary
                     with 'high' and 'low' keys specifying the valid range.

    Returns:
        DataFrame with enforced ordinal hierarchy.
    '''
    for item in coerce_list:
        col = item[0]
        # Convert to numeric and coerce errors to NaN
        quant_df[col] = pd.to_numeric(quant_df[col], errors='coerce')
        # Check for decimals and convert to NaN
        quant_df[col] = quant_df[col].apply(
            lambda x: np.nan if pd.notna(x) and x % 1 != 0 else x)
        # Enforce the specified range
        quant_df[col] = quant_df[col].apply(
            lambda x, item=item: np.nan if pd.notna(x) and (
                x > item[1]['high'] or x < item[1]['low']) else x)
    return quant_df


def one_hot_encoding(df: pd.DataFrame, column_list: list):
    '''
    Args:
        Takes the dataframe and list of columns to encode, 
    Returns:
        Entire Dataframe containing all completed tranformations with the one-hot encoded columns appended
    '''
    df = pd.concat(
        [df, pd.get_dummies(df[column_list], dtype='Int64')], axis=1)
    df.drop(columns=column_list, inplace=True)
    return df


def load_preprocessing_variables():
    """load and return all the user input variables saved to joblib for preprocessing
    Returns:
        dictionary: (key: value)
            "drop": drop_quant_cols_list
            "add": add_quant_cols_list
            "encode": columns_to_encode
            "ord_trans": ordinal_cols_to_transform_list
            "ord_coerce": ordinal_quant_columns_to_coerce
            "zeroes_nan": zeros_to_nan_col_list
            "other_col_to_nan": df_cols_to_trans_nan
            "not_outliers": cols_not_to_remove_outliers"""

    # Load Joblibs
    preproc_dict = {
        'quanttypes': joblib.load(
            'modules/joblib_preprocessing/quant_columns_dtypes_dict.joblib'),
        "drop": joblib.load(
            'modules/joblib_preprocessing/drop_quant_cols_list.joblib'),
        "add": joblib.load(
            'modules/joblib_preprocessing/add_quant_cols_list.joblib'),
        "encode": joblib.load(
            'modules/joblib_preprocessing/columns_to_encode.joblib'),
        "ord_trans": joblib.load(
            'modules/joblib_preprocessing/ordinal_cols_to_transform_list.joblib'),
        "ord_coerce": joblib.load(
            'modules/joblib_preprocessing/ordinal_quant_columns_to_coerce.joblib'),
        "zeroes_nan": joblib.load(
            'modules/joblib_preprocessing/zeros_to_nan_col_list.joblib'),
        "other_col_to_nan": joblib.load(
            'modules/joblib_preprocessing/df_cols_to_trans_nan.joblib'),
        "not_outliers": joblib.load(
            'modules/joblib_preprocessing/cols_not_to_remove_outliers.joblib')
    }
    return preproc_dict


def data_clean(dataframe):
    """peforms all Feature Exploration data transformation functions"""
    pp_dict = load_preprocessing_variables()
    if 'SalePrice' not in dataframe.columns:
        del pp_dict['quanttypes']['SalePrice']
    org_df = dataframe.copy()
    dataframe = select_columns_func(dataframe, pp_dict['quanttypes'])
    dataframe = check_negative_values(dataframe)
    dataframe = drop_columns_from_list(dataframe, pp_dict['drop'])
    dataframe = add_columns_to_dataframe(org_df, dataframe, pp_dict['add'])
    dataframe = one_hot_encoding(dataframe, pp_dict['encode'])
    dataframe = ordinal_hierarchy(org_df, dataframe, pp_dict['ord_trans'])
    dataframe = ordinal_hierarchy_enforce(dataframe, pp_dict['ord_coerce'])
    dataframe, _ = replace_zeros_with_nan(dataframe, pp_dict['zeroes_nan'])
    dataframe, _ = replace_values_based_on_another_column(
        org_df, dataframe, pp_dict['other_col_to_nan'])
    dataframe, _ = remove_outliers(dataframe, pp_dict['not_outliers'])
    if 'SalePrice' in dataframe.columns:
        dataframe = dataframe.dropna(subset=['SalePrice'])
    return dataframe


def perform_anova(df, predictor_column: str, input_columns: list):
    """calculates and checks assumptions for ANOVA based on the qualitative columns (input_columns) 
    against the predictor column"""
    for column in input_columns:

        grouped_data = [df[predictor_column][
            df[column] == category] for category in df[column].unique()]

        # Perform ANOVA test
        fvalue, pvalue = stats.f_oneway(*grouped_data)

        # Check significance
        if pvalue < 0.05:
            # therefore at least one of means across the categorical values
            # is not relatively equal to the others.
            # Reject Null hypothesis

            _, pvalue_levene = levene(*grouped_data)

            if pvalue_levene > 0.05:
                # print(f"{column} variances are homogeneous (Levine's test)")
                print(
                    f"""{column}: 
    Passed the ANOVA p-value significance check, 
    Passed the homogeneity of variances check (Levene's Test), 
    Normality test (Shapiro-Wilk test for residuals) for each category within {column} are:""")
                for category in df[column].unique():
                    residuals = df[predictor_column][df[column] == category]
                    if len(residuals) >= 3:
                        _, pvalue_shapiro = shapiro(residuals)

                        if pvalue_shapiro < 0.05:
                            print(
                                f"{category}: NO, residuals are not normally distributed ({len(residuals)} values)")
                        else:
                            print(
                                f"{category}: YES, residuals are normally distributed ({len(residuals)} values)")
                    else:
                        print(
                            f'{category}: NA, normality cannot be calculated (has less than 3 values)')
                print("")
