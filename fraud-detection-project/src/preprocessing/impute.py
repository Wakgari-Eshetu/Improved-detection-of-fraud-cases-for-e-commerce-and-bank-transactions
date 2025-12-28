def impute_missing_values(data, strategy='mean', columns=None):
    """
    Impute missing values in the dataset.

    Parameters:
    - data: pandas DataFrame containing the dataset.
    - strategy: str, the imputation strategy ('mean', 'median', 'most_frequent', 'constant').
    - columns: list of str, specific columns to impute. If None, all columns with missing values will be imputed.

    Returns:
    - data: pandas DataFrame with imputed values.
    """
    from sklearn.impute import SimpleImputer

    imputer = SimpleImputer(strategy=strategy)
    
    if columns is None:
        columns = data.columns[data.isnull().any()]

    data[columns] = imputer.fit_transform(data[columns])
    
    return data