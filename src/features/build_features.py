"""
This script loads in the training data and transforms it so that it can be used to trian models.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

def one_hot(input_df, columns):
    """
    Returns a dataframe with categorical columns transformed to dummy variables.
    """
    df = input_df.copy()

    for col in columns:
        dummies = pd.get_dummies(df[col].str.lower())
        dummies.drop(dummies.columns[-1], axis=1, inplace=True)
        df = df.drop(col, axis=1).merge(dummies, left_index=True, right_index=True)
    
    return df

def fill_features(df, features):
    """
    Standardize data frame to a list of given features. Features in DataFrame but
    not in features list will be dropped. Features not in the dataframe but in the
    list will be created.
    """
    existing = df.columns

    # Drop untrained features
    for feat in existing:
        if feat not in features:
            df.drop(feat, axis=1, inplace=True)

    # Add missing features
    for feat in features:
        if feat not in existing:
            df[feat] = 0

    # Return with column selection for ordering
    return df[features]

def clean_features(dataframe, features, target=None, fill=None):
    """Input features DataFrame, clean features by columns, 
    and return cleaned DataFrame.
    
    dataframe:      DataFrame containing all raw features including target.
    features:       list; features to clean.
    target:         str; name of target column.
    fill:            list; names of features to fill in zeros for if corresponding
                        columns not created during dummy variable creation.
    
    Returns: X_df; DataFrame of cleaned features
             Y_df; DataFrame of target variable
    """

    # Copy and split data frames
    X_df = dataframe[features].copy()
    if target:
        y_df = dataframe[target].copy()

    # Create dummy features
    dummies = DUMMY_FEATS.intersection(set(features))
    if dummies:
        X_df = one_hot(X_df, dummies)
    
    # Fill missing dummy features
    if fill:
        X_df = fill_features(X_df, fill)

    # Replace YearMade == 1000 with NaN
    if 'YearMade' in features:
        X_df.loc[X_df['YearMade'] == 1000, 'YearMade'] = X_df.loc[X_df['YearMade'] > 1000, 'YearMade'].median()

    # Parse year from datetime sold
    if 'saledate' in features:
        X_df['SaleYear'] = pd.to_datetime(X_df['saledate']).dt.year
        X_df['SaleMonth'] = pd.to_datetime(X_df['saledate']).dt.month
        X_df.drop('saledate', axis=1, inplace=True)

    ## All features
    # Impute NaN values with median
    X_df.fillna(X_df.median(axis=0), axis=0, inplace=True)

    if target:
        return X_df, y_df
    else:
        return X_df

##################################################################
if __name__ == "__main__":
    
    features = [
        'SalesID',      # Always included as key for predictions
        'MachineID',
        'ModelID',      # These features were selected via RFE
        'YearMade',
        'MachineHoursCurrentMeter',
        'UsageBand',
        'ProductSize',
        'ProductGroup',
        'Enclosure',
        'Enclosure_Type',
        'Hydraulics',
        'Tire_Size',
        ]

    DUMMY_FEATS = { # list all categorical features
                'UsageBand', 'fiModelDesc',
                'fiBaseModel', 'fiSecondaryDesc', 'fiModelSeries',
                'fiModelDescriptor', 'ProductSize', 'fiProductClassDesc',
                'state', 'ProductGroup', 'ProductGroupDesc',
                'Drive_System', 'Enclosure', 'Forks',
                'Pad_Type', 'Ride_Control', 'Stick',
                'Transmission', 'Turbocharged', 'Blade_Extension',
                'Blade_Width', 'Enclosure_Type', 'Engine_Horsepower',
                'Hydraulics', 'Pushblock', 'Ripper',
                'Scarifier', 'Tip_Control', 'Tire_Size',
                'Coupler', 'Coupler_System', 'Grouser_Tracks',
                'Hydraulics_Flow', 'Track_Type', 'Undercarriage_Pad_Width',
                'Stick_Length', 'Thumb', 'Pattern_Changer',
                'Grouser_Type', 'Backhoe_Mounting', 'Blade_Type',
                'Travel_Controls', 'Differential_Type', 'Steering_Controls'}

    # load the data
    df = pd.read_csv("data/raw/Train.csv")
    
    # Preprocess features and target
    X_df, y_df = clean_features(df, features, 'SalePrice')

    # Save list of dummy variables and numeric features
    trained_features = list(X_df.columns)

    # Separate `SalesID` for mapping back to predictions
    X_sid = X_df.pop('SalesID')

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_df, y_df)

    # write out the transformed data
    X_train.to_csv('data/processed/X_train.csv')
    y_train.to_csv('data/processed/y_train.csv')
    X_test.to_csv('data/processed/X_test.csv')
    y_test.to_csv('data/processed/y_test.csv')
    X_sid.to_csv('data/processed/X_sid.csv')

    