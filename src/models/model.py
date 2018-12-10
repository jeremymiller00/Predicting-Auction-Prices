import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_log_error, mean_squared_error

def rmsle(y_true, y_pred): 
    """Compute the Root Mean Squared Log Error of the y_pred and y_true values"""
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


def rmse(y_true, y_pred): 
    """Compute the Root Mean Squared Error of the y_pred and y_true values"""
    return np.sqrt(mean_squared_error(y_true, y_pred))

if __name__ == "__main__":
    
    # unzip and load the data
    unzip Train.zip
    df = pd.read_csv("data/Train.csv")
    
    features = [
        'SalesID' ,
        'MachineID' ,
        'ModelID' ,
        # 'datasource' ,
        'auctioneerID' ,
        'YearMade' ,
        # 'MachineHoursCurrentMeter',
        # 'UsageBand',
        # 'saledate',    # BE CAREFUL
        'fiModelDesc',
        # 'fiBaseModel',
        # 'fiSecondaryDesc',
        # 'fiModelSeries',
        # 'fiModelDescriptor',
        # 'ProductSize',
        # 'fiProductClassDesc',
        # 'state',
        # 'ProductGroup',
        # 'ProductGroupDesc',
        # 'Drive_System',
        # 'Enclosure',
        # 'Forks',
        # 'Pad_Type',
        # 'Ride_Control',
        # 'Stick',
        # 'Transmission',
        # 'Turbocharged',
        # 'Blade_Extension',
        # 'Blade_Width',
        # 'Enclosure_Type',
        # 'Engine_Horsepower',
        # 'Hydraulics',
        # 'Pushblock',
        # 'Ripper',
        # 'Scarifier',
        # 'Tip_Control',
        # 'Tire_Size',
        # 'Coupler',
        # 'Coupler_System',
        # 'Grouser_Tracks',
        # 'Hydraulics_Flow',
        # 'Track_Type',
        # 'Undercarriage_Pad_Width',
        # 'Stick_Length',
        # 'Thumb',
        # 'Pattern_Changer',
        # 'Grouser_Type',
        # 'Backhoe_Mounting',
        # 'Blade_Type',
        # 'Travel_Controls',
        # 'Differential_Type',
        # 'Steering_Controls'
    ]

    X_df, y_df = clean_features(df, ['YearMade', 'MachineID', 'ModelID', 'MachineHoursCurrentMeter', 'UsageBand'], 'SalePrice')
