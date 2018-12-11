# Auction Price Prediction

This was a one-day team case study to predict auction prices of heavy farming equipment. It demonstrates the main aspects of regression analysis.

The data set was challenging. There are many categorical features with low incidence. A significant amount of numerical features values are missing. Nonetheless we were able to create a regression model which performed somewhat better than simply predicting the mean. A comprehensive solution the problem would certainly require more complete data.

Our unitial regression model performed reasonable well. We achieved a RMSE of ~$15,599 compared to the standard deviation of the prices in the data set of ~$23,037. The residual plots and QQ plot led to question to quality of fit and explore regularized options:

![residual](reports/figures/baseline/bl_target_resid.png "Baseline Model Residuals")
![residual](reports/figures/baseline/model_id_resid.png "Baseline Model Residuals")
![residual](reports/figures/baseline/year_resid.png "Baseline Model Residuals")
![residual](reports/figures/baseline/bl_qq_plot.png "Baseline Model Residuals")
