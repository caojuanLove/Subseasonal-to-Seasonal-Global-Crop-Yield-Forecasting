from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import pandas as pd
import os
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import ast
import optuna
from optuna_integration.keras import KerasPruningCallback
from optuna.samplers import TPESampler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from scikeras.wrappers import KerasRegressor
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta, date


################################# LSTM Model Parameters ############################################################################


# Use Optuna for hyperparameter optimization, modify the storage path to the specified SQLite database path


################ Function Definitions ##################################################
def calculate_mare(y_actual, y_predicted):
    """
    Calculate Mean Absolute Relative Error (MARE).

    Parameters:
    y_actual (array-like): Actual values
    y_predicted (array-like): Predicted values

    Returns:
    float: MARE value
    """
    # Ensure no division by zero errors
    if any(y_actual == 0):
        raise ValueError("Actual values contain zeros, which may cause division by zero errors.")

    # Calculate MARE
    mare = np.mean(np.abs((y_actual - y_predicted) / y_actual))
    return mare


def calculate_region_bounds(gdf, offset=2):  # Added a 2° offset to fully cover the study area
    """
    Calculate the geographic bounds of a specified region in a GeoDataFrame with an added offset.

    :param gdf: A GeoDataFrame containing geographic data.
    :param offset: A numeric value indicating the amount to expand the bounds by (default is 2).
    :return: A string representing the geographic bounds (N/W/S/E) of the region.
    """
    [minx, miny, maxx, maxy] = gdf.total_bounds  # Extract bounds

    # Apply offset and round the coordinates
    minx = round(minx - offset, 1)
    miny = round(miny - offset, 1)
    maxx = round(maxx + offset, 1)
    maxy = round(maxy + offset, 1)

    # Format the output as 'N/W/S/E'
    region_limits = f"{maxy}/{minx}/{miny}/{maxx}"
    return region_limits


def filter_dates(inputpath_base, hyear, harvest_point, institution, weeks):
    # Construct file path
    file_path = os.path.join(inputpath_base, '02_S2S', '01_dataori', 'ECMWF', 'CommonYear_Week.txt')

    # Read the CommonYear_Week.txt file
    with open(file_path, 'r') as file:
        lines = [line.strip() for line in file.readlines()]

    # Get end_dates from lines
    if harvest_point == 46:
        end_dates = hyear + '-' + '12-31'
    else:
        end_dates = hyear + '-' + lines[harvest_point]  # Extract the date corresponding to harvest_point+1

    # Read the Forecast_ECWMF_03.txt file
    file_path = os.path.join(inputpath_base, '02_S2S', '01_dataori', institution, 'Forecast_ECWMF_03.txt')
    with open(file_path, 'r') as file:
        lines = [line.strip() for line in file.readlines()]

    # Calculate date range
    end_date = datetime.strptime(end_dates, '%Y-%m-%d')
    eight_weeks_before = end_date - timedelta(days=8 * weeks - 1)  # 48 days

    # Filter dates within the given date range
    filtered_dates = [date for date in lines if eight_weeks_before < datetime.strptime(date, '%Y-%m-%d') < end_date]

    return filtered_dates


def retrieve_mx2t6_mn2t6(date, parameters, variable, hdates, inputpath_base, origen, step, num, region_limits, region,
                         country, institution, years):
    ECMWF_path = os.path.join(inputpath_base, '02_S2S', '02_Reforecast', institution, region, hdates[-5:])
    os.makedirs(ECMWF_path, exist_ok=True)
    save_path_pf = [ECMWF_path, '\\', institution, "_", variable, "_", country, "_pf_forecasttimefcst_", years, ".grib"]
    save_path_cf = [ECMWF_path, '\\', institution, "_", variable, "_", country, "_cf_forecasttimefcst_", years, ".grib"]
    from ecmwfapi import ECMWFDataServer
    server = ECMWFDataServer()
    string = ""
    server.retrieve({
        "class": "s2",
        "dataset": "s2s",
        "date": date,  # Should be the model version date
        "origin": origen,
        "expver": "prod",
        "hdate": hdates,  # Model initialization time (backward correction time)
        "levtype": 'sfc',
        "model": "glob",
        "param": parameters,  # Separate variables
        "step": step,
        "stream": "enfh",
        "time": "00:00:00",
        "type": "cf",  # Control Forecast
        "area": region_limits,
        "target": string.join(save_path_cf),  # File name to be modified
        # "grid": "0.125/0.125"
    })
    print(variable + ' cf download completed')

    string = ""

    server.retrieve({
        "class": "s2",
        "dataset": "s2s",
        "date": date,
        "expver": "prod",
        "hdate": hdates,  # Model initialization time (backward correction time)
        "levtype": "sfc",
        "model": "glob",
        "origin": origen,
        "param": parameters,  # Separate variables
        "step": step,
        "stream": "enfh",
        "time": "00:00:00",
        "number": num,
        "type": "pf",  # Perturbed Forecast
        "area": region_limits,
        "target": string.join(save_path_pf),  # File name to be modified
        #  "grid": "0.125/0.125"
    })
    print(variable + ' pf download completed')


def retrieve_tp_10_u_10v_ssr(date, parameters, variable, hdates, inputpath_base, origen, step, num, region_limits,
                             region, country, institution, years):
    ECMWF_path = os.path.join(inputpath_base, '02_S2S', '02_Reforecast', institution, region, hdates[-5:])
    os.makedirs(ECMWF_path, exist_ok=True)
    save_path_pf = [ECMWF_path, '\\', institution, "_", variable, "_", country, "_pf_forecasttimefcst_", years, ".grib"]
    from ecmwfapi import ECMWFDataServer
    server = ECMWFDataServer()
    string = ""
    server.retrieve({
        "class": "s2",
        "dataset": "s2s",
        "date": date,
        "expver": "prod",
        "hdate": hdates,  # Model initialization time (backward correction time)
        "levtype": "sfc",
        "model": "glob",
        "origin": origen,
        "param": parameters,  # Separate variables
        "step": step,
        "stream": "enfh",
        "time": "00:00:00",
        "number": num,
        "type": "pf",  # Perturbed Forecast
        "area": region_limits,
        "target": string.join(save_path_pf),  # File name to be modified
        #  "grid": "0.125/0.125"
    })
    print(variable + ' pf download completed')

    # Download control forecast (CF), null value mode
    string = ""
    save_path_cf = [ECMWF_path, '\\', institution, "_", variable, "_", country, "_cf_forecasttimefcstt_", years,
                    ".grib"]
    server.retrieve({
        "class": "s2",
        "dataset": "s2s",
        "date": date,
        "origin": origen,
        "hdate": hdates,  # Model initialization time (backward correction time)
        "expver": "prod",
        "levtype": "sfc",
        "model": "glob",
        "param": parameters,  # Separate variables
        "step": step,
        "stream": "enfh",
        "time": "00:00:00",
        "type": "cf",  # Control Forecast
        "area": region_limits,
        "target": string.join(save_path_cf),  # File name to be modified
        #  "grid": "0.125/0.125"
    })
    print(variable + ' cf download completed')


def retrieve_2d_2t(date, parameters, variable, hdates, inputpath_base, origen, step, num, region_limits, region,
                   country, institution, years):
    ECMWF_path = os.path.join(inputpath_base, '02_S2S', '02_Reforecast', institution, region, hdates[-5:])
    os.makedirs(ECMWF_path, exist_ok=True)
    save_path_pf = [ECMWF_path, '\\', institution, "_", variable, "_", country, "_pf_forecasttimefcst_", years, ".grib"]
    from ecmwfapi import ECMWFDataServer
    server = ECMWFDataServer()
    string = ""
    server.retrieve({
        "class": "s2",
        "dataset": "s2s",
        "date": date,
        "expver": "prod",
        "levtype": "sfc",
        "model": "glob",
        "origin": origen,
        "hdate": hdates,  # Model initialization time (backward correction time)
        "param": parameters,  # Separate variables
        "step": step,
        "stream": "enfh",
        "time": "00:00:00",
        "number": num,
        "type": "pf",  # Perturbed Forecast
        "area": region_limits,
        "target": string.join(save_path_pf),  # File name to be modified
        #  "grid": "0.125/0.125"
    })
    print(variable + ' pf download completed')

    # Download control forecast (CF), null value mode
    string = ""
    save_path_cf = [ECMWF_path, '\\', institution, "_", variable, "_", country, "_cf_forecasttimefcst_", years, ".grib"]
    server.retrieve({
        "class": "s2",
        "dataset": "s2s",
        "date": date,
        "origin": origen,
        "hdate": hdates,  # Model initialization time (backward correction time)
        "expver": "prod",
        "levtype": "sfc",
        "model": "glob",
        "param": parameters,  # Separate variables
        "step": step,
        "stream": "enfh",
        "time": "00:00:00",
        "type": "cf",  # Control Forecast
        "area": region_limits,
        "target": string.join(save_path_cf),  # File name to be modified
        # "grid": "0.125/0.125"
    })
    print(variable + ' cf download completed')


################################# Extraction of GS Growing Period Data ############################################################################
def prepare_and_model_data(startyear, start_point, harvest_point, data_yield_ori_Visall, VI_select2, static):
    # Define static features

    if start_point < harvest_point:
        # Generate list of weekly data features
        # ['_CDD' ,'_HDD' ,'_GDD']
        weeks = [f'Week{week}_Pre' for week in range(start_point, harvest_point + 1)] + \
                [f'Week{week}_Tmin' for week in range(start_point, harvest_point + 1)] + \
                [f'Week{week}_Solar' for week in range(start_point, harvest_point + 1)] + \
                [f'Week{week}_Tmean' for week in range(start_point, harvest_point + 1)] + \
                [f'Week{week}_VPD' for week in range(start_point, harvest_point + 1)] + \
                [f'Week{week}_wind_speed' for week in range(start_point, harvest_point + 1)] + \
                [f'Week{week}_Tmax' for week in range(start_point, harvest_point + 1)] + \
                [f'Week{week}_CDD' for week in range(start_point, harvest_point + 1)] + \
                [f'Week{week}_HDD' for week in range(start_point, harvest_point + 1)] + \
                [f'Week{week}_GDD' for week in range(start_point, harvest_point + 1)] + \
                [f'Week{week}_SPEI' for week in range(start_point, harvest_point + 1)] + \
                [f'Week{week}{VI_select2}' for week in range(start_point, harvest_point + 1)]
        #  [f'Week{week}{VI_select1}' for week in range(start_point, harvest_point+1)] + \
        data_yield_ori_selected = data_yield_ori_Visall[weeks + static]
    else:
        # Reconstruct weekly features for cross-year growing periods
        weeks_select_list = list(range(start_point, 47)) + list(range(1, harvest_point + 1))
        weeks = [f'Week{week}_Pre' for week in weeks_select_list] + \
                [f'Week{week}_Tmin' for week in weeks_select_list] + \
                [f'Week{week}_Solar' for week in weeks_select_list] + \
                [f'Week{week}_Tmean' for week in weeks_select_list] + \
                [f'Week{week}_VPD' for week in weeks_select_list] + \
                [f'Week{week}_wind_speed' for week in weeks_select_list] + \
                [f'Week{week}_Tmax' for week in weeks_select_list] + \
                [f'Week{week}_CDD' for week in weeks_select_list] + \
                [f'Week{week}_HDD' for week in weeks_select_list] + \
                [f'Week{week}_GDD' for week in weeks_select_list] + \
                [f'Week{week}_SPEI' for week in weeks_select_list] + \
                [f'Week{week}{VI_select2}' for week in weeks_select_list]
        data_yield_ori_selected = data_yield_ori_Visall[weeks + static]
        # Rename and impute missing dynamic features for the year prior to startyear using mean imputation
        weeks1 = [f'Week{week}_Pre' for week in range(start_point, 47)] + \
                 [f'Week{week}_Tmin' for week in range(start_point, 47)] + \
                 [f'Week{week}_Solar' for week in range(start_point, 47)] + \
                 [f'Week{week}_Tmean' for week in range(start_point, 47)] + \
                 [f'Week{week}_VPD' for week in range(start_point, 47)] + \
                 [f'Week{week}_wind_speed' for week in range(start_point, 47)] + \
                 [f'Week{week}_Tmax' for week in range(start_point, 47)] + \
                 [f'Week{week}_CDD' for week in range(start_point, 47)] + \
                 [f'Week{week}_HDD' for week in range(start_point, 47)] + \
                 [f'Week{week}_GDD' for week in range(start_point, 47)] + \
                 [f'Week{week}_SPEI' for week in range(start_point, 47)] + \
                 [f'Week{week}{VI_select2}' for week in range(start_point, 47)]
        mean = data_yield_ori_selected[
            weeks1].mean()  # Impute missing dynamic features for the year prior to startyear using mean imputation
        data_yield_ori_Visall_lastyear = data_yield_ori_selected[weeks1 + ['year', 'idJoin']]
        data_yield_ori_Visall_lastyear['year'] = data_yield_ori_Visall_lastyear['year'] + 1
        # data_yield_ori_Visall_lastyear = data_yield_ori_Visall_lastyear[data_yield_ori_Visall_lastyear['year']2002]

        data_lastyear_2000 = data_yield_ori_Visall_lastyear.groupby(['idJoin']).mean().reset_index()
        data_lastyear_2000['year'] = startyear
        data_lastyear_2000 = data_lastyear_2000[weeks1 + ['year', 'idJoin']]
        data_yield_ori_Visall_lastyear = pd.concat([data_lastyear_2000, data_yield_ori_Visall_lastyear])

        data_yield_ori_Visall_currentyears = data_yield_ori_selected.drop(weeks1, axis=1)
        data_yield_ori_selected = data_yield_ori_Visall_lastyear.merge(data_yield_ori_Visall_currentyears,
                                                                       on=['year', 'idJoin'], how='inner')
    # Select data
    # Export data as CSV file
    return data_yield_ori_selected


########################## Output Only Three Vegetation Indices ######################################################

def find_nearest_point_left_of_target(change_points, target):
    # Filter all points less than the target value
    points_left_of_target = change_points[change_points < target]

    if len(points_left_of_target) == 0:
        # Return None if no points are found to the left of the target
        return None

    # Calculate the absolute differences between these points and the target
    diffs = np.abs(points_left_of_target - target)

    # Find the index of the minimum difference
    min_diff_index = np.argmin(diffs)

    # Return the point closest to the target
    return points_left_of_target[min_diff_index]


def plot_vis_correlation_with_yield1(data_yield_ori_Visall, outpath_fig3):
    # Extract correlation data
    corrdata = data_yield_ori_Visall.drop(['Country', 'idJoin'], axis=1)
    y = data_yield_ori_Visall[Yield_type]
    correlation_results = corrdata.corrwith(y)

    # Add Glass LAI and PML_GPP (if applicable)
    weeks = [f'Week{week}_EVI' for week in range(1, 47)] + \
            [f'Week{week}_KNDVI' for week in range(1, 47)] + \
            [f'Week{week}_NDVI' for week in range(1, 47)]

    filtered_correlation = correlation_results.loc[weeks]

    # Extract data for EVI, KNDVI, and NDVI separately
    evi_data = filtered_correlation[filtered_correlation.index.str.contains('_EVI')]
    kndvi_data = filtered_correlation[filtered_correlation.index.str.contains('_KNDVI')]
    ndvi_data = filtered_correlation[filtered_correlation.index.str.contains('_NDVI')]

    # Extract week labels
    weeks_labels = [f'Week{week}' for week in range(1, 47)]

    # Plot correlation line chart
    fig, ax1 = plt.subplots(figsize=(14, 8))
    ax1.plot(weeks_labels, evi_data.values, marker='o', label='EVI')
    ax1.plot(weeks_labels, kndvi_data.values, marker='o', label='KNDVI')
    ax1.plot(weeks_labels, ndvi_data.values, marker='o', label='NDVI')

    ax1.set_ylabel('Correlation with Yield', color='g')

    plt.xlabel('Week')
    plt.title('VIS Correlation with Yield ')
    xticks = range(1, 46 + 1)
    plt.xticks(xticks[::4], rotation=45)  # Display every 4th week
    fig.legend(loc="upper left", bbox_to_anchor=(0, 1), bbox_transform=ax1.transAxes)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpath_fig3)
    # plt.show()
    return evi_data, kndvi_data, ndvi_data


# Seven Vegetation Indices
def plot_vis_correlation_with_yield(data_yield_ori_Visall, outpath_fig3):
    # Extract correlation data
    corrdata = data_yield_ori_Visall.drop(['Country', 'idJoin'], axis=1)
    y = data_yield_ori_Visall['Yield']
    correlation_results = corrdata.corrwith(y)

    # Add Glass LAI and PML_GPP
    weeks = [f'Week{week}_EVI' for week in range(1, 47)] + \
            [f'Week{week}_KNDVI' for week in range(1, 47)] + \
            [f'Week{week}_NDVI' for week in range(1, 47)] + \
            [f'Week{week}_modis_Gpp' for week in range(1, 47)] + \
            [f'Week{week}_modis_Fpar' for week in range(1, 47)] + \
            [f'Week{week}_modis_LAI' for week in range(1, 47)] + \
            [f'Week{week}_PML_Gpp' for week in range(1, 47)]

    filtered_correlation = correlation_results.loc[weeks]

    # Extract data for each vegetation index separately
    evi_data = filtered_correlation[filtered_correlation.index.str.contains('_EVI')]
    kndvi_data = filtered_correlation[filtered_correlation.index.str.contains('_KNDVI')]
    ndvi_data = filtered_correlation[filtered_correlation.index.str.contains('_NDVI')]
    modis_Gpp_data = filtered_correlation[filtered_correlation.index.str.contains('_modis_Gpp')]
    Fpar_data = filtered_correlation[filtered_correlation.index.str.contains('_Fpar')]
    modis_LAI_data = filtered_correlation[filtered_correlation.index.str.contains('_modis_LAI')]
    PML_Gpp_data = filtered_correlation[filtered_correlation.index.str.contains('_PML_Gpp')]

    # Extract week labels
    weeks_labels = [f'Week{week}' for week in range(1, 47)]

    # Plot correlation line chart
    fig, ax1 = plt.subplots(figsize=(14, 8))
    print(len(evi_data.values))
    ax1.plot(weeks_labels, evi_data.values, marker='o', label='EVI')
    ax1.plot(weeks_labels, kndvi_data.values, marker='o', label='KNDVI')
    ax1.plot(weeks_labels, ndvi_data.values, marker='o', label='NDVI')
    ax1.plot(weeks_labels, modis_Gpp_data.values, marker='o', label='modis_Gpp')
    ax1.plot(weeks_labels, Fpar_data.values, marker='o', label='Fpar')
    ax1.plot(weeks_labels, modis_LAI_data.values, marker='o', label='modis_LAI')
    ax1.plot(weeks_labels, PML_Gpp_data.values, marker='o', label='PML_Gpp')

    ax1.set_ylabel('Correlation with Yield', color='g')

    plt.xlabel('Week')
    plt.title('VIS Correlation with Yield')
    xticks = range(1, 46 + 1)
    plt.xticks(xticks[::4], rotation=45)  # Display every 4th week
    fig.legend(loc="upper left", bbox_to_anchor=(0, 1), bbox_transform=ax1.transAxes)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpath_fig3)
    # plt.show()
    return evi_data, kndvi_data, ndvi_data, modis_Gpp_data, Fpar_data, modis_LAI_data, PML_Gpp_data


def analyze_correlations_and_plot(data_yield_ori_Visall, inputpath_base, VI_select):
    outpath_fig1 = os.path.join(inputpath_base, '06_figure', f'{VI_select}_every_years_R.jpg')
    weeks = [f'Week{week}{VI_select}' for week in range(1, 47)]
    weeks_labels = [f'Week{week}' for week in range(1, 47)]
    data = data_yield_ori_Visall[weeks + ['year', 'Yield', 'idJoin']]
    data = data.dropna(subset=['Yield'])

    # Calculate correlations for each group (grouped by idJoin)
    def calculate_correlations(group):
        return group.drop(['idJoin'], axis=1).corr()['year']

    grouped = data.groupby(['idJoin'])
    correlations = grouped.apply(calculate_correlations)

    # Plot correlation boxplot
    fig, ax1 = plt.subplots(figsize=(14, 8))
    correlations[weeks].boxplot(column=weeks, ax=ax1, rot=90)
    ax1.set_ylabel('Correlation with Yield', color='g')
    ax2 = ax1.twinx()  # Create a second y-axis

    # Calculate mean values across all years
    filtered_data_select = [col for col in data_yield_ori_Visall.columns if VI_select in col]
    mean_all_years = pd.DataFrame(data_yield_ori_Visall[filtered_data_select]).mean()
    SG_mean_all_years = savgol_filter(mean_all_years, window_length=5, polyorder=3)  # Savitzky-Golay filter

    # Plot mean line and smoothed line
    ax2.plot(range(1, len(mean_all_years) + 1), mean_all_years, marker='s', label='Mean ' + VI_select + ' (All Years)',
             color='g')
    ax2.plot(range(1, len(mean_all_years) + 1), SG_mean_all_years, label='Filtered Mean ' + VI_select + ' (SG)',
             linestyle='--', color='purple')
    ax2.set_ylabel(VI_select[1:], color='g')
    ax2.tick_params(axis='y', labelcolor='g')

    plt.xlabel('Week')
    plt.title(f'VIs Correlation with Yield')
    xticks = range(1, len(mean_all_years) + 1)
    plt.xticks(xticks[::4], rotation=45)  # Display every 4th week
    fig.legend(loc="upper left", bbox_to_anchor=(0, 1), bbox_transform=ax1.transAxes)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpath_fig1)
    plt.show()


################################# Automatically Constructed Accuracy Evaluation Metrics ############################################################################
def calculate_rrmse1(y_true, y_pred):
    """
    Calculate RRMSE (Relative Root Mean Square Error) using the mean of actual y as reference

    Parameters:
    y_true -- Array or list of true values
    y_pred -- Array or list of predicted values

    Returns:
    rrmse -- Relative Root Mean Square Error (expressed as a percentage)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate RMSE
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # Calculate mean of true values
    mean_y_true = np.mean(y_true)

    # Calculate RRMSE
    rrmse = (rmse / mean_y_true) * 100

    return rrmse


def calculate_rrmse2(y_true, y_pred):
    """
    Calculate rRMSE (Relative Root Mean Square Error) using each actual y value as reference

    Parameters:
    y_true -- Array or list of true values
    y_pred -- Array or list of predicted values

    Returns:
    rrmse -- Relative Root Mean Square Error (expressed as a percentage)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate rRMSE
    rrmse = np.sqrt(np.mean(((y_true - y_pred) / y_true) ** 2)) * 100

    return rrmse


# Define custom nRMSE evaluation function
def calculate_nrmse(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    nrmse = rmse / (y_true.max() - y_true.min())
    return nrmse * 100


def calculate_acc(y_true, y_pred):
    # Calculate means of observed and predicted values
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)

    # Calculate anomalies (deviations from mean)
    anomaly_true = y_true - mean_true
    anomaly_pred = y_pred - mean_pred

    # Calculate ACC (Anomaly Correlation Coefficient)
    numerator = np.sum(anomaly_true * anomaly_pred)
    denominator = np.sqrt(np.sum(anomaly_true ** 2) * np.sum(anomaly_pred ** 2))

    acc = numerator / denominator
    return acc


########################################## Generate S2S and Historical Data for Final Model Prediction ########################################################


def aggre_8days_modis(dynamic_features, dates, data_new):
    data_new1 = pd.DataFrame()
    for feature in dynamic_features:
        columns = [col for col in data_new.columns if feature in col]
        data = data_new[columns]
        if data.shape[1] < len(dates):
            date_columns = pd.to_datetime(data.columns.str.replace(feature, ''), format='%Y_%m_%d')
            data = data.T
            data.index = date_columns
            full_index = pd.date_range(start=dates[0], end=dates[-1], freq='8D')
            data = data.reindex(full_index).sort_index()
        else:
            data = pd.DataFrame(data.T.values, index=dates)
        data = data.T
        data.columns = [f'Week{week}_{feature[1:]}' for week in range(1, 47)]
        data_new1 = pd.concat([data_new1, data], axis=1)
        data_new1['idJoin'] = data_new['idJoin']
    return data_new1


def prepare_forecast_model_InputData(ii, ECMWF_path, origen, inputpath_base,
                                     startyear, endyear, T_upper, T_lower, dynamic_features,
                                     soil_feature, loc_feature, Year_feature, union_feature, region='I'):
    ########################### Read Harvest Dates ###################################
    inputpath_S2s = os.path.join(ECMWF_path, '04_mergeData', 'S2S', origen, ii)
    inputpath_hist = os.path.join(ECMWF_path, '04_mergeData', 'hist_similar', origen, ii)

    outputpath_S2s = os.path.join(ECMWF_path, '05_WeekData', 'S2S', ii)
    outputpath_hist = os.path.join(ECMWF_path, '05_WeekData', 'hist_similar', ii)

    os.makedirs(outputpath_S2s, exist_ok=True)
    os.makedirs(outputpath_hist, exist_ok=True)

    slected_dynamic_features, slected_static = extract_selected_variables(inputpath_base)
    slected_static = slected_static + ['actual_yield', 'idJoin']

    for current_year in list(range(startyear, endyear + 1))[0:20]:
        data_ori = pd.read_csv(os.path.join(inputpath_base, '01_data', '05_buildmodel', '03_modeldata', 'data_ori.csv'))
        data_ori.set_index(['idJoin', 'year'], inplace=True)

        ############## S2S ECMWF Data Preparation ###########################
        data_S2S = pd.read_csv(os.path.join(inputpath_S2s, str(current_year) + '.csv'))  # process_data_8agr
        data_new1 = process_data_8agr(data_S2S, current_year, T_upper, T_lower, dynamic_features, soil_feature,
                                      loc_feature, Year_feature, union_feature)
        data_new1 = Link_yield(data_ori, data_new1, start_point, harvest_point, slected_dynamic_features,
                               slected_static)
        data_new1.set_index(['idJoin', 'year'], inplace=True)

        nan_columns = data_new1.columns[data_new1.isna().all()]
        data_new1[nan_columns] = data_ori[nan_columns]
        data_new1.to_csv(os.path.join(outputpath_S2s, str(current_year) + '.csv'))

        ############## Historical Data Preparation ###########################
        dataall_hist = pd.read_csv(os.path.join(inputpath_hist, str(current_year) + '.csv'))
        data_new1 = process_data(dataall_hist, current_year, T_upper, T_lower, dynamic_features, soil_feature,
                                 loc_feature, Year_feature, union_feature)

        data_new1 = Link_yield(data_ori, data_new1, start_point, harvest_point, slected_dynamic_features,
                               slected_static)
        data_new1.set_index(['idJoin', 'year'], inplace=True)

        nan_columns = data_new1.columns[data_new1.isna().all()]
        data_new1[nan_columns] = data_ori[nan_columns]
        data_new1.to_csv(os.path.join(outputpath_hist, str(current_year) + '.csv'))


def aggre_8days(dynamic_features, dates, data_new):
    data_new1 = pd.DataFrame()
    for feature in dynamic_features:
        columns = [col for col in data_new.columns if feature in col]
        data = data_new[columns]
        if data.shape[1] < len(dates):
            date_columns = pd.to_datetime(data.columns.str.replace(feature, ''), format='%Y_%m_%d')
            data = data.T
            data.index = date_columns
            full_index = pd.date_range(start=dates[0], end=dates[-1])
            data = data.reindex(full_index).sort_index()
        else:
            data = pd.DataFrame(data.T.values, index=dates)
        if feature == '_Pre':
            data = data.T.resample('8D', axis=1).sum()  # Sum precipitation over 8-day periods
            data.columns = [f'Week{week}_{feature[1:]}' for week in range(1, 47)]
        else:
            data = data.T.resample('8D', axis=1).mean()  # Average other variables over 8-day periods
            data.columns = [f'Week{week}_{feature[1:]}' for week in range(1, 47)]
        data_new1 = pd.concat([data_new1, data], axis=1)
    return data_new1


def extreme_temperature(dates, Tmax, Tmin, T_upper, T_lower):
    GDD = np.where(Tmax < T_lower, 0,
                   (np.minimum(Tmax, T_upper) + np.maximum(Tmin, T_lower)) / 2 - T_lower)  # Growing Degree Days
    HDD = np.maximum(Tmax, T_upper) - T_upper  # Heating Degree Days
    CDD = np.minimum(Tmin, T_lower) - T_lower  # Cooling Degree Days
    GDD = pd.DataFrame(GDD.T, index=dates).T.resample('8D', axis=1).sum()  # Sum GDD over 8-day periods
    HDD = pd.DataFrame(HDD.T, index=dates).T.resample('8D', axis=1).sum()  # Sum HDD over 8-day periods
    CDD = pd.DataFrame(CDD.T, index=dates).T.resample('8D', axis=1).sum()  # Sum CDD over 8-day periods
    CDD_df = pd.DataFrame(data=CDD.values, columns=[f'Week{week}_CDD' for week in range(1, 47)])
    HDD_df = pd.DataFrame(data=HDD.values, columns=[f'Week{week}_HDD' for week in range(1, 47)])
    GDD_df = pd.DataFrame(data=GDD.values, columns=[f'Week{week}_GDD' for week in range(1, 47)])
    return CDD_df, HDD_df, GDD_df


def thornthwaite(T, lat=45):
    """Thornthwaite method for Potential Evapotranspiration (PET) calculation"""
    I = (T / 5.0) ** 1.514
    a = (6.75e-7) * I ** 3 - (7.71e-5) * I ** 2 + (1.79e-2) * I + 0.49239
    PET = 16 * ((10 * T / I) ** a)
    return PET


def spei(dates, Pre, Tmean):
    precipitation = pd.DataFrame(Pre.T, index=dates)
    Tmean = pd.DataFrame(Tmean.T, index=dates)

    # Transpose dataframes to have dates as rows and features as columns
    precipitation = precipitation.T
    Tmean = Tmean.T
    PET = thornthwaite(Tmean)
    # Calculate difference between precipitation and PET
    D = precipitation - PET
    # Calculate cumulative values every 8 days
    D_resampled = D.resample('8D', axis=1).sum()

    # D_resampled = D.T.resample('8D').sum().T
    # Calculate SPEI (Standardized Precipitation-Evapotranspiration Index)
    def compute_spei(series, scale):
        """Calculate SPEI"""
        # Cumulative sum
        cum_sum = series.cumsum()
        # Calculate mean and standard deviation
        mean = cum_sum.mean()
        std = cum_sum.std()
        # Standardize
        spei = (cum_sum - mean) / std
        return spei

    spei_values = D_resampled.apply(lambda x: compute_spei(x, scale=1), axis=1)
    spei_df = pd.DataFrame(data=spei_values.values, columns=[f'Week{week}_SPEI' for week in range(1, 47)])
    return spei_df


###################### Aggregation to 8-Day Scale Calculation ###################################################################
def process_data_8agr(data_new, year, T_upper, T_lower, dynamic_features, soil_feature, loc_feature, Year_feature,
                      union_feature):
    # Extract specific columns  
    Tmin_columns = [col for col in data_new.columns if '_Tmin' in col]
    Tmin = data_new[Tmin_columns].values
    Tmean_columns = [col for col in data_new.columns if '_Tmean' in col]
    Tmean = data_new[Tmean_columns].values
    Tmax_columns = [col for col in data_new.columns if '_Tmax' in col]
    Tmax = data_new[Tmax_columns].values
    Pre_columns = [col for col in data_new.columns if '_Pre' in col]
    Pre = data_new[Pre_columns].values

    # Calculate dates  
    days = Pre.shape[1]
    dates = pd.date_range(start=str(year) + '-01-01', periods=days, freq='D')
    data_new['year'] = year

    # Calculate extreme meteorological indices  
    spei_df = spei(dates, Pre, Tmean)
    CDD_df, HDD_df, GDD_df = extreme_temperature(dates, Tmax, Tmin, T_upper, T_lower)

    # Process other dynamic features  
    data_new1 = aggre_8days(dynamic_features, dates, data_new)

    # Merge all DataFrames  
    combined_df = pd.concat([CDD_df, HDD_df, GDD_df, spei_df, data_new1,
                             data_new[soil_feature + loc_feature + Year_feature + union_feature]],
                            axis=1)

    return combined_df


def prepare_and_export_data(start_point, harvest_point, data_yield_ori_Visall, slected_dynamic_features,
                            slected_static):
    # Define static features
    weeks = [];
    weeks1 = []
    if start_point < harvest_point:
        # Generate list of weekly data features
        # ['_CDD' ,'_HDD' ,'_GDD']

        for feature in slected_dynamic_features:
            weeks_sel = [f'Week{week}_{feature}' for week in range(start_point, harvest_point + 1)]
            weeks = weeks + weeks_sel
        #  [f'Week{week}{VI_select1}' for week in range(start_point, harvest_point+1)] + \
        data_yield_ori_selected = data_yield_ori_Visall[weeks + slected_static]
    else:
        # Reconstruct weekly features for cross-year growing periods
        weeks_select_list = list(range(start_point, 47)) + list(range(1, harvest_point + 1))
        for feature in slected_dynamic_features:
            weeks_sel = [f'Week{week}_{feature}' for week in weeks_select_list]
            weeks = weeks + weeks_sel
        #  [f'Week{week}{VI_select1}' for week in range(start_point, harvest_point+1)] + \
        data_yield_ori_selected = data_yield_ori_Visall[weeks + slected_static]
        # Rename and impute missing dynamic features for 2000 using mean imputation
        for feature in slected_dynamic_features:
            weeks_sel = [f'Week{week}_{feature}' for week in weeks_select_list]
            weeks = weeks + weeks_sel
        for feature in slected_dynamic_features:
            weeks_sel = [f'Week{week}_{feature}' for week in range(start_point, 47)]
            weeks1 = weeks1 + weeks_sel
        mean = data_yield_ori_selected[weeks1].mean()  # Impute missing dynamic features for 2000 using mean imputation
        data_yield_ori_Visall_lastyear = data_yield_ori_selected[weeks1 + ['year', 'idJoin']]
        data_yield_ori_Visall_lastyear['year'] = data_yield_ori_Visall_lastyear['year'] + 1
        # data_yield_ori_Visall_lastyear = data_yield_ori_Visall_lastyear[data_yield_ori_Visall_lastyear['year']2002]

        data_lastyear_2000 = data_yield_ori_Visall_lastyear.groupby(['idJoin']).mean().reset_index()
        data_lastyear_2000['year'] = 2001
        data_lastyear_2000 = data_lastyear_2000[weeks1 + ['year', 'idJoin']]
        data_yield_ori_Visall_lastyear = pd.concat([data_lastyear_2000, data_yield_ori_Visall_lastyear])

        data_yield_ori_Visall_currentyears = data_yield_ori_selected.drop(weeks1, axis=1)
        data_yield_ori_selected = data_yield_ori_Visall_lastyear.merge(data_yield_ori_Visall_currentyears,
                                                                       on=['year', 'idJoin'], how='inner')
    # Select data
    # Export data as CSV file
    return data_yield_ori_selected


def Link_yield(data_ori, data_new1, start_point, harvest_point, slected_dynamic_features, slected_static):
    # Set index  

    data_new1.set_index(['idJoin', 'year'], inplace=True)

    # Extract column names containing 'Yield' (assuming 'Yield' is the target column, not 'ield')
    yield_columns = [col for col in data_ori.columns if 'ield' in col]

    # Copy yield information from data_ori to data_new1  
    data_new1[yield_columns] = data_ori[yield_columns]

    # Reset index  
    data_new1 = data_new1.reset_index()

    # Call prepare_and_export_data function to process and export data  
    # Note: It is assumed that the prepare_and_export_data function can process and return the processed data  
    # If it only prepares and exports data without returning anything, the logic of this function may need to be adjusted  
    processed_data = prepare_and_export_data(start_point, harvest_point, data_new1, slected_dynamic_features,
                                             slected_static)

    # If prepare_and_export_data does not return data, processed_data may not be needed or should be set to None  
    # Alternatively, if this function is only used for export and does not return the processed DataFrame, this line may not be necessary  

    # Return the processed data (if prepare_and_export_data returns data)  
    return processed_data


def extract_selected_variables(inputpath_base):
    inpath_dates = os.path.join(inputpath_base, '01_data', '05_buildmodel', '04_selectFeatures', 'selectFeatures.txt')
    # Construct file path
    gs_infornamtion = pd.read_csv(inpath_dates, sep='\t', header=None)
    gs_infornamtion.columns = ['slected_dynamic_features', 'slected_static', 'regionID']
    gs_infornamtion['slected_dynamic_features'] = gs_infornamtion['slected_dynamic_features'].apply(ast.literal_eval)
    gs_infornamtion['slected_static'] = gs_infornamtion['slected_static'].apply(ast.literal_eval)
    return gs_infornamtion


########################################## Visualization of Factor Comparison (Full Year vs. Growing Period) for 2022 Across All Forecast Periods: Similar Years, S2S, S2S_corr, and Current Year ########################################
def extract_all_dates(outpath_S2s, current_year):
    # Construct file path
    outpath_dates = os.path.join(outpath_S2s, 'multipleDates.txt')

    # Read date data
    with open(outpath_dates, 'r', encoding='utf-8') as file:
        dates = file.read()
        plant_doy, ForecastStart_doy, ForecastEnd_doy, harvest_doy = dates.strip().split('\t')

    # Convert date format (DOY: Day of Year)
    plant_doy = str(current_year) + plant_doy.replace('-', '')
    forecast_start_doy = str(current_year) + ForecastStart_doy.replace('-', '')
    forecast_end_doy = str(current_year) + ForecastEnd_doy.replace('-', '')
    harvest_doy = str(current_year) + harvest_doy.replace('-', '')

    return [plant_doy, forecast_start_doy, forecast_end_doy, harvest_doy]


def plot_climate_data(data_hist, data_base, data_S2S_new, data_S2S_new_corr, feature, forecast_start_doy,
                      forecast_end_doy, plant_doy, harvest_doy, outpath_fig1, gs=True):
    """
    Plot the mean and standard deviation intervals for different datasets related to a specific feature.

    :param data_hist: DataFrame with historical data.
    :param data_base: DataFrame with base data (current year's observed data).
    :param data_S2S_new: DataFrame with S2S forecast data.
    :param data_S2S_new_corr: DataFrame with bias-corrected S2S forecast data.
    :param feature: String, feature of interest to plot.
    :param forecast_start_doy: String, forecast start date (YYYYMMDD).
    :param forecast_end_doy: String, forecast end date (YYYYMMDD).
    :param plant_doy: String, planting date (YYYYMMDD).
    :param harvest_doy: String, harvest date (YYYYMMDD).
    :param outpath_fig1: String, output file path for the plot.
    :param gs: Boolean, whether to filter by growing season (default: True).
    """
    columns = [col for col in data_hist.columns if feature in col]  # Historical data is pre-sorted
    if gs:
        index_plant = columns.index(plant_doy + '_' + feature)
        index_harvest = columns.index(harvest_doy + '_' + feature)
        columns = columns[index_plant:index_harvest]
    else:
        pass
    data_hist_feature = data_hist[columns]
    data_base_feature = data_base[columns]
    data_S2S_feature = data_S2S_new[columns]
    data_S2S_feature_corr = data_S2S_new_corr[columns]

    data_hist_feature_mean = data_hist_feature.mean()
    data_hist_feature_std = data_hist_feature.std()
    data_base_feature_mean = data_base_feature.mean()
    data_base_feature_std = data_base_feature.std()
    data_S2S_feature_mean = data_S2S_feature.mean()
    data_S2S_feature_std = data_S2S_feature.std()
    data_S2S_feature_corr_mean = data_S2S_feature_corr.mean()
    data_S2S_feature_corr_std = data_S2S_feature_corr.std()

    fig, ax1 = plt.subplots(figsize=(14, 8))
    x_label = [x.replace(feature, '').replace('_', '') for x in data_hist_feature.columns.tolist()]

    ax1.plot(x_label, data_base_feature_mean, marker='o', label='Base Feature Mean', color='blue')
    ax1.fill_between(x_label, (data_base_feature_mean - data_base_feature_std).values,
                     (data_base_feature_mean + data_base_feature_std).values, color='blue', alpha=0.3)

    ax1.plot(x_label, data_hist_feature_mean, marker='o', label='Historical Feature Mean', color='red')
    ax1.fill_between(x_label, (data_hist_feature_mean - data_hist_feature_std).values,
                     (data_hist_feature_mean + data_hist_feature_std).values, color='red', alpha=0.3)

    ax1.plot(x_label, data_S2S_feature_mean, marker='o', label='S2S Feature Mean', color='green')
    ax1.fill_between(x_label, (data_S2S_feature_mean - data_S2S_feature_std).values,
                     (data_S2S_feature_mean + data_S2S_feature_std).values, color='green', alpha=0.3)

    ax1.plot(x_label, data_S2S_feature_corr_mean, marker='o', label='S2S Feature Corr Mean', color='purple')
    ax1.fill_between(x_label, (data_S2S_feature_corr_mean - data_S2S_feature_corr_std).values,
                     (data_S2S_feature_corr_mean + data_S2S_feature_corr_std).values, color='purple', alpha=0.3)

    ax1.set_xticks(range(0, len(x_label), 10))
    ax1.set_xticklabels(x_label[::10], rotation=45)
    ax1.axvline(x=forecast_start_doy, color='green', linestyle=':', linewidth=2, label='Forecast Start DOY')
    ax1.axvline(x=forecast_end_doy, color='green', linestyle='--', linewidth=2, label='Forecast End DOY')
    ax1.axvline(x=plant_doy, color='red', linestyle=':', linewidth=2, label='Plant DOY')
    ax1.axvline(x=harvest_doy, color='red', linestyle='--', linewidth=2, label='Harvest DOY')

    ax1.set_ylabel(feature, color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    fig.legend(loc="upper left", bbox_to_anchor=(0, 1), bbox_transform=ax1.transAxes)
    plt.savefig(outpath_fig1)
    plt.show()


########################################Find the most similar year's VI and meteorological variables (sowing period-forecast period) using DTW; supplement factors from forecast period-harvest period with the found similar year's features/S2S#############################

def merge_S2S_ECMWF(ii, ECMWF_path, origen, inputpath_base, startyear, endyear, start_date_doy, harvest_date_doy,
                    pre_filename, climateFeatures, VI_select, data_climate, data_all_Vis, yearly_data_ViS_sorted,
                    yearly_data_climate, data_all_climate):
    outpath_S2s = os.path.join(ECMWF_path, '04_mergeData', 'S2S', origen, ii)
    outpath_hist = os.path.join(ECMWF_path, '04_mergeData', 'hist_similar', origen, ii)
    inputpath_S2S = os.path.join(ECMWF_path, '03_outputData', origen, ii)
    os.makedirs(outpath_hist, exist_ok=True)
    os.makedirs(outpath_S2s, exist_ok=True)

    similar_year_df = pd.DataFrame()
    S2S_data_all = pd.read_csv(os.path.join(inputpath_S2S, 'allyear.csv'))

    df_list = []
    for current_year in list(range(startyear, endyear + 1))[0:20]:
        data_base = pd.read_csv(os.path.join(inputpath_base, '01_data', '04_GEEdownloadData', '01_DailyData',
                                             pre_filename + str(current_year) + '.csv')).iloc[:, 1:]
        data_base.set_index('idJoin', inplace=True)
        S2S_data = pd.read_csv(os.path.join(inputpath_S2S, str(current_year) + '.csv'))
        S2S_data.set_index('idJoin', inplace=True)
        #  common_columns = S2S_data.columns.intersection(data_base.columns)
        plant_doy = str(current_year) + start_date_doy;
        ForecastStart_doy = str(current_year) + '-' + ii;
        ForecastEnd_doy = S2S_data.columns[-1][0:8]
        ForecastEnd_doy = ForecastEnd_doy[0:4] + '-' + ForecastEnd_doy[4:6] + '-' + ForecastEnd_doy[6:]
        harvest_doy = str(current_year) + harvest_date_doy
        print(
            f'Sowing date of {current_year}: {plant_doy}; Forecast start date: {ForecastStart_doy}; Forecast end date: {ForecastEnd_doy}; Harvest date: {harvest_doy}')
        data_hist = data_base.copy()

        for feature in climateFeatures:
            ID_plant = plant_doy[4:].replace('-', '') + '_' + feature
            ID_ForecastStart = ForecastStart_doy[4:].replace('-', '') + '_' + feature
            ID_ForecastEnd = ForecastEnd_doy[4:].replace('-', '') + '_' + feature
            ID_harvest_doy = harvest_doy[4:].replace('-', '') + '_' + feature
            yearly_data_climate_sel = yearly_data_climate[
                [col for col in yearly_data_climate.columns if feature in col]]
            yearly_data_climate_sel_sorted = sort_columns_by_date_climate(yearly_data_climate_sel)  # Sort by date
            yearly_data_climate_sel_sorted.set_index(yearly_data_climate['year'], inplace=True)
            most_similar_year = find_most_similar_year(yearly_data_climate_sel_sorted, startyear, endyear, current_year,
                                                       ID_plant, ID_ForecastStart)
            df_list.append([ii, current_year, feature, most_similar_year])

            ### Supplement meteorological data from ID_ForecastStart to ID_harvest_doy of the similar year
            filtered_columns = [col for col in data_all_climate.columns if feature in col]
            filtered_Forecast = data_all_climate.loc[
                data_all_climate['year'] == most_similar_year, filtered_columns + ['idJoin']]
            filtered_Forecast = filtered_Forecast.loc[:, ID_ForecastStart:ID_harvest_doy].set_index(
                filtered_Forecast['idJoin'])
            select_columns = filtered_Forecast.columns.tolist()
            select_columns = [str(current_year) + x for x in select_columns]
            data_hist[select_columns] = filtered_Forecast
            ##############Vegetation Index########################################################################
        ID_plantVis = ID_plant[0:2] + '_' + ID_plant[2:4];
        ID_ForecastStartVis = ID_ForecastStart[0:2] + '_' + ID_ForecastStart[2:4]
        # print(ID_plantVis);print(ID_ForecastStartVis)
        # yearly_data_ViS_sorted.set_index('year',inplace=True)
        most_similar_year = find_most_similar_year(yearly_data_ViS_sorted, startyear, endyear, current_year,
                                                   ID_plantVis,
                                                   ID_ForecastStartVis)  # Masking is applied when calculating similarity
        df_list.append([ii, current_year, 'VI', most_similar_year])

        ## For vegetation index: only supplement data after the forecast period, retain data before the forecast period
        filtered_Forecast = data_all_Vis.loc[data_all_climate['year'] == most_similar_year]
        filtered_Forecast = filtered_Forecast.drop('year', axis=1)
        filtered_Forecast.set_index('idJoin', inplace=True)
        filtered_Forecast = sort_columns_by_date(filtered_Forecast).loc[:, ID_ForecastStartVis:]

        filtered_Forecast.rename(columns=lambda x: str(current_year) + '_' + x + '_' + VI_select, inplace=True)
        select_columns = filtered_Forecast.columns.tolist()
        data_hist[select_columns] = filtered_Forecast
        data_hist.to_csv(os.path.join(outpath_hist, str(current_year) + '.csv'));

        ### Supplement historical data with S2S forecast meteorological data; same start date, downloaded data ends at harvest period, no need for judgment
        ### No bias correction
        data_S2S_new = data_hist.copy()
        common_columns = S2S_data.columns.intersection(data_S2S_new.columns)
        data_S2S_new[common_columns] = S2S_data[common_columns]  # Replace with forecast data

        ##############Perform Bias Correction###########################################

        data_S2S_new.to_csv(os.path.join(outpath_S2s, str(current_year) + '.csv'));
        #  data_S2S_new_corr = S2S_correct(data_hist, data_climate, S2S_data_all, climateFeatures, current_year,flag=1)#basic_quantile
        # data_S2S_new_corr.to_csv(os.path.join(outpath_S2s, str(current_year)+'_basic_quantile_corr.csv'));
        data_S2S_new_corr = S2S_correct(data_hist, data_climate, S2S_data_all, climateFeatures, current_year,
                                        flag=2)  # modified_quantile
        data_S2S_new_corr.to_csv(os.path.join(outpath_S2s, str(current_year) + '_modified_quantile_corr.csv'));
    outpath_dates = os.path.join(outpath_S2s, 'multipleDates.txt')
    with open(outpath_dates, 'w') as file:
        file.write(f'{plant_doy[4:]}\t{ForecastStart_doy[4:]}\t{ForecastEnd_doy[4:]}\t{harvest_doy[4:]}\n')

    similar_year_df = pd.DataFrame(data=df_list,
                                   columns=['ForecastStartDoy', 'current_year', 'feature', 'similar_year'])
    similar_year_df.to_csv(os.path.join(ECMWF_path, '04_mergeData', 'similar_year_summary' + str(ii) + '.csv'),
                           index=False)


def extract_dates(inputpath_base, institution, region):
    # Return the week numbers and corresponding days of start and end dates, as well as the selected VI
    file_path = os.path.join(inputpath_base, '02_S2S', '01_dataori', institution, 'CommonYear_Week.txt')
    inpath_dates = os.path.join(inputpath_base, '01_data', '05_buildmodel', '02_extractdates', 'gs_three_periods.txt')
    # Read lines from file and remove whitespace characters at both ends
    with open(file_path, 'r') as file:
        lines = [line.strip() for line in file.readlines()]

    # Read start point and harvest point from another file
    file_path = os.path.join(inputpath_base, '02_S2S', '01_dataori', institution, 'CommonYear_Week.txt')
    inpath_dates = os.path.join(inputpath_base, '01_data', '05_buildmodel', '02_extractdates', 'gs_three_periods.txt')
    # Read lines from file and remove whitespace characters at both ends
    with open(file_path, 'r') as file:
        lines = [line.strip() for line in file.readlines()]

    # Read start point and harvest point from another file
    gs_infornamtion = pd.read_csv(inpath_dates, sep='\t', header=None)
    gs_infornamtion.columns = ['start_point', 'peak', 'harvest_point', 'VI_select2', 'regions']
    harvest_point = gs_infornamtion[gs_infornamtion['regions'] == region]['harvest_point'].values[0]
    start_point = gs_infornamtion[gs_infornamtion['regions'] == region]['start_point'].values[0]
    VI_select2 = gs_infornamtion[gs_infornamtion['regions'] == region]['VI_select2'].values[0]

    # Convert harvest_point and start_point from strings to integers
    harvest_point = int(
        harvest_point)  # Aggregation into 8-day periods is backward aggregation, e.g., 01-01 refers to aggregated indicators from 01-01 to 01-08
    start_point = int(start_point)

    # Get corresponding dates from lines based on the indices of harvest_point and start_point
    if harvest_point == 46:
        harvest_date_doy = '-' + '12-31'
    else:
        harvest_date_doy = '-' + lines[harvest_point]
    start_date_doy = '-' + lines[start_point]

    # Return start date and harvest date
    return start_point, harvest_point, start_date_doy, harvest_date_doy, VI_select2


# Use DTW to find the similar year for NDVI imputation
# # Find KNDVI with similar trajectory, assuming forecast date is 0804 (dynamic), forecast year is dynamic
def replace_ndvi_data(crop, ForecastDate, VI, Forecastyear, inputpath, data_hist, region):
    # Find KNDVI with similar trajectory, assuming forecast date is 0804 (dynamic), forecast year is dynamic
    # data_hist = pd.read_csv(inputpath + '\\'+crop+'_'+region+'_' + str(Forecastyear) + '.csv')
    KNDVI_columns = [col for col in data_hist.columns if VI in col]
    ForecastDate = datetime.strptime(ForecastDate, "%Y%m%d")
    data_all_region1 = pd.DataFrame(columns=['year'] + KNDVI_columns)
    startyear = 2001

    for ii in range(startyear, Forecastyear + 1):
        data_new = pd.read_csv(inputpath + '\\' + crop + '_' + region + '_' + str(ii) + '.csv').iloc[:, 1:]
        KNDVI_columns = [col for col in data_new.columns if '_KNDVI' in col]
        data_new = data_new[KNDVI_columns].div(data_new['area'], axis=0)
        data_new.columns = [col.replace(str(ii) + '_', str(Forecastyear) + '_') for col in data_new.columns]
        data_new['year'] = str(ii)
        data_all_region1 = pd.concat([data_all_region1, data_new], axis=0)

    selected_columns = [
        column for column in KNDVI_columns
        if datetime.strptime(column.split('_')[0] + '_' + column.split('_')[1] + '_' + column.split('_')[2],
                             "%Y_%m_%d") <= ForecastDate
    ]

    data_all_region1 = data_all_region1.replace(0, np.nan)
    data_all_region1_fill = data_all_region1.rolling(window=9, min_periods=1, center=True, axis=1).mean()
    data_all_region1 = data_all_region1.fillna(data_all_region1_fill)

    dataNDVI_before = data_all_region1.groupby('year').mean().reset_index()[['year'] + selected_columns]
    dataNDVI_before['year'] = dataNDVI_before['year'].astype(int)
    dataNDVI_before = dataNDVI_before.set_index('year')

    data_Forecast = dataNDVI_before.loc[Forecastyear]
    dtw_distances = {}

    for year in range(startyear, Forecastyear):
        data_year = dataNDVI_before.loc[year]
        distance, path = fastdtw(data_Forecast, data_year)
        dtw_distances[year] = distance

    most_similar_by_dtw = min(dtw_distances, key=dtw_distances.get)
    dataNDVI_similaryear = data_all_region1[data_all_region1['year'] == str(most_similar_by_dtw)]
    columns_after = [col for col in dataNDVI_similaryear.columns if col not in selected_columns + ['year']]

    # KNDVI_Forecast = pd.concat([data_hist[selected_columns], dataNDVI_similaryear[columns_after]], axis=1)
    data_hist[columns_after] = dataNDVI_similaryear[columns_after]

    return data_hist, most_similar_by_dtw


def sort_columns_by_date(df):
    # Get list of column names
    columns = df.columns.tolist()

    # Define natural sort key function
    def natural_sort_key(s):
        month, day = map(int, s.split('_'))
        return (month, day)

    # Sort column names using natural sort
    sorted_columns = sorted(columns, key=natural_sort_key)

    # Reorder DataFrame according to sorted column names
    df_sorted = df[sorted_columns]

    return df_sorted


def sort_columns_by_date_climate(df):
    # Get list of column names
    columns = df.columns.tolist()

    # Define natural sort key function
    def natural_sort_key(s):
        month = int(s[0:2])
        day = int(s[2:4])
        return (month, day)

    # Sort column names using natural sort
    sorted_columns = sorted(columns, key=natural_sort_key)

    # Reorder DataFrame according to sorted column names
    df_sorted = df[sorted_columns]

    return df_sorted


def find_most_similar_year(data, startyear, endyear, current_year, ID_plant, ID_ForecastStart):
    """
    Find the most similar year to the given year using Dynamic Time Warping (DTW) distance.

    Parameters:
    - data: DataFrame containing multi-year data.
    - startyear: Start year of the data range.
    - endyear: End year of the data range.
    - current_year: Target year to find similarities for.
    - ID_plant: Start column index (corresponding to sowing date).
    - ID_ForecastStart: End column index (corresponding to forecast start date).

    Returns:
    - Integer: The most similar year.
    """
    # Select data for specific columns
    data_sel = data.loc[:, ID_plant:ID_ForecastStart]
    # Initialize distance dictionary
    dtw_distances = {}
    # Generate list of other years (exclude current year)
    other_years = [x for x in range(startyear, endyear + 1) if x != current_year]
    # Get data for current year

    # Iterate over other years and calculate DTW distance with current year
    for compare_year in other_years:
        data_sel_filtered = data_sel.loc[(data_sel.index == current_year) | (data_sel.index == compare_year)].dropna(
            how='any', axis=1)
        data_Forecast = data_sel_filtered.loc[current_year]
        data_year = data_sel_filtered.loc[compare_year]
        distance, path = fastdtw(data_Forecast, data_year)
        dtw_distances[compare_year] = distance

    # Find the year with the minimum DTW distance
    most_similar_by_dtw = min(dtw_distances, key=dtw_distances.get)

    return most_similar_by_dtw


# Usage Example
# The following code assumes you have a DataFrame `yearly_data_climate_sel_sorted`,
# and defined variables `ID_plant`, `ID_ForecastStart`, `startyear`, `endyear`, and `year`
# similar_year = find_most_similar_year(yearly_data_climate_sel_sorted, startyear, endyear, year, ID_plant, ID_ForecastStart)


def quantile_correction(obs_data, mod_data, sce_data, modified=True):
    cdf = ECDF(mod_data)
    p = cdf(sce_data) * 100
    cor = np.subtract(*[np.nanpercentile(x, p) for x in [obs_data, mod_data]])
    if modified:
        mid = np.subtract(*[np.nanpercentile(x, 50) for x in [obs_data, mod_data]])
        g = np.true_divide(*[np.nanpercentile(x, 50) for x in [obs_data, mod_data]])

        iqr_obs_data = np.subtract(*np.nanpercentile(obs_data, [75, 25]))
        iqr_mod_data = np.subtract(*np.nanpercentile(mod_data, [75, 25]))

        f = np.true_divide(iqr_obs_data, iqr_mod_data)
        cor = g * mid + f * (cor - mid)
        return sce_data + cor
    else:
        return sce_data + cor


###############Bias Correction for Forecast Data#################################################
def S2S_correct(data_hist, data_climate, S2S_data_all, climateFeatures, current_year, flag=1):
    data_S2S_new = data_hist.copy()

    for feature in climateFeatures:
        # All historical meteorological data
        data_all_climate_columns = [
            col for col in data_climate.columns
            if feature in col and not col.startswith(str(current_year))
        ]
        hist_climate = data_climate[data_all_climate_columns + ['idJoin']]
        hist_climate.set_index('idJoin', inplace=True)
        hist_climate.rename(columns=lambda x: x[0:8], inplace=True)
        hist_climate = hist_climate.T
        hist_climate.index.name = 'date'
        hist_climate.index = pd.to_datetime(hist_climate.index, format='%Y%m%d')

        # Forecast data
        S2S_columns = [col for col in S2S_data_all.columns if feature in col]
        S2S_data_sel = S2S_data_all[S2S_columns + ['idJoin']]
        S2S_data_sel.set_index('idJoin', inplace=True)
        S2S_data_sel.rename(columns=lambda x: x[0:8], inplace=True)
        S2S_data_sel = S2S_data_sel.T
        S2S_data_sel.index.name = 'date'
        S2S_data_sel.index = pd.to_datetime(S2S_data_sel.index, format='%Y%m%d')
        S2S_data_sel = S2S_data_sel[
            hist_climate.columns]  # Ensure consistency with historical administrative regions (may have more regions than GEE-extracted data)

        # Historical forecast data (exclude current year)
        hist_forecats_S2S = S2S_data_sel[S2S_data_sel.index.year != current_year]
        hist_climate = hist_climate.loc[hist_forecats_S2S.index]

        # Current year's forecast data
        current_forecats_S2S = S2S_data_sel[S2S_data_sel.index.year == current_year]

        # Perform quantile correction
        if flag == 1:
            current_forecats_S2S_correct = quantile_correction(
                hist_climate.values.flatten(),
                hist_forecats_S2S.values.flatten(),
                current_forecats_S2S.values.flatten(), modified=False)
        else:
            current_forecats_S2S_correct = quantile_correction(
                hist_climate.values.flatten(),
                hist_forecats_S2S.values.flatten(),
                current_forecats_S2S.values.flatten(), modified=True)

        current_forecats_S2S.iloc[:, :] = np.reshape(current_forecats_S2S_correct, current_forecats_S2S.shape)
        current_forecats_S2S_correct = current_forecats_S2S.T
        current_forecats_S2S_correct.rename(
            columns=lambda x: x.strftime('%Y%m%d') + '_' + feature,
            inplace=True
        )

        # Update corresponding columns in data_S2S_new
        common_columns = data_S2S_new.columns.intersection(current_forecats_S2S_correct.columns)
        data_S2S_new[common_columns] = current_forecats_S2S_correct[common_columns]

    return data_S2S_new
