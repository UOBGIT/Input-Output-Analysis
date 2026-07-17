#%%
import os
import pandas as pd
import numpy as np

#%%
def create_demand_shock(shock_dict, sector_list):
    """
    Converts a dictionary of sector shocks into a properly indexed numpy array.
    
    Parameters:
    shock_dict (dict): Keys are sector names, values are the demand shock levels.
    sector_list (list): The ordered list of sector names matching the IO table indices.
    
    Returns:
    np.array: A 1D numpy array of demand shocks with length equal to len(sector_list).
    """
    # Initialize an array of zeros with the length of your sector list
    delta_Y = np.zeros(len(sector_list))
    
    # Iterate through the user's dictionary
    for sector_name, shock_value in shock_dict.items():
        try:
            # Find the integer index of the sector in your master list
            idx = sector_list.index(sector_name)
            
            # Apply the shock to the correct position in the array
            delta_Y[idx] = shock_value
            
        except ValueError:
            # If the user provides a sector name that isn't in the list, warn them
            print(f"Warning: Sector '{sector_name}' not found in sector_list. Skipping.")
            
    return delta_Y

#%%
def simulate_demand_shock(L_inverse, delta_Y, A_m, sector_list, remuneration_array, NPT_array, DFA_array, OS_array, total_value_added_array):
    delta_X = np.dot(L_inverse, delta_Y)

    # Wrap in a pandas Series for readability
    delta_X_series = pd.Series(delta_X, index=sector_list, name='Total Output Impact')

    # Calculate how much of that output growth leaks overseas to buy foreign parts
    # Delta_M = A_m * Delta_X
    delta_M = np.dot(A_m, delta_X)

    # Wrap in a pandas Series
    delta_M_series = pd.Series(delta_M, index=sector_list, name='Import Leakage')

    # Domestic Value Add Series
    delta_DC_series = pd.Series(delta_X-delta_M, index = sector_list, name = "Domestic Content Impact")
    
    ## Calculating changes in Total Value Added

    delta_TVA = np.dot(total_value_added_array, delta_X)

    delta_TVA_series = pd.Series(delta_TVA, index = sector_list, name = "Total Change in Value Added")

    ## Calculating changes in 4 components of value added
    delta_remuneration = np.dot(remuneration_array, delta_X)
    delta_NPT = np.dot(NPT_array, delta_X)
    delta_DFA = np.dot(DFA_array, delta_X)
    delta_OS = np.dot(OS_array, delta_X)

    delta_remuneration_series = pd.Series(delta_remuneration, index = sector_list, name = "Change in Remuneration of Employees")
    delta_NPT_series = pd.Series(delta_NPT, index = sector_list, name = "Change in Net Production Tax Paid")
    delta_DFA_series = pd.Series(delta_DFA, index = sector_list, name = "Change in Depreciation of Fixed Assets")
    delta_OS_series = pd.Series(delta_OS, index = sector_list, name = "Change in Operating Surplus")

    
    # Combine the results into a clean table
    results_df = pd.concat([delta_X_series, delta_M_series, delta_DC_series, delta_remuneration_series, delta_NPT_series, delta_DFA_series, delta_OS_series, delta_TVA_series], axis=1)
    results_df.loc['Total'] = results_df.sum(numeric_only=True)

    return results_df
    


    

#%%
def generate_va_matrices(value_added_df):
    ### Creating a diagonal Total Value Added Dataframe

    sector_list = value_added_df.columns

    total_value_added_array = np.zeros((len(sector_list), len(sector_list)))
    remuneration_array = np.zeros((len(sector_list), len(sector_list)))
    NPT_array = np.zeros((len(sector_list), len(sector_list)))
    DFA_array = np.zeros((len(sector_list), len(sector_list)))
    OS_array = np.zeros((len(sector_list), len(sector_list)))

    total_value_added_df = pd.DataFrame(total_value_added_array, index = sector_list, columns = sector_list)
    remuneration_df = pd.DataFrame(remuneration_array, index = sector_list, columns = sector_list)
    NPT_df = pd.DataFrame(NPT_array, index = sector_list, columns = sector_list)
    DFA_df = pd.DataFrame(DFA_array, index = sector_list, columns = sector_list)
    OS_df = pd.DataFrame(OS_array, index = sector_list, columns = sector_list)

    print(total_value_added_df)

    for sector in sector_list:
        print(value_added_df.at["Total Value Added", sector])
        total_value_added_df.at[sector, sector] = value_added_df.at["Total Value Added", sector]
        remuneration_df.at[sector, sector] = value_added_df.at["Remuneration of Employee", sector]
        NPT_df.at[sector, sector] = value_added_df.at["Net Production Tax", sector]
        DFA_df.at[sector, sector] = value_added_df.at["Depreciation of Fixed Asset", sector]
        OS_df.at[sector, sector] = value_added_df.at["Operating Surplus", sector]

    total_value_added_array = total_value_added_df.values
    remuneration_array = remuneration_df.values
    NPT_array = NPT_df.values
    DFA_array = DFA_df.values
    OS_array = OS_df.values

    return total_value_added_array, remuneration_array, NPT_array, DFA_array, OS_array

#%%
def simulate_targeted_price_shock(L_inverse, A_m, value_added_df, import_price_shocks, primary_price_shock_matrix, sector_list):
    """
    Calculates price changes allowing for sector-specific primary input shocks.
    
    Parameters:
    primary_price_shock_matrix: A numpy array of shape (42, 4). 
                                Each row is a sector, each column is a VA component.
                                Values are percentage changes (e.g., 5 for 5%).
    """
    V_df = value_added_df.drop('Total Value Added')
    V = V_df.values  
    L_inverse_T = L_inverse.T
    A_m_T = A_m.T

    V=V_df.T # Shape: (42, 4)

    # Ensure shocks are numpy arrays
    import_shocks_np = np.asarray(import_price_shocks)
    pv_shock_matrix = np.asarray(primary_price_shock_matrix) # Shape: (42, 4)

    # --- THE NEW MATH ---
    # 1. Element-wise multiplication to get the specific cost push for each sector/component
    # V * pv_shock_matrix means: (baseline wage share * wage shock) for every cell
    localized_va_push = V * (pv_shock_matrix / 100) # Divide by 100 here to convert % to decimal
    
    # 2. Sum across the 4 VA columns (axis=1) to get the total primary cost push per sector
    # Results in a (42, 1) vector
    total_primary_push = np.sum(localized_va_push, axis=1)

    # 3. Calculate import push (stays the same, it was already sector-specific via A_m)
    import_push = np.dot(A_m_T, (import_shocks_np / 100))

    # 4. Total direct cost push
    total_direct_push = import_push + total_primary_push

    # 5. Apply the Leontief Inverse to get the final domestic price changes
    delta_P = np.dot(L_inverse_T, total_direct_push)

    # Convert back to percentage points for display
    domestic_price_pctchanges = delta_P * 100

    return pd.Series(domestic_price_pctchanges, index=sector_list, name="Targeted Price Changes (%)")

#%%

def linkage_calculator(L_inverse, sector_list):
    
    # Backward Linkage = Column Sums (axis=0 sums vertically down the columns)
    # This answers: "If I shock this sector's final demand, how much total output happens?"
    backward_linkage = np.sum(L_inverse, axis=0)

    std_backward_linkage = backward_linkage/backward_linkage.mean()
    
    # Forward Linkage = Row Sums (axis=1 sums horizontally across the rows)
    # This answers: "If this sector produces 1 unit, how much does it ripple through the economy?"
    forward_linkage = np.sum(L_inverse, axis=1)

    std_forward_linkage = forward_linkage/forward_linkage.mean()
    
    # Assemble into a clean DataFrame
    linkages_df = pd.DataFrame({
        'Backward Linkage (Output Mult.)': backward_linkage,
        'Standardized Backward Linkage': std_backward_linkage,
        'Forward Linkage': forward_linkage,
        'Standardized Forward Linkage': std_forward_linkage
    }, index=sector_list)

    filtered_linkages = linkages_df[(linkages_df["Standardized Backward Linkage"] > 1) & (linkages_df["Standardized Forward Linkage"] > 1)]
    
    return linkages_df, filtered_linkages

