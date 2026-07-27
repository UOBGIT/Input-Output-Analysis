#%%
import os
import pandas as pd
import numpy as np

#%%
# ==========================================
# Standardized Way to Clean Dataframes
# ==========================================

def clean_IO_df(df):
    try:
        df.index = df["Unnamed: 0"]
        df = df.drop(columns = ["Unnamed: 0"])
        df = df.rename_axis(None, axis = 0)
        return df
    except:
        print("Error in processing dataframe...")
        return None

#%%
# ==========================================
# FUNCTION TO CREATE 3 REQUIRED MATRICES
# ==========================================

def calculate_io_coefficients(IDT_df, IIT_df, PI_df, total_output):
    """
    Calculates Domestic, Imported, and Value Added technical coefficients.
    
    Parameters:
    IDT_df (pd.DataFrame): Domestic intermediate transactions (Index: Sectors, Columns: Sectors).
    IIT_df (pd.DataFrame): Imported intermediate transactions (Index: Sectors, Columns: Sectors).
    PI_df (pd.DataFrame): Primary inputs / Value added (Index: VA Categories, Columns: Sectors).
    total_output (pd.Series or pd.DataFrame): Total output (Index: Sectors).
    
    Returns:
    tuple: (DTC_df, ITC_df, value_added_df)
    """
    # If total_output is passed as a DataFrame (e.g., 1 column), squeeze it to a Series
    if isinstance(total_output, pd.DataFrame):
        total_output = total_output.squeeze()
        
    # Safety Check: Ensure indices match (Pandas will throw an error if they don't align)
    if not IDT_df.columns.equals(total_output.index):
        raise ValueError("Column names of IDT/IIT/PI do not match the index of total_output!")

    # 1. Domestic Technical Coefficients (A^d)
    # axis=1 means "divide each column by the corresponding value in total_output"
    DTC_df = IDT_df.divide(total_output, axis=1)
    
    # 2. Imported Technical Coefficients (A^m)
    ITC_df = IIT_df.divide(total_output, axis=1)
    
    # 3. Value Added Coefficients (VA / X)
    value_added_df = PI_df.divide(total_output, axis=1)
    
    # Optional: Replace infinities with NaN just in case a sector had 0 total output
    DTC_df = DTC_df.replace([np.inf, -np.inf], np.nan)
    ITC_df = ITC_df.replace([np.inf, -np.inf], np.nan)
    value_added_df = value_added_df.replace([np.inf, -np.inf], np.nan)

    return DTC_df, ITC_df, value_added_df

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
def simulate_demand_shock(L_inverse, delta_Y, A_d, A_m, sector_list, remuneration_array, NPT_array, DFA_array, OS_array, total_value_added_array):

    I = np.identity(len(sector_list))

    # Calculate the "Deep Supply Chain" matrix: L - I - A
    deep_supply_chain_matrix = L_inverse - I - A_d

    ## ==========================================
    ## 1. TOTAL OUTPUT DECOMPOSITION
    ## ==========================================
    delta_X_total = np.dot(L_inverse, delta_Y)
    delta_X_tier1 = delta_Y  # Direct
    delta_X_tier2 = np.dot(A_d, delta_Y) # 1st Wave
    delta_X_tier3 = np.dot(deep_supply_chain_matrix, delta_Y) # Deep Supply Chain

    ## ==========================================
    ## 2. IMPORT LEAKAGE DECOMPOSITION
    ## ==========================================
    delta_M_total = np.dot(A_m, delta_X_total)
    delta_M_tier2 = np.dot(A_m, delta_Y) # Imports to make the direct goods
    delta_M_tier3 = np.dot(A_m, delta_X_tier3) # Imports to make the deep supply chain goods

    ## ==========================================
    ## 3. VALUE ADDED (GDP) DECOMPOSITION
    ## ==========================================
    delta_VA_total = np.dot(total_value_added_array, delta_X_total)
    delta_VA_tier1 = np.dot(total_value_added_array, delta_Y) # VA from direct production
    delta_VA_tier2 = np.dot(total_value_added_array, delta_X_tier2) # VA from 1st wave suppliers
    delta_VA_tier3 = np.dot(total_value_added_array, delta_X_tier3) # VA from deep supply chain

    ## ==========================================
    ## 4. VA COMPONENTS (Total and Decomposed)
    ## ==========================================
    delta_remuneration = np.dot(remuneration_array, delta_X_total)
    delta_remuneration_tier1 = np.dot(remuneration_array, delta_Y)
    delta_remuneration_tier2 = np.dot(remuneration_array, delta_X_tier2)
    delta_remuneration_tier3 = np.dot(remuneration_array, delta_X_tier3)

    delta_NPT = np.dot(NPT_array, delta_X_total)
    delta_NPT_tier1 = np.dot(NPT_array, delta_Y)
    delta_NPT_tier2 = np.dot(NPT_array, delta_X_tier2)
    delta_NPT_tier3 = np.dot(NPT_array, delta_X_tier3)

    delta_DFA = np.dot(DFA_array, delta_X_total)
    delta_DFA_tier1 = np.dot(DFA_array, delta_Y)
    delta_DFA_tier2 = np.dot(DFA_array, delta_X_tier2)
    delta_DFA_tier3 = np.dot(DFA_array, delta_X_tier3)

    delta_OS = np.dot(OS_array, delta_X_total)
    delta_OS_tier1 = np.dot(OS_array, delta_Y)
    delta_OS_tier2 = np.dot(OS_array, delta_X_tier2)
    delta_OS_tier3 = np.dot(OS_array, delta_X_tier3)

    ## ==========================================
    ## ASSEMBLE MAIN DATAFRAME (Clean Summary)
    ## ==========================================
    results_df = pd.concat([
        pd.Series(delta_X_total, index=sector_list, name='Total Output Impact'),
        pd.Series(delta_M_total, index=sector_list, name='Total Import Leakage'),
        pd.Series(delta_X_total - delta_M_total, index=sector_list, name="Domestic Net Content Impact"),
        pd.Series(delta_VA_total, index=sector_list, name="Total Change in Value Added"),
        pd.Series(delta_remuneration, index=sector_list, name="Change in Remuneration"),
        pd.Series(delta_NPT, index=sector_list, name="Change in Taxes"),
        pd.Series(delta_DFA, index=sector_list, name="Change in Depreciation"),
        pd.Series(delta_OS, index=sector_list, name="Change in Operating Surplus")
        
    ], axis=1)
    results_df.loc['Total'] = results_df.sum(numeric_only=True)

    ## ==========================================
    ## ASSEMBLE DECOMPOSITION DATAFRAME (The Deep Dive)
    ## ==========================================
    direct_indirect_df = pd.concat([
        # Output Tiers
        pd.Series(delta_X_tier1, index=sector_list, name="Output: Direct (Initial Shock)"),
        pd.Series(delta_X_tier2, index=sector_list, name="Output: 1st Wave (Immediate Suppliers)"),
        pd.Series(delta_X_tier3, index=sector_list, name="Output: 2nd+ Wave (Deep Supply Chain)"),
        # Import Tiers
        pd.Series(delta_M_tier2, index=sector_list, name="Imports: 1st Wave"),
        pd.Series(delta_M_tier3, index=sector_list, name="Imports: 2nd+ Wave"),
        # GDP Tiers
        pd.Series(delta_VA_tier1, index=sector_list, name="Value Added: Direct"),
        pd.Series(delta_VA_tier2, index=sector_list, name="Value Added: 1st Wave"),
        pd.Series(delta_VA_tier3, index=sector_list, name="Value Added: 2nd+ Wave"),
        # Remuneration Tiers
        pd.Series(delta_remuneration_tier1, index=sector_list, name="Remuneration: Direct"),
        pd.Series(delta_remuneration_tier2, index=sector_list, name="Remuneration: 1st Wave"),
        pd.Series(delta_remuneration_tier3, index=sector_list, name="Remuneration: 2nd+ Wave"),
        # Net Production Tax Tiers
        pd.Series(delta_NPT_tier1, index=sector_list, name="NPT: Direct"),
        pd.Series(delta_NPT_tier2, index=sector_list, name="NPT: 1st Wave"),
        pd.Series(delta_NPT_tier3, index=sector_list, name="NPT: 2nd+ Wave"), 
        # Depreciation of Fixed Asset
        pd.Series(delta_DFA_tier1, index=sector_list, name="DFA: Direct"),
        pd.Series(delta_DFA_tier2, index=sector_list, name="DFA: 1st Wave"),
        pd.Series(delta_DFA_tier3, index=sector_list, name="DFA: 2nd+ Wave"),
        # Operating Surplus
        pd.Series(delta_OS_tier1, index=sector_list, name="OS: Direct"),
        pd.Series(delta_OS_tier2, index=sector_list, name="OS: 1st Wave"),
        pd.Series(delta_OS_tier3, index=sector_list, name="OS: 2nd+ Wave")
    ], axis=1)
    direct_indirect_df.loc['Total'] = direct_indirect_df.sum(numeric_only=True)

    return results_df, direct_indirect_df    


    


    

#%%
def generate_va_matrices(value_added_df, sector_list):
    ### Creating a diagonal Total Value Added Dataframe

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
    delta_P_direct_series = pd.Series(total_direct_push, index = sector_list, name = "Direct Price Changes")

    # 5. Apply the Leontief Inverse to get the final domestic price changes
    delta_P = np.dot(L_inverse_T, total_direct_push)
    delta_P_series = pd.Series(delta_P, index = sector_list, name = "Total Price Changes (%)")

    # 6. Calculate indirect price changes
    delta_P_indirect = delta_P - total_direct_push
    delta_P_indirect_series = pd.Series(delta_P_indirect, index = sector_list, name = "Indirect Price Changes")

    # Convert back to percentage points for display
    domestic_price_pctchanges = delta_P * 100

    results_df = pd.concat([
        100*delta_P_series, 100*delta_P_direct_series, 100*delta_P_indirect_series
        
    ], axis=1)

    return results_df

#%%

def linkage_calculator(L_inverse, A_m, sector_list):
    
    # Backward Linkage = Column Sums (axis=0 sums vertically down the columns)
    # This answers: "If I shock this sector's final demand, how much total output happens?"
    backward_linkage = np.sum(L_inverse, axis=0)

    std_backward_linkage = backward_linkage/backward_linkage.mean()
    
    # Forward Linkage = Row Sums (axis=1 sums horizontally across the rows)
    # This answers: "If this sector produces 1 unit, how much does it ripple through the economy?"
    forward_linkage = np.sum(L_inverse, axis=1)

    std_forward_linkage = forward_linkage/forward_linkage.mean()

    # 1. Multiply to get the full 42x42 matrix of import requirements
    import_intensity_matrix = np.dot(A_m, L_inverse)

    # 2. Sum down the columns (axis=0) to get the 1x42 vector
    import_intensity_vector = np.sum(import_intensity_matrix, axis=0)

    # 3. Wrap in a pandas Series for your app
    import_intensity_series = pd.Series(import_intensity_vector, index=sector_list, name="Import Dependency Ratio")
    
    # Assemble into a clean DataFrame
    linkages_df = pd.DataFrame({
        'Backward Linkage (Output Mult.)': backward_linkage,
        'Forward Linkage': forward_linkage,
        'Import Dependency Ratio': import_intensity_series,
        'Standardized Backward Linkage': std_backward_linkage,
        'Standardized Forward Linkage': std_forward_linkage
    }, index=sector_list)

    filtered_linkages = linkages_df[(linkages_df["Standardized Backward Linkage"] > 1) & (linkages_df["Standardized Forward Linkage"] > 1)]
    
    return linkages_df, filtered_linkages




