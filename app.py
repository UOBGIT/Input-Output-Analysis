import streamlit as st
import pandas as pd
import numpy as np

# Import the functions you saved in Step 2
from input_output_model import (
    create_demand_shock,
    simulate_demand_shock,
    generate_va_matrices, 
    simulate_targeted_price_shock, 
    linkage_calculator
)

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Input Output Analysis App", layout="wide")
st.title("China Input Output Analysis Engine")
st.markdown("Use China's 2023 Input Output data to forecast the impacts of demand and price shocks.")

# --- CATEGORIZATION FRAMEWORK ---
# Used to organize the sidebar inputs
sector_categories = {
    "1. Agriculture & Mining": [
        "Farming, Forestry, Animal Husbandry, Fishery and Services",
        "Coal",
        "Petroleum & Natural Gas",
        "Metal",  
        "Non Metal Mineral and Other Mining"
    ],
    "2. Light & Consumer Manufacturing": [
        "Foods & Tobacco",
        "Textile",
        "Garment and Apparel, Footwear, Headgear, Leather, Down & Related Products",
        "Wood Processing & Furniture",
        "Paper Making, Printing, Cultural & Cultural, Educational & Sports Articles"
    ],
    "3. Heavy, High-Tech & Capital Manufacturing": [
        "Petroleum, Coking and Nuclear Fuel Processing",
        "Chemical Product",
        "Non Metal Mineral Product", 
        "Metal Smelting & Pressing",
        "Fabricated Metal Product",
        "General Equipment",
        "Special Purpose Equipment",
        "Transportation Equipment",
        "Electrical Machinery & Equipment",
        "Communication, Computers & Other Electronic Equipment",
        "Instrument & Meter",
        "Other Mfg Product & Waste Material",
        "Fabricated Metal Product, Machine & Equipment Repair"
    ],
    "4. Utilities, Infrastructure & Construction": [
        "Electricity, Heat Production & Supply",
        "Gas Production & Supply",
        "Water Production & Supply",
        "Water Conservancy, Environment & Utility Management",
        "Construction Industry"
    ],
    "5. Modern & Traditional Services": [
        "Wholesale & Retail",
        "Transport, Storage & Post",
        "Accommodation and Catering",
        "Information Transmission, Software and Information Technology Service",
        "Financial Intermediation",
        "Real Estate",
        "Leasing and Commercial Service",
        "Scientific Research & Development",
        "Polytechincal Services",
        "Resident, Repair and Other Services",
        "Education",
        "Health Care & Social Work",
        "Culture, Sport & Entertainment",
        "Public Administration, Social Security & Social Organization"
    ]
}

# Explicit list of VA components to ensure strict ordering for matrix math
va_components_list = [
    "Remuneration of Employee", 
    "Net Production Tax", 
    "Depreciation of Fixed Asset", 
    "Operating Surplus"
]

file_path = "Input Output Table.xlsx"

### DOWNLOAD AND CLEAN DATA

DTC_df = pd.read_excel(file_path, sheet_name = "Domestic Technical Coefficients")
ITC_df = pd.read_excel(file_path, sheet_name = "Imported Technical Coefficients")

DTC_df.index = DTC_df["Unnamed: 0"]
DTC_df = DTC_df.drop(columns = ["Unnamed: 0"])
ITC_df.index = ITC_df["Unnamed: 0"]
ITC_df = ITC_df.drop(columns = ["Unnamed: 0"])

DTC_df = DTC_df.rename_axis(None, axis=0)
ITC_df = ITC_df.rename_axis(None, axis=0)

n = DTC_df.shape[0]

# Commented out st.dataframe to keep main page clean, 
# but you can uncomment it to debug:
# st.dataframe(DTC_df)

# ==========================================
# PHASE 2: Building the Prediction Engine
# ==========================================

I = np.identity(n)
A_d = DTC_df.values
A_m = ITC_df.values
L_matrix = I - A_d
L_inverse = np.linalg.inv(L_matrix)
L_inverse_df = pd.DataFrame(L_inverse, index=DTC_df.index, columns=DTC_df.columns)

sector_list = DTC_df.columns.tolist() # Converted to list for easier indexing

# ==========================================
# Downloading Value Added Dataframe
# ==========================================

value_added_df = pd.read_excel(file_path, sheet_name = "Value Added By Industry")
value_added_df.index = value_added_df["Unnamed: 0"]
value_added_df = value_added_df.drop(columns = ["Unnamed: 0"])
value_added_df = value_added_df.rename_axis(None, axis=0)

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

for sector in sector_list:
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

# ==========================================
# INITIALIZE SHOCK VARIABLES (Crucial step!)
# We must create these empty variables before the popups so the 
# execution block at the bottom doesn't throw an error if a button isn't clicked.
# ==========================================
primary_shock_matrix = np.zeros((len(sector_list), len(va_components_list)))
import_price_shock_dict = {sector: 0.0 for sector in sector_list}

short_va_names = ["Wages", "Taxes", "Deprec.", "Profit"]

# ==========================================
# SIDEBAR 1: DEMAND SHOCK UI
# ==========================================
st.sidebar.header("⚙️ Demand Shock Configuration")
st.sidebar.markdown("Set exogenous demand shocks (in Billions RMB).")

shock_dict = {}

# Create expandable sections for each category
for category, sectors in sector_categories.items():
    with st.sidebar.expander(category):
        for sector in sectors:
            # Safety check: only create input if sector actually exists in your Excel data
            if sector in sector_list:
                shock_value = st.number_input(
                    label=sector,
                    min_value=-1000.0, # Allow negative shocks (e.g. demand destruction)
                    max_value=10000.0,
                    value=0.0,        # Default is 0 as requested
                    step=10.0,
                    key=f"shock_{sector}" # Unique key for Streamlit
                )
                shock_dict[sector] = shock_value
            else:
                # Helpful warning if there's a typo in the dictionary above
                st.sidebar.warning(f"'{sector}' not found in data!")


# Add a run button so the math doesn't recalculate on every single keystroke
run_simulation = st.sidebar.button("🚀 Run Demand Simulation", type="primary", use_container_width=True)

st.sidebar.divider() # Visual separator

# ==========================================
# SIDEBAR 2: PRICE SHOCK UI
# ==========================================
st.sidebar.header("💰 Price Shock")
st.sidebar.markdown("Set percentage changes in prices (e.g., 5 for +5%).")


# --- PRIMARY INPUTS POPOVER ---
with st.sidebar.popover("🎯 Targeted Primary Input Prices"):
    apply_by_sector = st.checkbox("Apply by sector", key="apply_by_sector_cb")

    if not apply_by_sector:
        # ECONOMY-WIDE MODE
        st.markdown("*Applies uniform % change to all sectors*")
        cols = st.columns(4)
        for j, va_comp in enumerate(va_components_list):
            with cols[j]:
                st.markdown(f"<span style='font-size: 0.9em; font-weight: bold;'>{short_va_names[j]}</span>", unsafe_allow_html=True)
                val = st.number_input(
                    label=va_comp, label_visibility="collapsed", 
                    min_value=-100.0, max_value=500.0, value=0.0, step=1.0, 
                    key=f"econ_wide_{j}"
                )
                # Apply to entire column in the matrix
                primary_shock_matrix[:, j] = val
                
    else:
        # SECTOR-SPECIFIC MODE (Now has full screen width!)
        for category, sectors in sector_categories.items():
            with st.expander(category):
                # Small column headers
                header_cols = st.columns([3, 1, 1, 1, 1]) 
                for j, name in enumerate(short_va_names):
                    with header_cols[j+1]:
                        st.markdown(f"<span style='font-size: 0.85em; font-weight: bold; text-align: center;'>{name}</span>", unsafe_allow_html=True)
                
                st.markdown("<hr style='margin: 0px; border-top: 1px solid #555;'>", unsafe_allow_html=True)
                
                for sector in sectors:
                    if sector in sector_list:
                        i = sector_list.index(sector) 
                        
                        row_cols = st.columns([3, 1, 1, 1, 1])
                        with row_cols[0]:
                            st.markdown(f"<span style='font-size: 0.85em;'>{sector}</span>", unsafe_allow_html=True)
                        
                        for j, va_comp in enumerate(va_components_list):
                            with row_cols[j+1]:
                                val = st.number_input(
                                    label=va_comp, label_visibility="collapsed", 
                                    min_value=-100.0, max_value=500.0, value=0.0, step=1.0, 
                                    key=f"targeted_{i}_{j}"
                                )
                                # Apply to specific cell in the matrix
                                primary_shock_matrix[i, j] = val


# --- IMPORTED INPUTS POPOVER ---
with st.sidebar.popover("📦 Imported Input Prices"):
    st.markdown("*Set % change in price of imported intermediate goods*")
    for category, sectors in sector_categories.items():
        with st.expander(category):
            for sector in sectors:
                if sector in sector_list:
                    # 2 columns: Name | Input (Plenty of room in a popover)
                    cols = st.columns([4, 1])
                    with cols[0]:
                        st.markdown(f"<span style='font-size: 0.85em;'>{sector}</span>", unsafe_allow_html=True)
                    with cols[1]:
                        val = st.number_input(
                            label=sector, label_visibility="collapsed", 
                            min_value=-100.0, max_value=500.0, value=0.0, step=1.0, 
                            key=f"import_price_{sector}"
                        )
                        import_price_shock_dict[sector] = val

run_price_sim = st.sidebar.button("📈 Run Price Simulation", type="primary", use_container_width=True)

st.sidebar.divider() # Visual separator

# ==========================================
# SIDEBAR 3: Linkage Table
# ==========================================
st.sidebar.header("🔗 Linkages")
st.sidebar.markdown("Explore industry connections")


show_linkages = st.sidebar.button("🔗 Generate Table", type="primary", use_container_width=True)
filtered = st.sidebar.checkbox("Show only above-average linkages")

# ==========================================
# EXECUTE SIMULATION (Triggered by button)
# ==========================================
if run_simulation:
    st.subheader("Simulation Results (in Billions RMB)")
    
    # 1. Convert the UI dictionary into the numpy array using your function
    delta_Y = create_demand_shock(shock_dict, sector_list)
    
    # 2. Run your simulation function
    results_df = simulate_demand_shock(
        L_inverse_df, 
        delta_Y, 
        A_m, 
        sector_list, 
        remuneration_array, 
        NPT_array, 
        DFA_array, 
        OS_array, 
        total_value_added_array
    )
    
    # 3. Display the results
    # Format numbers to 2 decimal places for cleaner UI
    st.dataframe(results_df.style.format("{:.4f}"), use_container_width=True)

if run_price_sim:
    st.subheader("📈 Price Simulation Results")
    
    # 1. Convert UI dicts to numpy arrays (reusing the demand shock function!)
    import_shocks_np = create_demand_shock(import_price_shock_dict, sector_list)
    
    # 2. Run simulation
    price_results_df = simulate_targeted_price_shock(
        L_inverse_df, A_m, value_added_df, 
        import_shocks_np, primary_shock_matrix, sector_list
    )
    
    # 3. Display
    st.dataframe(price_results_df.to_frame().style.format("{:.2f}%"), use_container_width=True)

if show_linkages:
    linkages_df, filtered_linkages = linkage_calculator(L_inverse, sector_list)

    if filtered:
        st.subheader("🔗 Filtered Inter-Industry Linkage Table")
        st.dataframe(filtered_linkages)

    else:
        st.subheader("🔗 Inter-Industry Linkage Table")
        st.dataframe(linkages_df)
    

    