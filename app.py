import streamlit as st
import pandas as pd
import numpy as np
import io

# Import the functions you saved in Step 2
from input_output_model import (
    clean_IO_df,
    calculate_io_coefficients,
    create_demand_shock,
    simulate_demand_shock,
    generate_va_matrices, 
    simulate_targeted_price_shock, 
    linkage_calculator
)

from io_processor import (
    is_number, 
    get_rid_of_numbers,
    clean_input_output_data,
    create_sector_list,
    populate_IT_matrix,
    create_value_added_list,
    populate_primary_input_matrix,
    create_final_use_columns,
    populate_final_use_matrix,
    populate_total_output_vector,
    populate_total_input_vector,
    check_value_added,
    check_final_use,
    check_domestic_total_output,
    check_imported_total_output,
    check_total_input,
    check_input_output_equality
)

def load_and_process_data(uploaded_file):
    df = pd.read_excel(uploaded_file)
    df_index, parsed_names, logs = clean_input_output_data(df) ## Create identification table
    st.code("\n".join(logs), language="text")
    sector_list, logs = create_sector_list(parsed_names) ## Create list of sector names (normally 42)
    st.code("\n".join(logs), language="text")

    ## Generate and populate intermediate transaction matrices

    IDT_df = pd.DataFrame(index = sector_list, columns = sector_list)
    IIT_df = pd.DataFrame(index = sector_list, columns = sector_list) 
    IDT_df, logs = populate_IT_matrix(IDT_df, df_index, parsed_names, domestic = True)
    st.code("\n".join(logs), language="text")
    IIT_df, logs = populate_IT_matrix(IIT_df, df_index, parsed_names, domestic = False)
    st.code("\n".join(logs), language="text")
    
    ## Generate and populate primary input matrix

    value_added_list, logs = create_value_added_list(parsed_names)
    st.code("\n".join(logs), language="text")
    PI_df = pd.DataFrame(index = value_added_list, columns = sector_list)
    PI_df, logs = populate_primary_input_matrix(PI_df, df_index, parsed_names)
    st.code("\n".join(logs), language="text")

    ## Generate and populate final demand matrices

    final_use_columns, logs = create_final_use_columns(parsed_names)
    st.code("\n".join(logs), language="text")

    FDD_df = pd.DataFrame(index = sector_list, columns = final_use_columns)
    FDI_df = pd.DataFrame(index = sector_list, columns = final_use_columns)
    FDD_df, logs = populate_final_use_matrix(FDD_df, df_index, parsed_names, domestic = True)
    st.code("\n".join(logs), language="text")
    FDI_df, logs = populate_final_use_matrix(FDI_df, df_index, parsed_names, domestic = False)
    st.code("\n".join(logs), language="text")

    ## Generate and populate total output vectors

    total_output_domestic = pd.DataFrame(index = sector_list, columns = ["Total Domestic Output"])
    total_output_imported = pd.DataFrame(index = sector_list, columns = ["Total Imported Output"])
    total_output_domestic, total_output_imported, logs = populate_total_output_vector(total_output_domestic, total_output_imported, df_index, parsed_names)
    st.code("\n".join(logs), language="text")
    ## Generate and populate total input vectors 

    total_input = pd.DataFrame(index = sector_list, columns = ["Total Input"])
    total_input, logs = populate_total_input_vector(total_input, df_index, parsed_names)
    st.code("\n".join(logs), language="text")

    return IDT_df, IIT_df, PI_df, FDD_df, FDI_df, total_output_domestic, total_output_imported, total_input

def verify_io_data(IDT_df, IIT_df, PI_df, FDD_df, FDI_df, total_output_domestic, total_output_imported, total_input):
    logs = check_value_added(PI_df)
    st.code("\n".join(logs), language="text")
    logs = check_final_use(FDD_df)
    st.code("\n".join(logs), language="text")
    logs = check_domestic_total_output(IDT_df, FDD_df, total_output_domestic)
    st.code("\n".join(logs), language="text")
    logs = check_imported_total_output(IIT_df, FDI_df, total_output_imported)
    st.code("\n".join(logs), language="text")
    logs = check_total_input(IDT_df, IIT_df, PI_df, total_input)
    st.code("\n".join(logs), language="text")
    logs = check_input_output_equality(total_input, total_output_domestic)
    st.code("\n".join(logs), language="text")

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Input Output Analysis App", layout="wide")
st.title("China Input Output Analysis Engine")
st.markdown("Use China's 2023 Input Output data to forecast the impacts of demand and price shocks.")
st.markdown("Data is taken from China's National Bureau of Statistics 2023 Non-competitive Input Output database. Feel free to download the formatted input output table below for your own reference.")


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



# ==========================================
# NEW WAY TO GET THE 3 MAIN MATRICES
# ==========================================

file_path = "Input Output Table.xlsx"

# ==========================================
# Download and Clean Dataframes
# ==========================================

IDT_df = pd.read_excel(file_path, sheet_name = "1. IDT (Domestic)")
IIT_df = pd.read_excel(file_path, sheet_name = "2. IIT (Imported)")
PI_df = pd.read_excel(file_path, sheet_name = "3. Primary Inputs")
total_output = pd.read_excel(file_path, sheet_name = "6. Total Output (Domestic)")

IDT_df = clean_IO_df(IDT_df)
IIT_df = clean_IO_df(IIT_df)
PI_df = clean_IO_df(PI_df)
total_output = clean_IO_df(total_output)

# ==========================================
# CREATE MATRICES FOR ALGEBRAIC MANIPULATION
# ==========================================

DTC_df, ITC_df, value_added_df = calculate_io_coefficients(IDT_df, IIT_df, PI_df, total_output)
n = DTC_df.shape[0]

sector_list = value_added_df.columns
total_value_added_array, remuneration_array, NPT_array, DFA_array, OS_array = generate_va_matrices(value_added_df, sector_list)

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

# Initialize session state variables for auto-populating UI
if 'primary_econ_shocks' not in st.session_state:
    st.session_state.primary_econ_shocks = [0.0, 0.0, 0.0, 0.0] # Wages, Taxes, Deprec, OS
if 'import_econ_shocks' not in st.session_state:
    st.session_state.import_econ_shocks = 0.0 # Single number for all imports
if 'import_price_shock_dict' not in st.session_state:
    st.session_state.import_price_shock_dict = {sector: 0.0 for sector in sector_list}
if 'fd_econ_shocks' not in st.session_state:
    st.session_state.fd_econ_shocks = [0.0, 0.0, 0.0, 0.0] # Capital, Inventory, Consumption, Exports



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
st.sidebar.header("⚙️ Demand Shock")

fd_components = ["Capital", "Inventory", "Consumption", "Exports"]
demand_shock_dict = {}

with st.sidebar.popover("🎯 Targeted Final Demand Shock"):
    st.markdown("*Input values in Billions RMB. Components will be summed per sector.*")
    apply_fd_by_sector = st.checkbox("Apply by sector", key="fd_by_sector_cb")

    if not apply_fd_by_sector:
        # ---------------------------------------------------------
        # ECONOMY-WIDE MODE
        # ---------------------------------------------------------
        cols = st.columns(4)
        for j, comp in enumerate(fd_components):
            with cols[j]:
                st.markdown(f"<span style='font-size: 0.9em; font-weight: bold;'>{comp}</span>", unsafe_allow_html=True)
                # Save input to session state so it isn't lost when clicking the checkbox
                val = st.number_input(
                    label=comp, label_visibility="collapsed", 
                    min_value=-1000.0, max_value=10000.0, 
                    value=st.session_state.fd_econ_shocks[j], step=10.0, 
                    key=f"fd_econ_wide_{j}"
                )
                st.session_state.fd_econ_shocks[j] = val
        
        # Sum the 4 components and apply to every sector
        total_economy_shock = sum(st.session_state.fd_econ_shocks)
        for sector in sector_list:
            demand_shock_dict[sector] = total_economy_shock
                
    else:
        # ---------------------------------------------------------
        # SECTOR-SPECIFIC MODE
        # ---------------------------------------------------------
        for category, sectors in sector_categories.items():
            with st.expander(category):
                # Small column headers
                header_cols = st.columns([3, 1, 1, 1, 1]) 
                for j, name in enumerate(fd_components):
                    with header_cols[j+1]:
                        st.markdown(f"<span style='font-size: 0.75em; font-weight: bold; text-align: center;'>{name}</span>", unsafe_allow_html=True)
                
                st.markdown("<hr style='margin: 0px; border-top: 1px solid #555;'>", unsafe_allow_html=True)
                
                for sector in sectors:
                    if sector in sector_list:
                        # Use global index to prevent duplicate key errors across categories
                        global_idx = sector_list.index(sector)
                        
                        row_cols = st.columns([3, 1, 1, 1, 1])
                        with row_cols[0]:
                            st.markdown(f"<span style='font-size: 0.85em;'>{sector}</span>", unsafe_allow_html=True)
                        
                        sector_total_shock = 0.0
                        for j, comp in enumerate(fd_components):
                            with row_cols[j+1]:
                                # THE MAGIC: Use the economy-wide shock as the default value!
                                val = st.number_input(
                                    label=comp, label_visibility="collapsed", 
                                    min_value=-1000.0, max_value=10000.0, 
                                    value=st.session_state.fd_econ_shocks[j], # <--- Auto-populate
                                    step=10.0, key=f"fd_sec_{global_idx}_{j}"
                                )
                                sector_total_shock += val
                        
                        # Add the summed total for this specific sector to the dictionary
                        demand_shock_dict[sector] = sector_total_shock

run_demand_sim = st.sidebar.button("🚀 Run Demand Simulation", type="primary", use_container_width=True)

st.sidebar.divider()

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
                # Save input to session state so it isn't lost when clicking the checkbox
                val = st.number_input(
                    label=va_comp, label_visibility="collapsed", 
                    min_value=-100.0, max_value=500.0, value=st.session_state.primary_econ_shocks[j], step=1.0, 
                    key=f"econ_wide_{j}"
                )
                st.session_state.primary_econ_shocks[j] = val
                primary_shock_matrix[:, j] = val
                
    else:
        # SECTOR-SPECIFIC MODE
        for category, sectors in sector_categories.items():
            with st.expander(category):
                header_cols = st.columns([3, 1, 1, 1, 1]) 
                for j, name in enumerate(short_va_names):
                    with header_cols[j+1]:
                        st.markdown(f"<span style='font-size: 0.85em; font-weight: bold; text-align: center;'>{name}</span>", unsafe_allow_html=True)
                st.markdown("<hr style='margin: 0px; border-top: 1px solid #555;'>", unsafe_allow_html=True)
                
                for sector in sectors:
                    if sector in sector_list:
                        global_idx = sector_list.index(sector)
                        row_cols = st.columns([3, 1, 1, 1, 1])
                        with row_cols[0]:
                            st.markdown(f"<span style='font-size: 0.85em;'>{sector}</span>", unsafe_allow_html=True)
                        
                        for j, va_comp in enumerate(va_components_list):
                            with row_cols[j+1]:
                                # THE MAGIC: Use the economy-wide shock as the default value!
                                val = st.number_input(
                                    label=va_comp, label_visibility="collapsed", 
                                    min_value=-100.0, max_value=500.0, 
                                    value=st.session_state.primary_econ_shocks[j], # <--- Auto-populate
                                    step=1.0, key=f"targeted_{global_idx}_{j}"
                                )
                                primary_shock_matrix[global_idx, j] = val


# --- IMPORTED INPUTS POPOVER ---
with st.sidebar.popover("📦 Imported Input Prices"):
    import_apply_by_sector = st.checkbox("Apply by sector", key="import_apply_by_sector_cb")
    
    if not import_apply_by_sector:
        # NEW ECONOMY-WIDE MODE FOR IMPORTS
        st.markdown("*Applies uniform % change to all imported inputs*")
        st.session_state.import_econ_shocks = st.number_input(
            label="All Imported Inputs", label_visibility="visible",
            min_value=-100.0, max_value=500.0, 
            value=st.session_state.import_econ_shocks, step=1.0, 
            key="import_econ_wide_single"
        )
        # Apply to all sectors in the background dictionary
        for sector in sector_list:
            st.session_state.import_price_shock_dict[sector] = st.session_state.import_econ_shocks

    else:
        # SECTOR-SPECIFIC MODE FOR IMPORTS
        for category, sectors in sector_categories.items():
            with st.expander(category):
                for sector in sectors:
                    if sector in sector_list:
                        cols = st.columns([4, 1])
                        with cols[0]:
                            st.markdown(f"<span style='font-size: 0.85em;'>{sector}</span>", unsafe_allow_html=True)
                        with cols[1]:
                            # THE MAGIC: Use the economy-wide shock as the default value!
                            val = st.number_input(
                                label=sector, label_visibility="collapsed", 
                                min_value=-100.0, max_value=500.0, 
                                value=st.session_state.import_econ_shocks, # <--- Auto-populate
                                step=1.0, key=f"import_price_{sector}"
                            )
                            st.session_state.import_price_shock_dict[sector] = val

run_price_sim = st.sidebar.button("📈 Run Price Simulation", type="primary", use_container_width=True)

st.sidebar.divider() # Visual separator

# ==========================================
# SIDEBAR 3: Deeper Analysis Tools
# ==========================================
st.sidebar.header("🔎 Deeper Analysis Tools")
st.sidebar.markdown("Explore industry connections")

show_linkages = st.sidebar.button("🔗 Generate Linkage Table", type="primary", use_container_width=True)
filtered = st.sidebar.checkbox("Show only above-average linkages")

st.sidebar.markdown("Explore production functions & demand multipliers")
# Combined button for both matrices
show_TCs = st.sidebar.button("📊 Show Technical Coefficients", type="primary", use_container_width=True)
show_leontief = st.sidebar.button("📈 Show Leontief Inverse Matrix", type="primary", use_container_width=True)

# ==========================================
# SIDEBAR 4: Input Output Spreadsheet Download
# ==========================================
st.sidebar.divider() # Visual separator

with open(file_path, "rb") as f:
    excel_data = f.read()

# 2. Add the download button widget
st.sidebar.download_button(
    label="📥 Download Formatted 2023 Input Output Table",
    data=excel_data,
    file_name="China_2023_input_output_table.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# ============================================
# SIDEBAR 5: Clean Up New Input Output Data
# ============================================

st.sidebar.divider()
st.sidebar.header("Clean Input Output Data")

uploaded_file = st.sidebar.file_uploader("Upload Raw IO Data", type=['xlsx'])

# 1. THE CLEANING BLOCK (Only runs if button is clicked)
if uploaded_file is not None:
    clean_new_data = st.sidebar.button("Clean Data", type="primary", use_container_width=True)

    if clean_new_data:
        with st.spinner("Parsing through data..."):
            IDT_df, IIT_df, PI_df, FDD_df, FDI_df, total_output_domestic, total_output_imported, total_input = load_and_process_data(uploaded_file)

        # Save them to session state
        st.session_state.io_dataframes = {
            'IDT': IDT_df,
            'IIT': IIT_df,
            'PI': PI_df,
            'FDD': FDD_df,
            'FDI': FDI_df,
            'total_output_dom': total_output_domestic,
            'total_output_import': total_output_imported,
            'total_input': total_input
        }
        with st.spinner("Verifying data..."):
            verify_io_data(IDT_df, IIT_df, PI_df, FDD_df, FDI_df, total_output_domestic, total_output_imported, total_input)
        st.success("Data successfully parsed and ready for download!")


# 2. THE DOWNLOAD BLOCK (Runs on every page load, but only shows if data exists)
if 'io_dataframes' in st.session_state:
    
    if st.sidebar.button("📥 Download Organized Data (Excel)", type="primary", use_container_width=True):
        with st.spinner("Packaging 8 DataFrames into Excel workbook..."):
            
            # Retrieve your 8 dataframes from session state
            IDT_df = st.session_state.io_dataframes['IDT']
            IIT_df = st.session_state.io_dataframes['IIT']
            PI_df = st.session_state.io_dataframes['PI']
            FDD_df = st.session_state.io_dataframes['FDD']
            FDI_df = st.session_state.io_dataframes['FDI']
            total_output_domestic = st.session_state.io_dataframes['total_output_dom']
            total_output_imported = st.session_state.io_dataframes['total_output_import']
            total_input = st.session_state.io_dataframes['total_input']

            # Create an in-memory buffer
            buffer = io.BytesIO()

            # Write each DataFrame to a different sheet inside the buffer
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                IDT_df.to_excel(writer, sheet_name='1. Transactions (Domestic)', index=True)
                IIT_df.to_excel(writer, sheet_name='2. Transactions (Imported)', index=True)
                PI_df.to_excel(writer, sheet_name='3. Primary Inputs', index=True)
                FDD_df.to_excel(writer, sheet_name='4. Final Demand (Domestic)', index=True)
                FDI_df.to_excel(writer, sheet_name='5. Final Demand (Imported)', index=True)
                total_output_domestic.to_excel(writer, sheet_name='6. Total Output (Domestic)', index=True)
                total_output_imported.to_excel(writer, sheet_name='7. Total Output (Imported)', index=True)
                total_input.to_excel(writer, sheet_name='8. Total Input', index=True)

            # CRITICAL STEP: Reset the buffer's cursor to the beginning
            buffer.seek(0)

            # Trigger the download
            st.sidebar.download_button(
                label="⬇️ Click here to download IO_Data_Separated.xlsx",
                data=buffer,
                file_name="IO_Data_Separated.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

# ==========================================
# EXECUTE SIMULATION (Triggered by button)
# ==========================================
# ==========================================
# MAIN PAGE: EXECUTION & DISPLAY LOGIC
# ==========================================

# ---------------------------------------------------------
# STEP 1: BUTTON CLICKS (Calculate and Save Data)
# ---------------------------------------------------------

if run_demand_sim:
    with st.spinner("Calculating ripple effects..."):
        delta_Y = create_demand_shock(demand_shock_dict, sector_list)
        st.session_state.main_demand_results, st.session_state.breakdown_demand_results = simulate_demand_shock(
            L_inverse_df, delta_Y, DTC_df, ITC_df, sector_list, 
            remuneration_array, NPT_array, DFA_array, OS_array, total_value_added_array
        )
        st.session_state.active_view = 'demand'

if run_price_sim:
    with st.spinner("Calculating price pass-through..."):
        import_shocks_np = create_demand_shock(st.session_state.import_price_shock_dict, sector_list)
        st.session_state.price_results = simulate_targeted_price_shock(
            L_inverse_df, ITC_df, value_added_df, 
            import_shocks_np, primary_shock_matrix, sector_list
        )
        st.session_state.active_view = 'price'

if show_linkages:
    with st.spinner("Calculating linkages..."):
        st.session_state.linkage_results, st.session_state.filtered_linkage_results = linkage_calculator(L_inverse, A_m, sector_list)
        st.session_state.active_view = 'linkages'

if show_TCs:
    # Technical coefficients don't need calculation, just change the view
    st.session_state.active_view = 'tcs'

if show_leontief:
    st.session_state.active_view = 'leontief'


# ---------------------------------------------------------
# STEP 2: DISPLAY LOGIC (Strict If/Elif to prevent stacking)
# ---------------------------------------------------------

current_view = st.session_state.get('active_view')

if current_view == 'demand':
    st.subheader("📊 Demand Simulation Results")
    st.dataframe(st.session_state.main_demand_results.style.format("{:.2f}"), use_container_width=True)
    
    if st.checkbox("🔍 Show Supply Chain Wave Breakdown", key="demand_breakdown_cb"):
        sorted_breakdown = st.session_state.breakdown_demand_results.sort_values(
            by="Output: 2nd+ Wave (Deep Supply Chain)", ascending=False
        )
        st.dataframe(sorted_breakdown.style.format("{:.2f}"), use_container_width=True)

elif current_view == 'price':
    st.subheader("📈 Price Simulation Results")
    if st.checkbox("🔍 Show Direct vs. Indirect Price Pass-Through", key="price_breakdown_cb"):
        st.dataframe(st.session_state.price_results.style.format("{:.2f}%"), use_container_width=True)
    else:
        st.dataframe(st.session_state.price_results[['Total Price Changes (%)']].style.format("{:.2f}%"), use_container_width=True)

elif current_view == 'linkages':
    if st.session_state.get('filtered') and st.sidebar.checkbox("Show only above-average linkages"):
        st.subheader("🔗 Filtered Inter-Industry Linkage Table")
        st.dataframe(st.session_state.filtered_linkage_results)
    else:
        st.subheader("🔗 Inter-Industry Linkage Table")
        st.dataframe(st.session_state.linkage_results)

elif current_view == 'tcs':
    st.subheader("Domestic Technical Coefficients Matrix")
    st.dataframe(DTC_df.style.format("{:.4f}"), use_container_width=True)
    st.subheader("Imported Technical Coefficients Matrix")
    st.dataframe(ITC_df.style.format("{:.4f}"), use_container_width=True)

elif current_view == 'leontief':
    st.subheader("Leontief Inverse Matrix")
    st.dataframe(L_inverse_df.T.style.format("{:.4f}"), use_container_width=True)


    


    
