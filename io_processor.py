#%%

# ---------------------------------------------------------
# IMPORTING NECESSARY PACKAGES
# ---------------------------------------------------------

import os
import pandas as pd
import numpy as np

#%%

# ----------------------------------------------------------------------------------
# QUICKLY DEFINING FUNCTION TO DEAL WITH EXTRA NUMBERS IN FRONT OF INDUSTRY NAMES
# ----------------------------------------------------------------------------------

def is_number(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

def get_rid_of_numbers(mystring):
    string_split = mystring.split(' ', 1)
    if is_number(string_split[0]):
        return string_split[1]
    else:
        return mystring

#%%

# ------------------------------------------------------------------------------------------------------------
# INITIAL DATA CLEANING, DATA SHOULD BE ORGANIZED IN A DATAFRAME AFTER THIS WITH COLUMNS AS STRING IDENTIFIERS
# ------------------------------------------------------------------------------------------------------------
def clean_input_output_data(df):

    logs = []
    logs.append("Attempting initial data cleaning...")
    new_df = df.drop(0)
    new_df = new_df.iloc[:, 1:]
    data_series = new_df.iloc[0]
    parsed_names = data_series.index.str.split(': ', expand=True)
    #print(new_df.shape)
    #print(df)
    #print(parsed_names) ## PARSED NAMES SHOULD BE A MULTI INDEX OF STRINGS

    # DATA PREPARATION, TRANSPOSES DATAFRAME AND PREPARES A VALUE COLUMN AND COLUMNS FOR EACH STRING VALUE

    new_df=new_df.T

    new_df = new_df.rename(columns = {new_df.columns[0]: "Value"})

    #print(new_df)

    # CREATES A DATAFRAME CALLED df_index WHICH HAS ALL THE VALUES AND IDENTIFYING COLUMNS FILLED OUT

    if len(parsed_names[0]) != 8:
        #print("WARNING: UNEXPECTED IDENTIFYING STRING LENGTH: DATA CLEANING CODE IS UNLIKELY TO WORK. PLEASE MANUALLY CHECK DATA STRUCTURE")
        logs.append("WARNING: UNEXPECTED IDENTIFYING STRING LENGTH: DATA CLEANING CODE IS UNLIKELY TO WORK. PLEASE MANUALLY CHECK DATA STRUCTURE")
    else:
        logs.append("SUCCESS: IDENTIFYING STRING LENGTH IS AS EXPECTED.")

    df_index = parsed_names.to_frame(index = False)
    df_index.index = new_df.index
    df_index['Value'] = new_df['Value']
    df_index["Value"] = df_index["Value"].astype(float)
    df_index.columns = ["Part 0", "Part 1", "Part 2", "Part 3", "Part 4", "Part 5", "Part 6", "Part 7", "Value"]
    #print(df_index.head(5))

    return df_index, parsed_names, logs


#%%

# ----------------------------------------------------------------------------------------------------------
# CREATES A LIST OF SECTOR NAMES - CREATES TWO LISTS FROM DIFFERENT METHODS AND CHECKS IF THEY ARE THE SAME
# ----------------------------------------------------------------------------------------------------------

def create_sector_list(parsed_names):
    logs = []
    sector_list = []
    sector_list_check = []

    logs.append("Creating list of sector names...")

    for parsed_name in parsed_names: ##### METHOD 1
        #if not pd.isna(parsed_name[-1]):
            #print(parsed_name[-1])
            #print(parsed_name)
        #if parsed_name[2] == "Final Use":
        
        #if parsed_name[2] == "Intermediate Input":
            #print(parsed_name)
        if parsed_name[1] == 'Input-Output':
            if parsed_name[2] == 'Final Use':
                if parsed_name[3] == 'Intermediate Input':
                    if parsed_name[4] == 'Total Output':
                        if not pd.isna(parsed_name[5]):
                            clean_name = get_rid_of_numbers(parsed_name[5])
                            if not(clean_name in sector_list):
                                #print(parsed_name)
                                sector_list.append(clean_name)

        if parsed_name[1] == 'Input-Output': ### METHOD 2
            if parsed_name[2] == 'Intermediate Use':
                if parsed_name[3] == 'Intermediate Input':
                    #print(parsed_name)
                    if (not pd.isna(parsed_name[4])) and (parsed_name[4] != "Intermediate Use"):
                        clean_name = get_rid_of_numbers(parsed_name[4])
                        if clean_name not in sector_list_check:
                            sector_list_check.append(clean_name)

    if set(sector_list) != set(sector_list_check):
        #print(f"WARNING: THE TWO METHODS FOR COMPUTING SECTOR LIST DO NOT MATCH. THEIR LENGTHS ARE {len(sector_list)} AND {len(sector_list_check)}. SELECTING THE SECOND SECTOR LIST BY DEFAULT.")
        logs.append(f"WARNING: THE TWO METHODS FOR COMPUTING SECTOR LIST DO NOT MATCH. THEIR LENGTHS ARE {len(sector_list)} AND {len(sector_list_check)}. SELECTING THE SECOND SECTOR LIST BY DEFAULT.")
    else:
        #print("SUCCESS! SECTOR LISTS MATCH")
        logs.append("SUCCESS!")

    #print(sector_list_check)
    return sector_list_check, logs


#%%

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# POPULATING THE INTERMEDIATE TRANSACTION DATAFRAMES USING HARDCODED STRING IDENTIFICATION (FIXED FOR TRANSPOSITION AND FLEXIBLE TO DOMESTIC/IMPORTED)
# ----------------------------------------------------------------------------------------------------------------------------------------------------

def populate_IT_matrix(IT_df, df_index, parsed_names, domestic = True):
    logs = []
    logs.append("Attempting to populate intermediate transaction matrix...")
    if domestic:
        source = "Domestic"
    else:
        source = "Imported"

    for parsed_name in parsed_names:
        if parsed_name[1] == "Input-Output":
            if parsed_name[2] == 'Intermediate Use':
                if parsed_name[3] == 'Intermediate Input':
                    if not(pd.isna(parsed_name[5])):
                        raw_row_name = parsed_name[5]
                        raw_col_name = parsed_name[4]
                        col_name = get_rid_of_numbers(raw_col_name)
                        row_name = get_rid_of_numbers(raw_row_name)
                        new_index = df_index.index[(df_index['Part 1'] == 'Input-Output') & (df_index['Part 2'] == 'Intermediate Use') & (df_index['Part 3'] == 'Intermediate Input')
                                        & (df_index['Part 4'] == raw_col_name) & (df_index['Part 5'] == raw_row_name) & (df_index['Part 6'] == source)]
                        current_value = df_index.loc[new_index[0]]["Value"]

                        if col_name != "Intermediate Use":
                            IT_df.at[row_name, col_name] = current_value

    if IT_df.isna().any().any():
        #print("NaN VALUES DETECTED: ATTEMPTING TO FILL THEM WITH OTHER STRING PATHS")
        logs.append("NaN VALUES DETECTED: ATTEMPTING TO FILL THEM WITH OTHER STRING PATHS")
        # This returns a list of tuples: [(row_index, column_name), ...]
        nan_locations = IT_df.isna().stack()[IT_df.isna().stack()].index.tolist()
        i=1
        # If you want to print them out to see what they are:
        for row, col in nan_locations:
            #print(f"#{i}. Missing value at Row: '{row}' | Column: '{col}'")
            logs.append(f"#{i}. Missing value at Row: '{row}' | Column: '{col}'")
            i+=1

        for parsed_name in parsed_names:
            if parsed_name[1] == "IOIU":
                if parsed_name[2] == 'Intermediate Input':
                    # print(parsed_name)
                    col_name = parsed_name[3]
                    raw_row_name = parsed_name[4]
                    row_name = get_rid_of_numbers(raw_row_name)
                    
                    new_index = df_index.index[(df_index['Part 1'] == 'IOIU') & (df_index['Part 2'] == 'Intermediate Input') & (df_index['Part 3'] == col_name) & (df_index["Part 4"] == raw_row_name) & (df_index['Part 5'] == source)]

                    

                    # print(new_index)
                    # print(row_name)
                    # print(col_name)

                    current_value = df_index.loc[new_index[0]]["Value"]

                    # print(current_value)
                    # print(IDT_df.at[row_name, col_name])

                    IT_df.at[row_name, col_name] = current_value

        if IT_df.isna().any().any():
            #print("NaN VALUES STILL DETECTED: PLEASE MANUALLY CHECK DATA")
            logs.append("NaN VALUES STILL DETECTED: PLEASE MANUALLY CHECK DATA")
        else:
            #print("SUCCESS: NaN VALUES SUCCESSFULLY HANDLED")
            logs.append("SUCCESS: NaN VALUES SUCCESSFULLY HANDLED")

    #print(IT_df)
    return IT_df, logs

#%%

# ----------------------------------------------------------------------------------------------------------
# PRODUCING VALUE ADDED LIST: HISTORICALLY THERE ARE 4 CATEGORIES + TOTAL VALUE ADDED
# ----------------------------------------------------------------------------------------------------------

def create_value_added_list(parsed_names):

    logs = []
    value_added_list = []

    logs.append("Attempting to create list of value added categories...")

    for parsed_name in parsed_names:
        if parsed_name[1] == "Input-Output":
            if parsed_name[2] == 'Intermediate Use':
                if parsed_name[3] == 'Value Added':
                    va_name = parsed_name[5]
                    if not(pd.isna(va_name)):
                        va_name = get_rid_of_numbers(va_name)
                        if not(va_name in value_added_list):
                            value_added_list.append(va_name)

    value_added_list.append("Total Value Added")

    if len(value_added_list) != 5:
        #print(f"NOTICE: THE LENGTH OF THE VALUE ADDED CATEGORIES IS DIFFERENT FROM WHAT IS HISTORICALLY EXPECTED ({len(value_added_list)} INSTEAD OF 5).")
        logs.append(f"NOTICE: THE LENGTH OF THE VALUE ADDED CATEGORIES IS DIFFERENT FROM WHAT IS HISTORICALLY EXPECTED ({len(value_added_list)} INSTEAD OF 5).")
    else:
        #print("SUCCESS! THE LENGTH OF THE VALUE ADDED CATEGORIES IS IN LINE WITH HISTORICAL EXPECTATION.")
        logs.append("SUCCESS! THE LENGTH OF THE VALUE ADDED CATEGORIES IS IN LINE WITH HISTORICAL EXPECTATION.")
   # print(value_added_list)
    return value_added_list, logs

#%%

# -------------------------------------------------------------------------------------------------------------------------
# POPULATING THE PRIMARY INPUT DATAFRAME USING HARDCODED STRING IDENTIFICATION
# -------------------------------------------------------------------------------------------------------------------------

def populate_primary_input_matrix(PI_df, df_index, parsed_names):

    logs = []
    logs.append("Attempting to populate primary input matrix...")
    for parsed_name in parsed_names:
        if parsed_name[1] == "Input-Output":
            if parsed_name[2] == 'Intermediate Use':
                if parsed_name[3] == 'Value Added':
                    raw_col_name = parsed_name[4]
                    col_name = get_rid_of_numbers(raw_col_name)
                    raw_sector_name = parsed_name[5]
                    # print(raw_sector_name)
                    if pd.isna(parsed_name[5]):
                        sector_name = "Total Value Added"
                        new_index = df_index.index[(df_index['Part 1'] == 'Input-Output') & (df_index['Part 2'] == 'Intermediate Use') & (df_index['Part 3'] == 'Value Added')
                                    & (df_index['Part 4'] == raw_col_name) & pd.isna(df_index['Part 5'])]
                    else:
                        sector_name = parsed_name[5]
                        new_index = df_index.index[(df_index['Part 1'] == 'Input-Output') & (df_index['Part 2'] == 'Intermediate Use') & (df_index['Part 3'] == 'Value Added')
                                    & (df_index['Part 4'] == raw_col_name) & (df_index['Part 5'] == raw_sector_name)]
                        
                    # print(col_name)
                    # print(sector_name)

                    
                    # print(new_index)
                    current_value = df_index.loc[new_index[0]]["Value"]
                    
                    if col_name != "Intermediate Use":
                        PI_df.at[sector_name, col_name] = current_value
                    # print(current_value)


    if PI_df.isna().any().any():
        #print("NaN VALUES DETECTED: PLEASE MANUALLY CHECK DATA")
        logs.append("NaN VALUES DETECTED: PLEASE MANUALLY CHECK DATA")
        # This returns a list of tuples: [(row_index, column_name), ...]
        nan_locations = PI_df.isna().stack()[PI_df.isna().stack()].index.tolist()
        i=1
        # If you want to print them out to see what they are:
        for row, col in nan_locations:
            #print(f"#{i}. Missing value at Row: '{row}' | Column: '{col}'")
            logs.append(f"#{i}. Missing value at Row: '{row}' | Column: '{col}'")
            i+=1         
    else:
        #print("SUCCESS: NO UNEXPECTED NaN VALUES.")
        logs.append("SUCCESS: NO UNEXPECTED NaN VALUES.")

    


    #print(PI_df)
    return PI_df, logs

#%%

# ----------------------------------------------------------------------------------------------------------
# PRODUCING FINAL USE COLUMN NAMES: HISTORICALLY THERE ARE 4 SUB COLUMNS AND FINAL USE
# ----------------------------------------------------------------------------------------------------------

def create_final_use_columns(parsed_names):
    logs = []
    final_use_headcolumns = []
    final_use_columns = []
    logs.append("Attempting to create list of final use categories...")
    for parsed_name in parsed_names:
        if parsed_name[1] == "Input-Output":
            if parsed_name[2] == 'Final Use':
                if parsed_name[3] == 'Intermediate Input':
                    if not(parsed_name[4] in final_use_headcolumns):
                        final_use_headcolumns.append(parsed_name[4])

                    if not(pd.isna(parsed_name[7])):
                        if not(parsed_name[5] in final_use_columns):
                            final_use_columns.append(parsed_name[5])

    check_clean = True

    final_use_columns.append("Final Use")

    if len(final_use_headcolumns) != 4:
        #print("WARNING: DIFFERENT NUMBER OF HEAD COLUMNS THAN EXPECTED. STANDARD IS (GROSS CAPITAL FORMATION, FINAL USE, IMPORT, TOTAL OUTPUT).")
        logs.append("WARNING: DIFFERENT NUMBER OF HEAD COLUMNS THAN EXPECTED. STANDARD IS (GROSS CAPITAL FORMATION, FINAL USE, IMPORT, TOTAL OUTPUT).")
        check_clean = False

    if len(final_use_columns) != 5:
        #print("WARNING: DIFFERENT NUMBER OF SUBCOLUMNS THAN EXPECTED. STANDARD IS (FIXED CAPITAL, INVENTORY, FINAL CONSUMPTION, EXPORT, AND FINAL USE).")
        logs.append("WARNING: DIFFERENT NUMBER OF SUBCOLUMNS THAN EXPECTED. STANDARD IS (FIXED CAPITAL, INVENTORY, FINAL CONSUMPTION, EXPORT, AND FINAL USE).")
        check_clean = False
    # print(final_use_headcolumns)
    # print(final_use_columns)
    if check_clean:
        logs.append("SUCCESS! COLUMN NAMES ARE AS EXPECTED.")

    return final_use_columns, logs

#%%

# --------------------------------------------------------------------------------------------------------------------------------
# POPULATING THE FINAL DEMAND MATRICES USING HARDCODED STRING IDENTIFICATION (FLEXIBLE BETWEEN DOMESTIC AND IMPORTED FINAL DEMAND)
# --------------------------------------------------------------------------------------------------------------------------------

def populate_final_use_matrix(FD_df, df_index, parsed_names, domestic = True):


    logs = []
    if domestic:
        source = "Domestic"
    else:
        source = "Imported"
    logs.append("Attempting to populate final use matrix...")

    for parsed_name in parsed_names:
        if parsed_name[1] == "Input-Output":
            if parsed_name[2] == 'Final Use':
                if parsed_name[3] == 'Intermediate Input':
                    if parsed_name[4] == "Gross Capital Formation": ## THIS SECTION POPULATES THE FIXED CAPITAL AND INVENTORY, PROVIDED THAT THEY'RE UNDER GROSS CAPITAL FORMATION
                        if not(pd.isna(parsed_name[6])):
                            # print(parsed_name)
                            raw_col_name = parsed_name[5]
                            raw_row_name = parsed_name[6]

                            row_name = get_rid_of_numbers(raw_row_name)

                            # print(raw_col_name)
                            # print(raw_row_name)
                            # print(row_name)

                            new_index = df_index.index[(df_index['Part 1'] == 'Input-Output') & (df_index['Part 2'] == 'Final Use') & (df_index['Part 3'] == 'Intermediate Input')
                                            & (df_index['Part 4'] == parsed_name[4]) & (df_index['Part 5'] == raw_col_name) & (df_index["Part 6"] == raw_row_name) & (df_index["Part 7"] == source)]



                            
                            # print(new_index)
                            current_value = df_index.loc[new_index[0]]["Value"]
                            
                            # print(current_value)
                            FD_df.at[row_name, raw_col_name] = current_value

                    elif parsed_name[4] == "Final Use": ## THIS SECTION POPULATES THE FINAL USE CATEGORIES, INCLUDING TOTAL FINAL USE
                        if not(pd.isna(parsed_name[6])):
                            # print(parsed_name)
                            if pd.isna(parsed_name[7]):
                                col_name = 'Final Use'
                                raw_row_name = parsed_name[5]
                                domimp = parsed_name[6]
                                new_index = df_index.index[(df_index['Part 1'] == 'Input-Output') & (df_index['Part 2'] == 'Final Use') & (df_index['Part 3'] == 'Intermediate Input')
                                            & (df_index['Part 4'] == parsed_name[4]) & (df_index['Part 5'] == parsed_name[5]) & (df_index["Part 6"] == source)]
                            else:
                                col_name = parsed_name[5]
                                raw_row_name = parsed_name[6]
                                domimp = parsed_name[7]
                                new_index = df_index.index[(df_index['Part 1'] == 'Input-Output') & (df_index['Part 2'] == 'Final Use') & (df_index['Part 3'] == 'Intermediate Input')
                                            & (df_index['Part 4'] == parsed_name[4]) & (df_index['Part 5'] == parsed_name[5]) & (df_index["Part 6"] == parsed_name[6]) & (df_index["Part 7"] == source)]

                            row_name = get_rid_of_numbers(raw_row_name)
                            

                            # print(new_index)

                            # print(col_name)
                            # print(row_name)
                            # print(domimp)

                            current_value = df_index.loc[new_index[0]]["Value"]

                            FD_df.at[row_name, col_name] = current_value

    if FD_df.isna().any().any():
        print("WARNING: NaN VALUES DETECTED. PLEASE MANUALLY REVIEW DATA STRUCTURE.")
        logs.append("WARNING: NaN VALUES DETECTED. PLEASE MANUALLY REVIEW DATA STRUCTURE.")
        # This returns a list of tuples: [(row_index, column_name), ...]
        nan_locations = FD_df.isna().stack()[FD_df.isna().stack()].index.tolist()
        i=1
        # If you want to print them out to see what they are:
        for row, col in nan_locations:
            #print(i)
            #print(f"Missing value at Row: '{row}' | Column: '{col}'")
            logs.append(f"#{i}. Missing value at Row: '{row}' | Column: '{col}'")
            i+=1
    else:
        #print("SUCCESS: NO NaN VALUES FOUND.")
        logs.append("SUCCESS: NO NaN VALUES FOUND.")


    # print(FD_df)
    # print(FD_df.shape)
    #display(FD_df)
    return FD_df, logs

#%%

# -------------------------------------------------------------------------------------------------------------------------
# RETRIEVING BOTH DOMESTIC TOTAL OUTPUT ('TOTAL OUTPUT') AND IMPORTED TOTAL OUTPUT ('IMPORT')
# -------------------------------------------------------------------------------------------------------------------------

def populate_total_output_vector(total_output_domestic, total_output_imported, df_index, parsed_names):

    logs = []
    logs.append("Attempting to populate total output vector...")
    for parsed_name in parsed_names:
        if parsed_name[1] == "Input-Output":
            if parsed_name[2] == 'Final Use':
                if parsed_name[3] == 'Intermediate Input':
                    if parsed_name[4] == 'Total Output':
                        if not(pd.isna(parsed_name[5])):

                            raw_row_name = parsed_name[5]
                            row_name = get_rid_of_numbers(raw_row_name)

                            new_index = df_index.index[(df_index['Part 1'] == 'Input-Output') & (df_index['Part 2'] == 'Final Use') & (df_index['Part 3'] == 'Intermediate Input')
                                            & (df_index['Part 4'] == "Total Output") & (df_index['Part 5'] == parsed_name[5]) & (df_index["Part 6"] == parsed_name[6])]
                            # print(new_index)
                            current_value = df_index.loc[new_index[0]]["Value"]

                            if parsed_name[6] == "Domestic":
                                total_output_domestic.at[row_name, "Total Domestic Output"] = current_value


                            # print(parsed_name)
                            
                            # print(row_name)
                            # print(current_value)

                    if parsed_name[4] == 'Import':
                        if not(pd.isna(parsed_name[5])):

                            raw_row_name = parsed_name[5]
                            row_name = get_rid_of_numbers(raw_row_name)

                            new_index = df_index.index[(df_index['Part 1'] == 'Input-Output') & (df_index['Part 2'] == 'Final Use') & (df_index['Part 3'] == 'Intermediate Input')
                                            & (df_index['Part 4'] == "Import") & (df_index['Part 5'] == parsed_name[5]) & (df_index["Part 6"] == parsed_name[6])]
                            # print(new_index)
                            current_value = df_index.loc[new_index[0]]["Value"]

                            if parsed_name[6] == "Imported":
                                total_output_imported.at[row_name, "Total Imported Output"] = current_value


                            # print(parsed_name)
                            
                            # print(row_name)
                            # print(current_value)

    import_dirty = total_output_imported.isna().any().any()
    domestic_dirty = total_output_domestic.isna().any().any()

    if import_dirty:
        #print("WARNING: IMPORTED TOTAL OUTPUT HAS NaN VALUES. PLEASE MANUALLY CHECK DATA.")
        logs.append("WARNING: IMPORTED TOTAL OUTPUT HAS NaN VALUES. PLEASE MANUALLY CHECK DATA.")

    if domestic_dirty:
        #print("WARNING: DOMESTIC TOTAL OUTPUT HAS NaN VALUES. PLEASE MANUALLY CHECK DATA. ")
        logs.append("WARNING: IMPORTED TOTAL OUTPUT HAS NaN VALUES. PLEASE MANUALLY CHECK DATA.")
        domestic_clean = False

    if not(import_dirty) and not(domestic_dirty): 
        #print("SUCCESS: NO NaN VALUES DETECTED.")
        logs.append("SUCCESS: NO NaN VALUES DETECTED.")
                            
    return total_output_domestic, total_output_imported, logs

#%%

# -------------------------------------------------------------------------------------------------------------------------
# POPULATING TOTAL INPUT VECTOR - THIS ONE ALSO HAD ISSUES WITH NAN, BUT THE DATA IS FOUND ELSEWHERE
# -------------------------------------------------------------------------------------------------------------------------

def populate_total_input_vector(total_input, df_index, parsed_names):

    logs = []
    logs.append("Attempting to populate total input vector...")
    for parsed_name in parsed_names:
        if parsed_name[1] == "Input-Output":
            if parsed_name[2] == "Intermediate Use":
                if parsed_name[3] == 'Total Input':
                    if parsed_name[4] != "Intermediate Use":
                        raw_col_name = parsed_name[4]
                        sector_name = get_rid_of_numbers(raw_col_name)
                        new_index = df_index.index[(df_index['Part 1'] == 'Input-Output') & (df_index['Part 2'] == 'Intermediate Use') & (df_index['Part 3'] == 'Total Input')
                                        & (df_index['Part 4'] == raw_col_name)]
                        current_value = df_index.loc[new_index[0]]["Value"]
                        total_input.at[sector_name, "Total Input"] = current_value

    if total_input.isna().any().any():
        #print("NOTICE: NaN VALUES DETECTED: ATTEMPTING TO FILL THEM...")
        nan_locations = total_input.isna().stack()[total_input.isna().stack()].index.tolist()
        i=1
        # If you want to print them out to see what they are:
        for row, col in nan_locations:
            #print(i)
            #print(f"Missing value at Row: '{row}' | Column: '{col}'")
            i+=1

        for parsed_name in parsed_names:
            if parsed_name[1] == "Input-Output":
                if parsed_name[2] == 'Intermediate Use(IOIU)':
                    raw_col_name = parsed_name[4]
                    sector_name = get_rid_of_numbers(raw_col_name)
                    new_index = df_index.index[(df_index['Part 1'] == 'Input-Output') & (df_index['Part 2'] == 'Intermediate Use(IOIU)') & (df_index['Part 3'] == 'Total Input')
                                    & (df_index['Part 4'] == raw_col_name)]
                    current_value = df_index.loc[new_index[0]]["Value"]
                    total_input.at[sector_name, "Total Input"] = current_value

    if total_input.isna().any().any():
        #print("ERROR: NaN VALUES STILL DETECTED: PLEASE MANUALLY CHECK THE DATA STRUCTURE.")
        logs.append("ERROR: NaN VALUES STILL DETECTED: PLEASE MANUALLY CHECK THE DATA STRUCTURE.")
    else:
        #print("SUCCESS: NO UNEXPECTED NaN VALUES FOUND.")
        logs.append("SUCCESS: NO UNEXPECTED NaN VALUES FOUND.")

    return total_input, logs

#%%

# -------------------------------------------------------------------------------------------------------------------------
# CHECK THAT VALUE ADDED CATEGORIES SUM UP TO THE CORRECT NUMBER
# -------------------------------------------------------------------------------------------------------------------------

def check_value_added(PI_df):

    logs = []
    logs.append("Checking internal consistency of the value added matrix...")
    new_df = PI_df.T.drop(columns = ["Total Value Added"]).copy()

    new_df["TVA Check"] = new_df.sum(axis = 1)
    #print("===========================================")

    new_df['Equality Check'] = (abs(PI_df.T["Total Value Added"] - new_df["TVA Check"]) < 1e-4)
    new_df["TVA"] = PI_df.T["Total Value Added"]

    check_satisfied = True

    mask = new_df["Equality Check"] == False        # use == False to handle NaN safely
    for idx in new_df.index[mask]:
        #print(f"WARNING: The {idx} sector has value added categories which do not add up to Total Value Added - check further.")
        logs.append(f"WARNING: The {idx} sector has value added categories which do not add up to Total Value Added - check further.")
        check_satisfied = False

    if check_satisfied:
        #print("SUCCESS: VALUE ADDED CATEGORIES ADD UP TO TOTAL VALUE ADDED.")
        logs.append("SUCCESS: VALUE ADDED CATEGORIES ADD UP TO TOTAL VALUE ADDED.")

    return logs
    #display(new_df)

#%%

# -------------------------------------------------------------------------------------------------------------------------
# CHECK THAT FINAL DEMAND CATEGORIES SUM UP TO THE CORRECT NUMBER
# -------------------------------------------------------------------------------------------------------------------------

def check_final_use(FD_df):

    logs = []
    logs.append("Checking internal consistency of the final use matrix...")
    new_df = FD_df.drop(columns = ["Final Use"]).copy()

    new_df["Final Use Check"] = new_df.sum(axis = 1)
    #print("===========================================")

    new_df['Equality Check'] = (abs(FD_df["Final Use"] - new_df["Final Use Check"]) < 1e-4)
    new_df["Final Use"] = FD_df["Final Use"]

    check_satisfied = True

    mask = new_df["Equality Check"] == False        # use == False to handle NaN safely
    for idx in new_df.index[mask]:
        #print(f"WARNING: The {idx} sector has final demand categories which do not add up to Final Use - check further.")
        logs.append(f"WARNING: The {idx} sector has final demand categories which do not add up to Final Use - check further.")
        check_satisfied = False

    if check_satisfied:
        #print("SUCCESS: Final demand categories add up to Final Use.")
        logs.append("SUCCESS: Final demand categories add up to Final Use.")

    return logs


#%%

# -------------------------------------------------------------------------------------------------------------------------
# CHECK THAT INTERMEDIATE OUTPUTS + FINAL USE SUMS TO TOTAL DOMESTIC OUTPUT
# -------------------------------------------------------------------------------------------------------------------------

def check_domestic_total_output(IDT_df, FDD_df, total_output_domestic):

    logs = []
    logs.append("Checking internal consistency of domestic total output...")

    new_df = total_output_domestic.copy()
    new_df["Total Intermediate Production"] = IDT_df.sum(axis = 1)
    new_df["Final Use"] = FDD_df["Final Use"]
    new_df["Total Output Check"] = new_df["Total Intermediate Production"] + new_df["Final Use"]
    new_df["Equality Check"] = (abs(new_df["Total Output Check"] - new_df["Total Domestic Output"]) < 1e-4)

    check_satisfied = True

    mask = new_df["Equality Check"] == False        # use == False to handle NaN safely
    for idx in new_df.index[mask]:
        #print(f"WARNING: The {idx} sector does not satisfy the identity: total intermediate production + final use = total output. Check further.")
        logs.append(f"WARNING: The {idx} sector does not satisfy the identity: total intermediate production + final use = total output. Check further.")
        check_satisfied = False

    if check_satisfied:
        #print("SUCCESS: total intermediate production + final use = total output.")
        logs.append("SUCCESS: total intermediate production + final use = total output.")

    #display(new_df)
    return logs

#%%

# -------------------------------------------------------------------------------------------------------------------------
# CHECK THAT IMPORTED INTERMEDIATE OUTPUTS + FINAL USE SUMS TO TOTAL IMPORTS
# -------------------------------------------------------------------------------------------------------------------------

def check_imported_total_output(IIT_df, FDI_df, total_output_imported):

    logs = []
    logs.append("Checking internal consistency of domestic total output...")
    new_df = total_output_imported.copy()
    new_df["Total Intermediate Production"] = IIT_df.sum(axis = 1)
    new_df["Final Use"] = FDI_df["Final Use"]
    new_df["Total Output Check"] = new_df["Total Intermediate Production"] + new_df["Final Use"]
    new_df["Equality Check"] = (abs(new_df["Total Output Check"] - new_df["Total Imported Output"]) < 1e-4)

    check_satisfied = True

    mask = new_df["Equality Check"] == False        # use == False to handle NaN safely
    for idx in new_df.index[mask]:
        #print(f"WARNING: The {idx} sector does not satisfy the identity: total intermediate production + final use = total output. Check further.")
        logs.append(f"WARNING: The {idx} sector does not satisfy the identity: total intermediate production + final use = total output. Check further.")
        check_satisfied = False

    if check_satisfied:
        #print("SUCCESS: total intermediate production + final use = total output.")
        logs.append("SUCCESS: total intermediate production + final use = total output.")
    
    #display(new_df)
    return logs

#%%

# -------------------------------------------------------------------------------------------------------------------------
# CHECK THAT DOMESTIC TOTAL INPUTS + IMPORTED TOTAL INPUTS + TOTAL VALUE ADDED = TOTAL INPUTS
# -------------------------------------------------------------------------------------------------------------------------

def check_total_input(IDT_df, IIT_df, PI_df, total_input):

    logs = []
    logs.append("Checking internal consistency of total input...")

    new_df = total_input.T.copy()
    new_df.loc["Domestic Total"] = IDT_df.sum(axis = 0)
    new_df.loc["Imported Total"] = IIT_df.sum(axis = 0)
    new_df.loc["TVA"] = PI_df.loc["Total Value Added"]
    new_df.loc["Total Input Check"] = new_df.loc["Domestic Total"] + new_df.loc["Imported Total"] + new_df.loc["TVA"]

    new_df.loc["Equality Check"] = (abs(new_df.loc["Total Input Check"] - new_df.loc["Total Input"]) < 1e-4)

    check_satisfied = True
    
    mask = new_df.T["Equality Check"] == False        # use == False to handle NaN safely
    for idx in new_df.T.index[mask]:
        #print(f"WARNING: The {idx} sector does not satisfy the identity: total domestic inputs + total imported inputs + total value added = total inputs. Check further.")
        logs.append(f"WARNING: The {idx} sector does not satisfy the identity: total domestic inputs + total imported inputs + total value added = total inputs. Check further.")
        check_satisfied = False

    if check_satisfied:
        #print("SUCCESS: total domestic inputs + total imported inputs + total value added = total inputs.")
        logs.append("SUCCESS: total domestic inputs + total imported inputs + total value added = total inputs.")
    #display(new_df)

    return logs

#%%

# -------------------------------------------------------------------------------------------------------------------------
# CHECK THAT TOTAL INPUT = TOTAL OUTPUT
# -------------------------------------------------------------------------------------------------------------------------

def check_input_output_equality(total_input, total_output_domestic):

    logs = []
    logs.append("Checking internal consistency of total input and total output...")

    new_df = total_input.copy()
    new_df["Total Domestic Output"] = total_output_domestic["Total Domestic Output"]
    new_df["Equality Check"] = (abs(new_df["Total Domestic Output"] - new_df["Total Input"]) < 1e-4)

    mask = new_df["Equality Check"] == False        # use == False to handle NaN safely

    check_satisfied = True

    for idx in new_df.index[mask]:
        #print(f"WARNING: The {idx} sector does not satisfy the identity: total domestic output = total input. Check further.")
        logs.append(f"WARNING: The {idx} sector does not satisfy the identity: total domestic output = total input. Check further.")
        check_satisfied = False

    if check_satisfied:
        #print("SUCCESS: total domestic output = total input.")
        logs.append("SUCCESS: total domestic output = total input.")
    #display(new_df)

    return logs
