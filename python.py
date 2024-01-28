import tkinter as tk
import pandas as pd
from tkinter import ttk, filedialog, messagebox
import sqlite3
from sqlite3 import Error
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class MyGui:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Prototype")

        self.label = tk.Label(self.window, text="Welcome!")
        self.label.pack(pady=10)

        self.button = tk.Button(self.window, text = "Convert CSV to SQL", command = self.convert_to_sql)
        self.button.pack(pady=10)
        
        self.button1 = tk.Button(self.window, text="Load translated format", command = self.open_translated)
        self.button1.pack(pady=10)
        
        self.conn = None
        self.selected_db_name = None
        self.selected_table_name = None
        
    def calculate_statistics(self, condition):
        if self.selected_db_name and self.selected_table_name:
            try:
                # Read the data directly from the SQLite file
                condition_query = f"SELECT \"CIRAF ZONES\" FROM {self.selected_table_name} WHERE {condition}"
                df_filtered = pd.read_sql_query(condition_query, f"sqlite:///{self.selected_db_name}")
    
                if not df_filtered.empty:
                    # Replace "JUL-40" with corresponding values (28, 29, 30)
                    df_filtered['CIRAF ZONES'] = df_filtered['CIRAF ZONES'].replace('JUL-40', '28,29,30')
    
                    # Extract numerical values from "CIRAF ZONES" column
                    numerical_values = df_filtered['CIRAF ZONES'].apply(self.extract_numerical_values)
    
              
                    all_values = [item for sublist in numerical_values for item in sublist]
    
                    # Calculate statistics
                    if all_values:
                        mean_value = sum(all_values) / len(all_values)
                        mode_value = max(set(all_values), key=all_values.count)
                        median_value = sorted(all_values)[len(all_values)//2]
    
                        messagebox.showinfo("Statistics", f"Mean: {mean_value}\nMode: {mode_value}\nMedian: {median_value}")
                    else:
                        messagebox.showinfo("No Data", "No data matching the specified conditions found.")
    
                else:
                    messagebox.showinfo("No Data", "No data matching the specified conditions found.")
    
            except Error as e:
                messagebox.showerror("Error", f"Error calculating statistics: {str(e)}")
    
    def extract_numerical_values(self, value):
        # Function to extract numerical values from a string
        numerical_values = []
        for part in value.split(','):
            if '-' in part:
                # Handle number ranges
                start, end = map(self.safe_convert_to_int, part.split('-'))
                if start is not None and end is not None:
                    numerical_values.extend(range(start, end + 1))
            elif part.isdigit():
                # Handle single numbers
                numerical_values.append(int(part))
        return numerical_values
    
    def safe_convert_to_int(self, value):
        try:
            return int(value)
        except ValueError:
            return None



    
    def open_translated(self):
     
        db_dialog = tk.Toplevel(self.window)
        db_dialog.title("Select Database")

     
        self.selected_db_name = filedialog.askopenfilename(defaultextension=".db", filetypes=[("SQLite Database", "*.db")]) 

        if self.selected_db_name:
        
            table_names = self.get_table_names(self.selected_db_name)

            if not table_names:
                messagebox.showinfo("No Tables", "No tables found in the selected database.")
                db_dialog.destroy()
                return

            
            table_var = tk.StringVar(db_dialog)
            table_var.set(table_names[0])  

            table_menu = tk.OptionMenu(db_dialog, table_var, *table_names)
            table_menu.pack(pady=10)

            confirm_button = tk.Button(db_dialog, text="Confirm Selection", command=lambda: self.confirm_table_selection(table_var.get(), db_dialog))
            confirm_button.pack(pady=10)
        else:
            messagebox.showinfo("No Database Selected", "Please select a database.")

    def get_table_names(self, db_name):
        table_names = []
        try:
            ##############################################
            conn = sqlite3.connect(db_name)
            ##############################################
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            table_names = [row[0] for row in cursor.fetchall()]
        except Error as e:
            messagebox.showerror("Error", f"Error getting table names: {str(e)}")
        finally:
            if conn:
                conn.close()
        return table_names

    def confirm_table_selection(self, table_name, db_dialog):
        self.selected_table_name = table_name
        db_dialog.destroy()

        self.display_selected_table()

    def display_selected_table(self):
        if self.selected_db_name and self.selected_table_name:
            try:
                ##############################################
                conn = sqlite3.connect(self.selected_db_name)
                ##############################################
                query = f"SELECT * FROM {self.selected_table_name}"
                df = pd.read_sql_query(query, conn)
            
                table_window = tk.Toplevel(self.window)
                table_window.title(f"Table: {self.selected_table_name}")
    
                tree = ttk.Treeview(table_window)
                tree["columns"] = tuple(df.columns)
    
                for col in df.columns:
                    max_length = max(df[col].astype(str).apply(len).max(), len(col))
                    max_digits = max(df[col].apply(lambda x: len(str(abs(x))) if pd.notnull(x) and isinstance(x, (int, float)) else 0))
                    tree.column(col, anchor="center", width=max(max_length, max_digits) * 10)
                    tree.heading(col, text=col, anchor="center")
    
                for index, row in df.iterrows():
                    max_row_height = max(len(str(value)) for value in row)
                    tree.insert("", tk.END, values=tuple(row))
                    tree.rowconfigure(index, minsize=max_row_height * 20)
    
                scrollbar = ttk.Scrollbar(table_window, orient="vertical", command=tree.yview)
                tree.configure(yscrollcommand=scrollbar.set)
    
                tree.pack(fill="both", expand=True)
                scrollbar.pack(side="right", fill="y")
    
                if self.selected_table_name == "HF_SCHEDULE_reshaped":

                        label_frame_avg = tk.LabelFrame(table_window, text="Calculate Averages")
                        label_frame_avg.pack(pady=10)

                        self.stat_button1 = tk.Button(label_frame_avg, text="For POWR > 90", command=lambda: self.calculate_statistics("POWR > 90"))
                        self.stat_button2 = tk.Button(label_frame_avg, text="For START >= 1100", command=lambda: self.calculate_statistics("START >= 1100"))
                        self.stat_button1.pack(side="left", padx=10)
                        self.stat_button2.pack(side="left", padx=10)

                        label_frame_graphs = tk.LabelFrame(table_window, text="Generate Graphs")
                        label_frame_graphs.pack(pady=10)

                        self.graph_button = tk.Button(label_frame_graphs, text="Generate Graphs", command=self.generate_graphs)
                        self.correlation_button = tk.Button(label_frame_graphs, text="Correlation", command=self.calculate_freq_ciraf_correlation)
                        self.graph_button.pack(side="left", padx=10)
                        self.correlation_button.pack(side="left", padx=10)

        
            except Error as e:
                messagebox.showerror("Error", f"Error displaying selected table: {str(e)}")
            finally:
                if conn:
                    conn.close()
        
    def clean_check(self, file_path):
        # Read CSV file
        df = pd.read_csv(file_path, encoding='iso-8859-1')

        # Check for null values
        null_columns = df.columns[df.isnull().any()]

        if not null_columns.empty:
            # Ask the user what to do with null values
            action = messagebox.askyesno("Null Values", f"The following columns have null values: {', '.join(null_columns)}\n\n"
                                                           "Do you want to delete rows with null values?")
            if action:
                df = df.dropna()
                messagebox.showinfo("Null Values", "Rows with null values have been deleted.")
            else:
                messagebox.showinfo("Null Values", "Null values have been left as they are.")

        # Check for duplicate rows
        if df.duplicated().sum() > 0:
            # Ask the user what to do with duplicate rows
            action = messagebox.askyesno("Duplicate Rows", "There are duplicate rows in the data.\n\n"
                                                            "Do you want to delete duplicate rows?")
            if action:
                df = df.drop_duplicates()
                messagebox.showinfo("Duplicate Rows", "Duplicate rows have been deleted.")
            else:
                messagebox.showinfo("Duplicate Rows", "Duplicate rows have been left as they are.")

        # Further processing or analysis can be added here

        return df

    def choose_file(self, file_type):
        if file_type == "CSV":
            file_path = filedialog.askopenfilename(
                title="Select CSV File",
                filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
            )
        if file_type == "DB":
            file_path = filedialog.askopenfilename(
                title="Select DB File",
                filetypes=[("Data Base Files", "*.db"), ("All Files", "*.*")],
            )
        return file_path

    def convert_to_sql(self):
        csv_file_path = self.choose_file("CSV")
        self.clean_check(csv_file_path)
    
        if csv_file_path:
            try:
                db_dialog = tk.Toplevel()
                db_dialog.title("Database Choice")
                db_dialog.geometry("+%d+%d" % (self.window.winfo_x() + 50, self.window.winfo_y() + 50))
    
                def open_existing_db_callback():
                    self.open_existing_db(csv_file_path)
                    db_dialog.destroy()
    
                open_db_button = tk.Button(db_dialog, text="Open Existing Database", command=open_existing_db_callback)
                open_db_button.pack(pady=10)
    
                def create_new_db_callback():
                    self.create_new_db(csv_file_path)
                    db_dialog.destroy()
    
                new_db_button = tk.Button(db_dialog, text="Create New Database", command=create_new_db_callback)
                new_db_button.pack(pady=10)
    
                cancel_button = tk.Button(db_dialog, text="Cancel", command=db_dialog.destroy)
                cancel_button.pack(pady=10)
    
                db_dialog.grab_set()
                db_dialog.wait_window()
    
            except Exception as e:
                messagebox.showerror("Error", f"Error during conversion: {str(e)}")


    def open_existing_db(self, csv_file_path):
        db_name = filedialog.askopenfilename(defaultextension=".db", filetypes=[("SQLite Database", "*.db")])
        if db_name:
            self.process_database(db_name, csv_file_path)

    def create_new_db(self, csv_file_path):
        db_name = filedialog.asksaveasfilename(defaultextension=".db", filetypes=[("SQLite Database", "*.db")])
        if db_name:
            self.process_database(db_name, csv_file_path)

    def process_database(self, db_name, csv_file_path):
        try:
            ##############################################
            self.conn = sqlite3.connect(db_name)
            ##############################################
            cursor = self.conn.cursor()
    
            df = pd.read_csv(csv_file_path, encoding='iso-8859-1')
    
            default_table_name = os.path.splitext(os.path.basename(csv_file_path))[0]
            table_name = default_table_name
    
            # Check if the table already exists in the database
            existing_tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", self.conn)
            if table_name in existing_tables.values:
                # If the table name already exists, append a suffix to distinguish it
                suffix = 1
                while f"{table_name}_{suffix}" in existing_tables.values:
                    suffix += 1
                table_name = f"{table_name}_{suffix}"
    
            df.to_sql(name=table_name, con=self.conn, index=False, if_exists='replace')
            messagebox.showinfo("Success", f"CSV data has been successfully converted to SQL table '{table_name}' in database '{db_name}'.")
            if any(column[1] == 'BROADCASTERCODE' for column in cursor.execute(f"PRAGMA table_info({table_name})")):
                self.data_manipulation(db_name, table_name, cursor)
    
        except Error as e:
            messagebox.showerror("Error", f"Error during conversion: {str(e)}")


    def data_manipulation(self, db_name, table_name, cursor):
        self.REMOVE(db_name, table_name, cursor)
        self.RESHAPE(db_name, table_name, cursor)

    def REMOVE(self, db_name, table_name, cursor):
        cursor.execute("PRAGMA foreign_keys=on;")
        cursor.execute(f"DELETE FROM {table_name} WHERE BROADCASTERCODE IN ('ADM', 'DWL', 'KBS')")
        self.conn.commit()

    def execute_sql_query(self, query, cursor):
        try:
            cursor.execute(query)
            return cursor.fetchall()
        except Error as e:
            messagebox.showerror("Error", f"Error executing SQL query: {str(e)}")
            return None

    def RESHAPE(self, db_name, table_name, cursor):
        cursor.execute("PRAGMA foreign_keys=on;")
        cursor.execute(f"ATTACH DATABASE '{db_name}' AS original_db;")
    
        cursor.execute(f"""
            CREATE TABLE {table_name}_reshaped AS
            SELECT
                A.FREQ,
                B.BROADCASTER AS BROADCASTER,
                A1.ADMINNAME AS ADMIN,
                A2.LANGUAGE AS LANGUAGE,
                A3.ANT AS "Antenna Type",
                A4.SITE AS Transmitter,
                A.*  -- Include all remaining original columns
            FROM {table_name} A
            LEFT JOIN BROADCASTER B ON A.BROADCASTERCODE = B.code
            LEFT JOIN ADMIN A1 ON A.ADMINCODE = A1.code
            LEFT JOIN LANGUAGE A2 ON A.LANGUAGECODE = A2.code
            LEFT JOIN ANT A3 ON A.ANTCODE = A3.code
            LEFT JOIN LOCATION A4 ON A.LOCCODE = A4.code
            WHERE A.FREQ IN ('5890', '6040', '7220', '9490', '9510')
                AND B.BROADCASTER IS NOT NULL
                AND A1.ADMINNAME IS NOT NULL
                AND A2.LANGUAGE IS NOT NULL
                AND A3.ANT IS NOT NULL
                AND A4.SITE IS NOT NULL;
        """)
    
        self.conn.commit()
        cursor.execute("DETACH DATABASE original_db;")
    
        messagebox.showinfo("Success", f"Data with specified 'FREQ' values has been extracted to the new table '{table_name}_reshaped'.")
        

    def generate_graphs(self):
        if self.selected_db_name and self.selected_table_name:
            try:
                query = f"SELECT BROADCASTER, LANGUAGE, \"CIRAF ZONES\", DAYS, FREQ FROM {self.selected_table_name} " \
                        f"WHERE FREQ IN ('5890', '6040', '7220', '9490', '9510')"
                df_filtered = pd.read_sql_query(query, f"sqlite:///{self.selected_db_name}")
    
                if not df_filtered.empty:
                    df_filtered['CIRAF ZONES'] = df_filtered['CIRAF ZONES'].apply(self.process_ciraf_zones)
                    df_flat = df_filtered.explode('CIRAF ZONES')
    
                    # Separate graphs for each category
                    self.plot_bar_count(df_flat, 'BROADCASTER', 'Frequency Count by Broadcaster', 'BROADCASTER')
                    self.plot_bar_count(df_flat, 'LANGUAGE', 'Frequency Count by Language', 'LANGUAGE')
                    self.plot_bar_count(df_flat, 'CIRAF ZONES', 'Frequency Count by CIRAF ZONE', 'CIRAF ZONES')
                    self.plot_bar_count(df_flat, 'DAYS', 'Frequency Count by Day', 'DAYS')
    
                else:
                    messagebox.showinfo("No Data", "No data matching the specified conditions found.")
    
            except Error as e:
                messagebox.showerror("Error", f"Error generating graphs: {str(e)}")

    def plot_bar_count(self, df, column, title, x_label):
        plt.figure(figsize=(12, 6))
        sns.countplot(data=df, x=column, hue='FREQ', palette='viridis', dodge=True)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel("Frequency Count")
        plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
        plt.legend(title='FREQ', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()



    def process_ciraf_zones(self, value):
        individual_zones = []

        for part in value.split(','):
            if '-' in part:
                # Handle number ranges
                range_parts = part.split('-')
                start, end = map(self.safe_convert_to_int, range_parts)

                if start is not None and end is not None:
                    individual_zones.extend(range(start, end + 1))
            elif part.isdigit():
                # Handle single numbers
                individual_zones.append(int(part))
            elif part.startswith("JUL-40"):
                # Handle JUL-40 case
                individual_zones.extend([28, 29, 30])
            elif '-' in part:
                # Handle mixed case like '3,4,5-7,JUL-40'
                mixed_parts = part.split(',')
                for mixed_part in mixed_parts:
                    if '-' in mixed_part:
                        start, end = map(self.safe_convert_to_int, mixed_part.split('-'))
                        if start is not None and end is not None:
                            individual_zones.extend(range(start, end + 1))
                    elif mixed_part.isdigit():
                        individual_zones.append(int(mixed_part))
            elif part:  # Handle the case where 'CIRAF ZONES' is a list
                individual_zones.extend(map(self.safe_convert_to_int, part.split()))

        return individual_zones
    
    def calculate_freq_ciraf_correlation(self):
        if self.selected_db_name and self.selected_table_name:
            try:
                # Query the data
                query = f"SELECT \"CIRAF ZONES\", FREQ FROM {self.selected_table_name} " \
                        f"WHERE FREQ IN ('5890', '6040', '7220', '9490', '9510')"
                df_filtered = pd.read_sql_query(query, f"sqlite:///{self.selected_db_name}")
    
                if not df_filtered.empty:
                    # Process CIRAF ZONES column
                    df_filtered['CIRAF ZONES'] = df_filtered['CIRAF ZONES'].apply(self.process_ciraf_zones)
    
                    # Create a new DataFrame with separate rows for each value in 'CIRAF ZONES' list
                    df_expanded = df_filtered.explode('CIRAF ZONES')
    
                    # Create a count matrix
                    freq_ciraf_matrix = pd.crosstab(df_expanded['FREQ'], df_expanded['CIRAF ZONES'])
    
                    # Calculate the correlation coefficient
                    correlation_coefficient = np.corrcoef(freq_ciraf_matrix, rowvar=False)
    
                    # Plot the heatmap
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(freq_ciraf_matrix, cmap='coolwarm', annot=True, fmt=".0f", linewidths=.5)
                    
                    # Display correlation coefficient in the title
                    plt.title(f"Correlation Heatmap between FREQ and CIRAF ZONES\nCorrelation Coefficient: {correlation_coefficient[0, 1]:.2f}")
    
                    plt.show()
    
                else:
                    messagebox.showinfo("No Data", "No data matching the specified conditions found.")
            except Error as e:
                messagebox.showerror("Error", f"Error calculating correlation: {str(e)}")

    def run(self):
        self.window.mainloop()


my_gui = MyGui()
my_gui.run()
