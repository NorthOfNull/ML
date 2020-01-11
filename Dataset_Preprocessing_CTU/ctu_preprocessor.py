# Dataset Preprocessing class for CTU-13 netflows
#      get_dataset()
#      fix_labels()
#      get_unique_labels()
#      print_flow_label_totals()
#      write_fixed_dataset()
#      show_dataset_head()
#


# Import Pandas library for DataFrame and related functions
import pandas as pd

# Dataset preprocessor class
class ctu_preprocessor:
    def __init__(self, input_path):
        self.path = input_path
        self.dataset = 0
        
        print("Created preprocessor object")
    
    def __del__(self):
        print("Destroying preprocessor object")
        print("")
        print("")

        
        
    # Function to load dataset from a given path string
    def get_dataset(self):
        print(self.path)

        try:
            # Read in .csv file data from the defined path
            raw_dataset = pd.read_csv(self.path)
            
            # Put data into Pandas DataFrame
            self.dataset = pd.DataFrame(raw_dataset)

        except:
            print("File Read Error")

            

    # Function to modify dataset DataFrame Label descriptions
    def fix_labels(self, extended):
        print("Started CTU Dataset Netflow Preprocessor!")
        print("     [1/2] Fixing hex values...")
        print("     [2/3] Filling empty fields...")
        print("     [3/3] Fixing labels...")
            
        # Iterate over each row
        for row in range(self.dataset.shape[0]):
            # [1/2]
            # Fix hex values in src and dest port fields
            src_port = str(self.dataset.at[row, 'Sport'])
            dest_port = str(self.dataset.at[row, 'Dport'])
            
            # Convert any hex in fields to int      
            # Empty field 'nan' checks - if empty, converts to "0"
            if "nan" in src_port:
                src_port = "0"
                self.dataset.at[row, 'Sport'] = src_port
                
            if "nan" in dest_port:
                dest_port = "0"
                self.dataset.at[row, 'Dport'] = dest_port
            
            # Convert all strings, even if hex, to ints
            self.dataset.at[row, 'Sport'] = int(str(src_port), 0)
            self.dataset.at[row, 'Dport'] = int(str(dest_port), 0)
            
            
            # [2/3]
            # Fill empty hex fields, to prevent NaN conflicts
            sTos = str(self.dataset.at[row, 'sTos']) 
            dTos = str(self.dataset.at[row, 'dTos'])
            
            # Empty field 'nan' checks - if empty, converts to "0"
            if "nan" in sTos:
                self.dataset.at[row, 'sTos'] = "0"
                
            if "nan" in dTos:
                self.dataset.at[row, 'dTos'] = "0"
            
            
            # [3/3]
            # Fix labels
            label = self.dataset.at[row, 'Label']
            
            if "Background" in label:
                self.dataset.at[row, 'Label'] = "Normal"
            elif "Normal" in label:
                self.dataset.at[row, 'Label'] = "Normal"
            elif "Botnet" in label:
                self.dataset.at[row, 'Label'] = "Botnet"
                
            # For extended datasets only
            # Fix other empty fields
            if extended:
                SrcWin = str(self.dataset.at[row, 'SrcWin'])
                DstWin = str(self.dataset.at[row, 'DstWin'])
                sHops = str(self.dataset.at[row, 'sHops'])
                dHops = str(self.dataset.at[row, 'dHops'])
                sTtl = str(self.dataset.at[row, 'sTtl'])
                dTtl = str(self.dataset.at[row, 'dTtl'])

                if "nan" in SrcWin:
                        self.dataset.at[row, 'SrcWin'] = "0"

                if "nan" in DstWin:
                    self.dataset.at[row, 'DstWin'] = "0"

                if "nan" in sHops:
                    self.dataset.at[row, 'sHops'] = "0"

                if "nan" in dHops:
                    self.dataset.at[row, 'dHops'] = "0"

                if "nan" in sTtl:
                    self.dataset.at[row, 'sTtl'] = "0"

                if "nan" in dTtl:
                    self.dataset.at[row, 'dTtl'] = "0"
        
        print("Finished preprocessing!!!")
        
    
    
    # Function to get unique labels into an associative array
    def get_unique_labels(self):
        dataset = self.dataset
        
        unique_labels = {}
        rows = dataset.shape[0]

        for row in range(rows):
            label = dataset.at[row, 'Label']

            if label not in unique_labels:
                unique_labels.update({label : ""})

        return unique_labels
    
    

    # Prints total number of botnet records
    def print_flow_label_totals(self):
        background_flows = 0
        normal_flows = 0
        botnet_flows = 0

        for row in range(self.dataset.shape[0]):    
            if dataset.at[row, 'Label'] == "Background":
                background_flows += 1
            if dataset.at[row, 'Label'] == "Normal":
                normal_flows += 1
            if dataset.at[row, 'Label'] == "Botnet":
                botnet_flows += 1

        print('Background flows = ', background_flows)
        print('Normal flows = ', normal_flows)
        print('Botnet flows = ', botnet_flows)
        
        

    # Function to write modified dataset to a new file
    def write_fixed_dataset(self, index, extended):
        if extended:
            fixed_dataset_dir = "../../Datasets/CTU-13/Pre-processed_Extended/"
        else:   
            fixed_dataset_dir = "../../Datasets/CTU-13/Pre-processed/"
            
        fixed_dataset_dir += str(index)
        fixed_dataset_dir += ".csv"

        print("Writing fixed dataset Dataframe to ", fixed_dataset_dir)
        
        try:
            self.dataset.to_csv(fixed_dataset_dir)
            print("File Write Successful!")
        except:
            print("to_csv - File Write Error")
            
    
    
    # Function to show the head of the object's loaded dataset
    def show_dataset_head(self):
        return self.dataset.head()
