#!/usr/bin/env python
# coding: utf-8

# Dataset Preprocessing class for ISOT netflows
#      get_dataset()
#      process_isot()
#      write_fixed_dataset()
#      show_dataset_head()


# Import Pandas library for DataFrame and related functions
import pandas as pd

# Dataset preprocessor class
class isot_preprocessor:   
    def __init__(self, input_path):
        self.path = input_path
        self.dataset = 0
        
        self.botnet_mac_addresses = ["aa:aa:aa:aa:aa:aa",
                                     "bb:bb:bb:bb:bb:bb",
                                     "cc:cc:cc:cc:cc:cc",
                                     "cc:cc:cc:dd:dd:dd"]
        
        print("Created isot_preprocessor object")
    
    def __del__(self):
        print("Destroying isot_preprocessor object")
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

            

    # Function to process isot netflow dataset
    # Modifies any hex values found in the source and destination port fields to decimal
    # Create dataset DataFrame Label descriptions from IP mapping
    def process_isot(self, extended):
        print("Started ISOT Dataset Netflow Preprocessor!")
        print("     [1/3] Fixing hex values...")
        print("     [2/3] Filling empty fields...")
        print("     [3/3] Creating labels...")       
        
        # Iterate over each row
        for row in range(self.dataset.shape[0]):
            # [1/2]
            # Fix hex values in src and dest port fields
            src_port = str(self.dataset.at[row, 'Sport'])
            dest_port = str(self.dataset.at[row, 'Dport'])

            # Sets any hex value in port fields to 0
            # This is due to the port numbers not being used in any supervised learning, but we need to maintain formatting
            if "0x" in src_port:
                src_port = "0"
                self.dataset.at[row, 'Sport'] = src_port

            if "0x" in dest_port:
                dest_port = "0"
                self.dataset.at[row, 'Dport'] = dest_port

            
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
            # Create labels depending upon source IP Address
            src_mac = str(self.dataset.at[row, 'SrcMac'])
            
            if src_mac in self.botnet_mac_addresses:
                self.dataset.at[row, 'Label'] = str("Botnet")
            else:
                self.dataset.at[row, 'Label'] = str("Normal")
            
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

        
        
    # Function to write modified dataset to a new file
    def write_fixed_dataset(self, extended):
        if extended:
            dataset_dir = "../../Datasets/ISOT Botnet 2010/Pre-processed_Extended/isot_botnet.csv"
        else:
            dataset_dir = "../../Datasets/ISOT Botnet 2010/Pre-processed/isot_botnet.csv"
        
        print("Writing fixed dataset Dataframe to ", dataset_dir)
        
        try:
            self.dataset.to_csv(dataset_dir)
            print("File Write Successful!")
        except:
            print("ERROR!   ----->   to_csv - File Write Error")
    
    
    
    # Function to show the head of the object's loaded dataset
    def show_dataset_head(self):
        return self.dataset.head()
        
    
    # Count how many of each flows are present in the dataset
    def count_flows(self):
        botnet_flows = 0
        normal_flows = 0
        
        for row in range(self.dataset.shape[0]):    
            if self.dataset.at[row, 'Label'] == "Normal":
                normal_flows += 1
            if self.dataset.at[row, 'Label'] == "Botnet":
                botnet_flows += 1
                
        return botnet_flows, normal_flows