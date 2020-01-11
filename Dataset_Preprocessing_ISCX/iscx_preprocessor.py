#!/usr/bin/env python
# coding: utf-8

# Dataset Preprocessing class for ISCX netflows
#      get_dataset()
#      process_iscx()
#      count_flows()
#      write_fixed_dataset()
#      show_dataset_head()


# Import Pandas library for DataFrame and related functions
import pandas as pd

# Dataset preprocessor class
class iscx_preprocessor:
    botnet_ips = ['192.168.2.112', '131.202.243.84', '192.168.5.122', '198.164.30.2', '192.168.2.110', '192.168.4.118', '192.168.2.113', '192.168.1.103', '192.168.4.120', '192.168.2.112', '192.168.2.109', '192.168.2.105', '147.32.84.180', '147.32.84.170', '147.32.84.150', '147.32.84.140', '147.32.84.130', '147.32.84.160', '10.0.2.15', '192.168.106.141', '192.168.106.131', '172.16.253.130', '172.16.253.131', '172.16.253.129', '172.16.253.240', '74.78.117.238', '158.65.110.24', '192.168.3.35', '192.168.3.25', '192.168.3.65', '172.29.0.116', '172.29.0.109', '172.16.253.132', '192.168.248.165', '10.37.130.4']

    
    def __init__(self, input_path):
        self.path = input_path
        self.dataset = 0
        
        print("Created iscx_preprocessor object")
    
    def __del__(self):
        print("Destroying iscx_preprocessor object")
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
    def process_iscx(self):
        print("Started ISCX Dataset Netflow Preprocessor!")
        print("     [1/2] Fixing hex values...")
        print("     [2/3] Filling empty fields...")
        print("     [3/3] Creating labels...")       
        
        # Iterate over each row
        for row in range(self.dataset.shape[0]):
            # [1/3]
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
            
            # Convert all port strings from hex to ints
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
            # Create labels depending upon source IP Address
            src_ip = str(self.dataset.at[row, 'SrcAddr'])
            
            if src_ip in self.botnet_ips:
                self.dataset.at[row, 'Label'] = str("Botnet")
            else:
                self.dataset.at[row, 'Label'] = str("Normal")
            
            
        print("Finished preprocessing!!!")

        
        
    # Function to write modified dataset to a new file
    def write_fixed_dataset(self):
        dataset_dir = "../../Datasets/ISCX Botnet 2014/Pre-processed/ISCX_Training.csv"
        
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
        
        # Iterate over each row
        for row in range(self.dataset.shape[0]):
            src_ip = str(self.dataset.at[row, 'SrcAddr'])
            
            if src_ip in self.botnet_ips:
                botnet_flows += 1
            else:
                normal_flows += 1
                
        return botnet_flows, normal_flows