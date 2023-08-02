import pandas as pd
import numpy as np
import os

from dataloaders.dataloader_base import BASE_DATA

# ========================================       BOSCH_HAR_DATA               =============================

class BOSCH_HAR_DATA(BASE_DATA):
    
    def __init__(self, args):
        self.used_cols = None
        
        self.col_names = ['rawMagnetometer_iPhone_magn_x',
                          'rawMagnetometer_iPhone_magn_y',
                          'rawMagnetometer_iPhone_magn_z', 
                          'deviceMotion_iPhone_roll',
                          'deviceMotion_iPhone_pitch', 
                          'deviceMotion_iPhone_yaw',
                          'deviceMotion_iPhone_gravity_x',
                          'deviceMotion_iPhone_gravity_y',
                          'deviceMotion_iPhone_garvity_z',
                          'deviceMotion_iPhone_magneticField_x',
                          'deviceMotion_iPhone_magneticField_y',
                          'deviceMotion_iPhone_magneticField_z',
                          'deviceMotion_iPhone_acc_x',
                          'deviceMotion_iPhone_acc_y',
                          'deviceMotion_iPhone_acc_z',
                          'deviceMotion_iPhone_rotationRate_x',
                          'deviceMotion_iPhone_rotationRate_y',
                          'deviceMotion_iPhone_rotationRate_z',
                          'relAltitude_AppleWatch_pressure',
                          'location_iPhone_speed',
                          'deviceMotion_AppleWatch_roll',
                          'deviceMotion_AppleWatch_pitch',
                          'deviceMotion_AppleWatch_yaw',
                          'deviceMotion_AppleWatch_gravity_x',
                          'deviceMotion_AppleWatch_gravity_y',
                          'deviceMotion_AppleWatch_garvity_z',
                          'deviceMotion_AppleWatch_magneticField_x',
                          'deviceMotion_AppleWatch_magneticField_y',
                          'deviceMotion_AppleWatch_magneticField_z',
                          'deviceMotion_AppleWatch_acc_x',
                          'deviceMotion_AppleWatch_acc_y',
                          'deviceMotion_AppleWatch_acc_z',
                          'deviceMotion_AppleWatch_rotationRate_x',
                          'deviceMotion_AppleWatch_rotationRate_y',
                          'deviceMotion_AppleWatch_rotationRate_z',
                          'rawRotationRate_iPhone_acc_x',
                          'rawRotationRate_iPhone_acc_y',
                          'rawRotationRate_iPhone_acc_z',
                          'location_AppleWatch_speed',
                          'relAltitude_iPhone_pressure',
                          'rawAccelerometer_iPhone_acc_x',
                          'rawAccelerometer_iPhone_acc_y',
                          'rawAccelerometer_iPhone_acc_z',
                          'WatchWorkout_HeartRate_AppleWatch_HeartRate',
                          'activity_id',
                          'sub',
                          'sub_id']
        
        
        self.pos_filter         = None
        self.sensor_filter      = None
        self.selected_cols      = None
        
        self.label_map = [(0, 'car'),
                          (1, "walking"), 
                          (2, "bike"), 
                          (3, "train"),
                          (4, "motorcycle")
                         ]
        self.drop_activities = []


        self.train_keys   = []
        self.vali_keys    = []
        self.test_keys    = []
        
        self.LOCV_keys = [
            [
                "0" ,
                "DFB2D69D-FEBF-4930-95AF-3B2EE3826645",
                "4A24314C-555A-4B9A-B385-FEC0C8C34CBE",
                "44059DF6-1526-4BBB-9BEF-C552C211C7D5",
                "680CF97D-E7A4-40E7-A1C0-9C5E23384C4D",
                "BF85AAD3-AD2F-4A1B-A33A-2F34C2DA2D3B",
                "870B4438-9DB8-4AD0-BBBB-3FF4CA3180EC"
            ],
            
            [
                "0B81EFAB-5E21-4AF4-98D8-7901EA8151EB",
                 "15FCC77E-B556-441C-812B-157039B69D0F",
                 "1609BE08-1A60-4FDD-B1B9-7D917E48C8E6",
                 "067EBE88-7885-480A-B3DE-F2B38C63E9C2",
                 "B84F03EC-DF46-4099-9C5E-74DF63D533B3"
            ]
        ]
        self.all_keys = ["0" , 
                         "DFB2D69D-FEBF-4930-95AF-3B2EE3826645",
                         "4A24314C-555A-4B9A-B385-FEC0C8C34CBE",
                         "44059DF6-1526-4BBB-9BEF-C552C211C7D5",
                         "680CF97D-E7A4-40E7-A1C0-9C5E23384C4D",
                         "BF85AAD3-AD2F-4A1B-A33A-2F34C2DA2D3B",
                         "870B4438-9DB8-4AD0-BBBB-3FF4CA3180EC",
                         "0B81EFAB-5E21-4AF4-98D8-7901EA8151EB",
                         "15FCC77E-B556-441C-812B-157039B69D0F",
                         "1609BE08-1A60-4FDD-B1B9-7D917E48C8E6",
                         "067EBE88-7885-480A-B3DE-F2B38C63E9C2",
                         "B84F03EC-DF46-4099-9C5E-74DF63D533B3"]
        self.sub_ids_of_each_sub = {}

        self.exp_mode     = args.exp_mode
        self.split_tag = "sub"
        
        self.file_encoding = {}  # no use 
        

        self.labelToId = {int(x[0]): i for i, x in enumerate(self.label_map)}
        self.all_labels = list(range(len(self.label_map)))

        self.drop_activities = [self.labelToId[i] for i in self.drop_activities]
        self.no_drop_activites = [item for item in self.all_labels if item not in self.drop_activities]
        super(BOSCH_HAR_DATA, self).__init__(args)
        
    def load_all_the_data(self, root_path):

        print(" ----------------------- load all the data -------------------")
        
        df = pd.read_csv(os.path.join(root_path,"d_merged_47.csv"))
        df["label"] = df["label"].astype(str)
        df["id"] = df["id"].astype(str)
        df=df.iloc[:,1:]
        
        
        
        df.columns = self.col_names
  
        for sub in df["sub"].unique():
            temp_sub = df[df["sub"]==sub]
            self.sub_ids_of_each_sub[sub] = list(temp_sub["sub_id"].unique())
            
        df = df.set_index('sub_id')
        df = df[list(df.columns)[:-2]+["sub"]+["activity_id"]]
        
        label_mapping = {item[1]:item[0] for item in self.label_map}
        
        df["activity_id"] = df["activity_id"].map(label_mapping)
        df["activity_id"] = df["activity_id"].map(self.labelToId)
        df.dropna(inplace=True)
        data_y = df.iloc[:,-1]
        data_x = df.iloc[:,:-1]

        data_x = data_x.reset_index()
        # sub_id, sensor1, sensor2... sensorn, sub, 

        return data_x, data_y