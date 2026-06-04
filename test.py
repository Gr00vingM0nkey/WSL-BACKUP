
import numpy as np
import pandas as pd
"""
Map_Size = 5
map = np.full((Map_Size,Map_Size), "0")
map[1,1]="1"#@
map[0,0]="2"#M

print(map)

def Move_Left(Character_Location):
        Character = map[Character_Location[0], Character_Location[1]]
        map[Character_Location[0], Character_Location[1]] = "0"
        map[Character_Location[0], max(Character_Location[1] - 1, 0)] = Character

Move_Left([1,1])

print(map)"""
"""
full = np.full(100, 0)
print(full)
print(np.linspace(0,len(full), len(full)).round())

print(
            "################################################\n"+
            "################################################\n"+
            "#################  NOT FOUND  ##################\n"+
            "################################################\n"+
            "################################################\n"

        )
"""


nvda = pd.read_csv("Finance_Data/NVDA.csv")
# print(nvda.head())
x = nvda["Date"]

print(x)

import pandas as pd

# Sample DataFrame
df = pd.DataFrame({'Full_Name': ['John Smith', 'Jane Doe', 'Alex Jones']})

# Split the string by space and assign to new columns
df['First_Name'] = df['Full_Name'].str.split(' ', n=1).str[0]
print(df)
