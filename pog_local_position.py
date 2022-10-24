import json
import numpy as np

with open("pog_local_position.json",'r') as file:
    product_local_pos=json.load(file)
    
cam_4=product_local_pos["cameras"][0]["cabinets"]
print(cam_4)
cam_7=product_local_pos["cameras"][1]["cabinets"]
cam_t=[cam_4,cam_7]


def pd_coord(cam_num, sh_num, row,col):
    row -= 1
    col -= 1
    list_result=[k["annotation"] for k in cam_t[cam_num] if k["shelfNum"]==sh_num]
    #list_result=np.array(list_result)
    #list_result.shape
    print(list_result)
    if list_result:
        list_result=list_result[0]
        print(list_result[row][row])
        x=list_result[row][col*3:(col+1)*3][0]
        y=list_result[row][col*3:(col+1)*3][1]
        return x,y
    else:
        return 0,0