import h5py as h5
import numpy as np
Squence_Path = "Zebrafish_Data/Fish_test_150.txt"
Label = []
Sequence_List = []
Label_List = []
List_A = [1, 0, 0, 0]
List_T = [0, 0, 1, 0]
List_C = [0, 0, 0, 1]
List_G = [0, 1, 0, 0]
List_N = [0, 0, 0, 0]
kk = 0
for line in open(Squence_Path):
    kk += 1
    print(kk)
    list_seq = line.split()
    Label.append(int(list_seq[len(list_seq)-1])) #创建Label
    Squence_Tem = []
    for i in range(len(list_seq)-1):
        if(list_seq[i] == 'A' or list_seq[i] == 'a'):
            Squence_Tem.append(List_A)
        elif(list_seq[i] == 'T' or list_seq[i] == 'U' or list_seq[i] == 't' or list_seq[i] == 'u'):
            Squence_Tem.append(List_T)
        elif(list_seq[i] == 'C' or list_seq[i] == 'c'):
            Squence_Tem.append(List_C)
        elif(list_seq[i] == 'G' or list_seq[i] == 'g'):
            Squence_Tem.append(List_G)
        else:
            Squence_Tem.append(List_N)
    Sequence_List.append(Squence_Tem)

F = h5.File("Zebrafish_Data/Fish_test_150.h5", "w")
Label = np.array(Label)
Sequence_List = np.array(Sequence_List)
print(Label.shape)
print(Sequence_List.shape)
F.create_dataset("Label", data=Label)
F.create_dataset("Sequence", data=Sequence_List)
F.close()