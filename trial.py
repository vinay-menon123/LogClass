# filename1 = "csv_to_txt_converted.txt"
# filename2 = open("words.txt","w")
# with open(filename1) as file:
#     for line in file:
#         filename2.write(line[17:])


# file = open("words.txt","r")
# f = open("occurences.txt","w")

# data = file.readlines()
# print(len(data))
# count = 0
# for i in range(len(data)):
#     f.write(str(data.count(data[i])))
#     f.write("\n")

# print(data)   
# print(data.count(data[0]))
# f.close()
# file.close()

#from ssl import OP_NO_RENEGOTIATION



# f_r = open("result_APP1.txt","r")
# f = open("weight_APP1.txt","w")

# #data = file.readlines()
# data_r = f_r.readlines()
# #print(len(data))
# # print(data[45][:16]=="Seq/QuantXXXXXXX")
# count = 0
# #print(data_r[5028])
# #print(data_r[5028]=="Seq/Quant\n")
# for i in range(len(data_r)):
#     if(data_r[i]=="Seq/Quant\n" or data_r[i]=="NewTemplate\n"):
#         count+=1
#         #print("hmm")
#         f.write("1\n")
#         continue
#     else:
#        f.write("-1\n")
# # f.close()

# # #print(count)
# # f = open("weight2.txt","w")
# # for i in range(len(data_r)):
# #     f.write(data1[i])
# #     f.write("\n")

# f.close()
# f_r.close()
# import pandas as pd

# df = pd.read_csv("LogClass/data/intern/aPP1/test_2022-04-26_07_59_40.csv")

# # df.drop(["timestamp","id","raw_logs","log_file_name","sorted_raw_logs","original_index","to_keep","danger","danger_word","template","template_id","score","variables","encoded_nums","encoded_cats","categorical_variables","base_score","rare_sequence","sequence","danger_base_score"],axis=1,inplace=True)
         
# # print(df.head())

# df["result"].to_csv('result_APP1.txt', sep="\n", index=False, header = False)
# df["level"].to_csv('weight2.txt', sep="\n", index=False, header = False)

import eif

#clf = eif.iForest()