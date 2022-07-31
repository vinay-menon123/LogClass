# f = open("words2.txt","r")
# data = f.readlines()

# for i in range(len(data)):
#     data[i] = data[i][3:]
# f.close()
# f = open("words2.txt","w")
# for i in range(len(data)):
#     f.write(data[i])
# f.close()


# from multiprocessing import Pool
# from tqdm import tqdm

# def process_line(line):
#     label = line[0].strip()
#     msg = ' '.join(line[1].strip().split()[1:])
#   #  msg = remove_parameters(msg)
#     if msg:
#         msg = ' '.join((label, msg))
#         msg = ''.join((msg, '\n'))
#         return msg
#     return ''
# output = "Only_Inf.txt"
# with open(output, "w", encoding='latin-1') as f:
#     gtruth = "weight1.txt"
#     rawlog = "words1.txt"
#     with open(gtruth, 'r', encoding='latin-1') as IN:
#         line_count = sum(1 for line in IN)
#     with open(gtruth, 'r', encoding='latin-1') as in_gtruth:
#         with open(rawlog, 'r', encoding='latin-1') as in_log:
#             IN = zip(in_gtruth, in_log)
#             with Pool() as pool:
#                 results = pool.imap(process_line, IN, chunksize=10000)
#                 f.writelines(tqdm(results, total=line_count))



# output = open("Only_Inf.txt","w")
# gtruth  = open("weight1.txt","r")
# rawlog = open("words1.txt","r")

# data_t = gtruth.readlines()
# data_f = rawlog.readlines()
# for i in range(len(data_t)):
#     output.write(data_t[i].strip())
#     output.write(" ")
#     output.write(data_f[i])
#     #output.write("\n")

# output.close()
# gtruth.close()
# rawlog.close()


                
# filename1 = "words1.txt"
# filename2 = open("words1_mod.txt","w")
# count = 0
# with open(filename1) as file:
#     for line in file:
#         if(line[0:3]=="END"):
#             filename2.write(line[4:])
#         if(line[0:5]=="START"):
#             filename2.write(line[6:])
#         if(line[0:4]=="INFO"):
#             filename2.write(line[5:])
#         if(line[0:6]=="REPORT"):
#             filename2.write(line[7:])
# print(count)
# filename2.close()