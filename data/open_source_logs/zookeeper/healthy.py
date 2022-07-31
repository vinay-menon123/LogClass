f_w = open("weight2_60.txt","w")
for i in range(18842):
    f_w.write("1\n")
f_w.close()
f_r = open("weight2_60.txt","r")
data = f_r.readlines()
for i in range(800,11305+800,1):
    data[i] = "-1\n"
f_r.close()
f_w = open("weight2_60.txt","w")
for i in range(18842):
    f_w.write(data[i])
f_w.close()
print(data.count("-1\n"))