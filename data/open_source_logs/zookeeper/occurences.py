file = open("rawlog.log","r")

data = file.readlines()
print(data)
count = 0
l1 = []
for i in data:
    if i not in l1:
        l1.append(i)
file.close()
print(len(l1))