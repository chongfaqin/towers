result=[]
with open("data/test","r") as file:
    for line in file.readlines():
        result.append(line.strip())

print(",".join(result))
