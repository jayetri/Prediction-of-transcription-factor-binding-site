dict={'A':"1,0,0,0","C":"0,1,0,0","G":"0,0,1,0","T":"0,0,0,1"}
inp = open("test.csv", "r")
out = open("proc_test.csv","w")
s=inp.read()
sequence=s.strip()
sequence=sequence.split('\n')
sequence=sequence[1:]
for line in sequence:
	#print line
	line=line.strip()
	line=line.split(",")
	#print line	
	#label=line[2]
	sequences=line[1]
	encoding=[]
	for seq in sequences:
			encoding.append(dict[seq])
	'''
	if label=="1":
		label="0,1"
	else:
		label="1,0"
	'''
	#hot_enc.append(label)
	encoding=",".join(encoding)+"\n"
	out.write(encoding)
out.close()
