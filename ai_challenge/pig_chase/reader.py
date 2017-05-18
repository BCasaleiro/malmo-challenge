import sys

f = open(sys.argv[1],'r')
lines = f.read().split('\n')

val = []
step = []
for line in lines:
	line_split = line.split(' ')
	#print line_split
	if(line_split[3]=="Training/reward"):
		#print line_split[3]
		step.append(line_split[1])
		val.append(line_split[len(line_split)-1])

h1 = [0 for i in range(50)]
h2 = [0 for i in range(50)]
h3 = [0 for i in range(50)]
h4 = [0 for i in range(50)]

for i, v in enumerate(val):
	if(i<((len(val)/4))):
		h1[int(float(v))+25] = h1[int(float(v))+25] + 1 
	elif (i<((len(val)/4)*2)):
		h2[int(float(v))+25] = h2[int(float(v))+25] + 1
	elif (i<((len(val)/4)*3)):
		h3[int(float(v))+25] = h3[int(float(v))+25] + 1
	elif (i<((len(val)/4)*4)):
		h4[int(float(v))+25] = h4[int(float(v))+25] + 1


def print_array(array):
	for a in array:
		print a
	print "-----"

print_array(h1)
print_array(h2)
print_array(h3)
print_array(h4)
