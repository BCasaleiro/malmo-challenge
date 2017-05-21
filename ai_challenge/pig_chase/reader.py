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

h1 = [0 for i in range(51)]

for i, v in enumerate(val):
	h1[int(float(v))+25] = h1[int(float(v))+25] + 1 

def print_array(array):
	for a in array:
		print a
	print "-----"

print_array(h1)
