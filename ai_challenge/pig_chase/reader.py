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

for v in val:
	print v

print "-----"

for s in step:
	print s