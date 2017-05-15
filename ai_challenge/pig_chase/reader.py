

f = open('test01.txt','r')
lines = f.read().split('\n')
print lines

for i in range(len(lines)):
	line_split = line.split(' ')
	number = line_split[len(line_split)-1]
	print number