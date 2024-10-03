import sys

def extract(split):
    f = open(split+ '.txt', 'r')
    lines = f.readlines()
    f.close()
    f = open(split+ '.promptKeys', 'w')
    for line in lines:
        line = line.split('=')[0]
        f.write(line + '\n')
    f.close()



extract(sys.argv[1])

