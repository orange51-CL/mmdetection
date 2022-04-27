num_blocks = [64, 128, 256, 512]
width_multiplier = [1, 1, 1, 2]
c = []

for i in range(4):
    print(i)
    c.append(int(num_blocks[i] * width_multiplier[i]))

print(c)
