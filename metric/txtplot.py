import matplotlib.pyplot as plt


# dataset = 'pleiades'
dataset = 'WV3'

# method1 = 'bic'
# method2 = 'LGCNet'
# method2 = 'SRCNN'
# method1 = 'PECNN'

# method3 = 'CARS_densely_B5_NCAT'
# method2 = 'CARS_densely_B5_T'
# method2 = 'CARS_densely_B8_T'

# method1 = 'CARS_flat_B5'
# method2 = 'CARS_densely_B102'
# method3 = 'CARS_densely_B10_T'

method1 = 'CARS_densely_B5_T'
method2 = 'CARS_densely_B8_T'
method3 = 'CARS_densely_B10_T'

# method1 = 'CARS_flat_B5_T'
# method2 = 'CARS_densely_B10_T2'
# method3 = 'CARS_densely_B10_TT'

# method1 = 'CARS_flat_B10'
input_txt1 = './test_log/{}_{}.txt'.format(dataset,method1)
input_txt2 = './test_log/{}_{}.txt'.format(dataset,method2)
input_txt3 = './test_log/{}_{}.txt'.format(dataset,method3)

# input_txt1 = f'./test_log/{dataset}_{method1}.txt'
# input_txt2 = f'./test_log/{dataset}_{method2}.txt'
# input_txt3 = f'./test_log/{dataset}_{method3}.txt'
x = []
y = []
z = []
w = []
r = []
f1 = open(input_txt1)
f2 = open(input_txt2)
f3 = open(input_txt3)
for line in f1 :
    
    
    line = line.strip('\n')
    line = line.lstrip()
    line = line.split(':')
    # print(line)
    # print(line[0].split('-'))
    # print(line[1].split(','))
    if line[0].split('-')[0] == 'Average':
        break
    else:
    
        x.append(int(line[0].split('-')[0].strip().split(' ')[-1]))
    
        # y.append(float(line[1].split(',')[0]))
        y.append(float(line[-2].split(',')[0]))
    # z.append(float(line[2].split(']')[0]))
    # w.append(float(line[3].split(']')[0]))
f1.close

for line in f2 :
    
    
    line = line.strip('\n')
    line = line.lstrip()
    line = line.split(':')
    # print(line)
    # print(line[0].split('-'))
    # print(line[1].split(','))
    if line[0].split('-')[0] == 'Average':
        break
    else:
    
        # x.append(int(line[0].split('-')[0].strip().split(' ')[-1]))
    
        # z.append(float(line[1].split(',')[0]))
        z.append(float(line[-2].split(',')[0]))
    # z.append(float(line[2].split(']')[0]))
    # w.append(float(line[3].split(']')[0]))
f2.close
for line in f3 :
    
    
    line = line.strip('\n')
    line = line.lstrip()
    line = line.split(':')
    # print(line)
    # print(line[0].split('-'))
    # print(line[1].split(','))
    if line[0].split('-')[0] == 'Average':
        break
    else:
    
        # x.append(int(line[0].split('-')[0].strip().split(' ')[-1]))
    
        # w.append(float(line[1].split(',')[0]))
        
        w.append(float(line[-2].split(',')[0]))
    # z.append(float(line[2].split(']')[0]))
    # w.append(float(line[3].split(']')[0]))
f3.close

plt.plot(x, y, marker = '.', label = 'B = 5')
plt.plot(x, z, marker = '+', label = 'B = 8')
plt.plot(x, w, marker = '+', label = 'B = 10')
plt.xticks(x[0:len(x):10], x[0:len(x):10], rotation = 45)
plt.margins(0)
plt.xlabel('Image number')
plt.ylabel('NIQE')
# plt.ylabel('PSNR(dB)')
plt.title('Test results on Pleiades dataset')
plt.tick_params(axis = 'both')
plt.show()