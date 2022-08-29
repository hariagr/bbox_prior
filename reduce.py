import csv
import random

if __name__ == '__main__':

    count = []
    reduced = []

    with open('train_reduced.csv') as trn_file:
        read_file = csv.reader(trn_file)
        for line in read_file:
            count.append(line[0])
    
    count = list(set(count))
    ll = len(count)
    print(ll)
    
    imgs = random.sample(range(0,ll), int(ll/10))
    # for x in imgs:
    #     reduced.append(count[x])

    for i,x in enumerate(count):
        if i not in imgs:
            reduced.append(x)
    # count = count[:int(ll/20)]
    print(len(reduced))

    points = []
    boxes = []

    # Opens a csv file
    with open('train_reduced.csv') as rd_file:
        rd_file = csv.reader(rd_file)
        with open('reduced_train_points_10.csv', 'a') as trn_file:
            writer = csv.writer(trn_file)
            writer.writerow(['image', 'xmin', 'ymin', 'xmax', 'ymax', 'label'])
            for line in rd_file:
                if line[0] in reduced and line[0] != 'image':
                    xx = str(float(line[1])+(float(line[3])-float(line[1]))/2)
                    yy = str(float(line[2])+(float(line[4])-float(line[2]))/2)
                    writer.writerow([line[0],xx,yy,xx,yy,line[5]])

    # # Opens a csv file
    # with open('boxes_only_.csv', 'a') as trn_file:
    #     writer = csv.writer(trn_file)
    #     writer.writerow(('image', 'xmin', 'ymin', 'xmax', 'ymax','label'))
    #     for line in boxes:
    #         writer.writerow(line)

    # with open('points_only_30.csv', 'a') as trn_file:
    #     writer = csv.writer(trn_file)
    #     writer.writerow(('image', 'xmin', 'ymin', 'xmax', 'ymax','label'))
    #     for line in points:
    #         writer.writerow(line)