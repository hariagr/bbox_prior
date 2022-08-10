import csv

if __name__ == '__main__':

    classes = ['leuko', 'eryth', 'epith', 'epithn', 'cryst', 'cast', 'mycete']
    count = [0, 0, 0, 0, 0, 0, 0]

    for ind, cl in enumerate(classes):
        with open('train_reduced.csv') as trn_file:
            read_file = csv.reader(trn_file)
            for line in read_file:
                if line[5] == cl:
                    count[ind] += 1
    
    print(count)
    count = [i*0.9 for i in count] 
    print(count)
    points = []
    boxes = []
    
    pclasses = ['p-leuko', 'p-eryth', 'p-epith', 'p-epithn', 'p-cryst', 'p-cast', 'p-mycete']
    # for ind, cl in enumerate(classes):
    with open('train_reduced.csv') as trn_file:
        read_file = csv.reader(trn_file)
        for line in read_file:
            for i,cls in enumerate(classes):
                if cls == line[5]:
                    ind = i
                    break
            if count[ind] > 0 and line[5] != 'label':
                count[ind] -= 1
                xx = str(float(line[1])+(float(line[3])-float(line[1]))/2)
                yy = str(float(line[2])+(float(line[4])-float(line[2]))/2)
                points.append([line[0],xx,yy,xx,yy,line[5]])
            elif count[ind] <= 0:
                boxes.append(line)

    # Opens a csv file
    with open('reduced_boxes_only_10.csv', 'a') as trn_file:
        writer = csv.writer(trn_file)
        writer.writerow(['image', 'xmin', 'ymin', 'xmax', 'ymax','label'])
        for line in boxes:
            writer.writerow(line)

    with open('reduced_points_only_10.csv', 'a') as trn_file:
        writer = csv.writer(trn_file)
        writer.writerow(['image', 'xmin', 'ymin', 'xmax', 'ymax','label'])
        for line in points:
            writer.writerow(line)
