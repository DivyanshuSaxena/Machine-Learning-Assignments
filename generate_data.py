with open('ass2_data/test.json') as f:
    content = f.readlines()
content = [x.strip() for x in content]

with open('ass2_data/test.json', 'w') as f:
    f.write('[')
    for item in content[:-1]:
        f.write("%s, \n" % item)
    f.write("%s ]" % content[-1])