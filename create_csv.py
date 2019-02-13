import sys
import os.path

if __name__ == "__main__":
    if len(sys.argv) != 2:
	print "usage: create_csv <base_path>"
	sys.exit(1)

    BASE_PATH=sys.argv[1]
    SEPARATOR=";"
    label = 0

    Train = open("face_csv.txt","w")
    Test = open("face_csv_test.txt","w")

    for dirname, dirnames, filenames in os.walk(BASE_PATH):
        for subdirname in dirnames:
	    subject_path = os.path.join(dirname, subdirname)
	    count = 0
	    for filename in os.listdir(subject_path):
		count += 1
		abs_path = "%s/%s" % (subject_path, filename)
		if count == 10:
		    Test.write("%s%s%d\n" % (abs_path, SEPARATOR, int(subdirname)))
		else:
		    Train.write("%s%s%d\n" % (abs_path, SEPARATOR, int(subdirname)))
	    label = label+1
