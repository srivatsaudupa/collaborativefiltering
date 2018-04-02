from cosine_for_item import Cosine_For_Item as Cosine
import sys

def main():
	if len(sys.argv) == 1:
		K = 200
	else:
		K = int(sys.argv[1])
	print(K)
	cosObj = Cosine("test5.txt", K)
	cosObj.compute()
	print("test5.txt processed. result5.txt produced")

	cosObj = Cosine("test10.txt", K)
	cosObj.compute()
	print("test10.txt processed. result10.txt produced")

	cosObj = Cosine("test20.txt", K)
	cosObj.compute()
	print("test20.txt processed. result20.txt produced")

main()