import sys, json
# extract training data CSV from graphql dump
# python convert_to_csv.py ies-rectangle.json

def fixStr(messyStr):
	#                                             \ 
	return messyStr.replace("\\n","\n").replace("\\\"","\"")

inputfilename = sys.argv[1]
segments = inputfilename.split(".")
outfilename = "".join(segments[:-1]) + ".csv"

infile = open(inputfilename)

data = fixStr(json.load(infile)["data"]["trainingData"]["training"])
infile.close()
outfile = open(outfilename, 'w')
print(data,file=outfile)
outfile.close()