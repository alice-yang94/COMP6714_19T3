import pickle
import time
import project_part1 as project_part1

with open('test_500docs.pickle', 'rb') as handle:
    contents_dict = pickle.load(handle)

start = time.time()
index = project_part1.InvertedIndex()
index.index_documents(contents_dict)
end = time.time()
diff = end - start
print(diff)