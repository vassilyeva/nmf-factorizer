
original_text 		= 'datasets/Freebase15k/entityWords.txt'
new_text = 'datasets/Test/entityWords.txt'
new_vocab 		= 'datasets/Test/word2id.txt'


fout = open(new_text, 'w')
words = set()

test_size = 10

count = 0
with open(original_text, 'r') as fin:
	for line in fin:
		if count == test_size: break
		text = (line.split('\t')[2]).split()[:20]
		words |= set(text)
		fout.write('\t'.join(['blah', 'blah', '']) + ' '.join(text) + '\n')
		count += 1
fin.close()
fout.close()

with open(new_vocab, 'w') as fout:
	for i, w in enumerate(words):
		fout.write(w + '\t' + str(i) + '\n')
fout.close()


