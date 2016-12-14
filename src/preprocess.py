import re
import string
import os
from nltk.stem import WordNetLemmatizer

data_dir = '/home/hchang65/Gutenberg/txt'
output_dir = '../data/holmes_train_data'
regex = re.compile('[%s]' % re.escape(string.punctuation))
lemmatizer = WordNetLemmatizer()

def tokenize(s):
    s = s.lower()
    s = regex.sub('', s)
    splits = s.split()
    tokens = []

    for token in splits:
        try:
            tokens.append(lemmatizer.lemmatize(token))
        except UnicodeDecodeError:
            continue
    # tokens = [lemmatizer.lemmatize(token) for token in splits]
    return ' '.join(tokens)

if __name__ == '__main__':
    # Read all input filenames
    files = []
    with open('../data/holmes_train_files.txt', 'r') as f:
        for line in f:
            files.append((os.path.join(data_dir, line.strip()), \
                          os.path.join(output_dir, line.strip())))

    for filename, out_filename in files:
        print 'Preprocessing %s ...' % filename
        out = open(out_filename, 'w')

        with open(filename, 'r') as f:
            flag = False
            sentence = []
            while True:
                line = f.readline()
                if not line: # EOF
                    break
        
                if not line.strip(): # empty line
                    if flag and sentence: # Ending a paragraph
                        out.write(' '.join(sentence) + '\n')
                        sentence = []
                        flag = False

                    continue

                sentence.append(tokenize(line.strip()))
                flag = True

        if sentence:
            out.write(' '.join(sentence) + '\n')
        out.close()


