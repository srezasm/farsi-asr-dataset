import ast
from glob import glob
import sys
from editdistpy import levenshtein

def distance(s1, s2):
    return levenshtein.distance(s1, s2, sys.maxsize)

def get_sentence(filename):
    with open(filename, 'r') as f:
        dictionary = ast.literal_eval(f.read())

    sentence = ''
    for r in dictionary['results']:
        if len(r['alternatives']) != 1:
            raise Exception(f'Unexpected number of alternatives: {r}')
        
        content = r['alternatives'][0]['content']

        if r in ['.', ',', '!', '?', '؟']:
            sentence += content
        else:
            sentence += ' ' + content

    return sentence

# json_files = glob('/home/srezas/Programming/projects/farsi-asr-dataset/samples/sm/*.json')
# json_files = sorted(json_files)

# sentences = []
# for json_file in json_files:
#     sentences.append(get_sentence(json_file))

sentence = get_sentence('/home/srezas/Programming/projects/farsi-asr-dataset/audio_validation/virgool.json')

with open('output.txt', 'w') as f:
    f.write(sentence)