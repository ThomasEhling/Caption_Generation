# env/python3
# Libraries Importation

import argparse
from pickle import load
from pickle import dump
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu
from numpy import argmax
import pyprind
import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# Set the path as global variables
PATH_CAPTIONS = "../data/captions.txt"
PATH_TRAIN_IMAGES = "../data/Flickr_8k.trainImages.txt"
PATH_TEST_IMAGES = "../data/Flickr_8k.testImages.txt"

# marker for the beguining and the end of a caption
# we cheve chosen french words for start and end
# doing so, it will not be misinterpreted with english words
marker_start = 'debut'
marker_end = 'fin'


# extract the datasets from :
def get_flickr8k_datasets():

    # init the list :
    flicker_datasets = []

    # for our two files
    for i in range (2) :

        # we load the images :
        path = PATH_TRAIN_IMAGES if i == 0 else PATH_TEST_IMAGES
        my_list = open(path, 'r')
        my_file = my_list.read()
        my_list.close()

        my_set = []
        for elem in my_file.split('\n'):
            if len(elem) >= 1:
                val = elem.split('.')[0]
                my_set.append(val)
        flicker_datasets.append(my_set)

    return flicker_datasets


# create datasets of cleaned captions from the "captions.txt" file
# according to the ids found with the FLICKr 8k datasets
def get_captions_set(flickr8_datasets):

    # Open the caption files
    file_captions = open(PATH_CAPTIONS, 'r')
    my_captions = file_captions.read()
    file_captions.close()

    # Init the list of captions
    captions_sets = []

    # for each of our dataset : training and testing
    for i in range(2):

        # init the captions set
        capt_set = dict()

        # get the corresponding list of ids
        list_images = flickr8_datasets[i]

        for lines in my_captions.split('\n'):

            # extract the elements for each lines
            elems = lines.split()

            # the first element is the ids
            # the other ones are the caption associated with the id
            img_key = elems[0]

            # if the image is in the targeted data set
            if img_key in list_images:

                # we check if a value exists
                keys = capt_set.keys()
                if img_key not in keys:
                    capt_set[img_key] = list()

                # save the captions with the keywords "debut" and "fin"
                # this is french for "start" and "end"
                # we decided to use this because it will not be misinterpreted with english words
                capt_set[img_key].append(marker_start + ' ' + ' '.join(elems[1:]) + ' ' + marker_end)

        # add the resulting list of captions to our datasets
        captions_sets.append(capt_set)

    return captions_sets

# We attribute a number id. It will be used to compare with the tokenizer file which already have a mapping of integers
# to words
def convert_word_to_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


# Predict a caption with our model
def predict_caption(model, tokenizer, photo, max_len_caption):

    # initialize with cap start
    capt = marker_start

    # for every word in the sentence
    for i in range(max_len_caption):

        sequence = tokenizer.texts_to_sequences([capt])[0]
        sequence = pad_sequences([sequence], maxlen=max_len_caption)

        prediction = model.predict([photo, sequence], verbose=0)
        prediction = argmax(prediction)

        found = False
        my_word = None
        for word, index in tokenizer.word_index.items():
            if index == prediction:
                my_word = word
                found = True
                break

        if not found:
            break

        # append as input for generating the next word
        capt += ' ' + my_word

        # detect the marker end
        if my_word == marker_end:
            break

    return capt


def main():

    parser_description = "You are using a program to test your model which has already been train"
    parser = argparse.ArgumentParser(description=parser_description)
    parser.add_argument("--model", '-m',  help="You should Give the path of the model file (.h5)\
     you want to use", default='../data/model_our_4.h5')
    parser.add_argument("--features", '-f', help='You have to give the path of the features file you wan to use ',
                        default='../data/features_our.pkl')

    args = parser.parse_args()

    # get the model from the given path
    model_path = args.model
    model = load_model(model_path)

    # get the features from the given path
    features_path = args.features
    features = load(open(features_path, 'rb'))

    # Extract datasets ids from flickr8k
    # 0 for Training set
    # 1 for Testing set
    fliskr8k_datasets = get_flickr8k_datasets()
    print('\nWe have ', len(fliskr8k_datasets[0]), "images in our training set")
    print('We have ', len(fliskr8k_datasets[1]), "images in our testing set")

    # Get the features extracted from the images in the data-set
    test_features = {f: features[f] for f in fliskr8k_datasets[1]}
    print('\n TWe have', len(test_features)," features for the test")

    # get captions datasets from the preious ids
    # 0 for Training set
    # 1 for Testing set
    captions_sets = get_captions_set(fliskr8k_datasets)
    print('\nWe have ', len(captions_sets[0]), "captions in our training set")
    print('We have ', len(captions_sets[1]), "captions in our testing set")

    # extract the list of captions for training
    my_training_captions = list()
    for key in captions_sets[0].keys():
        [my_training_captions.append(d) for d in captions_sets[0][key]]

    # extract the list of captions for testing
    my_testing_captions = list()
    for key in captions_sets[1].keys():
        [my_testing_captions.append(d) for d in captions_sets[1][key]]

    # We set the maximum lenght of a captions to the longest caption
    max_len_caption = max(len(ind.split()) for ind in my_training_captions)
    print("\nThe maximum lenght for a caption is ", max_len_caption)

    # creation of a tokenizer
    my_tokenizer = Tokenizer()
    my_tokenizer.fit_on_texts(my_training_captions)
    dump(my_tokenizer, open("../data/tokenizer.pkl","wb"))

    # init the values for evaluation
    originals = []
    predictions= []

    # use pyprind to set a progression bar
    # n = 100
    # bar = pyprind.ProgBar(n, stream=sys.stdout)

    # for every element of our testing set
    for img_id, img_captions in captions_sets[1].items():

        # we save the originals captions
        originals.append([d.split() for d in img_captions])

        prediction = predict_caption(model, my_tokenizer, test_features[img_id], max_len_caption)

        # and save it
        predictions.append(prediction.split())

        # bar.update()

    # calculate BLEU score
    print("Objective BLEU-1 : 0.69")
    print('Final BLEU-1:', corpus_bleu(originals, predictions, weights=(1.0, 0, 0, 0)))


if __name__ == '__main__':
    main()
