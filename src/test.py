# env/python3
# Libraries Importation

import argparse
import sys
from pickle import load
from keras.preprocessing.sequence import pad_sequences
from numpy import argmax
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.models import load_model
from keras.models import Model

# Specific for Mac to resolve conflict
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# Markers initialization : these are the french version for start and end (we are french).
# Doing that won't cause any misunderstanding of the caption
cap_start = 'debut'
cap_end = 'fin'

########################################################################################################################
########################################################################################################################
# initialization of  the cnn values

my_optimizer = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

loss_functions = ['categorical_crossentropy', 'mean_squared_error', 'mean_absolute_error', 'squared_hinge']
my_loss = loss_functions[0]

activation_functions = ['relu', 'softmax', 'sigmoid', 'elu']
my_activation = activation_functions[0]
my_activation_1 = activation_functions[1]

my_pool_size = (2, 2)
my_stride = (2, 2)

# Careful ! we have 5 block in our model, so we need 5 values here
receptive_fields = [64, 128, 256, 512, 512]

# Careful ! we have 3 block in our model, so we need 3 values here
dense_values = [4096, 4096, 1000]

dropout = 0.5


# create the model
def create_model():

    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
    model.add(Conv2D(receptive_fields[0], (3, 3), activation=my_activation))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(receptive_fields[0], (3, 3), activation=my_activation))
    model.add(MaxPooling2D(pool_size=my_pool_size, strides=my_stride))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(receptive_fields[1], (3, 3), activation=my_activation))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(receptive_fields[1], (3, 3), activation=my_activation))
    model.add(MaxPooling2D(pool_size=my_pool_size, strides=my_stride))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(receptive_fields[2], (3, 3), activation=my_activation))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(receptive_fields[2], (3, 3), activation=my_activation))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(receptive_fields[2], (3, 3), activation=my_activation))
    model.add(MaxPooling2D(pool_size=my_pool_size, strides=my_stride))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(receptive_fields[3], (3, 3), activation=my_activation))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(receptive_fields[3], (3, 3), activation=my_activation))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(receptive_fields[3], (3, 3), activation=my_activation))
    model.add(MaxPooling2D(pool_size=my_pool_size, strides=my_stride))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(receptive_fields[4], (3, 3), activation=my_activation))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(receptive_fields[4], (3, 3), activation=my_activation))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(receptive_fields[4], (3, 3), activation=my_activation))
    model.add(MaxPooling2D(pool_size=my_pool_size, strides=my_stride))

    model.add(Flatten())
    model.add(Dense(dense_values[0], activation=my_activation))
    model.add(Dropout(dropout))
    model.add(Dense(dense_values[1], activation=my_activation))
    #     model.add(Dropout(dropout))
    #     model.add(Dense(dense_values[2], activation=my_activation_1))

    return model

########################################################################################################################
########################################################################################################################


# create our own model
def load_our_model():
    model = create_model()
    model.compile(optimizer=my_optimizer, loss=my_loss)

    # re-structure the model
    print(model.layers.pop())
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)

    return model


# load a pre-made model
def load_vgg_model():
    # get the vgg model from Keras
    model = VGG16()

    # re-structure the model to our need
    print(model.layers.pop())
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)

    return model


########################################################################################################################
########################################################################################################################


# Predict a caption with our model
def predict_caption(model,  picture_features, tokenizer, max_caption_size):

    # Caption initialization => should math with the function prepare_caption (clean)
    cap = cap_start

    # iterate over the whole length of the sequence
    for i in range(max_caption_size):

        sequence = tokenizer.texts_to_sequences([cap])[0]
        sequence = pad_sequences([sequence], maxlen=max_caption_size)

        predicted_caps = model.predict([picture_features, sequence], verbose=0)
        predicted_caps = argmax(predicted_caps)

        # Research of the word based on the id in the token file
        found = False
        my_word = None
        for word, index in tokenizer.word_index.items():
            if index == predicted_caps:
                my_word = word
                found = True
                break
        # If nothing is found
        if not found:
            break

        # Construct the caption by adding a word with a space each time
        cap += ' ' + my_word
        # if the word end is meet => end of the caption. Stop when the final reference is meet.
        if my_word == cap_end:
            break

    return cap


########################################################################################################################
########################################################################################################################


def main():

    parser_description = "You are using  an amazing program for generating a caption on a picture"

    parser = argparse.ArgumentParser(description=parser_description)
    parser.add_argument('--picture', '-p', help="Give the path of the picture as 1st argument",
                        default=None)
    parser.add_argument("--token", '-t',  help="You shouldGive the path of the tokenizer-file file\
     you want to use", default='../data/tokenizer.pkl')
    parser.add_argument("--model", '-m',  help="You shouldGive the path of the model file (.h5)\
     you want to use", default='../data/model_our_4.h5')

    args = parser.parse_args()
    picture_path = args.picture
    token_path = args.token
    model_path = args.model

    # It should not work if nothing is add
    if picture_path is None:
        parser.print_help()
        sys.exit()

    # load the tokenizer
    tokenizer = load(open(token_path, 'rb'))
    # pre-define the max sequence length (from training)
    max_length = 34
    # load the model from the path of the .h5 file which contains information of the model
    model = load_model(model_path)

    # prepare the picture and extraction of features
    user_choice = input("Do you want to use our model features (Y) or VGG16 features (N) ?")
    if user_choice == "Y" or user_choice == "":
        model_features = load_our_model()
    else:
        model_features = load_vgg_model()

    # Load the picture with a size which fit with the model (224,224) and extract the features from it
    image = load_img(picture_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    # Features extraction
    features_picture = model_features.predict(image, verbose=0)

    # Predict a caption link to the features extracted
    predicted_caption = predict_caption(model, features_picture, tokenizer, max_length)
    print("\n The caption of your picture is probably:\n")
    print(predicted_caption)
    print('\n')

########################################################################################################################
########################################################################################################################


if __name__ == '__main__':
    main()


########################################################################################################################
########################################################################################################################
