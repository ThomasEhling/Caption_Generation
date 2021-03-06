# Caption_Generation
Deep Learning Photo Caption Generator

Final Project of the Computer Vision class at Illinois Institute of technology, Fall semester 2018.

Developed with **Benjamin Scialom.**

The whole project and code is based on the following tutorial : 
https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/

We implemented our own model for the feature extraction, based on the vgg16 architecture, to compare the final scores. The NLP part is exactly the same.

We are using google colab to accelerate the data preparation and training process.

The architecture between our files is the following :

<img src=https://imgur.com/lJnjydH.png width="600px"/>

If you wish to implement the project, please copy the "/data" folder into your drive and mount it locally.

The /data folder contains also all the model weigths.

The /doc folder contains both of our report, for any more details please refer to the poriginal tutorial or these reports.

The /src folder contains all the source code.

The main model architecture is :
<img src=https://imgur.com/MrYAQ61.png width="600px"/>

Our Final results are :
<img src=https://imgur.com/I8eLTkw.png width="600px"/>

We used the Flickr8K dataset, so we are require to cite here : M. Hodosh, P. Young and J. Hockenmaier (2013) "Framing Image Description as a Ranking Task: Data, Models and Evaluation Metrics", Journal of Artifical Intellegence Research, Volume 47, pages 853-899
http://www.jair.org/papers/paper3994.html
