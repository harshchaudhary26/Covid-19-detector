We address the issue of COVID-19 detection from X-ray images in this paper. Since the major problem was relating to Dataset, in our project we have used a varied dataset which comprises of Chest X-Ray images of people whose age varies from age 14 to 84. It also has an almost 1:1 Male to Female ratio. We use Chest X-Ray images instead of CT Scan Images to reduce the processing time of the model. We improve the accuracy and reduce the loss of the model by filtering only one view of the Chest X-Ray which is the PA view (instead of the common AP view). In this study, we developed a CNN model for COVID-19 identification from chest radiography pictures under the assumption that radiologists must first distinguish COVID-19 X-rays from normal chest X rays before classifying and detecting COVID-19 to isolate and treat the patient effectively. As a result, we decide to use CNN to make one of the following predictions: Normal or COVID-19. With these needs in mind, we created our straightforward CNN architecture, which consists of four parallel layers with a total of 16 filters in three distinct sizes (3-by- 3, 5-by-5, and 9-by 9). Then, rectified linear unit (ReLU) and batch normalization are used. Future research directions, and in progress work, contain segmenting the lung region from chest X-rays and removing other artefact such as text and medical device traces on chest X-rays. Data from other sources need to be incorporate to build CNN models that can be generalized and not biased towards a specific country, such as China/Italy, or a targeted population.
