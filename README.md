# ct-lung-segmentations
Examples of medical image segmentation using publically available annoymized data from Kaggle and other online data sources

# Dataset creation

 - I suspect some of my issues around `TypeError: new() missing 1 required positional argument: 'x'` that I get when I run my [micro-CT segmentation notebook](mct_segmentation.ipynb) arise from improper creation of my dataset, including how to convert floats to ints and save as a file and then reload retaining value.
 - This is likely the case because when I use premade datasets like mnist I can train on my cloud machine, but when I use a homemade dataset, I get the error (even though my custom dataset works on my local PC!)
 - Hopefully if I follow other examples of making datasets I can overcome these challenges.

1. [chest CT dataset creation notebook](https://github.com/mandrakedrink/ChestCTSegmentation/blob/master/dataset_creation.ipynb)

 - Cites this notebook: [dicom jpg](https://www.kaggle.com/rashmibanthia/dicom-jpg)
