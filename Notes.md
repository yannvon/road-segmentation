# Notes

## Exploratory Data Analysis

parkings can be mistaken for roads

objects can cover the roads (tree?)

roads are not necessarily horizontal or vertical 

##Links 
https://www.azavea.com/blog/2017/05/30/deep-learning-on-aerial-imagery/



## Keras things to consider

- validation data used by keras are the last elements in the numpy array given, maybe shuffle first?
- look at shuffle option for fit
- look at class weight option to counter balance under representation of roads !
- 



##Things to write in the report

- Why a CNN? eg why not regression
- cite paper and articles to justify
- state of the art
- in the intro, our contribution (do a model, compare with transfer learning)
- expliquer methodologie (baseline, relevant things we did)
- data augmentation (shift, rotation, random sampling)
- window
- overfitting
- weight (because 75% of background)

