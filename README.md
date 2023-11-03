# CS473 - ERD Clustering

## Group Name: Hintonians
## Members:
* Anvit Sinha
* Arnob Kazi
* Dan Ding
* Srikar Lanka
* Vinh Tran

The `CS473_Project_colab.ipynb` file in the repository is set up so that all the cells in the notebook can be run 
sequentially to generate the desired output in the zip folder with the name hintonians.zip.

The notebook clones this repo and sets up the files as per the default Configurations settings.

The only additional things needed are the grading.zip and the model weights zip files. 

The zip file for the model weights can be downloaded from:<br>
`https://purdue0-my.sharepoint.com/:f:/g/personal/sinha102_purdue_edu/EjeivgmR47xGvWgRJ0X_XF8BmcOoDrERSjFBJgiv691rPw?e=kYcbZs`
<br>This link is only valid till 12/22/2023.

The grading.zip file to be used for grading is expected to follow the structure 
described in Campuswire post #308 when unzipped, and should be placed in the content/ directory of colab:
```
grading/
│
├── sample_a/
│   ├── sample_a.png
│   └── question.txt
│
├── sample_b/
│   ├── sample_b.png
│   └── question.txt
│
├── sample_c/
│   ├── sample_c.png
│   └── question.txt
│
...
│
└── sample_j/
    ├── sample_j.png
    └── question.txt
```


Before running the cells, make sure your content/ directory has the following files:<br>

```
content/
├── best_weights_ver5.zip
└── grading.zip
```

If the directory structure you are using is different, make sure to update the paths as needed in the Configuration cell.

At this point, running all cells will generate the required output.

<br> NOTE:  the `sample_data` directory present in colab by default is not used, but doesn't affect the 
functioning of the program even if it is not removed.