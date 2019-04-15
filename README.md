Feature matching based object detection on images using SIFT and homography transformation. Images used in the example below can be found in `data/desk` directory. Arguments passed to python `run_matching.py` script are:
1. Image containing full scene where objects might be placed (`data\desk\original_5.jpg`)
2. List of images containing cropped objects that you want to be found (`data\desk\asus_3.jpg data/desk/tissues_1.jpg`)
```
python matching\run_matching.py data\desk\original_5.jpg data\desk\asus_3.jpg data/desk/tissues_1.jpg
```

![ASUS matching](https://image.ibb.co/gb6Lro/match_desk1.jpg)
![Tissues matching](https://image.ibb.co/gQxtBo/match_desk2.jpg)
![Full object detection](https://image.ibb.co/gzB3Bo/match_desk_full.jpg)
