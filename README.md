# Study of Hardware-Software Co-design of Vehicle(Car) detection using HOG and SVM

RnD project done under Prof. Sachin Patkar as a part of RnD course project (EE 692)

## Softwares Used

Vivado WebPACK 2019.1 

## Python codes 

Can be found inside ```CarDetection_python_codes``` directory and it is originally taken from:
[HOG-SVM-python](https://github.com/jianlong-yuan/HOG-SVM-python)

### Important links to the dataset
Information can be found [here](https://cogcomp.seas.upenn.edu/Data/Car/)

Dataset can be downloaded from [here](https://github.com/Menuka5/Data-sets-for-opencv-classifier-training), go inside ```Other Image Datasets collected``` folder

### Training

	- Uses train.py file
	- If you have scikit learn version 0.22 and cv2 version as 4.1.2 then you can directly use the file svm.model otherwise, you need to re-train and create your own model file
	
### Testing
	-Use test.py file
	



## Create HOG-SVM accelerator IP form HLS

Using the normal HLS synthesis flow create, synthesize and export the IP 

## Create Vivado project using TCL


```tcl

source <path-to-build.tcl>/build.tcl

```
### Running Behavioral Simulation using MicroBlaze

I have published an article on medium.com for how I ran Behavioral Simulation.

Link to the post: [Behavioral RTL Simulation with MicroBlaze](https://medium.com/@sapphire.sharma1996/behavioral-rtl-simulation-with-microblaze-131671e86f04)


## Debugging any error encountered while sourcing build.tcl
	- ERROR: [BD_TCL-115] The following IPs are not found in the IP Catalog.
	Solution:
	- Change your IP repository path of HOG-SVM accelerator obtained from HLS in line no 153 of build.tcl




## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Some minor issues that I faced while uploading
[warning: LF will be replaced by CRLF and Special characters appear](https://github.com/gobuffalo/buffalo/issues/1189)

<!--## License
[MIT](https://choosealicense.com/licenses/mit/) -->
