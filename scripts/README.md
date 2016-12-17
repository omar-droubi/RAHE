## Scripts
These are miscellaneous scripts to process the output of RAHE, described as follow:

1. **check_duplicate.py**: checks the output files for duplicate results.
2. **merge_results.py**: when processing the VOC dataset on multiple machines, this scripts merges the results in a single file for each class. Before running the script open it and modify the directories inside of it to match yours.
3. **avgArea.py**: Prints the average area convered by objects inside the image.
4. **NIObjects.py**: ooutput a MATLAB matrix contaitng the number of intersecting objects in an VOC dataset.
5. **objects.py**: outputs a MATALB matrix conating the number of objects in VOC dataset images.

To run these scripts is favorable to use Anaconda package to insure to missing libraries.