# morphing-with-delaunay
A Python project which aims to morph two given images by generating in-between frames as much as user wants. The project uses the concepts of affine transformation and Delaunay triangulation for calculations. This project is done under the guidance of [Dr. Anukriti Bansal](https://www.lnmiit.ac.in/Employee_ProfileNew.aspx?nDeptID=iagma) for MPA subject of [LNMIIT](https://www.lnmiit.ac.in/).

## Before execution
Python libraries needed for running the script:
* [Numpy](https://numpy.org/)
* [OpenCV](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_tutorials.html)
* [OS](https://docs.python.org/3/library/os.html)

## To execute
1. Open the terminal and change the directory to the folder where the code is present and use the following command: `python3 Morphing_with_Delaunay.py [-h] [-outf OUTFOLDER] [-m] <ipath> <fpath> <N>`
    
    1.1. Positional arguments:
    * `ipath` - Path of the initial image
    * `fpath` - Path of the final image
    * `N` - No. of frames to generate (must be greater than 1)
    
    1.2. Optional arguments
    * `-h`, `--help` - show the help message and exit
    * `-outf OUTFOLDER`, `--outfolder OUTFOLDER` - To specify output folder path. By default it is `Morphing_Output/` in the current directory.
    * `-m`, `--monochrome` - Ignore the color of images.
2. Select an equal number of points on both initial and final image in the same order and press any key if you are finished.
3. Triangulation will be displayed. To exit, press any key.
4. The Code is executing and progress bar can be seen on the terminal.
5. For video generation put the following code in the terminal - `ffmpeg -framerate 5 -i ./Morphing_Output/Frame_%d.jpg Morphing.mp4`
6. For more clarity you can watch [this](https://github.com/anshuljain21120/morphing-with-delaunay/raw/main/Executing%20Morphing_with_Delaunay.mp4).

### Some examples
**Basic** - `python3 Morphing_with_Delaunay.py ./Bush.jpg ./Clinton.jpg 5`

**With Specified Output folder** - `python3 Morphing_with_Delaunay.py ./Bush.jpg ./Clinton.jpg 5 -outf OUTPUT` OR `python3 Morphing_with_Delaunay.py ./Bush.jpg ./Clinton.jpg 5 -outfolder OUTPUT`

**With monochrome images** - `python3 Morphing_with_Delaunay.py ./Bush.jpg ./Clinton.jpg 5 -m` OR `python3 Morphing_with_Delaunay.py ./Bush.jpg ./Clinton.jpg 5 -monochrome`
