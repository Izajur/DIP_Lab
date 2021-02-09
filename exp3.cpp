#include <bits/stdc++.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
using namespace cv;
using namespace std;

const int file_max = 16, width_max = 9, filter_max = 9;
int file_id = 0, width = 1, filter = 2;

string directory[] = {"Noisy_Images/Cameraman_Salt_Pepper_0_02.jpg", "Noisy_Images/Cameraman_Salt_Pepper_0_005.jpg",
					"Noisy_Images/Cameraman_Salt_Pepper_0_08.jpg", "Noisy_Images/Camerman_Gaussian_0.005.jpg", 
					"Noisy_Images/Camerman_Gaussian_0_05.jpg", "Noisy_Images/Pepper_Gaussian_0_01.jpg", 
					"Noisy_Images/Pepper_Gaussian_0_005.jpg", "Noisy_Images/Pepper_Salt_Pepper_0_02.jpg",
					"Noisy_Images/Pepper_Salt_Pepper_0_005.jpg", "Noisy_Images/Pepper_Salt_Pepper_0_08.jpg", 
					"Normal_Images/jetplane.jpg", "Normal_Images/lake.jpg", "Normal_Images/lena_gray_512.jpg",
					"Normal_Images/livingroom.jpg", "Normal_Images/mandril_gray.jpg", "Normal_Images/pirate.jpg",
					"Normal_Images/walkbridge.jpg"};

Mat input;

bool check(int x, int y, int n, int m) {
	if(x-width/2>=0 and x+width/2<n and y-width/2>=0 and y+width/2<m)
		return 1;
	return 0;
}

int truncate(int x) {
	if(x<0)
		return 0;
	if(x>255)
		return 255;
	return x;
} 

Mat mean_filter() {
	double kernel[width_max][width_max], size = width*width, value;
	for(int i=0; i<width; i++)
		for(int j=0; j<width; j++)
			kernel[i][j] = 1/size;

	Mat output = input.clone();

	for(int i=0; i<input.rows; i++) {
		for(int j=0; j<input.cols; j++) {
			if(!check(i,j,input.rows,input.cols)) {
				output.at<uchar>(i, j) = 0;
		    	continue;
			}
		    value = 0;
			for(int x=0; x<width; x++) {
				for(int y=0; y<width; y++) {
					value += (double)input.at<uchar>(i+x-width/2, j+y-width/2)*kernel[x][y];
				}
			}
			output.at<uchar>(i, j) = value;
		}
	}
	return output;
}

Mat median_filter() {
	Mat output = input.clone();

	int size = width*width;
	vector<int> arr (size); 

	for(int i=0; i<input.rows; i++) {
		for(int j=0; j<input.cols; j++) {
			if(!check(i,j,input.rows,input.cols)) {
				output.at<uchar>(i, j) = 0;
		    	continue;
			}
			for(int x=0, k=0; x<width; x++) {
				for(int y=0; y<width; y++) {
					arr[k++] = input.at<uchar>(i+x-width/2, j+y-width/2);
				}
			}
			
			sort(arr.begin(), arr.end());
			output.at<uchar>(i, j) = arr[size/2];
		}
	}
	return output;
}

Mat prewitt_filter() {
	int kernelH[width_max][width_max], kernelV[width_max][width_max], size = width*width, value1, value2;
	for(int i=0; i<width; i++) {
		for(int j=0; j<width; j++) {
			kernelV[i][j] = j>width/2 ? -1 : 1;
			kernelH[i][j] = i>width/2 ? -1 : 1;
			if(i==width/2)
				kernelH[i][j] = 0;
			if(j==width/2)
				kernelV[i][j] = 0;
		}
	}
	Mat output = input.clone();

	for(int i=0; i<input.rows; i++) {
		for(int j=0; j<input.cols; j++) {
			if(!check(i,j,input.rows,input.cols)) {
				output.at<uchar>(i, j) = 0;
		    	continue;
			}
		    value1 = value2 = 0;
			for(int x=0; x<width; x++) {
				for(int y=0; y<width; y++) {
					value1 += input.at<uchar>(i+x-width/2, j+y-width/2)*kernelH[x][y];
					value2 += input.at<uchar>(i+x-width/2, j+y-width/2)*kernelV[x][y];
				}
			}
			output.at<uchar>(i, j) = truncate(sqrt(value1*value1+value2*value2));
		}
	}
	return output;
}

Mat laplacian_filter() {
	double kernel[width_max][width_max] = {-1}, size = width*width, value;
	kernel[width/2][width/2] = width*width-1;

	Mat output = input.clone();

	for(int i=0; i<input.rows; i++) {
		for(int j=0; j<input.cols; j++) {
			if(!check(i,j,input.rows,input.cols)) {
				output.at<uchar>(i, j) = 0;
		    	continue;
			}
		    value = 0;
			for(int x=0; x<width; x++) {
				for(int y=0; y<width; y++) {
					value += (double)input.at<uchar>(i+x-width/2, j+y-width/2)*kernel[x][y];
				}
			}
			output.at<uchar>(i, j) = value;
		}
	}
	return output;
}



static void on_file_change(int, void*);
static void on_width_change(int, void*);
static void on_filter_change(int, void*);


static void on_file_change(int, void*) {
	input = imread(directory[file_id], 0);
	imshow("Tracker", input);
	imshow("OUTPUT", input);
}

static void on_width_change(int, void*) {
	if(!(width&1))
		width++;
	on_filter_change(filter, 0);
}

static void on_filter_change(int, void*) {
	Mat ouput;
	switch(filter) {
		case 1:
			ouput = mean_filter();
			break;
		case 2:
			ouput = median_filter();
			break;
		case 3:
			ouput = prewitt_filter();
			break;
		case 4:
			ouput = laplacian_filter();
			break;
		default:
			ouput = input;
	}
	imshow("OUTPUT", ouput);
}

int main() {
    
	namedWindow("Tracker", WINDOW_AUTOSIZE);
    createTrackbar("File Name", "Tracker", &file_id, file_max, on_file_change);
    createTrackbar("Kernel width", "Tracker", &width, width_max, on_width_change);
    createTrackbar("Filter Type", "Tracker", &filter, filter_max, on_filter_change); 
    on_file_change(0, 0);

    waitKey(0);
    return 0;
}