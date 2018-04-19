#include "imgFeat.h"

void feat::detectHarrisCornersOpencv(const Mat& imgSrc, Mat& imgDst, int blockSize, int kSize, double alpha)
{
	Mat gray;
	if (imgSrc.channels() == 3)
	{
		cvtColor(imgSrc, gray, CV_BGR2GRAY);
	}
	else
	{
		gray = imgSrc.clone();
	}

	Mat cornerStrength;
	cornerHarris(gray, cornerStrength, blockSize, kSize, alpha);

	double maxStrength;
	double minStrength;
	minMaxLoc(cornerStrength, &minStrength, &maxStrength);

	Mat dilated;
	Mat localMax;
	Mat element = getStructuringElement(MORPH_RECT, Size(blockSize, blockSize));
	dilate(cornerStrength, dilated, element);
	compare(cornerStrength, dilated, localMax, CMP_EQ);

	Mat cornerMap;
    double qualityLevel = 0.01;
    double th = qualityLevel*maxStrength;
    threshold(cornerStrength, cornerMap, th, 255, THRESH_BINARY);
    cornerMap.convertTo(cornerMap, CV_8U);
    
    bitwise_and(cornerMap, localMax, cornerMap);

	imgDst = cornerMap.clone();
}

void feat::detectHarrisCorners(const Mat& imgSrc, Mat& imgDst, int blockSize, int kSize, double alpha)
{
	Mat gray;
	if (imgSrc.channels() == 3)
	{
		cvtColor(imgSrc, gray, CV_BGR2GRAY);
	}
	else
	{
		gray = imgSrc.clone();
	}
	gray.convertTo(gray, CV_64F);

	Mat xKernel = (Mat_<double>(1,3) << -1, 0, 1);
	Mat yKernel = xKernel.t();

	Mat Ix,Iy;
	filter2D(gray, Ix, CV_64F, xKernel);
	filter2D(gray, Iy, CV_64F, yKernel);

	Mat Ix2,Iy2,Ixy;
	Ix2 = Ix.mul(Ix);
	Iy2 = Iy.mul(Iy);
	Ixy = Ix.mul(Iy);

	Mat gaussKernel = getGaussianKernel(7, 1);
	filter2D(Ix2, Ix2, CV_64F, gaussKernel);
	filter2D(Iy2, Iy2, CV_64F, gaussKernel);
	filter2D(Ixy, Ixy, CV_64F, gaussKernel);
	

	Mat cornerStrength(gray.size(), gray.type());
	for (int i = 0; i < gray.rows; i++)
	{
		for (int j = 0; j < gray.cols; j++)
		{
			double det_m = Ix2.at<double>(i,j) * Iy2.at<double>(i,j) - Ixy.at<double>(i,j) * Ixy.at<double>(i,j);
			double trace_m = Ix2.at<double>(i,j) + Iy2.at<double>(i,j);
			cornerStrength.at<double>(i,j) = det_m - alpha * trace_m *trace_m;
		}
	}
	// threshold
	double maxStrength;
	minMaxLoc(cornerStrength, NULL, &maxStrength, NULL, NULL);
	Mat dilated;
	Mat localMax;
	Mat element = getStructuringElement(MORPH_RECT, Size(blockSize, blockSize));
	dilate(cornerStrength, dilated, element);
	compare(cornerStrength, dilated, localMax, CMP_EQ);

	Mat tmp1, tmp2, tmp3, tmp4, tmp5;
	cornerStrength.convertTo(tmp1, CV_8U);
	dilated.convertTo(tmp2, CV_8U);
	localMax.convertTo(tmp3, CV_8U);
	
	Mat cornerMap;
	double qualityLevel = 0.01;
	double thresh = qualityLevel * maxStrength;
	cornerMap = cornerStrength > thresh;
	cornerMap.convertTo(tmp4, CV_8U);
	bitwise_and(cornerMap, localMax, cornerMap);

	cornerMap.convertTo(tmp5, CV_8U);
	imshow("cornerStrength", tmp1);
	imshow("dilated", tmp2);
	imshow("localMax", tmp3);
	imshow("cornerMap1", tmp4);
	imshow("cornerMap2", tmp5);
	waitKey(-1);
	

	imgDst = cornerMap.clone();
	
}

void feat::drawCornerOnImage(Mat& image, const Mat&binary)
{
    Mat_<uchar>::const_iterator it = binary.begin<uchar>();
    Mat_<uchar>::const_iterator itd = binary.end<uchar>();
	int cornerNum = 0;
    for (int i = 0; it != itd; it++, i++)
    {
        if (*it)
		{
            circle(image, Point(i%image.cols, i / image.cols), 3, Scalar(0, 255, 0), 1);
			cornerNum += 1;
		}
    }
	std::cout << "Number of corners: " << cornerNum <<std::endl;
}

